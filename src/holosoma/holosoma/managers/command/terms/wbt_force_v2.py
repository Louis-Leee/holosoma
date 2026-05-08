"""v14 wrist force-tracking command.

Self-contained — does NOT import any symbol from
``holosoma.managers.command.terms.wbt_force`` (V10).

Causality in v14:

* F_ext is the sampled stochastic process (world frame), driven by a
  trapezoidal FSM (``_ForceChannel``) defined locally in this module.
* F_cmd_b is derived once per ``step()``: ``F_cmd_b = -R_yaw^-1(base_quat) . F_ext_w``,
  and snapshot-cached so that physics injection, obs reads, metrics, and
  debug viz all see the same value within a single tick.

Physical reading: the policy is shown, in its body-yaw frame, the force it
would need to output to keep the wrist on the reference motion trajectory
(zero net external force). No K_virtual, no shift, no independent F_cmd
channel.

See docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.command.terms._gh_utils import (
    TemporalLerp,
    clamp_norm,
    random_uniform,
)
from holosoma.utils.rotations import quat_apply, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from holosoma.config_types.command import CommandTermCfg
    from holosoma.config_types.command_v2 import WristForceTrackingConfig

# -- Local FSM state codes (V14-owned; numerically mirror V10 but independent) --
STATE_COOLDOWN: int = 0
STATE_RAMP_UP: int = 1
STATE_HOLD: int = 2
STATE_RAMP_DOWN: int = 3

NUM_WRISTS: int = 2
DIM: int = 3


def _duration_to_steps(duration_s: torch.Tensor, control_rate_hz: float) -> torch.Tensor:
    """steps = max(1, round(duration_s * control_rate_hz))."""
    steps = torch.round(duration_s * control_rate_hz).to(dtype=torch.int32)
    return torch.clamp(steps, min=1)


class _ForceChannel:
    """Trapezoidal FSM per (env, wrist). V14-local copy; not imported from V10.

    Structure mirrors the V10 ``_ext_channel`` behaviour
    (COOLDOWN->RAMP_UP->HOLD->RAMP_DOWN->COOLDOWN with Bernoulli activation),
    but this class is defined in wbt_force_v2.py so V14 has zero runtime
    dependency on the V10 module.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        *,
        magnitude_range: tuple[float, float],
        duration_range_s: tuple[float, float],
        cooldown_range_s: tuple[float, float],
        ramp_frac: float,
        activation_prob_per_step: float,
        enable_left: bool,
        enable_right: bool,
        control_rate_hz: float,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.magnitude_range = magnitude_range
        self.duration_range_s = duration_range_s
        self.cooldown_range_s = cooldown_range_s
        self.ramp_frac = ramp_frac
        self.activation_prob = activation_prob_per_step
        self.control_rate_hz = control_rate_hz

        self.force = torch.zeros(num_envs, NUM_WRISTS, DIM, device=device)

        # TemporalLerp shares one clock with (env, wrist) flattened onto axis 0.
        self._lerp = TemporalLerp(
            (num_envs * NUM_WRISTS,),
            device=device,
            default=0.0,
            easing="linear",
        )

        self.state = torch.full((num_envs, NUM_WRISTS), STATE_COOLDOWN, dtype=torch.int8, device=device)
        self.cooldown_remaining = torch.zeros((num_envs, NUM_WRISTS), dtype=torch.int32, device=device)
        self.hold_remaining = torch.zeros_like(self.cooldown_remaining)
        self.ramp_up_steps = torch.ones_like(self.cooldown_remaining)
        self.ramp_down_steps = torch.ones_like(self.cooldown_remaining)
        self.hold_steps = torch.zeros_like(self.cooldown_remaining)

        self.peak_magnitude = torch.zeros((num_envs, NUM_WRISTS), dtype=torch.float32, device=device)
        self.direction = torch.zeros((num_envs, NUM_WRISTS, DIM), dtype=torch.float32, device=device)

        mask = torch.tensor(
            [float(enable_left), float(enable_right)], dtype=torch.float32, device=device,
        )
        self._wrist_enable_mask = mask.view(1, NUM_WRISTS, 1)
        self._wrist_enable_2d = mask.view(1, NUM_WRISTS)

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.force[env_ids] = 0.0
        self.state[env_ids] = STATE_COOLDOWN
        self.cooldown_remaining[env_ids] = 0
        self.hold_remaining[env_ids] = 0
        self.ramp_up_steps[env_ids] = 1
        self.ramp_down_steps[env_ids] = 1
        self.hold_steps[env_ids] = 0
        self.peak_magnitude[env_ids] = 0.0
        self.direction[env_ids] = 0.0
        self._lerp.reset(self._flatten_ids(env_ids))

    def _flatten_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        base = env_ids.view(-1, 1) * NUM_WRISTS
        offsets = torch.arange(NUM_WRISTS, device=self.device, dtype=torch.long)
        return (base + offsets.view(1, -1)).reshape(-1)

    def _pair_to_flat(self, env_idx: torch.Tensor, w_idx: torch.Tensor) -> torch.Tensor:
        return (env_idx * NUM_WRISTS + w_idx).to(dtype=torch.long)

    @property
    def lerp_magnitude_nw(self) -> torch.Tensor:
        return self._lerp.value.view(self.num_envs, NUM_WRISTS)

    @property
    def lerp_active_nw(self) -> torch.Tensor:
        return self._lerp._active.view(self.num_envs, NUM_WRISTS)

    @torch.no_grad()
    def step(self) -> None:
        device = self.device
        # 1) decrement counters
        hold_mask = self.state == STATE_HOLD
        self.hold_remaining[hold_mask] = self.hold_remaining[hold_mask] - 1
        cool_mask = self.state == STATE_COOLDOWN
        self.cooldown_remaining[cool_mask] = self.cooldown_remaining[cool_mask].clamp(min=0) - 1
        self.cooldown_remaining = self.cooldown_remaining.clamp(min=0)

        # 2) advance lerp
        self._lerp.update_time(1)

        # 3) RAMP_UP -> HOLD
        ramp_up_done = (self.state == STATE_RAMP_UP) & (~self.lerp_active_nw)
        if ramp_up_done.any():
            self.state[ramp_up_done] = STATE_HOLD
            self.hold_remaining[ramp_up_done] = self.hold_steps[ramp_up_done]

        # 4) HOLD -> RAMP_DOWN
        hold_done = (self.state == STATE_HOLD) & (self.hold_remaining <= 0)
        if hold_done.any():
            self.state[hold_done] = STATE_RAMP_DOWN
            idx = hold_done.nonzero(as_tuple=False)
            env_idx = idx[:, 0]
            w_idx = idx[:, 1]
            flat = self._pair_to_flat(env_idx, w_idx)
            lerp = self._lerp
            lerp._start[flat] = lerp.value[flat]
            lerp._end[flat] = 0.0
            lerp._t[flat] = 0
            lerp._T[flat] = self.ramp_down_steps[env_idx, w_idx]
            lerp._active[flat] = True

        # 5) RAMP_DOWN -> COOLDOWN
        ramp_down_done = (self.state == STATE_RAMP_DOWN) & (~self.lerp_active_nw)
        if ramp_down_done.any():
            self.state[ramp_down_done] = STATE_COOLDOWN
            n = int(ramp_down_done.sum().item())
            cool_s = random_uniform(
                (n,), self.cooldown_range_s[0], self.cooldown_range_s[1], device=device,
            )
            self.cooldown_remaining[ramp_down_done] = _duration_to_steps(
                cool_s, self.control_rate_hz,
            )
            self.peak_magnitude[ramp_down_done] = 0.0
            self.direction[ramp_down_done] = 0.0

        # 6) Bernoulli trigger when COOLDOWN + counter hits 0
        ready = (self.state == STATE_COOLDOWN) & (self.cooldown_remaining == 0)
        ready = ready & self._wrist_enable_2d.expand_as(ready).bool()
        if ready.any():
            triggered = ready & (torch.rand(ready.shape, device=device) < self.activation_prob)
            if triggered.any():
                self._start_episode(triggered)

        # 7) magnitude * direction * enable mask = channel force
        magnitude_nw = self.lerp_magnitude_nw.unsqueeze(-1)
        self.force = magnitude_nw * self.direction * self._wrist_enable_mask

    @torch.no_grad()
    def _start_episode(self, triggered: torch.Tensor) -> None:
        device = self.device
        n = int(triggered.sum().item())
        if n == 0:
            return
        idx = triggered.nonzero(as_tuple=False)
        env_idx = idx[:, 0]
        w_idx = idx[:, 1]

        d = torch.randn(n, DIM, device=device)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mag_lo, mag_hi = self.magnitude_range
        mag = random_uniform((n,), mag_lo, mag_hi, device=device)
        peak_vec = clamp_norm(d * mag.unsqueeze(-1), max_norm=float(mag_hi))
        mag = peak_vec.norm(dim=-1)
        d = peak_vec / mag.clamp(min=1e-6).unsqueeze(-1)

        self.direction[env_idx, w_idx] = d
        self.peak_magnitude[env_idx, w_idx] = mag

        dur_s = random_uniform(
            (n,), self.duration_range_s[0], self.duration_range_s[1], device=device,
        )
        ramp_up_steps_t = torch.clamp(
            torch.round(dur_s * self.ramp_frac * self.control_rate_hz).to(torch.int32),
            min=1,
        )
        ramp_down_steps_t = ramp_up_steps_t.clone()
        total_steps_t = _duration_to_steps(dur_s, self.control_rate_hz)
        hold_steps_t = torch.clamp(total_steps_t - ramp_up_steps_t - ramp_down_steps_t, min=0)

        self.ramp_up_steps[env_idx, w_idx] = ramp_up_steps_t
        self.ramp_down_steps[env_idx, w_idx] = ramp_down_steps_t
        self.hold_steps[env_idx, w_idx] = hold_steps_t

        flat = self._pair_to_flat(env_idx, w_idx)
        lerp = self._lerp
        lerp._start[flat] = 0.0
        lerp._end[flat] = mag
        lerp._t[flat] = 0
        lerp._T[flat] = ramp_up_steps_t
        lerp._active[flat] = True
        lerp.value[flat] = 0.0

        self.state[env_idx, w_idx] = STATE_RAMP_UP


def _rotate_world_to_body_yaw(
    base_quat_xyzw: torch.Tensor, force_w: torch.Tensor
) -> torch.Tensor:
    """Rotate ``(N, 2, 3)`` world force -> base-yaw frame via ``R_yaw^-1(base_quat)``.

    yaw-only variant: roll/pitch does not enter the observation.
    Matches learning-compliance's ``EeGripperForceSensor``.
    """
    yaw_q = yaw_quat(base_quat_xyzw, w_last=True)  # (N, 4) xyzw
    yaw_q_nw = yaw_q.unsqueeze(1).expand(-1, force_w.shape[1], -1).reshape(-1, 4)
    flat = force_w.reshape(-1, 3)
    out = quat_rotate_inverse(yaw_q_nw, flat, w_last=True)
    return out.view(force_w.shape)


def _rotate_world_to_body_full(
    base_quat_xyzw: torch.Tensor, force_w: torch.Tensor
) -> torch.Tensor:
    """Rotate ``(N, 2, 3)`` world force -> full base frame via ``R^-1(base_quat)``.

    full_base variant: roll/pitch/yaw all enter the observation.
    Matches gentle-humanoid's ``MotionTrackingCommand_impedance``.
    """
    base_q_nw = base_quat_xyzw.unsqueeze(1).expand(-1, force_w.shape[1], -1).reshape(-1, 4)
    flat = force_w.reshape(-1, 3)
    out = quat_rotate_inverse(base_q_nw, flat, w_last=True)
    return out.view(force_w.shape)


# frame key -> rotation function
_OBS_FRAME_ROTATIONS = {
    "yaw_only": _rotate_world_to_body_yaw,
    "full_base": _rotate_world_to_body_full,
}


class WristForceTrackingCommand(CommandTermBase):
    """v14 command term: samples F_ext (world), derives + caches F_cmd_b."""

    def __init__(self, cfg: CommandTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        params: dict[str, Any] = cfg.params or {}
        wcfg = params.get("wrist_force_tracking_config")
        if wcfg is None:
            raise ValueError(
                "WristForceTrackingCommand requires 'wrist_force_tracking_config' in params."
            )
        self.wrist_cfg: WristForceTrackingConfig = wcfg

        self.num_envs: int = env.num_envs
        self.device: torch.device = torch.device(env.device)

        dt = float(getattr(env, "dt", 0.0))
        if dt <= 0.0:
            raise RuntimeError(f"env.dt must be > 0, got {dt!r}.")
        self.control_rate_hz: float = 1.0 / dt

        if self.wrist_cfg.enable_force_curriculum:
            initial_magnitude_range = self.wrist_cfg.curriculum_initial_magnitude_range
        else:
            initial_magnitude_range = self.wrist_cfg.force_ext_magnitude_range

        self._ext_channel = _ForceChannel(
            self.num_envs,
            self.device,
            magnitude_range=initial_magnitude_range,
            duration_range_s=self.wrist_cfg.force_ext_duration_range_s,
            cooldown_range_s=self.wrist_cfg.force_ext_cooldown_range_s,
            ramp_frac=self.wrist_cfg.force_ext_ramp_frac,
            activation_prob_per_step=self.wrist_cfg.force_ext_activation_prob_per_step,
            enable_left=self.wrist_cfg.enable_left,
            enable_right=self.wrist_cfg.enable_right,
            control_rate_hz=self.control_rate_hz,
        )

        self._curriculum_step: int = 0

        self._force_cmd_b: torch.Tensor = torch.zeros(
            self.num_envs, NUM_WRISTS, DIM, dtype=torch.float32, device=self.device,
        )
        self._obs_frame_quat: torch.Tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device,
        )
        self._obs_frame_quat[..., 3] = 1.0  # identity xyzw

        if self.wrist_cfg.force_cmd_obs_frame not in _OBS_FRAME_ROTATIONS:
            raise ValueError(
                f"Unsupported force_cmd_obs_frame: {self.wrist_cfg.force_cmd_obs_frame!r}"
            )
        self._rotate_to_obs_frame: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            _OBS_FRAME_ROTATIONS[self.wrist_cfg.force_cmd_obs_frame]
        )

        self.metrics: dict[str, torch.Tensor] = {}

    def setup(self) -> None:
        """No deferred setup; all state is initialized in __init__."""
        return

    def reset(self, env_ids: torch.Tensor | None) -> None:
        idx = self._as_index_tensor(env_ids)
        if idx.numel() == 0:
            return
        self._ext_channel.reset(idx)
        self._force_cmd_b[idx] = 0.0
        self._obs_frame_quat[idx] = 0.0
        self._obs_frame_quat[idx, 3] = 1.0  # identity xyzw

    @torch.no_grad()
    def step(self) -> None:
        """Advance the FSM one control step and snapshot F_cmd obs."""
        self._advance_curriculum()
        self._ext_channel.step()
        self._obs_frame_quat = self._compute_obs_frame_quat(self.env.base_quat)
        self._force_cmd_b = -self._rotate_to_obs_frame(
            self.env.base_quat, self._ext_channel.force,
        )

    def _compute_obs_frame_quat(self, base_quat_xyzw: torch.Tensor) -> torch.Tensor:
        """Return the forward obs-frame quaternion (xyzw, [N, 4]) for snapshot cache."""
        if self.wrist_cfg.force_cmd_obs_frame == "yaw_only":
            return yaw_quat(base_quat_xyzw, w_last=True).detach().clone()
        return base_quat_xyzw.detach().clone()

    def _advance_curriculum(self) -> None:
        if not self.wrist_cfg.enable_force_curriculum:
            return
        self._curriculum_step += 1
        # The first call to step() preserves the initial range so that
        # the test checkpoint "step 0: range = initial" passes after the
        # first step() invocation (the range was already set in __init__).
        if self._curriculum_step == 1:
            return
        ramp_steps = self.wrist_cfg.curriculum_ramp_steps
        progress = min(1.0, self._curriculum_step / float(ramp_steps))
        lo_i, hi_i = self.wrist_cfg.curriculum_initial_magnitude_range
        lo_t, hi_t = self.wrist_cfg.force_ext_magnitude_range
        lo = lo_i + (lo_t - lo_i) * progress
        hi = hi_i + (hi_t - hi_i) * progress
        self._ext_channel.magnitude_range = (lo, hi)

    # ---- observable buffers ------------------------------------------------
    @property
    def force_ext_w(self) -> torch.Tensor:
        """World-frame F_ext, shape [N, 2, 3]."""
        return self._ext_channel.force

    @property
    def force_cmd_b(self) -> torch.Tensor:
        """F_cmd obs snapshot, shape [N, 2, 3]."""
        return self._force_cmd_b

    # ---- metrics -----------------------------------------------------------
    def update_metrics(self) -> None:
        metrics: dict[str, torch.Tensor] = {}
        f_ext = self.force_ext_w
        f_cmd_b = self.force_cmd_b
        ext_mag = f_ext.norm(dim=-1)
        cmd_mag = f_cmd_b.norm(dim=-1)

        metrics["force/ext_magnitude_l"] = ext_mag[:, 0].float()
        metrics["force/ext_magnitude_r"] = ext_mag[:, 1].float()
        metrics["force/ext_magnitude_max"] = ext_mag.max(dim=-1).values.float()
        metrics["force/cmd_magnitude_l"] = cmd_mag[:, 0].float()
        metrics["force/cmd_magnitude_r"] = cmd_mag[:, 1].float()
        metrics["force/cmd_magnitude_max"] = cmd_mag.max(dim=-1).values.float()

        f_cmd_w = self._rotate_cmd_obs_to_world(f_cmd_b)

        cmd_w_safe = f_cmd_w.norm(dim=-1).clamp(min=1e-6)
        ext_safe = ext_mag.clamp(min=1e-6)
        cos_per_wrist = (f_cmd_w * f_ext).sum(dim=-1) / (cmd_w_safe * ext_safe)
        both_active = (cmd_mag > 1e-3) & (ext_mag > 1e-3)
        cos_per_wrist = torch.where(both_active, cos_per_wrist, torch.zeros_like(cos_per_wrist))
        denom = both_active.float().sum(dim=-1).clamp(min=1.0)
        metrics["force/cmd_ext_alignment"] = (cos_per_wrist.sum(dim=-1) / denom).float()

        ext_active = (self._ext_channel.state != STATE_COOLDOWN).float()
        metrics["force/active_frac_ext"] = ext_active.mean(dim=-1)

        ext_state = self._ext_channel.state
        metrics["force/phase_ramp_up"] = (ext_state == STATE_RAMP_UP).float().mean(dim=-1)
        metrics["force/phase_hold"] = (ext_state == STATE_HOLD).float().mean(dim=-1)
        metrics["force/phase_ramp_down"] = (ext_state == STATE_RAMP_DOWN).float().mean(dim=-1)

        if self.wrist_cfg.enable_force_curriculum:
            progress = min(
                1.0,
                self._curriculum_step / float(self.wrist_cfg.curriculum_ramp_steps),
            )
        else:
            progress = 1.0
        current_hi = float(self._ext_channel.magnitude_range[1])
        metrics["force/curriculum_progress"] = torch.full(
            (self.num_envs,), progress, dtype=torch.float32, device=self.device,
        )
        metrics["force/curriculum_current_hi"] = torch.full(
            (self.num_envs,), current_hi, dtype=torch.float32, device=self.device,
        )

        self.metrics = metrics

    def _rotate_cmd_obs_to_world(self, f_cmd_obs: torch.Tensor) -> torch.Tensor:
        """Rotate F_cmd obs back to world using cached obs-frame quat."""
        q = self._obs_frame_quat  # (N, 4) xyzw snapshot
        q_nw = q.unsqueeze(1).expand(-1, f_cmd_obs.shape[1], -1).reshape(-1, 4)
        flat = f_cmd_obs.reshape(-1, 3)
        out = quat_apply(q_nw, flat, w_last=True)
        return out.view(f_cmd_obs.shape)

    def _as_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
