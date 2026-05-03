"""Wrist compliance command — per-env/per-wrist F_cmd + F_ext + K buffers.

Implements the trapezoidal F_cmd / F_ext state machine described in
``docs/plans/2026-05-03-wbt-wrist-force-v10.md`` (Task 2). Two independent
channels share the same ``COOLDOWN -> RAMP_UP -> HOLD -> RAMP_DOWN ->
COOLDOWN`` FSM driven by ``_gh_utils.TemporalLerp``:

* ``F_cmd``: the 6-D force (two wrists x 3 world-axes but stored in
  body-yaw frame) the policy is asked to produce.
* ``F_ext``: the world-frame sim-injected disturbance.

``k_virtual`` is sampled **once per episode** (GH kp pattern) and stays
constant until the next reset.

The term exposes :py:attr:`force_cmd_b`, :py:attr:`force_ext_w`, and
:py:attr:`k_virtual` tensors, plus a ``metrics`` dict filled by
:py:meth:`update_metrics` (wired into wandb via the env's
``_update_log_dict`` in Task 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.command.terms._gh_utils import (
    TemporalLerp,
    clamp_norm,
    random_uniform,
)

if TYPE_CHECKING:
    from holosoma.config_types.command import CommandTermCfg, WristComplianceConfig

# ---- State machine states (per-env per-wrist int8 codes) -------------------
STATE_COOLDOWN: int = 0
STATE_RAMP_UP: int = 1
STATE_HOLD: int = 2
STATE_RAMP_DOWN: int = 3

# Number of wrists (left, right).
NUM_WRISTS: int = 2

# Spatial dimension (x, y, z).
DIM: int = 3


def _duration_to_steps(duration_s: torch.Tensor, control_rate_hz: float) -> torch.Tensor:
    """Convert duration in seconds to a (>=1) step count tensor.

    ``steps = max(1, round(duration_s * control_rate_hz))``. Uses ``round``
    (not floor/ceil) to avoid a one-sided bias. ``duration_s == 0`` maps to
    ``steps == 1`` so TemporalLerp never divides by 0.
    """
    steps = torch.round(duration_s * control_rate_hz).to(dtype=torch.int32)
    return torch.clamp(steps, min=1)


class _ForceChannel:
    """One independent F_cmd or F_ext channel with its own trapezoidal FSM.

    Layout:

    * ``force_<frame>``: ``[N, 2, 3]`` current force on each wrist.
    * ``state``: ``[N, 2]`` int8 FSM state.
    * ``cooldown_remaining`` / ``hold_remaining`` / ``ramp_*``: ``[N, 2]`` int32
      step counters.
    * ``peak_magnitude``: ``[N, 2]`` float32 (currently active peak).
    * ``direction``: ``[N, 2, 3]`` float32 (unit vector of the current peak).
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
        self.enable_left = enable_left
        self.enable_right = enable_right
        self.control_rate_hz = control_rate_hz

        # Force tensor — shape [N, 2, 3]. Populated via TemporalLerp's magnitude
        # output * direction.
        self.force = torch.zeros(num_envs, NUM_WRISTS, DIM, device=device)

        # TemporalLerp drives a per-(env, wrist) scalar magnitude in [0, peak].
        # TemporalLerp shares a single clock across all dims after the first,
        # so we must flatten (env, wrist) into the first dim. Shape becomes
        # (N * 2,) with a trailing (1,) unused; we treat the lerp as 1-D.
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

        # Per-wrist enable mask. A disabled wrist stays at force=0 forever.
        mask = torch.tensor(
            [float(enable_left), float(enable_right)],
            dtype=torch.float32,
            device=device,
        )
        self._wrist_enable_mask = mask.view(1, NUM_WRISTS, 1)
        self._wrist_enable_2d = mask.view(1, NUM_WRISTS)

    # ---- lifecycle ---------------------------------------------------------
    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        # Zero everything on reset.
        self.force[env_ids] = 0.0
        self.state[env_ids] = STATE_COOLDOWN
        # cooldown=0 lets the Bernoulli roll trigger immediately.
        self.cooldown_remaining[env_ids] = 0
        self.hold_remaining[env_ids] = 0
        self.ramp_up_steps[env_ids] = 1
        self.ramp_down_steps[env_ids] = 1
        self.hold_steps[env_ids] = 0
        self.peak_magnitude[env_ids] = 0.0
        self.direction[env_ids] = 0.0

        # Reset the flat (N*2,) lerp entries for every wrist on each env_id.
        flat_ids = self._flatten_ids(env_ids)
        self._lerp.reset(flat_ids)

    # ---- flat-index helpers ------------------------------------------------
    def _flatten_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Expand a per-env index tensor to ``env_idx * 2 + wrist_idx``."""
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        base = env_ids.view(-1, 1) * NUM_WRISTS
        offsets = torch.arange(NUM_WRISTS, device=self.device, dtype=torch.long)
        return (base + offsets.view(1, -1)).reshape(-1)

    def _pair_to_flat(self, env_idx: torch.Tensor, w_idx: torch.Tensor) -> torch.Tensor:
        return (env_idx * NUM_WRISTS + w_idx).to(dtype=torch.long)

    # The lerp value seen per-env per-wrist, shape [N, 2].
    @property
    def lerp_magnitude_nw(self) -> torch.Tensor:
        return self._lerp.value.view(self.num_envs, NUM_WRISTS)

    # An active-mask per-env per-wrist.
    @property
    def lerp_active_nw(self) -> torch.Tensor:
        return self._lerp._active.view(self.num_envs, NUM_WRISTS)

    @torch.no_grad()
    def step(self) -> None:
        """Advance the FSM one control step for all envs / wrists."""
        device = self.device

        # 1) Advance per-step counters.
        hold_mask = self.state == STATE_HOLD
        self.hold_remaining[hold_mask] = self.hold_remaining[hold_mask] - 1
        cool_mask = self.state == STATE_COOLDOWN
        self.cooldown_remaining[cool_mask] = self.cooldown_remaining[cool_mask].clamp(min=0) - 1
        self.cooldown_remaining = self.cooldown_remaining.clamp(min=0)

        # 2) Advance TemporalLerp.
        self._lerp.update_time(1)

        # 3) RAMP_UP -> HOLD transition when lerp reports done.
        ramp_up_done = (self.state == STATE_RAMP_UP) & (~self.lerp_active_nw)
        if ramp_up_done.any():
            self.state[ramp_up_done] = STATE_HOLD
            self.hold_remaining[ramp_up_done] = self.hold_steps[ramp_up_done]

        # 4) HOLD -> RAMP_DOWN transition.
        hold_done = (self.state == STATE_HOLD) & (self.hold_remaining <= 0)
        if hold_done.any():
            self.state[hold_done] = STATE_RAMP_DOWN
            env_w_idx = hold_done.nonzero(as_tuple=False)
            env_idx = env_w_idx[:, 0]
            w_idx = env_w_idx[:, 1]
            flat = self._pair_to_flat(env_idx, w_idx)

            lerp = self._lerp
            lerp._start[flat] = lerp.value[flat]
            lerp._end[flat] = 0.0
            lerp._t[flat] = 0
            lerp._T[flat] = self.ramp_down_steps[env_idx, w_idx]
            lerp._active[flat] = True

        # 5) RAMP_DOWN -> COOLDOWN transition (lerp reports done again).
        ramp_down_done = (self.state == STATE_RAMP_DOWN) & (~self.lerp_active_nw)
        if ramp_down_done.any():
            self.state[ramp_down_done] = STATE_COOLDOWN
            n = int(ramp_down_done.sum().item())
            cool_s = random_uniform(
                (n,),
                self.cooldown_range_s[0],
                self.cooldown_range_s[1],
                device=device,
            )
            cool_steps = _duration_to_steps(cool_s, self.control_rate_hz)
            self.cooldown_remaining[ramp_down_done] = cool_steps
            self.peak_magnitude[ramp_down_done] = 0.0
            self.direction[ramp_down_done] = 0.0

        # 6) Bernoulli trigger in COOLDOWN when cooldown_remaining == 0.
        ready = (self.state == STATE_COOLDOWN) & (self.cooldown_remaining == 0)
        enable_mask_bool = self._wrist_enable_2d.expand_as(ready).bool()
        ready = ready & enable_mask_bool
        if ready.any():
            triggered = ready & (torch.rand(ready.shape, device=device) < self.activation_prob)
            if triggered.any():
                self._start_episode(triggered)

        # 7) Apply magnitude * direction to produce the channel force.
        magnitude_nw = self.lerp_magnitude_nw.unsqueeze(-1)  # (N, 2, 1)
        self.force = magnitude_nw * self.direction * self._wrist_enable_mask

    # ---- internals ---------------------------------------------------------
    @torch.no_grad()
    def _start_episode(self, triggered: torch.Tensor) -> None:
        """Sample direction + magnitude + durations; kick off RAMP_UP."""
        device = self.device
        n = int(triggered.sum().item())
        if n == 0:
            return

        env_w_idx = triggered.nonzero(as_tuple=False)
        env_idx = env_w_idx[:, 0]
        w_idx = env_w_idx[:, 1]

        # Direction: unit-sphere uniform sample.
        d = torch.randn(n, DIM, device=device)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Magnitude: uniform in [lo, hi]; clamp_norm ensures scalar cap.
        mag_lo, mag_hi = self.magnitude_range
        mag = random_uniform((n,), mag_lo, mag_hi, device=device)
        # Note: mag is already in range; keep clamp_norm as a defensive cap.
        peak_vec = d * mag.unsqueeze(-1)
        peak_vec = clamp_norm(peak_vec, max_norm=float(mag_hi))
        mag = peak_vec.norm(dim=-1)  # may shrink slightly due to clamp_norm
        d = peak_vec / mag.clamp(min=1e-6).unsqueeze(-1)

        self.direction[env_idx, w_idx] = d
        self.peak_magnitude[env_idx, w_idx] = mag

        # Duration: ramp_up + hold + ramp_down = duration.
        dur_s = random_uniform(
            (n,),
            self.duration_range_s[0],
            self.duration_range_s[1],
            device=device,
        )
        ramp_up_steps_t = torch.clamp(
            torch.round(dur_s * self.ramp_frac * self.control_rate_hz).to(torch.int32),
            min=1,
        )
        ramp_down_steps_t = ramp_up_steps_t.clone()
        total_steps_t = _duration_to_steps(dur_s, self.control_rate_hz)
        # hold = total - ramp_up - ramp_down, floor at 0.
        hold_steps_t = torch.clamp(total_steps_t - ramp_up_steps_t - ramp_down_steps_t, min=0)

        self.ramp_up_steps[env_idx, w_idx] = ramp_up_steps_t
        self.ramp_down_steps[env_idx, w_idx] = ramp_down_steps_t
        self.hold_steps[env_idx, w_idx] = hold_steps_t

        # Kick off RAMP_UP: lerp magnitude 0 -> peak over ramp_up_steps.
        flat = self._pair_to_flat(env_idx, w_idx)
        lerp = self._lerp
        lerp._start[flat] = 0.0
        lerp._end[flat] = mag
        lerp._t[flat] = 0
        lerp._T[flat] = ramp_up_steps_t
        lerp._active[flat] = True
        lerp.value[flat] = 0.0

        self.state[env_idx, w_idx] = STATE_RAMP_UP


class WristComplianceCommand(CommandTermBase):
    """Command term that owns the per-env F_cmd / F_ext / K buffers.

    F_cmd is stored in the robot's body-yaw frame (reward + observation
    consume it in that frame). F_ext is stored in world frame (the env
    subclass rotates it to body frame before ``set_external_force_and_torque``).
    """

    def __init__(self, cfg: CommandTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        params: dict[str, Any] = cfg.params or {}

        # Pull the frozen WristComplianceConfig from params.
        wcfg = params.get("wrist_compliance_config")
        if wcfg is None:
            raise ValueError(
                "WristComplianceCommand requires 'wrist_compliance_config' "
                "in params (a WristComplianceConfig dataclass)."
            )
        self.wrist_cfg: WristComplianceConfig = wcfg

        self.num_envs: int = env.num_envs
        self.device: torch.device = torch.device(env.device)

        # control_rate_hz = 1 / env.dt (env.dt is the control step seconds).
        dt = float(getattr(env, "dt", 0.0))
        if dt <= 0.0:
            raise RuntimeError(f"env.dt must be > 0 for duration-to-steps conversion, got {dt!r}.")
        self.control_rate_hz: float = 1.0 / dt

        self._cmd_channel = _ForceChannel(
            self.num_envs,
            self.device,
            magnitude_range=self.wrist_cfg.force_cmd_magnitude_range,
            duration_range_s=self.wrist_cfg.force_cmd_duration_range_s,
            cooldown_range_s=self.wrist_cfg.force_cmd_cooldown_range_s,
            ramp_frac=self.wrist_cfg.force_cmd_ramp_frac,
            activation_prob_per_step=self.wrist_cfg.force_cmd_activation_prob_per_step,
            enable_left=self.wrist_cfg.enable_left,
            enable_right=self.wrist_cfg.enable_right,
            control_rate_hz=self.control_rate_hz,
        )
        self._ext_channel = _ForceChannel(
            self.num_envs,
            self.device,
            magnitude_range=self.wrist_cfg.force_ext_magnitude_range,
            duration_range_s=self.wrist_cfg.force_ext_duration_range_s,
            cooldown_range_s=self.wrist_cfg.force_ext_cooldown_range_s,
            ramp_frac=self.wrist_cfg.force_ext_ramp_frac,
            activation_prob_per_step=self.wrist_cfg.force_ext_activation_prob_per_step,
            enable_left=self.wrist_cfg.enable_left,
            enable_right=self.wrist_cfg.enable_right,
            control_rate_hz=self.control_rate_hz,
        )

        # Per-env per-wrist virtual stiffness (resampled on reset, constant
        # for the episode). Shape [N, 2].
        self.k_virtual = torch.full(
            (self.num_envs, NUM_WRISTS),
            float(self.wrist_cfg.k_virtual_range[0]),
            dtype=torch.float32,
            device=self.device,
        )

        # wandb metrics dict (filled by update_metrics).
        self.metrics: dict[str, torch.Tensor] = {}

    # ---- lifecycle ---------------------------------------------------------
    def setup(self) -> None:
        """No additional setup beyond __init__."""
        # Keep init values; explicit reset at env start is issued by manager.
        return

    def reset(self, env_ids: torch.Tensor | None) -> None:
        idx = self._as_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        self._cmd_channel.reset(idx)
        self._ext_channel.reset(idx)

        # Resample K per-env per-wrist uniformly in [lo, hi].
        lo, hi = self.wrist_cfg.k_virtual_range
        if lo == hi:
            self.k_virtual[idx] = lo
        else:
            self.k_virtual[idx] = random_uniform((idx.shape[0], NUM_WRISTS), lo, hi, device=self.device)

    def step(self) -> None:
        self._cmd_channel.step()
        self._ext_channel.step()

    # ---- observable buffers -----------------------------------------------
    @property
    def force_cmd_b(self) -> torch.Tensor:
        """Body-yaw-frame F_cmd, shape [N, 2, 3] (left, right)."""
        return self._cmd_channel.force

    @property
    def force_ext_w(self) -> torch.Tensor:
        """World-frame F_ext, shape [N, 2, 3] (left, right)."""
        return self._ext_channel.force

    # ---- metrics ----------------------------------------------------------
    def update_metrics(self) -> None:
        """Populate ``self.metrics`` for wandb consumption.

        Keys mirror Task 9.5 "Command-owned" contract. All values are
        ``torch.Tensor`` on ``self.device`` with dtype float32 and shape
        ``[N]`` (TensorAverageMeterDict averages across envs downstream).
        """
        metrics: dict[str, torch.Tensor] = {}

        f_cmd = self._cmd_channel.force  # [N, 2, 3]
        f_ext = self._ext_channel.force  # [N, 2, 3]

        cmd_mag = f_cmd.norm(dim=-1)  # [N, 2]
        ext_mag = f_ext.norm(dim=-1)  # [N, 2]

        metrics["force/cmd_magnitude_l"] = cmd_mag[:, 0].float()
        metrics["force/cmd_magnitude_r"] = cmd_mag[:, 1].float()
        metrics["force/cmd_magnitude_max"] = cmd_mag.max(dim=-1).values.float()

        metrics["force/ext_magnitude_l"] = ext_mag[:, 0].float()
        metrics["force/ext_magnitude_r"] = ext_mag[:, 1].float()
        metrics["force/ext_magnitude_max"] = ext_mag.max(dim=-1).values.float()

        # cos(F_cmd, F_ext) per wrist, then average; only where both non-zero.
        cmd_mag_safe = cmd_mag.clamp(min=1e-6)
        ext_mag_safe = ext_mag.clamp(min=1e-6)
        cos_per_wrist = (f_cmd * f_ext).sum(dim=-1) / (cmd_mag_safe * ext_mag_safe)
        both_active = (cmd_mag > 1e-3) & (ext_mag > 1e-3)
        cos_per_wrist = torch.where(
            both_active,
            cos_per_wrist,
            torch.zeros_like(cos_per_wrist),
        )
        # Avoid /0 when no wrist has both channels active on an env.
        denom = both_active.float().sum(dim=-1).clamp(min=1.0)
        metrics["force/cmd_ext_alignment"] = (cos_per_wrist.sum(dim=-1) / denom).float()

        metrics["force/k_virtual_l"] = self.k_virtual[:, 0].float()
        metrics["force/k_virtual_r"] = self.k_virtual[:, 1].float()

        # Activity fraction per env (mean over two wrists).
        cmd_active = (self._cmd_channel.state != STATE_COOLDOWN).float()
        ext_active = (self._ext_channel.state != STATE_COOLDOWN).float()
        metrics["force/active_frac_cmd"] = cmd_active.mean(dim=-1)
        metrics["force/active_frac_ext"] = ext_active.mean(dim=-1)

        # F_cmd phase occupancies (averaged across wrists).
        cmd_state = self._cmd_channel.state
        metrics["force/phase_ramp_up"] = (cmd_state == STATE_RAMP_UP).float().mean(dim=-1)
        metrics["force/phase_hold"] = (cmd_state == STATE_HOLD).float().mean(dim=-1)
        metrics["force/phase_ramp_down"] = (cmd_state == STATE_RAMP_DOWN).float().mean(dim=-1)

        self.metrics = metrics

    # ---- helpers ----------------------------------------------------------
    def _as_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
