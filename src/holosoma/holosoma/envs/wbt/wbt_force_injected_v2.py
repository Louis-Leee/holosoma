"""v14 WBT env with wrist-only force injection + debug arrows.

Self-contained. Directly inherits the baseline ``WholeBodyTrackingManager``
and implements its own:

* body-id resolution for the two wrists
* ``_apply_force_in_physics_step`` — injects world F_ext into the sim via
  per-wrist body-frame rotation, constrained to the two wrist ``body_ids``
* ``_rotate_force_world_to_body`` — uses each wrist's ``body_quat_w``
* ``draw_debug_viz`` + ``_wrap_simulator_draw_hook``
* ``_update_log_dict`` — V14 env-owned metrics

Does NOT import or subclass ``WholeBodyTrackingForceInjected`` (V10).
The implementation is modeled after V10 but maintained as an independent
copy; V10 can change without affecting V14.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt_force_v2 import WristForceTrackingCommand

if TYPE_CHECKING:
    from holosoma.config_types.command_v2 import WristForceTrackingConfig

# Color constants for debug arrows (RGB tuples in [0,1]).
# F_cmd = orange (actor-observed force command)
# F_ext = green (sim-injected external force)
COLOR_F_CMD: tuple[float, float, float] = (1.0, 0.5, 0.0)
COLOR_F_EXT: tuple[float, float, float] = (0.0, 0.8, 0.0)

# Debug arrow min-norm threshold (skip zero arrows).
_F_NORM_MIN_N: float = 1e-3

class WholeBodyTrackingForceInjectedV2(WholeBodyTrackingManager):
    """V14 WBT env with wrist-only force injection.

    Sibling of ``WholeBodyTrackingForceInjected`` (V10): both directly extend
    ``WholeBodyTrackingManager``; neither inherits the other.
    """

    # Command term key used by V14 (see docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md).
    _CMD_TERM_KEY: str = "wrist_force_tracking_command"

    def __init__(self, tyro_config, *, device) -> None:  # type: ignore[no-untyped-def]
        super().__init__(tyro_config, device=device)

        # Resolve wrist body ids (isaac indices into simulator._robot).
        wrist_cmd = self._get_wrist_command()
        wcfg: WristForceTrackingConfig = wrist_cmd.wrist_cfg

        self._left_wrist_isaac_id = int(self.simulator.find_rigid_body_indice(wcfg.left_wrist_body_name))
        self._right_wrist_isaac_id = int(self.simulator.find_rigid_body_indice(wcfg.right_wrist_body_name))

        # Int64 tensor — IsaacLab expects a long-dtype body_ids tensor.
        self._wrist_body_ids_t = torch.tensor(
            [self._left_wrist_isaac_id, self._right_wrist_isaac_id],
            device=self.simulator.sim_device,
            dtype=torch.long,
        )

        # Debug snapshot: world-frame force per isaac body id (for diagnostics /
        # wandb env-owned metrics).
        self.last_applied_force_w_by_body_id: dict[int, torch.Tensor] = {}

        # Wire env-level arrow drawing into the simulator's per-render hook.
        self._wrap_simulator_draw_hook()

    # ------------------------------------------------------------------ #
    # Command resolution
    # ------------------------------------------------------------------ #

    def _get_wrist_command(self, *, require: bool = True) -> WristForceTrackingCommand:
        term = self.command_manager.get_state(self._CMD_TERM_KEY)
        if term is None:
            if require:
                raise RuntimeError(
                    f"WholeBodyTrackingForceInjectedV2 requires a "
                    f"'{self._CMD_TERM_KEY}' command term to be registered."
                )
            return None  # type: ignore[return-value]
        if not isinstance(term, WristForceTrackingCommand):
            raise TypeError(
                f"'{self._CMD_TERM_KEY}' must be a WristForceTrackingCommand instance, "
                f"got {type(term).__name__}."
            )
        return term

    # ------------------------------------------------------------------ #
    # Force injection in physics substep
    # ------------------------------------------------------------------ #

    def _apply_force_in_physics_step(self) -> None:
        """Apply action torques (via super) + push F_ext onto the two wrist bodies."""
        super()._apply_force_in_physics_step()

        wrist_cmd = self._get_wrist_command(require=False)
        if wrist_cmd is None:
            # Term not registered (e.g. running with a baseline preset) —
            # defensive fallback: no injection.
            return

        force_w = wrist_cmd.force_ext_w  # (N, 2, 3) world frame
        forces_body = self._rotate_force_world_to_body(force_w)

        self.simulator._robot.set_external_force_and_torque(  # type: ignore[attr-defined]
            forces=forces_body,
            torques=torch.zeros_like(forces_body),
            env_ids=None,
            body_ids=self._wrist_body_ids_t,
        )

        # Debug snapshot. Detach + clone so downstream code can't mutate it.
        self.last_applied_force_w_by_body_id[self._left_wrist_isaac_id] = (
            force_w[:, 0].detach().clone()
        )
        self.last_applied_force_w_by_body_id[self._right_wrist_isaac_id] = (
            force_w[:, 1].detach().clone()
        )

    def _rotate_force_world_to_body(self, force_w: torch.Tensor) -> torch.Tensor:
        """Rotate ``(N, 2, 3)`` world-frame force to each wrist's body frame.

        Uses each wrist's ``body_quat_w`` (wxyz convention, IsaacLab).
        """
        from isaaclab.utils.math import quat_apply_inverse

        sim = self.simulator
        # body_quat_w is wxyz in IsaacLab.
        left_q = sim._robot.data.body_quat_w[:, self._left_wrist_isaac_id]  # type: ignore[attr-defined]
        right_q = sim._robot.data.body_quat_w[:, self._right_wrist_isaac_id]  # type: ignore[attr-defined]

        left_f_b = quat_apply_inverse(left_q, force_w[:, 0])
        right_f_b = quat_apply_inverse(right_q, force_w[:, 1])
        return torch.stack([left_f_b, right_f_b], dim=1)

    # ------------------------------------------------------------------ #
    # Debug viz
    # ------------------------------------------------------------------ #

    def _wrap_simulator_draw_hook(self) -> None:
        """Chain env's draw_debug_viz onto the simulator's per-render hook.

        IsaacSim calls ``simulator.draw_debug_viz()`` once per render when
        ``debug_viz_enabled`` is True. We preserve the simulator's own
        behaviour (e.g. virtual_gantry markers) and append our arrows after.
        """
        sim = self.simulator
        original = sim.draw_debug_viz

        def _chained_draw_debug_viz() -> None:
            original()
            try:
                self.draw_debug_viz()
            except Exception:
                # Swallow draw errors so a viz glitch can't take down a run.
                return

        sim.draw_debug_viz = _chained_draw_debug_viz  # type: ignore[method-assign]

    @torch.no_grad()
    def draw_debug_viz(self) -> None:
        """Draw F_ext (green) and F_cmd (orange) force arrows at the two wrists of env 0."""
        sim = self.simulator

        # Require an actively-initialised draw handle (headless → None).
        if not getattr(sim, "draw", None):
            return

        wrist_cmd = self._get_wrist_command(require=False)
        if wrist_cmd is None:
            return

        wcfg: WristForceTrackingConfig = wrist_cmd.wrist_cfg
        scale = float(wcfg.debug_arrow_scale_n_per_m)

        env_id = 0
        body_ids = [self._left_wrist_isaac_id, self._right_wrist_isaac_id]

        # Import locally to defer draw_adapter import under the right sim type.
        from holosoma.utils.draw import draw_line, draw_sphere

        # F_ext (world frame) and F_cmd (obs-frame → world) for env 0.
        f_ext_w = wrist_cmd.force_ext_w[env_id]  # (2, 3)
        f_cmd_obs_batched = wrist_cmd.force_cmd_b[env_id : env_id + 1]  # (1, 2, 3)
        f_cmd_w_batched = wrist_cmd._rotate_cmd_obs_to_world(f_cmd_obs_batched)
        f_cmd_w = f_cmd_w_batched[0]  # (2, 3)

        for wrist_idx, body_id in enumerate(body_ids):
            wrist_pos_w = (
                sim._robot.data.body_pos_w[env_id, body_id].detach().cpu()  # type: ignore[attr-defined]
            )
            # F_ext arrow (green).
            self._draw_arrow(draw_line, draw_sphere, sim, wrist_pos_w, f_ext_w[wrist_idx], scale, COLOR_F_EXT, env_id)
            # F_cmd arrow (orange).
            self._draw_arrow(draw_line, draw_sphere, sim, wrist_pos_w, f_cmd_w[wrist_idx], scale, COLOR_F_CMD, env_id)

    @staticmethod
    def _draw_arrow(
        draw_line_fn,
        draw_sphere_fn,
        sim,
        start_pos_w: torch.Tensor,
        force_w: torch.Tensor,
        scale_n_per_m: float,
        color: tuple[float, float, float],
        env_id: int,
    ) -> None:
        """Draw one arrow: line + end-sphere. Skip tiny forces."""
        norm = float(force_w.norm().item())
        if norm < _F_NORM_MIN_N:
            return
        delta = (force_w / scale_n_per_m).detach().cpu()
        end = start_pos_w + delta
        draw_line_fn(sim, start_pos_w, end, color, env_id)
        draw_sphere_fn(sim, end, 0.015, color, env_id)

    # ------------------------------------------------------------------ #
    # Wandb metrics
    # ------------------------------------------------------------------ #

    def _update_log_dict(self) -> None:
        super()._update_log_dict()

        wrist_cmd = self._get_wrist_command(require=False)
        if wrist_cmd is None:
            return

        # Command-owned metrics.
        wrist_cmd.update_metrics()
        self.log_dict.update(wrist_cmd.metrics)

        # Env-owned metrics.
        wcfg: WristForceTrackingConfig = wrist_cmd.wrist_cfg
        self.log_dict.update(self._env_owned_force_metrics(wrist_cmd, wcfg))

    def _env_owned_force_metrics(
        self,
        wrist_cmd: WristForceTrackingCommand,
        wcfg: WristForceTrackingConfig,
    ) -> dict[str, torch.Tensor]:
        """V14 env metrics: applied-body-force sanity + active-window termination
        monitor + wrist reaction diagnostic."""
        metrics: dict[str, torch.Tensor] = {}
        device = self.device

        # Applied body-frame F norms (sanity for the world→body rotation path).
        applied_l = self.last_applied_force_w_by_body_id.get(self._left_wrist_isaac_id)
        applied_r = self.last_applied_force_w_by_body_id.get(self._right_wrist_isaac_id)
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        metrics["force/applied_f_body_l"] = (
            applied_l.norm(dim=-1).float().to(device) if applied_l is not None else zeros
        )
        metrics["force/applied_f_body_r"] = (
            applied_r.norm(dim=-1).float().to(device) if applied_r is not None else zeros
        )

        # Active-window flag: any channel active (not COOLDOWN)?
        ext_any_active = (wrist_cmd._ext_channel.state != 0).any(dim=-1).float()
        metrics["force/active_window_flag"] = ext_any_active

        # Termination-rate monitor: track resets coincident with active ext force.
        last_reset_ids = getattr(getattr(self, "reset_manager", None), "last_reset_ids", None)
        if last_reset_ids is not None:
            reset_mask = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            reset_mask[last_reset_ids.to(device=device, dtype=torch.long)] = 1.0
            prev_active = getattr(self, "_prev_active_window_flag", None)
            if prev_active is not None and prev_active.shape == ext_any_active.shape:
                metrics["force/term_rate_active_ext"] = reset_mask * prev_active
                metrics["force/term_rate_inactive_ext"] = reset_mask * (1.0 - prev_active)
        self._prev_active_window_flag = ext_any_active.detach().clone()

        # Optional wrist-reaction diagnostic (only if backend exposes wrench).
        reaction_metrics = self._try_compute_wrist_reaction_metrics(wrist_cmd)
        metrics.update(reaction_metrics)
        return metrics

    def _try_compute_wrist_reaction_metrics(
        self,
        wrist_cmd: WristForceTrackingCommand,
    ) -> dict[str, torch.Tensor]:
        """Read wrist body_incoming_wrench_b (if available) and compare against F_cmd_b.

        Log-only diagnostic for the Q1 closure mechanism. Returns an empty
        dict if the simulator backend does not expose ``body_incoming_wrench_b``.
        """
        sim = self.simulator
        wrench_buf = getattr(getattr(sim, "_robot", None), "data", None)
        if wrench_buf is None or not hasattr(wrench_buf, "body_incoming_wrench_b"):
            return {}

        wrench_b = wrench_buf.body_incoming_wrench_b  # (N, num_bodies, 6)
        if wrench_b is None:
            return {}

        left_f_b = wrench_b[:, self._left_wrist_isaac_id, :3]
        right_f_b = wrench_b[:, self._right_wrist_isaac_id, :3]
        f_cmd_w = wrist_cmd._rotate_cmd_obs_to_world(wrist_cmd.force_cmd_b)  # (N, 2, 3)
        from isaaclab.utils.math import quat_apply as _quat_apply

        lq = wrench_buf.body_quat_w[:, self._left_wrist_isaac_id]
        rq = wrench_buf.body_quat_w[:, self._right_wrist_isaac_id]
        left_f_w = _quat_apply(lq, left_f_b)
        right_f_w = _quat_apply(rq, right_f_b)

        def _cos_and_ratio(
            reaction_w: torch.Tensor, cmd_w: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            r_n = reaction_w.norm(dim=-1).clamp(min=1e-6)
            c_n = cmd_w.norm(dim=-1).clamp(min=1e-6)
            cos = (reaction_w * cmd_w).sum(dim=-1) / (r_n * c_n)
            ratio = r_n / c_n
            active = cmd_w.norm(dim=-1) > 1e-3
            cos = torch.where(active, cos, torch.zeros_like(cos))
            ratio = torch.where(active, ratio, torch.zeros_like(ratio))
            return cos.float(), ratio.float()

        lcos, lrat = _cos_and_ratio(left_f_w, f_cmd_w[:, 0])
        rcos, rrat = _cos_and_ratio(right_f_w, f_cmd_w[:, 1])
        return {
            "force/wrist_reaction_vs_cmd_cos_l": lcos,
            "force/wrist_reaction_vs_cmd_cos_r": rcos,
            "force/wrist_reaction_vs_cmd_mag_ratio_l": lrat,
            "force/wrist_reaction_vs_cmd_mag_ratio_r": rrat,
        }
