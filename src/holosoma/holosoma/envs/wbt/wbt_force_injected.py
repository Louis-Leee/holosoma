"""WBT env subclass that injects wrist-only external forces + F_cmd/F_ext arrows.

Extends :class:`WholeBodyTrackingManager` with:

* ``_apply_force_in_physics_step``: rotates the command term's world-frame
  F_ext into body-local frame and writes it to the simulator for the two
  wrist bodies **only** (red line 1).
* ``draw_debug_viz``: renders F_ext (green) and F_cmd (orange) arrows at
  the two wrists of env 0 when the simulator's debug viz is enabled.
  Wired into the simulator's per-render hook in ``__init__``.
* ``_update_log_dict``: calls the command term's ``update_metrics()`` and
  merges "env-owned" wrist-tracking / applied-force keys into ``log_dict``
  for wandb (Task 9.5 contract).

All changes are additive — the base :class:`WholeBodyTrackingManager` and
its subclasses continue to work without force injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt_force import WristComplianceCommand

if TYPE_CHECKING:
    from holosoma.config_types.command import WristComplianceConfig

# Color constants for debug arrows (RGB tuples in [0,1]).
# F_cmd = orange (actor-observed force command)
# F_ext = green (sim-injected external force)
COLOR_F_CMD: tuple[float, float, float] = (1.0, 0.5, 0.0)
COLOR_F_EXT: tuple[float, float, float] = (0.0, 0.8, 0.0)

# Debug arrow min-norm threshold (skip zero arrows).
_F_NORM_MIN_N: float = 1e-3


class WholeBodyTrackingForceInjected(WholeBodyTrackingManager):
    """WBT env that enables wrist-force injection + debug arrow visualization."""

    def __init__(self, tyro_config, *, device) -> None:  # type: ignore[no-untyped-def]
        super().__init__(tyro_config, device=device)

        # Resolve wrist body ids (isaac indices into simulator._robot).
        wrist_cmd = self._get_wrist_command()
        wcfg: WristComplianceConfig = wrist_cmd.wrist_cfg

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
        # We keep the simulator's own draw_debug_viz behaviour (virtual gantry)
        # and append our arrows after it.
        self._wrap_simulator_draw_hook()

    # ------------------------------------------------------------------ #
    # Force injection in physics substep
    # ------------------------------------------------------------------ #

    def _apply_force_in_physics_step(self) -> None:
        """Apply action torques + push F_ext onto the two wrist bodies."""
        super()._apply_force_in_physics_step()

        wrist_cmd = self._get_wrist_command(require=False)
        if wrist_cmd is None:
            # Term not registered (e.g. when running this class with a
            # baseline preset — defensive). Fall back to no injection.
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
        self.last_applied_force_w_by_body_id[self._left_wrist_isaac_id] = force_w[:, 0].detach().clone()
        self.last_applied_force_w_by_body_id[self._right_wrist_isaac_id] = force_w[:, 1].detach().clone()

    # ------------------------------------------------------------------ #
    # Debug viz
    # ------------------------------------------------------------------ #

    def _wrap_simulator_draw_hook(self) -> None:
        """Chain env's draw_debug_viz onto the simulator's hook.

        IsaacSim calls ``simulator.draw_debug_viz()`` once per render when
        ``debug_viz_enabled`` is True. We preserve the simulator's own
        behaviour (e.g. virtual_gantry markers) and append our arrow
        drawing after it.
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
        """Draw red/blue/purple force arrows at the two wrists of env 0."""
        sim = self.simulator

        # Require an actively-initialised draw handle (e.g. headless → None).
        if not getattr(sim, "draw", None):
            return

        wrist_cmd = self._get_wrist_command(require=False)
        if wrist_cmd is None:
            return

        wcfg: WristComplianceConfig = wrist_cmd.wrist_cfg
        scale = float(wcfg.debug_arrow_scale_n_per_m)

        env_id = 0
        body_ids = [self._left_wrist_isaac_id, self._right_wrist_isaac_id]

        # Import locally to defer draw_adapter import under the right sim type.
        from holosoma.utils.draw import draw_line, draw_sphere

        # F_ext (world frame) and F_cmd (body-yaw → world) for env 0.
        f_ext_w = wrist_cmd.force_ext_w[env_id]  # (2, 3) cpu-or-device
        f_cmd_b = wrist_cmd.force_cmd_b[env_id]  # (2, 3) body-yaw frame
        # Rotate F_cmd body-yaw → world using base_quat yaw. ``env.base_quat``
        # is xyzw (from ``simulator.base_quat = robot_root_states[:, 3:7]``).
        f_cmd_w = self._rotate_body_yaw_to_world(f_cmd_b, env_id)

        for wrist_idx, body_id in enumerate(body_ids):
            wrist_pos_w = (
                sim._robot.data.body_pos_w[env_id, body_id].detach().cpu()  # type: ignore[attr-defined]
            )

            # F_ext arrow (red).
            self._draw_arrow(
                draw_line,
                draw_sphere,
                sim,
                wrist_pos_w,
                f_ext_w[wrist_idx],
                scale,
                COLOR_F_EXT,
                env_id,
            )
            # F_cmd arrow (blue).
            self._draw_arrow(
                draw_line,
                draw_sphere,
                sim,
                wrist_pos_w,
                f_cmd_w[wrist_idx],
                scale,
                COLOR_F_CMD,
                env_id,
            )

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
        # radius=0.015 is a visual hint; the isaacsim adapter ignores it
        # but the interface is kept for future simulator ports.
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
        wcfg: WristComplianceConfig = wrist_cmd.wrist_cfg
        self.log_dict.update(self._env_owned_force_metrics(wrist_cmd, wcfg))

    def _env_owned_force_metrics(
        self,
        wrist_cmd: WristComplianceCommand,
        wcfg: WristComplianceConfig,
    ) -> dict[str, torch.Tensor]:
        """Compute env-side force metrics for the wandb log_dict.

        Reads ``self.simulator`` to get each wrist's current world position
        and uses the wrist command to compute virtual spring shifts +
        residual error.
        """
        metrics: dict[str, torch.Tensor] = {}

        device = self.device
        sim = self.simulator

        # Wrist world positions, shape [N, 2, 3].
        left_pos = sim._robot.data.body_pos_w[:, self._left_wrist_isaac_id]  # type: ignore[attr-defined]
        right_pos = sim._robot.data.body_pos_w[:, self._right_wrist_isaac_id]  # type: ignore[attr-defined]
        wrist_actual_w = torch.stack([left_pos, right_pos], dim=1)

        # Total force world-frame = F_ext + yaw_rotated(F_cmd).
        f_ext_w = wrist_cmd.force_ext_w
        f_cmd_w = self._rotate_body_yaw_to_world_batch(wrist_cmd.force_cmd_b)
        f_total_w = f_ext_w + f_cmd_w

        k_virtual = wrist_cmd.k_virtual.clamp(min=1e-3)
        shift = f_total_w / k_virtual.unsqueeze(-1)
        shift_norm = shift.norm(dim=-1)  # [N, 2]

        metrics["force/wrist_target_shift_l"] = shift_norm[:, 0].float().to(device)
        metrics["force/wrist_target_shift_r"] = shift_norm[:, 1].float().to(device)

        # Position residual vs the motion-target-shifted point. We follow the
        # reward formula: error = ||wrist_actual - (motion_target + shift)||.
        motion_target_w = self._motion_target_wrist_positions_w()
        target_shifted_w = motion_target_w + shift
        pos_error = (wrist_actual_w - target_shifted_w).norm(dim=-1)  # [N, 2]
        metrics["force/wrist_pos_error_l"] = pos_error[:, 0].float().to(device)
        metrics["force/wrist_pos_error_r"] = pos_error[:, 1].float().to(device)

        # Applied body-frame F norm, one per wrist (sanity for the
        # world→body rotation).
        applied_l = self.last_applied_force_w_by_body_id.get(self._left_wrist_isaac_id)
        applied_r = self.last_applied_force_w_by_body_id.get(self._right_wrist_isaac_id)
        if applied_l is not None:
            metrics["force/applied_f_body_l"] = applied_l.norm(dim=-1).float().to(device)
        else:
            metrics["force/applied_f_body_l"] = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        if applied_r is not None:
            metrics["force/applied_f_body_r"] = applied_r.norm(dim=-1).float().to(device)
        else:
            metrics["force/applied_f_body_r"] = torch.zeros(self.num_envs, dtype=torch.float32, device=device)

        return metrics

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_wrist_command(self, *, require: bool = True) -> WristComplianceCommand:
        term = self.command_manager.get_state("wrist_compliance_command")
        if term is None:
            if require:
                raise RuntimeError(
                    "WholeBodyTrackingForceInjected requires a "
                    "'wrist_compliance_command' command term to be registered."
                )
            return None  # type: ignore[return-value]
        if not isinstance(term, WristComplianceCommand):
            raise TypeError(
                f"'wrist_compliance_command' must be a WristComplianceCommand instance, got {type(term).__name__}."
            )
        return term

    def _rotate_force_world_to_body(self, force_w: torch.Tensor) -> torch.Tensor:
        """Rotate ``(N, 2, 3)`` world-frame force to each wrist's body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        sim = self.simulator
        # body_quat_w is wxyz in IsaacLab.
        left_q = sim._robot.data.body_quat_w[:, self._left_wrist_isaac_id]  # type: ignore[attr-defined]
        right_q = sim._robot.data.body_quat_w[:, self._right_wrist_isaac_id]  # type: ignore[attr-defined]

        left_f_b = quat_apply_inverse(left_q, force_w[:, 0])
        right_f_b = quat_apply_inverse(right_q, force_w[:, 1])
        return torch.stack([left_f_b, right_f_b], dim=1)

    def _rotate_body_yaw_to_world(self, force_b: torch.Tensor, env_id: int) -> torch.Tensor:
        """Rotate a single env's (2, 3) body-yaw F_cmd to world using base yaw.

        ``self.base_quat`` is xyzw (from ``simulator.base_quat``).
        """
        from holosoma.utils.rotations import quat_apply, yaw_quat

        base_q = self.base_quat[env_id : env_id + 1]  # (1, 4) xyzw
        yaw_q = yaw_quat(base_q, w_last=True)  # (1, 4) xyzw
        # Duplicate yaw_q over the 2 wrists.
        yaw_q2 = yaw_q.expand(2, -1)
        return quat_apply(yaw_q2, force_b, w_last=True)

    def _rotate_body_yaw_to_world_batch(self, force_b: torch.Tensor) -> torch.Tensor:
        """Rotate a batched (N, 2, 3) body-yaw force to world frame.

        ``self.base_quat`` is xyzw (from ``simulator.base_quat``).
        """
        from holosoma.utils.rotations import quat_apply, yaw_quat

        yaw_q = yaw_quat(self.base_quat, w_last=True)  # (N, 4) xyzw
        yaw_q_nw = yaw_q.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)  # (N*2, 4)
        flat = force_b.reshape(-1, 3)  # (N*2, 3)
        out = quat_apply(yaw_q_nw, flat, w_last=True)
        return out.view(force_b.shape)

    def _motion_target_wrist_positions_w(self) -> torch.Tensor:
        """Fetch the motion-target world-frame wrist positions, shape [N, 2, 3].

        Relies on the baseline ``motion_command`` term's ``body_pos_relative_w``
        plus its ``motion_cfg.body_names_to_track`` list. Falls back to
        ``wrist_actual_w`` if the motion command doesn't track wrists (in
        which case the residual metric collapses to the virtual-spring shift).
        """
        motion = self.command_manager.get_state("motion_command")
        sim = self.simulator

        wrist_cmd = self._get_wrist_command(require=False)
        wcfg = wrist_cmd.wrist_cfg if wrist_cmd is not None else None

        if (
            motion is None
            or wcfg is None
            or not hasattr(motion, "body_pos_relative_w")
            or not hasattr(motion, "motion_cfg")
        ):
            # Degenerate fallback: target = actual wrist (error → shift).
            left = sim._robot.data.body_pos_w[:, self._left_wrist_isaac_id]  # type: ignore[attr-defined]
            right = sim._robot.data.body_pos_w[:, self._right_wrist_isaac_id]  # type: ignore[attr-defined]
            return torch.stack([left, right], dim=1)

        body_names: list[str] = list(motion.motion_cfg.body_names_to_track)

        def _target(name: str, fallback_idx: int) -> torch.Tensor:
            if name in body_names:
                return motion.body_pos_relative_w[:, body_names.index(name)]
            # If the motion file doesn't track this wrist explicitly, fall
            # back to the sim's current wrist position so the error metric
            # is simply the virtual-spring shift.
            return sim._robot.data.body_pos_w[:, fallback_idx]  # type: ignore[attr-defined]

        left_target = _target(wcfg.left_wrist_body_name, self._left_wrist_isaac_id)
        right_target = _target(wcfg.right_wrist_body_name, self._right_wrist_isaac_id)
        return torch.stack([left_target, right_target], dim=1)
