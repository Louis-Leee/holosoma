"""WBT wrist-force reward: position tracking with a virtual-spring shift.

Implements the v10 reward formula::

    target_shifted = motion_target + (F_ext + F_cmd) / K_virtual
    error = ||wrist_actual - target_shifted||^2
    reward = exp(-error.mean(-1) / sigma^2)

``K_virtual`` is read from the ``wrist_compliance_command`` term's
``k_virtual`` buffer (per-env, per-wrist); the reward does **not**
accept K as a parameter. ``sigma`` + wrist body-name params come from
the reward cfg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt_force import WristComplianceCommand

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager

# Minimum K (N/m) used in the 1/K reward term — defensive against
# misconfigurations (Task 1 __post_init__ already asserts K > 0).
_K_FLOOR: float = 1e-3


def _get_wrist_command(env: WholeBodyTrackingManager) -> WristComplianceCommand:
    term = env.command_manager.get_state("wrist_compliance_command")
    if term is None:
        raise RuntimeError(
            "wrist_force_position_tracking_exp requires the 'wrist_compliance_command' command term to be registered."
        )
    if not isinstance(term, WristComplianceCommand):
        raise TypeError(f"'wrist_compliance_command' must be a WristComplianceCommand, got {type(term).__name__}.")
    return term


def _yaw_rotate_body_to_world(base_quat_wxyz: torch.Tensor, force_b_nw3: torch.Tensor) -> torch.Tensor:
    """Rotate ``(N, 2, 3)`` body-yaw force to world frame using base yaw."""
    from holosoma.utils.rotations import quat_apply, yaw_quat

    yaw_q = yaw_quat(base_quat_wxyz, w_last=False)  # (N, 4) wxyz
    yaw_q_nw = yaw_q.unsqueeze(1).expand(-1, force_b_nw3.shape[1], -1).reshape(-1, 4)
    flat = force_b_nw3.reshape(-1, 3)
    out = quat_apply(yaw_q_nw, flat, w_last=False)
    return out.view(force_b_nw3.shape)


def _wrist_actual_world(
    env: WholeBodyTrackingManager,
    left_body_name: str,
    right_body_name: str,
) -> torch.Tensor:
    """Read the two wrist world-frame positions, shape ``[N, 2, 3]``."""
    sim = env.simulator
    left_id = sim.find_rigid_body_indice(left_body_name)
    right_id = sim.find_rigid_body_indice(right_body_name)
    left_pos = sim._robot.data.body_pos_w[:, left_id]  # type: ignore[attr-defined]
    right_pos = sim._robot.data.body_pos_w[:, right_id]  # type: ignore[attr-defined]
    return torch.stack([left_pos, right_pos], dim=1)


def _wrist_motion_target_world(
    env: WholeBodyTrackingManager,
    left_body_name: str,
    right_body_name: str,
) -> torch.Tensor:
    """Read the motion-clip target wrist positions, shape ``[N, 2, 3]``.

    Falls back to the actual wrist position when the motion clip does not
    explicitly track that wrist (error metric then collapses to the
    virtual-spring shift).
    """
    motion = env.command_manager.get_state("motion_command")
    if motion is None or not hasattr(motion, "body_pos_relative_w"):
        return _wrist_actual_world(env, left_body_name, right_body_name)

    body_names = list(motion.motion_cfg.body_names_to_track)  # type: ignore[attr-defined]
    sim = env.simulator

    def _target(name: str) -> torch.Tensor:
        if name in body_names:
            return motion.body_pos_relative_w[:, body_names.index(name)]
        fallback_idx = sim.find_rigid_body_indice(name)
        return sim._robot.data.body_pos_w[:, fallback_idx]  # type: ignore[attr-defined]

    return torch.stack([_target(left_body_name), _target(right_body_name)], dim=1)


def wrist_force_position_tracking_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    left_wrist_body_name: str,
    right_wrist_body_name: str,
) -> torch.Tensor:
    """Position-tracking reward that includes a virtual-spring shift.

    Parameters
    ----------
    env
        Environment instance (must expose ``simulator``, ``base_quat``,
        ``command_manager``).
    sigma
        Length scale for the Gaussian kernel. Smaller ``sigma`` → sharper
        penalty on residuals.
    left_wrist_body_name, right_wrist_body_name
        Rigid-body names of the two wrists in the robot asset.

    Returns
    -------
    torch.Tensor
        Per-env reward, shape ``[N]``, values in ``[0, 1]``.
    """
    wrist_cmd = _get_wrist_command(env)

    # F_total in world frame.
    f_ext_w = wrist_cmd.force_ext_w  # [N, 2, 3]
    f_cmd_w = _yaw_rotate_body_to_world(env.base_quat, wrist_cmd.force_cmd_b)
    f_total_w = f_ext_w + f_cmd_w

    k_virtual = wrist_cmd.k_virtual.clamp(min=_K_FLOOR).unsqueeze(-1)  # [N, 2, 1]
    shift = f_total_w / k_virtual

    motion_target_w = _wrist_motion_target_world(env, left_wrist_body_name, right_wrist_body_name)
    target_shifted_w = motion_target_w + shift

    wrist_actual_w = _wrist_actual_world(env, left_wrist_body_name, right_wrist_body_name)

    sq_err = torch.sum((target_shifted_w - wrist_actual_w) ** 2, dim=-1)  # [N, 2]
    return torch.exp(-sq_err.mean(dim=-1) / (sigma**2))
