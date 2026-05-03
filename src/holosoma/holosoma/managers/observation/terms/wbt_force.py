"""Observation terms that expose WBT wrist-force command / priv signals.

These terms read from the ``wrist_compliance_command`` command term
registered on the env's ``command_manager``:

* :func:`wrist_force_command` — F_cmd in body-yaw frame, shape ``[N, 6]``
  (actor + critic).
* :func:`wrist_force_ext_privileged` — F_ext in world frame, shape
  ``[N, 6]`` (critic-only privileged info).
* :func:`wrist_virtual_stiffness_command` — ``K_virtual`` per wrist,
  shape ``[N, 2]`` (actor + critic). Scaled by 0.01 in the preset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt_force import WristComplianceCommand

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_wrist_command(env: WholeBodyTrackingManager) -> WristComplianceCommand:
    term = env.command_manager.get_state("wrist_compliance_command")
    if term is None:
        raise RuntimeError(
            "wrist_force_* observation terms require the 'wrist_compliance_command' command term to be registered."
        )
    if not isinstance(term, WristComplianceCommand):
        raise TypeError(f"'wrist_compliance_command' must be a WristComplianceCommand, got {type(term).__name__}.")
    return term


def wrist_force_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Flatten F_cmd [N, 2, 3] → [N, 6]. Returns a clone so obs history
    buffers cannot mutate the underlying command state."""
    return _get_wrist_command(env).force_cmd_b.reshape(env.num_envs, 6).clone()


def wrist_force_ext_privileged(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Flatten F_ext world-frame [N, 2, 3] → [N, 6] (critic-only)."""
    return _get_wrist_command(env).force_ext_w.reshape(env.num_envs, 6).clone()


def wrist_virtual_stiffness_command(
    env: WholeBodyTrackingManager,
) -> torch.Tensor:
    """Return per-wrist K_virtual clone, shape [N, 2]."""
    return _get_wrist_command(env).k_virtual.clone()
