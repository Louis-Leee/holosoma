"""v14 wrist-force observation terms.

* ``wrist_force_command_v2``: actor+critic, F_cmd_b in base-frame (yaw-only or
  full-base depending on WristForceTrackingConfig.force_cmd_obs_frame), [N, 6].
  (F_cmd_b is derived inside the command term as -R_frame^-1(base_quat) . F_ext_w.)
* ``wrist_force_ext_privileged_v2``: critic-only, F_ext_w world frame, [N, 6].

Reads ``wrist_force_tracking_command`` (v14 key). v10 uses
``wrist_compliance_command`` so the two experiments coexist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt_force_v2 import WristForceTrackingCommand

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


_TERM_KEY = "wrist_force_tracking_command"


def _get_cmd(env: WholeBodyTrackingManager) -> WristForceTrackingCommand:
    term = env.command_manager.get_state(_TERM_KEY)
    if term is None:
        raise RuntimeError(
            f"v14 wrist-force obs terms require '{_TERM_KEY}' command term to be registered."
        )
    if not isinstance(term, WristForceTrackingCommand):
        raise TypeError(
            f"'{_TERM_KEY}' must be a WristForceTrackingCommand, got {type(term).__name__}."
        )
    return term


def wrist_force_command_v2(env: WholeBodyTrackingManager) -> torch.Tensor:
    return _get_cmd(env).force_cmd_b.reshape(env.num_envs, 6).clone()


def wrist_force_ext_privileged_v2(env: WholeBodyTrackingManager) -> torch.Tensor:
    return _get_cmd(env).force_ext_w.reshape(env.num_envs, 6).clone()
