"""v14 command presets — two variants for ablation on F_cmd obs frame.

* ``g1_29dof_wbt_force_v2_command``:         force_cmd_obs_frame="yaw_only"  (default)
* ``g1_29dof_wbt_force_v2_fullbase_command``: force_cmd_obs_frame="full_base"

See docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md for the
rationale (learning-compliance vs gentle-humanoid frame choices).
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.command_v2 import WristForceTrackingConfig
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command

_WRIST_FORCE_FUNC = (
    "holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand"
)

# Two configs — only force_cmd_obs_frame differs.
_CFG_YAW = WristForceTrackingConfig(force_cmd_obs_frame="yaw_only")
_CFG_FULLBASE = WristForceTrackingConfig(force_cmd_obs_frame="full_base")


def _term(cfg: WristForceTrackingConfig) -> CommandTermCfg:
    return CommandTermCfg(
        func=_WRIST_FORCE_FUNC,
        params={"wrist_force_tracking_config": cfg},
    )


def _build_preset(cfg: WristForceTrackingConfig):
    return replace(
        g1_29dof_wbt_command,
        setup_terms={
            **g1_29dof_wbt_command.setup_terms,
            "wrist_force_tracking_command": _term(cfg),
        },
        reset_terms={
            **g1_29dof_wbt_command.reset_terms,
            "wrist_force_tracking_command": _term(cfg),
        },
        step_terms={
            **g1_29dof_wbt_command.step_terms,
            "wrist_force_tracking_command": _term(cfg),
        },
    )


g1_29dof_wbt_force_v2_command = _build_preset(_CFG_YAW)
g1_29dof_wbt_force_v2_fullbase_command = _build_preset(_CFG_FULLBASE)

__all__ = [
    "g1_29dof_wbt_force_v2_command",
    "g1_29dof_wbt_force_v2_fullbase_command",
]
