"""Command preset for the WBT wrist-force experiment.

Extends the baseline ``g1_29dof_wbt_command`` with a second command term,
``wrist_compliance_command``, that owns the F_cmd / F_ext / K buffers
driven by :class:`WristComplianceCommand`. The motion command is reused
verbatim so all existing tracking behaviour is preserved.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.command import (
    CommandTermCfg,
    WristComplianceConfig,
)
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command

_WRIST_FORCE_FUNC = "holosoma.managers.command.terms.wbt_force:WristComplianceCommand"
_DEFAULT_WRIST_CFG = WristComplianceConfig()


def _wrist_force_term() -> CommandTermCfg:
    return CommandTermCfg(
        func=_WRIST_FORCE_FUNC,
        params={"wrist_compliance_config": _DEFAULT_WRIST_CFG},
    )


g1_29dof_wbt_force_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        **g1_29dof_wbt_command.setup_terms,
        "wrist_compliance_command": _wrist_force_term(),
    },
    reset_terms={
        **g1_29dof_wbt_command.reset_terms,
        "wrist_compliance_command": _wrist_force_term(),
    },
    step_terms={
        **g1_29dof_wbt_command.step_terms,
        "wrist_compliance_command": _wrist_force_term(),
    },
)


__all__ = ["g1_29dof_wbt_force_command"]
