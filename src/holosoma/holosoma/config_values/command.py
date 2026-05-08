"""Default command manager configurations."""

from holosoma.config_values.loco.g1.command import g1_29dof_command
from holosoma.config_values.loco.t1.command import t1_29dof_command
from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_command,
    g1_29dof_wbt_command_w_object,
)
from holosoma.config_values.wbt.g1.command_force import g1_29dof_wbt_force_command
from holosoma.config_values.wbt.g1.command_force_v2 import (
    g1_29dof_wbt_force_v2_command,
    g1_29dof_wbt_force_v2_fullbase_command,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_command,
    "g1_29dof": g1_29dof_command,
    "g1_29dof_wbt": g1_29dof_wbt_command,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_command_w_object,
    "g1_29dof_wbt_force": g1_29dof_wbt_force_command,
    "g1_29dof_wbt_force_v2": g1_29dof_wbt_force_v2_command,
    "g1_29dof_wbt_force_v2_fullbase": g1_29dof_wbt_force_v2_fullbase_command,
}
