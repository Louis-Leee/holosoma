"""v14 reward preset — alias for the baseline WBT reward preset.

v14 reward == original dataset reference-motion tracking reward. The F_ext
injection + F_cmd_b observation are the only mechanism that encodes the
force-tracking task; reward itself has no wrist-force term.
"""

from __future__ import annotations

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

g1_29dof_wbt_force_v2_reward = g1_29dof_wbt_reward

__all__ = ["g1_29dof_wbt_force_v2_reward"]
