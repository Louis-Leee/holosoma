"""Reward preset for the WBT wrist-force experiment.

Extends the baseline ``g1_29dof_wbt_reward`` with one additional term:
``wrist_force_position_tracking_exp`` (weight=2.0, sigma=0.3). K_virtual
is read from the wrist_compliance_command buffer at runtime and is NOT
passed as a reward parameter (see Task 5 signature).
"""

from __future__ import annotations

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

g1_29dof_wbt_force_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "wrist_force_position_tracking_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt_force:wrist_force_position_tracking_exp",
            params={
                "sigma": 0.3,
                "left_wrist_body_name": "left_wrist_yaw_link",
                "right_wrist_body_name": "right_wrist_yaw_link",
            },
            weight=2.0,
        ),
    }
)


__all__ = ["g1_29dof_wbt_force_reward"]
