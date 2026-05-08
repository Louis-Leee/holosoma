"""v14 reward preset must equal baseline reward (no wrist reward term)."""

from __future__ import annotations

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward
from holosoma.config_values.wbt.g1.reward_force_v2 import g1_29dof_wbt_force_v2_reward


def test_v2_reward_preset_is_baseline() -> None:
    assert g1_29dof_wbt_force_v2_reward is g1_29dof_wbt_reward or (
        set(g1_29dof_wbt_force_v2_reward.terms.keys())
        == set(g1_29dof_wbt_reward.terms.keys())
    )


def test_v2_has_no_wrist_force_reward_term() -> None:
    for name in g1_29dof_wbt_force_v2_reward.terms:
        assert "wrist_force" not in name, f"v14 must NOT carry wrist_force reward: {name}"
