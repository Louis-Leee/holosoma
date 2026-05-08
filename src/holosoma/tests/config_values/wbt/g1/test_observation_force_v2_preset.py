"""v14 observation preset — actor: baseline + F_cmd_b; critic: baseline + F_cmd_b + F_ext_w priv."""

from __future__ import annotations

from holosoma.config_values.wbt.g1.observation import actor_obs_shared
from holosoma.config_values.wbt.g1.observation_force_v2 import (
    g1_29dof_wbt_force_v2_observation,
)


def test_preset_adds_f_cmd_to_both_groups() -> None:
    actor_terms = g1_29dof_wbt_force_v2_observation.groups["actor_obs"].terms
    critic_terms = g1_29dof_wbt_force_v2_observation.groups["critic_obs"].terms
    assert "wrist_force_command_v2" in actor_terms
    assert "wrist_force_command_v2" in critic_terms


def test_preset_adds_f_ext_priv_only_to_critic() -> None:
    actor_terms = g1_29dof_wbt_force_v2_observation.groups["actor_obs"].terms
    critic_terms = g1_29dof_wbt_force_v2_observation.groups["critic_obs"].terms
    assert "wrist_force_ext_privileged_v2" not in actor_terms
    assert "wrist_force_ext_privileged_v2" in critic_terms


def test_preset_has_no_k_virtual_term() -> None:
    for group in g1_29dof_wbt_force_v2_observation.groups.values():
        for name in group.terms:
            assert "virtual_stiffness" not in name
            assert "k_virtual" not in name


def test_preset_preserves_baseline_actor_terms() -> None:
    actor_terms = g1_29dof_wbt_force_v2_observation.groups["actor_obs"].terms
    for key in actor_obs_shared.terms:
        assert key in actor_terms


def test_preset_history_length_matches_v10() -> None:
    assert g1_29dof_wbt_force_v2_observation.groups["actor_obs"].history_length == 10
    assert g1_29dof_wbt_force_v2_observation.groups["critic_obs"].history_length == 10
