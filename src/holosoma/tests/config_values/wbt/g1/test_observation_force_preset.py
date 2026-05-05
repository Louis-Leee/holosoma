"""Regression tests for the g1_29dof_wbt_force observation preset (Task 7).

Structural checks only — the actual per-step/stacked dimensions (162/1620
and 300/3000) are asserted by the isaacsim e2e test in Task 11.
"""

from __future__ import annotations

from holosoma.config_values.observation import DEFAULTS
from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation
from holosoma.config_values.wbt.g1.observation_force import (
    g1_29dof_wbt_force_observation,
)

_NEW_ACTOR_TERMS = ("wrist_force_command", "wrist_virtual_stiffness_command")
_NEW_CRITIC_PRIV_TERMS = ("wrist_force_ext_privileged",)


def test_new_preset_registered_in_defaults() -> None:
    assert DEFAULTS["g1_29dof_wbt_force"] is g1_29dof_wbt_force_observation


def test_baseline_preset_is_untouched() -> None:
    assert DEFAULTS["g1_29dof_wbt"] is g1_29dof_wbt_observation
    # Baseline must still have history_length=1 and no wrist-force terms.
    actor_terms = g1_29dof_wbt_observation.groups["actor_obs"].terms
    critic_terms = g1_29dof_wbt_observation.groups["critic_obs"].terms
    assert g1_29dof_wbt_observation.groups["actor_obs"].history_length == 1
    assert g1_29dof_wbt_observation.groups["critic_obs"].history_length == 1
    for t in (*_NEW_ACTOR_TERMS, *_NEW_CRITIC_PRIV_TERMS):
        assert t not in actor_terms
        assert t not in critic_terms


def test_force_preset_has_history_length_10_on_both_groups() -> None:
    assert g1_29dof_wbt_force_observation.groups["actor_obs"].history_length == 10
    assert g1_29dof_wbt_force_observation.groups["critic_obs"].history_length == 10


def test_actor_group_adds_only_expected_terms() -> None:
    actor_terms = g1_29dof_wbt_force_observation.groups["actor_obs"].terms
    baseline_actor = g1_29dof_wbt_observation.groups["actor_obs"].terms

    for t in _NEW_ACTOR_TERMS:
        assert t in actor_terms
    # Privileged term must NOT leak into actor group.
    for t in _NEW_CRITIC_PRIV_TERMS:
        assert t not in actor_terms
    # Baseline actor terms all retained.
    for t in baseline_actor:
        assert t in actor_terms


def test_critic_group_adds_actor_plus_privileged_terms() -> None:
    critic_terms = g1_29dof_wbt_force_observation.groups["critic_obs"].terms
    baseline_critic = g1_29dof_wbt_observation.groups["critic_obs"].terms

    for t in _NEW_ACTOR_TERMS + _NEW_CRITIC_PRIV_TERMS:
        assert t in critic_terms
    for t in baseline_critic:
        assert t in critic_terms


def test_wrist_virtual_stiffness_scale_is_001_on_both_groups() -> None:
    for grp in ("actor_obs", "critic_obs"):
        assert g1_29dof_wbt_force_observation.groups[grp].terms["wrist_virtual_stiffness_command"].scale == 0.01


def test_alphabetical_sort_yields_consistent_term_order() -> None:
    # Two groups should share a deterministic term-name ordering when
    # sorted — this matches the inference-side metadata contract.
    actor_terms = sorted(g1_29dof_wbt_force_observation.groups["actor_obs"].terms)
    critic_terms = sorted(g1_29dof_wbt_force_observation.groups["critic_obs"].terms)
    # actor ⊆ critic.
    assert set(actor_terms).issubset(set(critic_terms))
