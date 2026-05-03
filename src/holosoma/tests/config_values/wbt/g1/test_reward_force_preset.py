"""Regression tests for the g1_29dof_wbt_force reward preset (Task 8)."""

from __future__ import annotations

from holosoma.config_values.reward import DEFAULTS
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward
from holosoma.config_values.wbt.g1.reward_force import g1_29dof_wbt_force_reward


def test_new_preset_registered_in_defaults() -> None:
    assert DEFAULTS["g1_29dof_wbt_force"] is g1_29dof_wbt_force_reward


def test_baseline_preset_untouched() -> None:
    assert DEFAULTS["g1_29dof_wbt"] is g1_29dof_wbt_reward
    # Baseline must not have gained the wrist-force term.
    assert "wrist_force_position_tracking_exp" not in g1_29dof_wbt_reward.terms


def test_force_preset_has_baseline_terms_plus_wrist_force() -> None:
    for name in g1_29dof_wbt_reward.terms:
        assert name in g1_29dof_wbt_force_reward.terms
    assert "wrist_force_position_tracking_exp" in g1_29dof_wbt_force_reward.terms


def test_wrist_force_term_params_and_weight() -> None:
    term = g1_29dof_wbt_force_reward.terms["wrist_force_position_tracking_exp"]
    assert term.func == "holosoma.managers.reward.terms.wbt_force:wrist_force_position_tracking_exp"
    assert term.weight == 2.0
    assert term.params["sigma"] == 0.3
    assert term.params["left_wrist_body_name"] == "left_wrist_yaw_link"
    assert term.params["right_wrist_body_name"] == "right_wrist_yaw_link"
    # K_virtual must NOT be a reward parameter (red line).
    assert "K_virtual" not in term.params
    assert "k_virtual" not in term.params
