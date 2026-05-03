"""Regression tests for the g1_29dof_wbt_force command preset (Task 6)."""

from __future__ import annotations

from holosoma.config_values.command import DEFAULTS as COMMAND_DEFAULTS
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command
from holosoma.config_values.wbt.g1.command_force import g1_29dof_wbt_force_command


def test_new_preset_registered_in_defaults() -> None:
    assert "g1_29dof_wbt_force" in COMMAND_DEFAULTS
    assert COMMAND_DEFAULTS["g1_29dof_wbt_force"] is g1_29dof_wbt_force_command


def test_baseline_preset_is_untouched() -> None:
    assert COMMAND_DEFAULTS["g1_29dof_wbt"] is g1_29dof_wbt_command
    # The new preset does not mutate the baseline's term dicts.
    assert "wrist_compliance_command" not in g1_29dof_wbt_command.setup_terms
    assert "wrist_compliance_command" not in g1_29dof_wbt_command.reset_terms
    assert "wrist_compliance_command" not in g1_29dof_wbt_command.step_terms


def test_force_preset_retains_motion_command() -> None:
    assert "motion_command" in g1_29dof_wbt_force_command.setup_terms
    assert "motion_command" in g1_29dof_wbt_force_command.reset_terms
    assert "motion_command" in g1_29dof_wbt_force_command.step_terms


def test_force_preset_adds_wrist_compliance_command() -> None:
    for dct_name in ("setup_terms", "reset_terms", "step_terms"):
        terms = getattr(g1_29dof_wbt_force_command, dct_name)
        assert "wrist_compliance_command" in terms, dct_name
        assert (
            terms["wrist_compliance_command"].func == "holosoma.managers.command.terms.wbt_force:WristComplianceCommand"
        )


def test_wrist_compliance_cfg_has_expected_defaults() -> None:
    params = g1_29dof_wbt_force_command.setup_terms["wrist_compliance_command"].params
    cfg = params["wrist_compliance_config"]
    assert cfg.k_virtual_range == (100.0, 100.0)
    assert cfg.left_wrist_body_name == "left_wrist_yaw_link"
    assert cfg.right_wrist_body_name == "right_wrist_yaw_link"
    assert cfg.enable_left and cfg.enable_right
