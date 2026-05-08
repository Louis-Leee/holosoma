"""v14 command presets — two variants: yaw_only (default) + full_base."""

from __future__ import annotations

from dataclasses import asdict

from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command
from holosoma.config_values.wbt.g1.command_force_v2 import (
    g1_29dof_wbt_force_v2_command,
    g1_29dof_wbt_force_v2_fullbase_command,
)


def test_yaw_preset_adds_v14_term() -> None:
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_command.setup_terms
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_command.reset_terms
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_command.step_terms


def test_fullbase_preset_adds_v14_term() -> None:
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_fullbase_command.setup_terms
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_fullbase_command.reset_terms
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_fullbase_command.step_terms


def test_yaw_preset_uses_yaw_only_frame() -> None:
    term = g1_29dof_wbt_force_v2_command.step_terms["wrist_force_tracking_command"]
    cfg = term.params["wrist_force_tracking_config"]
    assert cfg.force_cmd_obs_frame == "yaw_only"


def test_fullbase_preset_uses_full_base_frame() -> None:
    term = g1_29dof_wbt_force_v2_fullbase_command.step_terms["wrist_force_tracking_command"]
    cfg = term.params["wrist_force_tracking_config"]
    assert cfg.force_cmd_obs_frame == "full_base"


def test_two_presets_differ_only_in_frame() -> None:
    """Two presets should differ ONLY in force_cmd_obs_frame."""
    yaw_term = g1_29dof_wbt_force_v2_command.step_terms["wrist_force_tracking_command"]
    full_term = g1_29dof_wbt_force_v2_fullbase_command.step_terms["wrist_force_tracking_command"]

    yaw_cfg_d = asdict(yaw_term.params["wrist_force_tracking_config"])
    full_cfg_d = asdict(full_term.params["wrist_force_tracking_config"])
    diffs = {k for k in yaw_cfg_d if yaw_cfg_d[k] != full_cfg_d.get(k)}
    assert diffs == {"force_cmd_obs_frame"}, (
        f"Two presets should differ only in force_cmd_obs_frame, actual diffs: {diffs}"
    )


def test_preset_does_not_carry_v10_term() -> None:
    for preset in (g1_29dof_wbt_force_v2_command, g1_29dof_wbt_force_v2_fullbase_command):
        assert "wrist_compliance_command" not in preset.setup_terms
        assert "wrist_compliance_command" not in preset.reset_terms
        assert "wrist_compliance_command" not in preset.step_terms


def test_preset_keeps_baseline_motion_command() -> None:
    for preset in (g1_29dof_wbt_force_v2_command, g1_29dof_wbt_force_v2_fullbase_command):
        for key in g1_29dof_wbt_command.setup_terms:
            assert key in preset.setup_terms


def test_preset_has_force_curriculum_disabled_by_default() -> None:
    """Both presets must default to enable_force_curriculum=False so
    CLI without a flag matches pre-curriculum behavior exactly. Enable via
    CLI --command...enable_force_curriculum=True for ablation."""
    for preset in (g1_29dof_wbt_force_v2_command, g1_29dof_wbt_force_v2_fullbase_command):
        term = preset.step_terms["wrist_force_tracking_command"]
        cfg = term.params["wrist_force_tracking_config"]
        assert cfg.enable_force_curriculum is False, (
            "preset must keep curriculum off; enable via CLI --command...enable_force_curriculum=True"
        )
