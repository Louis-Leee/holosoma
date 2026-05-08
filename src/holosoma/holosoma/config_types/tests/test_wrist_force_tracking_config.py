"""Regression tests for WristForceTrackingConfig (v14)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from holosoma.config_types.command_v2 import WristForceTrackingConfig

# pydantic wraps AssertionError raised in __post_init__ into ValidationError
_ConfigError = (AssertionError, ValidationError)


def test_defaults_construct_without_error() -> None:
    cfg = WristForceTrackingConfig()
    # v14 only keeps F_ext schedule (copied from v10 _ext_channel ranges)
    assert cfg.force_ext_magnitude_range == (5.0, 30.0)
    assert cfg.force_ext_duration_range_s == (1.0, 3.0)
    assert cfg.force_ext_cooldown_range_s == (0.5, 2.0)
    assert cfg.force_ext_ramp_frac == 0.25
    assert cfg.force_ext_activation_prob_per_step == 0.01
    assert cfg.enable_left is True
    assert cfg.enable_right is True
    assert cfg.left_wrist_body_name == "left_wrist_yaw_link"
    assert cfg.right_wrist_body_name == "right_wrist_yaw_link"
    assert cfg.debug_arrow_scale_n_per_m == 50.0
    # default obs frame is yaw_only (learning-compliance style); full_base is ablation.
    assert cfg.force_cmd_obs_frame == "yaw_only"
    # default force curriculum is off (matches current "direct 5-30 N" behavior); on is ablation.
    assert cfg.enable_force_curriculum is False
    assert cfg.curriculum_initial_magnitude_range == (0.0, 5.0)
    assert cfg.curriculum_ramp_steps == 10_000


def test_force_curriculum_accepts_enabled_and_disabled() -> None:
    cfg_off = WristForceTrackingConfig(enable_force_curriculum=False)
    cfg_on = WristForceTrackingConfig(
        enable_force_curriculum=True,
        curriculum_initial_magnitude_range=(0.0, 5.0),
        curriculum_ramp_steps=20_000,
    )
    assert cfg_off.enable_force_curriculum is False
    assert cfg_on.enable_force_curriculum is True
    assert cfg_on.curriculum_ramp_steps == 20_000


def test_curriculum_initial_range_must_not_exceed_target() -> None:
    """curriculum initial peak must not exceed target peak (must be ramp up, not ramp down)."""
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(
            enable_force_curriculum=True,
            curriculum_initial_magnitude_range=(0.0, 40.0),  # target hi only 30
            force_ext_magnitude_range=(5.0, 30.0),
        )


def test_curriculum_ramp_steps_must_be_positive() -> None:
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(
            enable_force_curriculum=True,
            curriculum_ramp_steps=0,
        )


def test_curriculum_initial_range_sanity() -> None:
    """lo >= 0, lo <= hi."""
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(curriculum_initial_magnitude_range=(-1.0, 5.0))
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(curriculum_initial_magnitude_range=(10.0, 5.0))


def test_force_cmd_obs_frame_accepts_both_variants() -> None:
    cfg_yaw = WristForceTrackingConfig(force_cmd_obs_frame="yaw_only")
    cfg_full = WristForceTrackingConfig(force_cmd_obs_frame="full_base")
    assert cfg_yaw.force_cmd_obs_frame == "yaw_only"
    assert cfg_full.force_cmd_obs_frame == "full_base"


def test_force_cmd_obs_frame_rejects_unknown_value() -> None:
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(force_cmd_obs_frame="world")  # type: ignore[arg-type]


def test_magnitude_lo_cannot_exceed_hi() -> None:
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(force_ext_magnitude_range=(30.0, 5.0))


def test_magnitude_lo_must_be_non_negative() -> None:
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(force_ext_magnitude_range=(-1.0, 30.0))


def test_ramp_frac_must_be_in_0_half() -> None:
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(force_ext_ramp_frac=0.6)


def test_debug_arrow_scale_must_be_positive() -> None:
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(debug_arrow_scale_n_per_m=0.0)


def test_activation_prob_must_be_in_range() -> None:
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(force_ext_activation_prob_per_step=0.0)
    with pytest.raises(_ConfigError):
        WristForceTrackingConfig(force_ext_activation_prob_per_step=1.1)


def test_no_k_virtual_attribute() -> None:
    """v14 explicitly drops K_virtual."""
    cfg = WristForceTrackingConfig()
    assert not hasattr(cfg, "k_virtual_range")


def test_no_independent_f_cmd_schedule() -> None:
    """v14 derives F_cmd from F_ext; no independent F_cmd schedule."""
    cfg = WristForceTrackingConfig()
    for field_name in (
        "force_cmd_magnitude_range",
        "force_cmd_duration_range_s",
        "force_cmd_cooldown_range_s",
        "force_cmd_ramp_frac",
        "force_cmd_activation_prob_per_step",
    ):
        assert not hasattr(cfg, field_name), f"v14 must not expose {field_name}"
