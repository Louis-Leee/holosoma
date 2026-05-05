"""Unit tests for :class:`WristComplianceConfig` (WBT wrist-force v10 Task 1)."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from holosoma.config_types.command import WristComplianceConfig

# pydantic wraps AssertionError raised in __post_init__ into ValidationError
# (tagged as assertion_error). We accept either shape — tests should assert
# "construction failed", not the exact exception class.
_ConfigError = (AssertionError, ValidationError)


def test_default_construction_succeeds() -> None:
    cfg = WristComplianceConfig()
    # K range defaults to the v1 constant (100, 100).
    assert cfg.k_virtual_range == (100.0, 100.0)
    assert cfg.left_wrist_body_name == "left_wrist_yaw_link"
    assert cfg.right_wrist_body_name == "right_wrist_yaw_link"
    assert cfg.enable_left and cfg.enable_right
    assert cfg.debug_arrow_scale_n_per_m == 50.0


def test_is_frozen() -> None:
    cfg = WristComplianceConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enable_left = False  # type: ignore[misc]


def test_k_virtual_range_lower_bound_must_be_positive() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(k_virtual_range=(0.0, 100.0))


def test_k_virtual_range_must_be_ordered() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(k_virtual_range=(200.0, 50.0))


def test_k_virtual_range_widened_v2_succeeds() -> None:
    cfg = WristComplianceConfig(k_virtual_range=(50.0, 300.0))
    assert cfg.k_virtual_range == (50.0, 300.0)


def test_debug_arrow_scale_must_be_positive() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(debug_arrow_scale_n_per_m=0.0)


def test_magnitude_range_lower_bound_must_be_non_negative() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(force_cmd_magnitude_range=(-1.0, 10.0))


def test_magnitude_range_must_be_ordered() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(force_ext_magnitude_range=(50.0, 10.0))


def test_duration_range_must_be_non_negative() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(force_cmd_duration_range_s=(-0.1, 1.0))


def test_cooldown_range_must_be_ordered() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(force_ext_cooldown_range_s=(2.0, 1.0))


def test_ramp_frac_in_bounds() -> None:
    with pytest.raises(_ConfigError):
        WristComplianceConfig(force_cmd_ramp_frac=-0.1)
    with pytest.raises(_ConfigError):
        WristComplianceConfig(force_ext_ramp_frac=0.6)
    # Boundary values accepted.
    WristComplianceConfig(force_cmd_ramp_frac=0.0)
    WristComplianceConfig(force_ext_ramp_frac=0.5)
