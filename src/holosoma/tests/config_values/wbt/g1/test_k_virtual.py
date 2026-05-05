"""Tests for the virtual stiffness constants used by WBT wrist-force training."""

from __future__ import annotations

from holosoma.config_values.wbt.g1 import _k_virtual


def test_k_virtual_range_is_positive_and_ordered() -> None:
    lo, hi = _k_virtual.K_VIRTUAL_RANGE_N_PER_M
    assert lo > 0.0, "K must be strictly positive (reward uses 1/K)"
    assert lo <= hi, "range lower bound must be <= upper bound"


def test_k_virtual_range_is_tuple_of_floats() -> None:
    rng = _k_virtual.K_VIRTUAL_RANGE_N_PER_M
    assert isinstance(rng, tuple), "K range must be a tuple for dataclass frozen use"
    assert len(rng) == 2
    assert all(isinstance(x, float) for x in rng)


def test_alias_matches_range_lower_bound() -> None:
    assert _k_virtual.K_VIRTUAL_RANGE_N_PER_M[0] == _k_virtual.G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M


def test_v1_deterministic_range() -> None:
    # v1 contract: range endpoints equal so sampling is deterministic.
    lo, hi = _k_virtual.K_VIRTUAL_RANGE_N_PER_M
    assert lo == hi == 100.0
