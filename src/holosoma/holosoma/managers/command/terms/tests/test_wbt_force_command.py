"""CPU unit tests for :class:`WristComplianceCommand` + its gh-utils helpers.

All tests run without isaacsim — we mock the environment with a tiny
``types.SimpleNamespace`` carrying ``num_envs``, ``device``, and ``dt``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.command import CommandTermCfg, WristComplianceConfig
from holosoma.managers.command.terms._gh_utils import (
    TemporalLerp,
    clamp_norm,
    random_uniform,
)
from holosoma.managers.command.terms.wbt_force import (
    NUM_WRISTS,
    STATE_RAMP_UP,
    WristComplianceCommand,
)


def _fake_env(num_envs: int = 4, dt: float = 0.02) -> SimpleNamespace:
    return SimpleNamespace(num_envs=num_envs, device="cpu", dt=dt)


def _make_term(
    num_envs: int = 4,
    dt: float = 0.02,
    *,
    wrist_cfg: WristComplianceConfig | None = None,
) -> WristComplianceCommand:
    wrist_cfg = wrist_cfg or WristComplianceConfig()
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force:WristComplianceCommand",
        params={"wrist_compliance_config": wrist_cfg},
    )
    term = WristComplianceCommand(cfg, _fake_env(num_envs=num_envs, dt=dt))
    term.reset(env_ids=None)
    return term


# ---------------------------------------------------------------------------
# _gh_utils: TemporalLerp / clamp_norm / random_uniform
# ---------------------------------------------------------------------------
def test_gh_temporal_lerp_basic_ramp() -> None:
    lerp = TemporalLerp((3, 1), device=torch.device("cpu"), default=0.0)
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    lerp.set(env_ids, end=torch.tensor([[10.0], [5.0]]), total_steps=4)
    # At t=0 we should still be at the start (0.0).
    assert torch.allclose(lerp.value[env_ids, 0], torch.zeros(2))
    lerp.update_time(1)
    # Linear easing → 25% of end.
    assert torch.allclose(lerp.value[0], torch.tensor([10.0 / 4.0]))
    assert torch.allclose(lerp.value[2], torch.tensor([5.0 / 4.0]))
    # Finish the ramp.
    lerp.update_time(3)
    assert torch.allclose(lerp.value[0], torch.tensor([10.0]))
    assert torch.allclose(lerp.value[2], torch.tensor([5.0]))
    assert not lerp._active[0].item()


def test_gh_clamp_norm_caps_magnitude_without_rotating() -> None:
    x = torch.tensor([[3.0, 0.0, 4.0]])  # norm 5
    y = clamp_norm(x, max_norm=2.0)
    assert pytest.approx(float(y.norm()), rel=1e-5) == 2.0
    # Direction preserved: unit vector unchanged up to floating error.
    assert torch.allclose(y / y.norm(), x / x.norm(), atol=1e-6)


def test_gh_random_uniform_range_and_device() -> None:
    torch.manual_seed(0)
    vals = random_uniform((1000,), 5.0, 10.0, device=torch.device("cpu"))
    assert vals.device.type == "cpu"
    assert vals.min().item() >= 5.0
    assert vals.max().item() <= 10.0
    assert vals.mean().item() == pytest.approx(7.5, abs=0.5)


# ---------------------------------------------------------------------------
# WristComplianceCommand basic state
# ---------------------------------------------------------------------------
def test_reset_zeroes_forces_and_sets_k_to_range() -> None:
    term = _make_term()
    assert term.force_cmd_b.shape == (4, NUM_WRISTS, 3)
    assert term.force_ext_w.shape == (4, NUM_WRISTS, 3)
    assert torch.all(term.force_cmd_b == 0.0)
    assert torch.all(term.force_ext_w == 0.0)
    assert term.k_virtual.shape == (4, NUM_WRISTS)
    # v1 range collapses to 100.0.
    assert torch.all(term.k_virtual == 100.0)


def test_k_virtual_resample_on_reset_widened_range() -> None:
    wrist_cfg = WristComplianceConfig(k_virtual_range=(50.0, 300.0))
    term = _make_term(num_envs=32, wrist_cfg=wrist_cfg)
    assert term.k_virtual.min().item() >= 50.0
    assert term.k_virtual.max().item() <= 300.0
    # Left and right must be independently sampled → some env must have L != R.
    assert (term.k_virtual[:, 0] != term.k_virtual[:, 1]).any().item()
    # A second reset reshuffles.
    k_before = term.k_virtual.clone()
    term.reset(env_ids=None)
    assert not torch.allclose(term.k_virtual, k_before)


def test_k_virtual_unchanged_during_step() -> None:
    wrist_cfg = WristComplianceConfig(k_virtual_range=(50.0, 300.0))
    term = _make_term(num_envs=8, wrist_cfg=wrist_cfg)
    k_before = term.k_virtual.clone()
    for _ in range(20):
        term.step()
    assert torch.equal(term.k_virtual, k_before)


# ---------------------------------------------------------------------------
# Trapezoidal profile + channel independence
# ---------------------------------------------------------------------------
def _force_left_activation(term: WristComplianceCommand, channel: str) -> None:
    """Mutate the F_cmd or F_ext channel so left wrist starts RAMP_UP now."""
    ch = term._cmd_channel if channel == "cmd" else term._ext_channel
    # Fix direction +x, magnitude 20 on env 0, left wrist (=0).
    ch.direction[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    ch.peak_magnitude[0, 0] = 20.0

    # TemporalLerp flat index for (env=0, wrist=0) = 0.
    flat = 0
    lerp = ch._lerp
    lerp._start[flat] = 0.0
    lerp._end[flat] = 20.0
    lerp._t[flat] = 0
    lerp._T[flat] = 4  # 4 steps of ramp-up
    lerp._active[flat] = True
    lerp.value[flat] = 0.0
    ch.hold_steps[0, 0] = 2
    ch.ramp_up_steps[0, 0] = 4
    ch.ramp_down_steps[0, 0] = 4
    ch.state[0, 0] = STATE_RAMP_UP


def test_trapezoidal_profile_rises_holds_and_falls() -> None:
    term = _make_term(num_envs=2)
    _force_left_activation(term, "cmd")

    peaks = []
    for _ in range(15):
        term.step()
        peaks.append(term.force_cmd_b[0, 0, 0].item())

    # RAMP_UP (4 steps): monotonic rise toward 20.
    rising = peaks[:4]
    assert all(rising[i + 1] > rising[i] for i in range(3))
    assert pytest.approx(rising[-1], abs=1e-5) == 20.0

    # HOLD: peak equals 20.0 for hold_steps.
    assert peaks[4] == pytest.approx(20.0, abs=1e-5)
    assert peaks[5] == pytest.approx(20.0, abs=1e-5)

    # Eventually force comes back down to 0 within the 15 step window.
    assert peaks[-1] == pytest.approx(0.0, abs=1e-5)


def test_cmd_and_ext_channels_are_independent() -> None:
    term = _make_term(num_envs=2)
    _force_left_activation(term, "cmd")
    _force_left_activation(term, "ext")
    term.step()
    # Both channels should now be non-zero on env0 / left wrist.
    assert term.force_cmd_b[0, 0, 0].item() > 0.0
    assert term.force_ext_w[0, 0, 0].item() > 0.0


def test_force_ext_norm_respects_magnitude_range() -> None:
    # ext magnitude range (0, 10); force command is disabled via prob=0.
    wrist_cfg = WristComplianceConfig(
        force_cmd_activation_prob_per_step=0.0,
        force_ext_activation_prob_per_step=1.0,  # trigger asap
        force_ext_magnitude_range=(5.0, 10.0),
        force_ext_cooldown_range_s=(0.0, 0.0),
        force_ext_duration_range_s=(0.5, 0.5),
        force_ext_ramp_frac=0.25,
    )
    term = _make_term(num_envs=64, wrist_cfg=wrist_cfg)
    # Step enough to drive all envs through at least one HOLD.
    max_mag_seen = 0.0
    for _ in range(60):
        term.step()
        mag = term.force_ext_w.norm(dim=-1).max().item()
        max_mag_seen = max(max_mag_seen, mag)
    # Must never exceed upper bound.
    assert max_mag_seen <= 10.0 + 1e-4
    # Must actually reach at least the lower bound at some point.
    assert max_mag_seen >= 5.0 - 1e-4


def test_enable_left_false_keeps_left_wrist_zero() -> None:
    wrist_cfg = WristComplianceConfig(
        enable_left=False,
        force_cmd_activation_prob_per_step=1.0,
        force_ext_activation_prob_per_step=1.0,
        force_cmd_cooldown_range_s=(0.0, 0.0),
        force_ext_cooldown_range_s=(0.0, 0.0),
    )
    term = _make_term(num_envs=8, wrist_cfg=wrist_cfg)
    for _ in range(20):
        term.step()
    assert torch.all(term.force_cmd_b[:, 0] == 0.0)
    assert torch.all(term.force_ext_w[:, 0] == 0.0)


def test_property_reads_are_stable_without_step() -> None:
    term = _make_term(num_envs=2)
    _force_left_activation(term, "ext")
    term.step()
    a = term.force_ext_w.clone()
    b = term.force_ext_w  # two reads without advancing time
    assert torch.equal(a, b)


def test_update_metrics_populates_all_command_owned_keys() -> None:
    term = _make_term(num_envs=4)
    _force_left_activation(term, "cmd")
    _force_left_activation(term, "ext")
    term.step()
    term.update_metrics()

    required = {
        "force/cmd_magnitude_l",
        "force/cmd_magnitude_r",
        "force/cmd_magnitude_max",
        "force/ext_magnitude_l",
        "force/ext_magnitude_r",
        "force/ext_magnitude_max",
        "force/cmd_ext_alignment",
        "force/k_virtual_l",
        "force/k_virtual_r",
        "force/active_frac_cmd",
        "force/active_frac_ext",
        "force/phase_ramp_up",
        "force/phase_hold",
        "force/phase_ramp_down",
    }
    assert required.issubset(set(term.metrics.keys()))

    for key in required:
        val = term.metrics[key]
        assert isinstance(val, torch.Tensor), f"{key} must be a Tensor"
        assert val.device.type == "cpu"
        assert val.dtype == torch.float32
        assert val.shape == (4,), f"{key} shape {val.shape}"

    # Value-range sanity
    assert (term.metrics["force/cmd_magnitude_l"] >= 0).all()
    assert (term.metrics["force/k_virtual_l"] == 100.0).all()  # v1
    align = term.metrics["force/cmd_ext_alignment"]
    assert torch.all(align >= -1.0 - 1e-5)
    assert torch.all(align <= 1.0 + 1e-5)
    for phase_key in (
        "force/phase_ramp_up",
        "force/phase_hold",
        "force/phase_ramp_down",
    ):
        p = term.metrics[phase_key]
        assert (p >= 0).all() and (p <= 1).all(), phase_key


def test_update_metrics_matches_expected_state_counts() -> None:
    # Verify the state counters move correctly when we force RAMP_UP on left.
    term = _make_term(num_envs=2)
    _force_left_activation(term, "cmd")
    term.step()
    term.update_metrics()
    # Env 0 has one wrist in RAMP_UP, the other in COOLDOWN → mean 0.5.
    phase_up = term.metrics["force/phase_ramp_up"][0].item()
    assert phase_up == pytest.approx(0.5, abs=1e-5)
