"""CPU-only unit tests for :mod:`holosoma.eval_agent_force`.

Covers:
  * CLI schema: both ``--enable-force-cmd`` and ``--enable-force-ext`` flags
    are registered (via tyro).
  * Config rewrite: :func:`_apply_force_eval_flags` and
    :func:`_rewrite_wrist_compliance_cfg` correctly zero each channel across
    all three command buckets (``setup_terms`` / ``reset_terms`` /
    ``step_terms``) in the 4 diagnostic modes.
  * Immutability: rewrite returns a new object; caller's input is not mutated.
  * Preflight contract: :func:`_assert_force_aware_ckpt` raises actionable
    ``RuntimeError`` for every failure mode (a/a'/b/c/d/e) described in the
    v11 eval plan §5.2.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import replace
from typing import Any

import pytest
import tyro
from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    WristComplianceConfig,
)
from holosoma.config_types.eval_force import ForceEvalConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

from holosoma import eval_agent_force

_WRIST_FORCE_FUNC = "holosoma.managers.command.terms.wbt_force:WristComplianceCommand"
_FORCE_AWARE_ENV_CLASS = (
    "holosoma.envs.wbt.wbt_force_injected.WholeBodyTrackingForceInjected"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrist_force_term(wcfg: WristComplianceConfig | None = None) -> CommandTermCfg:
    return CommandTermCfg(
        func=_WRIST_FORCE_FUNC,
        params={"wrist_compliance_config": wcfg if wcfg is not None else WristComplianceConfig()},
    )


def _make_force_aware_command(
    *,
    setup_present: bool = True,
    reset_present: bool = True,
    step_present: bool = True,
    setup_term: CommandTermCfg | None = None,
) -> CommandManagerCfg:
    term = setup_term if setup_term is not None else _make_wrist_force_term()
    setup_terms = {"wrist_compliance_command": term} if setup_present else {}
    reset_terms = {"wrist_compliance_command": _make_wrist_force_term()} if reset_present else {}
    step_terms = {"wrist_compliance_command": _make_wrist_force_term()} if step_present else {}
    return CommandManagerCfg(
        setup_terms=setup_terms,
        reset_terms=reset_terms,
        step_terms=step_terms,
    )


def _make_force_aware_observation() -> ObservationManagerCfg:
    # Minimal actor_obs group with the required ``wrist_force_command`` term.
    # Term func string is arbitrary — preflight only checks key presence.
    actor_terms = {
        "wrist_force_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt_force:wrist_force_command",
        ),
    }
    actor_obs = ObsGroupCfg(terms=actor_terms)
    return ObservationManagerCfg(groups={"actor_obs": actor_obs})


def make_force_aware_exp_config(**overrides: Any) -> ExperimentConfig:
    """Build a minimal force-aware ``ExperimentConfig`` for preflight tests.

    Only the fields :func:`_assert_force_aware_ckpt` and the rewrite helpers
    touch are populated; everything else defaults. Overrides replace any
    top-level field by name (e.g. ``command=None`` or ``env_class="..."``).
    """
    base_kwargs: dict[str, Any] = {
        "env_class": _FORCE_AWARE_ENV_CLASS,
        "command": _make_force_aware_command(),
        "observation": _make_force_aware_observation(),
    }
    base_kwargs.update(overrides)
    # ExperimentConfig is a dataclass — construct via ``replace`` on a default
    # instance so we keep the other non-trivial default subfields intact.
    return replace(ExperimentConfig(), **base_kwargs)


# ---------------------------------------------------------------------------
# CLI + rewrite tests
# ---------------------------------------------------------------------------


def test_cli_help_lists_both_flags() -> None:
    """Both ``--enable-force-cmd`` and ``--enable-force-ext`` appear in help."""
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit):
        tyro.cli(ForceEvalConfig, args=["--help"])
    out = buf.getvalue()
    assert "enable-force-cmd" in out
    assert "enable-force-ext" in out


def test_mode_0_zeros_both_channels_across_all_buckets() -> None:
    """Mode 0 (False, False) zeros F_cmd + F_ext in all 3 buckets."""
    exp = make_force_aware_exp_config()
    out = eval_agent_force._rewrite_wrist_compliance_cfg(
        exp,
        lambda c: eval_agent_force._apply_force_eval_flags(
            c, enable_force_cmd=False, enable_force_ext=False
        ),
    )
    cmd = out.command
    assert cmd is not None
    for bucket in (cmd.setup_terms, cmd.reset_terms, cmd.step_terms):
        wcfg = bucket["wrist_compliance_command"].params["wrist_compliance_config"]
        assert wcfg.force_cmd_magnitude_range == (0.0, 0.0)
        assert wcfg.force_cmd_activation_prob_per_step == 0.0
        assert wcfg.force_ext_magnitude_range == (0.0, 0.0)
        assert wcfg.force_ext_activation_prob_per_step == 0.0


def test_mode_1_zeros_only_force_ext() -> None:
    """Mode 1 (True, False) keeps F_cmd defaults, zeros F_ext."""
    exp = make_force_aware_exp_config()
    out = eval_agent_force._rewrite_wrist_compliance_cfg(
        exp,
        lambda c: eval_agent_force._apply_force_eval_flags(
            c, enable_force_cmd=True, enable_force_ext=False
        ),
    )
    cmd = out.command
    assert cmd is not None
    for bucket in (cmd.setup_terms, cmd.reset_terms, cmd.step_terms):
        wcfg = bucket["wrist_compliance_command"].params["wrist_compliance_config"]
        assert wcfg.force_cmd_magnitude_range == (5.0, 30.0)
        assert wcfg.force_cmd_activation_prob_per_step == 0.01
        assert wcfg.force_ext_magnitude_range == (0.0, 0.0)
        assert wcfg.force_ext_activation_prob_per_step == 0.0


def test_mode_2_zeros_only_force_cmd() -> None:
    """Mode 2 (False, True) zeros F_cmd, keeps F_ext defaults."""
    exp = make_force_aware_exp_config()
    out = eval_agent_force._rewrite_wrist_compliance_cfg(
        exp,
        lambda c: eval_agent_force._apply_force_eval_flags(
            c, enable_force_cmd=False, enable_force_ext=True
        ),
    )
    cmd = out.command
    assert cmd is not None
    for bucket in (cmd.setup_terms, cmd.reset_terms, cmd.step_terms):
        wcfg = bucket["wrist_compliance_command"].params["wrist_compliance_config"]
        assert wcfg.force_cmd_magnitude_range == (0.0, 0.0)
        assert wcfg.force_cmd_activation_prob_per_step == 0.0
        # Training default for ext magnitude is (0.0, 30.0).
        assert wcfg.force_ext_magnitude_range == (0.0, 30.0)
        assert wcfg.force_ext_activation_prob_per_step == 0.01


def test_mode_3_is_noop_identity() -> None:
    """Mode 3 (True, True) short-circuits: _apply_force_eval_flags returns
    the input config unchanged (``is`` identity), and ``main()`` skips the
    rewrite entirely. We check the short-circuit at the unit level."""
    cfg = WristComplianceConfig()
    returned = eval_agent_force._apply_force_eval_flags(
        cfg, enable_force_cmd=True, enable_force_ext=True
    )
    assert returned is cfg  # no replace() called, object identity preserved


def test_rewrite_is_immutable() -> None:
    """_rewrite_wrist_compliance_cfg returns a new object; the input
    ExperimentConfig and its nested command config are not mutated."""
    exp_before = make_force_aware_exp_config()
    cmd_before = exp_before.command
    assert cmd_before is not None
    setup_before = cmd_before.setup_terms["wrist_compliance_command"]
    wcfg_before = setup_before.params["wrist_compliance_config"]

    out = eval_agent_force._rewrite_wrist_compliance_cfg(
        exp_before,
        lambda c: eval_agent_force._apply_force_eval_flags(
            c, enable_force_cmd=False, enable_force_ext=False
        ),
    )

    # Returned object is a different ExperimentConfig.
    assert out is not exp_before
    assert out.command is not cmd_before
    # Original nested objects are still the same (input untouched).
    assert exp_before.command is cmd_before
    assert cmd_before.setup_terms["wrist_compliance_command"] is setup_before
    assert setup_before.params["wrist_compliance_config"] is wcfg_before
    # Original values are still training defaults.
    assert wcfg_before.force_cmd_magnitude_range == (5.0, 30.0)


# ---------------------------------------------------------------------------
# Preflight contract tests (_assert_force_aware_ckpt)
# ---------------------------------------------------------------------------


def test_preflight_rejects_none_command() -> None:
    """(a) command=None raises with 'command config is None'."""
    exp = make_force_aware_exp_config(command=None)
    with pytest.raises(RuntimeError, match="command config is None"):
        eval_agent_force._assert_force_aware_ckpt(exp)


@pytest.mark.parametrize(
    "missing_bucket",
    ["setup_terms", "reset_terms", "step_terms"],
)
def test_preflight_rejects_missing_bucket(missing_bucket: str) -> None:
    """(a') If any of the 3 buckets lacks the term, raise with its name."""
    flags = {
        "setup_present": missing_bucket != "setup_terms",
        "reset_present": missing_bucket != "reset_terms",
        "step_present": missing_bucket != "step_terms",
    }
    exp = make_force_aware_exp_config(command=_make_force_aware_command(**flags))
    with pytest.raises(
        RuntimeError,
        match=rf"command\.{missing_bucket} has no 'wrist_compliance_command' key",
    ):
        eval_agent_force._assert_force_aware_ckpt(exp)


def test_preflight_rejects_wrong_func_class() -> None:
    """(b) func that resolves to a different class raises with class name."""
    wrong_term = CommandTermCfg(
        func="holosoma.envs.wbt.wbt_manager:WholeBodyTrackingManager",
        params={"wrist_compliance_config": WristComplianceConfig()},
    )
    # All three buckets present, all pointing at the wrong class.
    exp = make_force_aware_exp_config(
        command=CommandManagerCfg(
            setup_terms={"wrist_compliance_command": wrong_term},
            reset_terms={"wrist_compliance_command": wrong_term},
            step_terms={"wrist_compliance_command": wrong_term},
        )
    )
    with pytest.raises(RuntimeError, match="resolves to WholeBodyTrackingManager"):
        eval_agent_force._assert_force_aware_ckpt(exp)


def test_preflight_rejects_wrong_params_type() -> None:
    """(c) params['wrist_compliance_config'] of wrong type raises with type."""
    bad_term = CommandTermCfg(
        func=_WRIST_FORCE_FUNC,
        params={"wrist_compliance_config": {"force_cmd_magnitude_range": (5, 30)}},
    )
    exp = make_force_aware_exp_config(
        command=CommandManagerCfg(
            setup_terms={"wrist_compliance_command": bad_term},
            reset_terms={"wrist_compliance_command": bad_term},
            step_terms={"wrist_compliance_command": bad_term},
        )
    )
    with pytest.raises(RuntimeError, match="is dict, expected WristComplianceConfig"):
        eval_agent_force._assert_force_aware_ckpt(exp)


def test_preflight_rejects_wrong_env_class() -> None:
    """(d) env_class not ending in WholeBodyTrackingForceInjected raises."""
    exp = make_force_aware_exp_config(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    )
    with pytest.raises(RuntimeError, match="WholeBodyTrackingForceInjected"):
        eval_agent_force._assert_force_aware_ckpt(exp)


def test_preflight_rejects_missing_actor_obs_term() -> None:
    """(e) actor_obs without ``wrist_force_command`` raises."""
    actor_obs = ObsGroupCfg(
        terms={
            "some_other_term": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:base_lin_vel"),
        }
    )
    observation = ObservationManagerCfg(groups={"actor_obs": actor_obs})
    exp = make_force_aware_exp_config(observation=observation)
    with pytest.raises(RuntimeError, match="Policy cannot be force-aware"):
        eval_agent_force._assert_force_aware_ckpt(exp)


def test_preflight_accepts_fully_valid_force_aware_config() -> None:
    """Smoke: a fully-valid force-aware config passes preflight silently."""
    exp = make_force_aware_exp_config()
    # Should not raise.
    eval_agent_force._assert_force_aware_ckpt(exp)
