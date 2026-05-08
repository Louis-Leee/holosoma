"""Tyro CLI registry tests — verifies all experiment presets are reachable
via the EXPERIMENTS dict used by AnnotatedExperimentConfig."""

from __future__ import annotations

import pytest
from holosoma.config_values.experiment import EXPERIMENTS
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

# ------------------------------ v14 (force v2) -----------------------------

V14_EXPERIMENT_KEYS = ("g1_29dof_wbt_force_v2", "g1_29dof_wbt_force_v2_fullbase")


@pytest.mark.parametrize("exp_key", V14_EXPERIMENT_KEYS)
def test_v2_exp_registered(exp_key: str) -> None:
    assert exp_key in EXPERIMENTS


@pytest.mark.parametrize("exp_key", V14_EXPERIMENT_KEYS)
def test_v2_preset_uses_v14_env_class(exp_key: str) -> None:
    preset = EXPERIMENTS[exp_key]
    assert preset.env_class.endswith("WholeBodyTrackingForceInjectedV2")


@pytest.mark.parametrize("exp_key", V14_EXPERIMENT_KEYS)
def test_v2_preset_reward_is_baseline(exp_key: str) -> None:
    preset = EXPERIMENTS[exp_key]
    assert set(preset.reward.terms.keys()) == set(g1_29dof_wbt_reward.terms.keys())


@pytest.mark.parametrize("exp_key", V14_EXPERIMENT_KEYS)
def test_v2_preset_does_not_inherit_v10_fields(exp_key: str) -> None:
    """Both V14 experiments must start from baseline. Termination / curriculum /
    randomization must match baseline, not leak from V10 preset."""
    base = EXPERIMENTS["g1_29dof_wbt"]
    v2 = EXPERIMENTS[exp_key]
    assert v2.termination == base.termination
    assert v2.curriculum == base.curriculum
    assert v2.randomization == base.randomization


@pytest.mark.parametrize("exp_key", V14_EXPERIMENT_KEYS)
def test_v2_actor_hidden_dims_explicit(exp_key: str) -> None:
    """V14 MLP dims must be explicitly written as [1024, 512, 256] (borrowed from V10)."""
    v2 = EXPERIMENTS[exp_key]
    assert v2.algo.config.module_dict.actor.layer_config.hidden_dims == [1024, 512, 256]
    assert v2.algo.config.module_dict.critic.layer_config.hidden_dims == [1024, 512, 256]


def test_v2_two_variants_differ_only_in_command() -> None:
    """The two V14 experiments must be identical except in their command preset (frame decides everything)."""
    yaw = EXPERIMENTS["g1_29dof_wbt_force_v2"]
    full = EXPERIMENTS["g1_29dof_wbt_force_v2_fullbase"]

    assert yaw.env_class == full.env_class
    assert yaw.observation is full.observation or (
        set(yaw.observation.groups) == set(full.observation.groups)
    )
    assert set(yaw.reward.terms) == set(full.reward.terms)
    yaw_term = yaw.command.step_terms["wrist_force_tracking_command"]
    full_term = full.command.step_terms["wrist_force_tracking_command"]
    assert yaw_term.params["wrist_force_tracking_config"].force_cmd_obs_frame == "yaw_only"
    assert full_term.params["wrist_force_tracking_config"].force_cmd_obs_frame == "full_base"
