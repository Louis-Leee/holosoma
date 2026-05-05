"""Regression tests for the g1_29dof_wbt_force experiment preset (Task 9)."""

from __future__ import annotations

from typing import cast

from holosoma.config_types.algo import PPOAlgoConfig, PPOConfig
from holosoma.config_values.experiment import DEFAULTS
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt, g1_29dof_wbt_force

ORIGINAL_ALGO = g1_29dof_wbt.algo


def _ppo_config(algo_config) -> PPOConfig:  # type: ignore[no-untyped-def]
    return cast("PPOAlgoConfig", algo_config).config


def test_new_experiment_registered_in_defaults() -> None:
    assert DEFAULTS["g1_29dof_wbt_force"] is g1_29dof_wbt_force


def test_baseline_algo_reference_untouched() -> None:
    assert DEFAULTS["g1_29dof_wbt"].algo is ORIGINAL_ALGO


def test_baseline_hidden_dims_unchanged() -> None:
    # Baseline still uses the smaller [512, 256, 128] MLP.
    cfg = _ppo_config(g1_29dof_wbt.algo)
    assert cfg.module_dict.actor.layer_config.hidden_dims == [512, 256, 128]


def test_force_env_class_points_to_new_subclass() -> None:
    assert g1_29dof_wbt_force.env_class == ("holosoma.envs.wbt.wbt_force_injected.WholeBodyTrackingForceInjected")


def test_force_uses_new_presets() -> None:
    from holosoma.config_values.wbt.g1.command_force import (
        g1_29dof_wbt_force_command,
    )
    from holosoma.config_values.wbt.g1.observation_force import (
        g1_29dof_wbt_force_observation,
    )
    from holosoma.config_values.wbt.g1.reward_force import (
        g1_29dof_wbt_force_reward,
    )

    assert g1_29dof_wbt_force.command is g1_29dof_wbt_force_command
    assert g1_29dof_wbt_force.observation is g1_29dof_wbt_force_observation
    assert g1_29dof_wbt_force.reward is g1_29dof_wbt_force_reward


def test_force_mlp_widened_to_1024_512_256() -> None:
    cfg = _ppo_config(g1_29dof_wbt_force.algo)
    assert cfg.module_dict.actor.layer_config.hidden_dims == [1024, 512, 256]
    assert cfg.module_dict.critic.layer_config.hidden_dims == [1024, 512, 256]


def test_force_algo_input_dim_matches_baseline() -> None:
    force_cfg = _ppo_config(g1_29dof_wbt_force.algo)
    base_cfg = _ppo_config(g1_29dof_wbt.algo)
    assert force_cfg.module_dict.actor.input_dim == base_cfg.module_dict.actor.input_dim == ["actor_obs"]
    assert force_cfg.module_dict.critic.input_dim == base_cfg.module_dict.critic.input_dim == ["critic_obs"]
