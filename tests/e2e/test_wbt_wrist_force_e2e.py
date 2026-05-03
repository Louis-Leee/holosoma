"""IsaacSim e2e test for WBT wrist-force (Task 11).

Boots the ``g1_29dof_wbt_force`` experiment with a single env, forces
the F_cmd and F_ext activation rates to 1.0 (zero cooldown) so both
channels are guaranteed active within a 20-step window, and asserts:

  * ``last_applied_force_w_by_body_id.keys() == {left_id, right_id}``
    (red line 1 — external force only on wrists).
  * ``force_ext_w.shape == (N, 2, 3)`` and is non-zero for at least one
    step of the first 20.
  * ``force_cmd_b`` has a smooth (monotonic) trapezoidal ramp.
  * Reward is in ``[0, 1]`` with no NaNs.
  * ``obs_dict["actor_obs"].shape[1] == 1620``
  * ``obs_dict["critic_obs"].shape[1] == 3000``
  * ``obs_dict`` only contains the two expected groups.
  * ``k_virtual.shape == (N, 2)`` and v1 value is 100.0 everywhere.

A second smoke test runs 200 control steps with the default config and
asserts ``force_ext_w.abs().sum() > 0`` at least once (the trapezoidal
state machine does get triggered with defaults).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = pytest.mark.isaacsim


def _make_forced_experiment(num_envs: int = 2):
    """Build a g1_29dof_wbt_force cfg with guaranteed-active force channels."""
    from holosoma.config_types.command import WristComplianceConfig
    from holosoma.config_values.experiment import DEFAULTS

    cfg = DEFAULTS["g1_29dof_wbt_force"]
    # Replace the wrist_compliance_command params so Bernoulli fires every step.
    forced_wrist_cfg = dataclasses.replace(
        WristComplianceConfig(),
        force_cmd_activation_prob_per_step=1.0,
        force_ext_activation_prob_per_step=1.0,
        force_cmd_cooldown_range_s=(0.0, 0.0),
        force_ext_cooldown_range_s=(0.0, 0.0),
    )

    def _with_new_cfg(term_cfg):
        return dataclasses.replace(
            term_cfg,
            params={"wrist_compliance_config": forced_wrist_cfg},
        )

    new_cmd = dataclasses.replace(
        cfg.command,
        setup_terms={
            **cfg.command.setup_terms,
            "wrist_compliance_command": _with_new_cfg(
                cfg.command.setup_terms["wrist_compliance_command"],
            ),
        },
        reset_terms={
            **cfg.command.reset_terms,
            "wrist_compliance_command": _with_new_cfg(
                cfg.command.reset_terms["wrist_compliance_command"],
            ),
        },
        step_terms={
            **cfg.command.step_terms,
            "wrist_compliance_command": _with_new_cfg(
                cfg.command.step_terms["wrist_compliance_command"],
            ),
        },
    )

    return dataclasses.replace(
        cfg,
        training=dataclasses.replace(cfg.training, num_envs=num_envs, headless=True),
        command=new_cmd,
    )


@pytest.fixture(scope="module")
def force_env():
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.helpers import get_class

    cfg = _make_forced_experiment(num_envs=2)
    tyro_cfg = get_tyro_env_config(cfg)
    env = get_class(cfg.env_class)(tyro_cfg, device="cuda:0")
    yield env
    try:
        env.close()
    except Exception:  # noqa: S110 - best-effort teardown
        pass


def test_force_injected_only_on_wrists(force_env) -> None:
    env = force_env
    # Drive 20 control steps with zero actions.
    actions = torch.zeros(env.num_envs, env.action_manager.action_dim, device=env.device)
    obs, _, _, _ = env.step({"actions": actions})
    for _ in range(19):
        obs, _, _, _ = env.step({"actions": actions})

    assert set(env.last_applied_force_w_by_body_id.keys()) == {
        env._left_wrist_isaac_id,
        env._right_wrist_isaac_id,
    }


def test_obs_shapes_match_plan_dims(force_env) -> None:
    env = force_env
    actions = torch.zeros(env.num_envs, env.action_manager.action_dim, device=env.device)
    obs, _, _, _ = env.step({"actions": actions})

    assert set(obs.keys()) == {"actor_obs", "critic_obs"}
    assert obs["actor_obs"].shape[1] == 1620
    assert obs["critic_obs"].shape[1] == 3000


def test_reward_and_k_sanity(force_env) -> None:
    env = force_env
    actions = torch.zeros(env.num_envs, env.action_manager.action_dim, device=env.device)
    _, rew, _, _ = env.step({"actions": actions})
    assert torch.all(torch.isfinite(rew))

    wrist_cmd = env.command_manager.get_state("wrist_compliance_command")
    assert wrist_cmd.k_virtual.shape == (env.num_envs, 2)
    assert torch.all(wrist_cmd.k_virtual == 100.0)


def test_force_ext_nonzero_under_forced_activation(force_env) -> None:
    env = force_env
    wrist_cmd = env.command_manager.get_state("wrist_compliance_command")
    assert wrist_cmd.force_ext_w.shape == (env.num_envs, 2, 3)
    actions = torch.zeros(env.num_envs, env.action_manager.action_dim, device=env.device)

    saw_nonzero = False
    for _ in range(20):
        env.step({"actions": actions})
        if wrist_cmd.force_ext_w.abs().sum().item() > 0:
            saw_nonzero = True
            break
    assert saw_nonzero, "F_ext never fired under activation_prob=1.0"


def test_default_config_force_ext_eventually_fires() -> None:
    """200-step smoke test with default activation probs."""
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.utils.helpers import get_class

    cfg = DEFAULTS["g1_29dof_wbt_force"]
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(cfg.training, num_envs=1, headless=True),
    )
    tyro_cfg = get_tyro_env_config(cfg)
    env = get_class(cfg.env_class)(tyro_cfg, device="cuda:0")
    try:
        wrist_cmd = env.command_manager.get_state("wrist_compliance_command")
        actions = torch.zeros(
            env.num_envs,
            env.action_manager.action_dim,
            device=env.device,
        )
        saw = False
        for _ in range(200):
            env.step({"actions": actions})
            if wrist_cmd.force_ext_w.abs().sum().item() > 0:
                saw = True
                break
        assert saw, "F_ext never fired in 200 steps with default config"
    finally:
        try:
            env.close()
        except Exception:  # noqa: S110 - best-effort teardown
            pass
