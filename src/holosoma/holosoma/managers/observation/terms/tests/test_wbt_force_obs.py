"""CPU unit tests for the three WBT wrist-force observation terms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.command import CommandTermCfg, WristComplianceConfig
from holosoma.managers.command.terms.wbt_force import (
    NUM_WRISTS,
    WristComplianceCommand,
)
from holosoma.managers.observation.terms.wbt_force import (
    wrist_force_command,
    wrist_force_ext_privileged,
    wrist_virtual_stiffness_command,
)


def _fake_env(num_envs: int = 4) -> SimpleNamespace:
    term = WristComplianceCommand(
        CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_force:WristComplianceCommand",
            params={"wrist_compliance_config": WristComplianceConfig()},
        ),
        SimpleNamespace(num_envs=num_envs, device="cpu", dt=0.02),
    )
    term.reset(env_ids=None)
    # Poke some non-zero values so shape+value checks can distinguish them.
    term._cmd_channel.force[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    term._cmd_channel.force[:, 1] = torch.tensor([4.0, 5.0, 6.0])
    term._ext_channel.force[:, 0] = torch.tensor([7.0, 8.0, 9.0])
    term._ext_channel.force[:, 1] = torch.tensor([10.0, 11.0, 12.0])
    term.k_virtual[:] = torch.tensor([100.0, 200.0])

    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        command_manager=SimpleNamespace(
            get_state=lambda name: term if name == "wrist_compliance_command" else None,
        ),
    )


def test_wrist_force_command_shape_and_values() -> None:
    env = _fake_env()
    obs = wrist_force_command(env)  # type: ignore[arg-type]
    assert obs.shape == (4, 6)
    # First wrist 3D block, second wrist 3D block.
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    torch.testing.assert_close(obs[0:1], expected)


def test_wrist_force_ext_privileged_shape_and_values() -> None:
    env = _fake_env()
    obs = wrist_force_ext_privileged(env)  # type: ignore[arg-type]
    assert obs.shape == (4, 6)
    expected = torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    torch.testing.assert_close(obs[0:1], expected)


def test_wrist_virtual_stiffness_command_shape_and_values() -> None:
    env = _fake_env()
    obs = wrist_virtual_stiffness_command(env)  # type: ignore[arg-type]
    assert obs.shape == (4, NUM_WRISTS)
    expected = torch.tensor([[100.0, 200.0]])
    torch.testing.assert_close(obs[0:1], expected)


def test_each_obs_term_returns_a_clone() -> None:
    """Mutating the obs tensor must not mutate the underlying buffer.

    This guards against obs-history buffers mutating the command state.
    """
    env = _fake_env()
    for term_fn, original_buffer in (
        (
            wrist_force_command,
            env.command_manager.get_state("wrist_compliance_command").force_cmd_b,
        ),
        (
            wrist_force_ext_privileged,
            env.command_manager.get_state("wrist_compliance_command").force_ext_w,
        ),
        (
            wrist_virtual_stiffness_command,
            env.command_manager.get_state("wrist_compliance_command").k_virtual,
        ),
    ):
        obs = term_fn(env)  # type: ignore[arg-type]
        before = original_buffer.clone()
        obs.zero_()
        assert torch.equal(original_buffer, before), f"{term_fn.__name__} aliased the underlying buffer"


def test_raises_when_command_not_registered() -> None:
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=SimpleNamespace(get_state=lambda _name: None),
    )
    with pytest.raises(RuntimeError, match="wrist_compliance_command"):
        wrist_force_command(env)  # type: ignore[arg-type]
