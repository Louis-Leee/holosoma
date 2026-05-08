"""Unit tests for wrist_force_command_v2 / wrist_force_ext_privileged_v2."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.command_v2 import WristForceTrackingConfig
from holosoma.managers.command.terms.wbt_force_v2 import WristForceTrackingCommand
from holosoma.managers.observation.terms.wbt_force_v2 import (
    wrist_force_command_v2,
    wrist_force_ext_privileged_v2,
)


class _CmdMgr:
    def __init__(self, term: WristForceTrackingCommand | None) -> None:
        self._term = term

    def get_state(self, name: str) -> Any:
        if name == "wrist_force_tracking_command":
            return self._term
        return None


def _make_env(num_envs: int = 4) -> Any:
    device = torch.device("cpu")
    base_quat = torch.zeros(num_envs, 4)
    base_quat[:, 3] = 1.0
    env = SimpleNamespace(num_envs=num_envs, device=device, dt=0.02, base_quat=base_quat)
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand",
        params={"wrist_force_tracking_config": WristForceTrackingConfig(
            force_ext_activation_prob_per_step=1.0,
        )},
    )
    term = WristForceTrackingCommand(cfg, env)
    term.reset(None)
    for _ in range(5):
        term.step()
    env.command_manager = _CmdMgr(term)
    return env


def test_wrist_force_command_v2_shape_and_values() -> None:
    env = _make_env()
    out = wrist_force_command_v2(env)
    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_wrist_force_ext_privileged_v2_shape() -> None:
    env = _make_env()
    out = wrist_force_ext_privileged_v2(env)
    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_obs_is_a_clone_not_a_view() -> None:
    env = _make_env()
    cmd = wrist_force_command_v2(env)
    cmd.fill_(123.0)
    term = env.command_manager.get_state("wrist_force_tracking_command")
    assert not torch.all(term.force_cmd_b == 123.0)


def test_missing_command_raises() -> None:
    env = SimpleNamespace(num_envs=1, command_manager=_CmdMgr(None))
    with pytest.raises(RuntimeError, match="wrist_force_tracking_command"):
        wrist_force_command_v2(env)
