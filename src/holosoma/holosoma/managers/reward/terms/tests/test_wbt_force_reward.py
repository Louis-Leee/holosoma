"""CPU unit tests for :func:`wrist_force_position_tracking_exp`.

All tests mock the environment with a tiny ``SimpleNamespace`` that
owns a mock simulator and a live ``WristComplianceCommand``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from holosoma.config_types.command import CommandTermCfg, WristComplianceConfig
from holosoma.managers.command.terms.wbt_force import WristComplianceCommand
from holosoma.managers.reward.terms.wbt_force import (
    _yaw_rotate_body_to_world,
    wrist_force_position_tracking_exp,
)

LEFT_ID = 11
RIGHT_ID = 22
LEFT_NAME = "left_wrist_yaw_link"
RIGHT_NAME = "right_wrist_yaw_link"


def _make_wrist_command(num_envs: int = 2, k_range=(100.0, 100.0)) -> WristComplianceCommand:
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force:WristComplianceCommand",
        params={
            "wrist_compliance_config": WristComplianceConfig(k_virtual_range=k_range),
        },
    )
    env = SimpleNamespace(num_envs=num_envs, device="cpu", dt=0.02)
    term = WristComplianceCommand(cfg, env)
    term.reset(env_ids=None)
    return term


def _make_env(
    wrist_cmd: WristComplianceCommand,
    *,
    num_envs: int = 2,
    base_quat_xyzw: torch.Tensor | None = None,
    num_bodies: int = 30,
    wrist_pos_offset: torch.Tensor | None = None,
) -> SimpleNamespace:
    body_pos_w = torch.zeros(num_envs, num_bodies, 3)
    body_pos_w[:, LEFT_ID] = torch.tensor([1.0, 0.2, 0.9])
    body_pos_w[:, RIGHT_ID] = torch.tensor([1.0, -0.2, 0.9])
    if wrist_pos_offset is not None:
        body_pos_w[:, LEFT_ID] += wrist_pos_offset
        body_pos_w[:, RIGHT_ID] += wrist_pos_offset

    sim = MagicMock()
    sim._robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=body_pos_w),
    )
    sim.find_rigid_body_indice = MagicMock(
        side_effect=lambda name: {LEFT_NAME: LEFT_ID, RIGHT_NAME: RIGHT_ID}[name],
    )

    if base_quat_xyzw is None:
        base_quat_xyzw = torch.zeros(num_envs, 4)
        base_quat_xyzw[:, 3] = 1.0  # xyzw identity (w at index 3)

    # No motion_command → motion target falls back to actual wrist pos,
    # so the residual is purely the virtual-spring shift.
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        simulator=sim,
        base_quat=base_quat_xyzw,
        command_manager=SimpleNamespace(
            get_state=lambda name: wrist_cmd if name == "wrist_compliance_command" else None,
        ),
    )


# ---------------------------------------------------------------------------
# Steady-state cases with motion_target == wrist_actual (identity base quat)
# ---------------------------------------------------------------------------
def test_zero_forces_yields_reward_one() -> None:
    term = _make_wrist_command()
    env = _make_env(term)
    r = wrist_force_position_tracking_exp(env, 0.3, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]
    assert r.shape == (2,)
    assert torch.all(torch.isfinite(r))
    assert torch.allclose(r, torch.ones(2), atol=1e-4)


def _expected_exp(shift_per_wrist_m: float, sigma: float) -> float:
    # error = mean over 2 wrists of ||shift||^2 = shift^2 (equal per wrist)
    return math.exp(-(shift_per_wrist_m**2) / (sigma**2))


def test_f_cmd_shift_produces_exp_penalty() -> None:
    term = _make_wrist_command()
    # F_cmd = 20N +x on each wrist, K=100 → shift 0.2 m on each wrist.
    term._cmd_channel.force[:, 0] = torch.tensor([20.0, 0.0, 0.0])
    term._cmd_channel.force[:, 1] = torch.tensor([20.0, 0.0, 0.0])

    env = _make_env(term)
    sigma = 0.3
    r = wrist_force_position_tracking_exp(env, sigma, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]

    expected = _expected_exp(0.2, sigma)
    torch.testing.assert_close(r, torch.full((2,), expected, dtype=r.dtype))


def test_f_ext_shift_produces_exp_penalty() -> None:
    term = _make_wrist_command()
    term._ext_channel.force[:, 0] = torch.tensor([0.0, 20.0, 0.0])
    term._ext_channel.force[:, 1] = torch.tensor([0.0, 20.0, 0.0])

    env = _make_env(term)
    sigma = 0.3
    r = wrist_force_position_tracking_exp(env, sigma, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]

    expected = _expected_exp(0.2, sigma)
    torch.testing.assert_close(r, torch.full((2,), expected, dtype=r.dtype))


def test_opposing_f_cmd_and_f_ext_cancel() -> None:
    term = _make_wrist_command()
    term._cmd_channel.force[:, 0] = torch.tensor([20.0, 0.0, 0.0])
    term._cmd_channel.force[:, 1] = torch.tensor([20.0, 0.0, 0.0])
    term._ext_channel.force[:, 0] = torch.tensor([-20.0, 0.0, 0.0])
    term._ext_channel.force[:, 1] = torch.tensor([-20.0, 0.0, 0.0])

    env = _make_env(term)
    r = wrist_force_position_tracking_exp(env, 0.3, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]
    # Shift = 0 → reward == 1.
    torch.testing.assert_close(r, torch.ones(2), atol=1e-5, rtol=0)


def test_yaw_rotation_vector_level_xyzw_identity() -> None:
    # xyzw identity → body-frame force passes through unchanged. Vector-level
    # comparison (not just norm), so a wrong-convention rotation would fail.
    identity_xyzw = torch.zeros(2, 4)
    identity_xyzw[:, 3] = 1.0
    force_b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]).expand(2, -1, -1).contiguous()
    out = _yaw_rotate_body_to_world(identity_xyzw, force_b)
    torch.testing.assert_close(out, force_b, atol=1e-6, rtol=0)


def test_yaw_rotation_vector_level_90deg_xyzw() -> None:
    # xyzw 90deg yaw around +z → body +x maps to world +y; body +y maps to -x;
    # body +z is yaw-invariant. Check component-by-component so same-magnitude
    # but wrong-direction outputs (e.g. +x, -y, rotated somewhere else on the
    # xy-plane) would fail this assertion.
    half = math.pi / 4
    base_q = torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])  # xyzw
    base_q = base_q.expand(3, -1).contiguous()
    # 3 batch envs × 2 wrists × 3 components — distinct vectors per wrist.
    force_b = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [[-2.0, 0.0, 0.0], [0.0, -3.0, 0.0]],
        ]
    )
    expected_w = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # +x→+y, +y→-x
            [[0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]],  # +z invariant, (1,1,1)→(-1,1,1)
            [[0.0, -2.0, 0.0], [3.0, 0.0, 0.0]],  # sign flips carried through
        ]
    )
    out = _yaw_rotate_body_to_world(base_q, force_b)
    torch.testing.assert_close(out, expected_w, atol=1e-6, rtol=0)


def test_yaw_rotation_maps_body_force_to_world() -> None:
    # base_quat 90deg yaw (xyzw) → body +x maps to world +y.
    yaw = math.pi / 2
    half = yaw / 2
    base_q = torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])  # xyzw
    base_q = base_q.expand(2, -1).contiguous()

    term = _make_wrist_command()
    # F_cmd body-frame +x, 20N → world +y 20N → shift 0.2m +y.
    term._cmd_channel.force[:, 0] = torch.tensor([20.0, 0.0, 0.0])
    term._cmd_channel.force[:, 1] = torch.tensor([20.0, 0.0, 0.0])

    env = _make_env(term, base_quat_xyzw=base_q)
    r = wrist_force_position_tracking_exp(env, 0.3, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]
    expected = _expected_exp(0.2, 0.3)
    torch.testing.assert_close(r, torch.full((2,), expected, dtype=r.dtype))


def test_per_wrist_k_distinct() -> None:
    # Left K=50 + right K=200 + same F_cmd=20N → left shift 0.4m, right 0.1m.
    term = _make_wrist_command(k_range=(50.0, 50.0))  # start uniform, then override
    term.k_virtual[:, 0] = 50.0
    term.k_virtual[:, 1] = 200.0
    term._cmd_channel.force[:, 0] = torch.tensor([20.0, 0.0, 0.0])
    term._cmd_channel.force[:, 1] = torch.tensor([20.0, 0.0, 0.0])

    env = _make_env(term)
    sigma = 0.3
    r = wrist_force_position_tracking_exp(env, sigma, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]

    # mean residual sq = mean(0.4^2, 0.1^2) = (0.16 + 0.01) / 2 = 0.085
    mean_sq = (0.4**2 + 0.1**2) / 2.0
    expected = math.exp(-mean_sq / sigma**2)
    torch.testing.assert_close(r, torch.full((2,), expected, dtype=r.dtype))


def test_reward_in_unit_interval_and_finite() -> None:
    term = _make_wrist_command()
    term._cmd_channel.force[:] = torch.tensor([30.0, 30.0, 30.0])
    env = _make_env(term)
    r = wrist_force_position_tracking_exp(env, 0.3, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]
    assert torch.all(torch.isfinite(r))
    assert torch.all(r >= 0.0)
    assert torch.all(r <= 1.0 + 1e-6)


def test_defensive_k_floor_prevents_nan() -> None:
    # Bypass __post_init__ by writing an invalid K post-hoc; reward must
    # not NaN because we clamp at _K_FLOOR.
    term = _make_wrist_command()
    term.k_virtual[:] = 0.0  # zero K
    term._cmd_channel.force[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    env = _make_env(term)
    r = wrist_force_position_tracking_exp(env, 0.3, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]
    assert torch.all(torch.isfinite(r))


def test_raises_when_command_missing() -> None:
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        simulator=MagicMock(),
        base_quat=torch.zeros(2, 4),
        command_manager=SimpleNamespace(get_state=lambda _name: None),
    )
    with pytest.raises(RuntimeError, match="wrist_compliance_command"):
        wrist_force_position_tracking_exp(env, 0.3, LEFT_NAME, RIGHT_NAME)  # type: ignore[arg-type]
