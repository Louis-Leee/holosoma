"""CPU-only mock-simulator tests for :class:`WholeBodyTrackingForceInjected`.

The real integration test lives in ``tests/e2e/test_wbt_wrist_force_e2e.py``
(isaacsim-marked); this file just exercises the draw_debug_viz logic,
the physics-step API contract, and the _update_log_dict wiring with a
fake simulator.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# ---- Install a fake holosoma.utils.draw before the subclass imports it. ----
# The real draw module has module-level simulator-type initialisation that
# requires a live sim config. These tests mock everything, so inject a stub
# with the two functions used by ``WholeBodyTrackingForceInjected``.
_fake_draw = types.ModuleType("holosoma.utils.draw")
_fake_draw.draw_line = MagicMock(name="draw_line")  # type: ignore[attr-defined]
_fake_draw.draw_sphere = MagicMock(name="draw_sphere")  # type: ignore[attr-defined]
sys.modules.setdefault("holosoma.utils.draw", _fake_draw)

from holosoma.config_types.command import CommandTermCfg, WristComplianceConfig  # noqa: E402
from holosoma.envs.wbt.wbt_force_injected import (  # noqa: E402
    COLOR_F_CMD,
    COLOR_F_EXT,
    WholeBodyTrackingForceInjected,
)
from holosoma.managers.command.terms.wbt_force import (  # noqa: E402
    NUM_WRISTS,
    WristComplianceCommand,
)

LEFT_ID = 11
RIGHT_ID = 22


def _make_wrist_command(
    num_envs: int = 2,
    *,
    wrist_cfg: WristComplianceConfig | None = None,
) -> WristComplianceCommand:
    wrist_cfg = wrist_cfg or WristComplianceConfig()
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force:WristComplianceCommand",
        params={"wrist_compliance_config": wrist_cfg},
    )
    env = SimpleNamespace(num_envs=num_envs, device="cpu", dt=0.02)
    term = WristComplianceCommand(cfg, env)
    term.reset(env_ids=None)
    return term


def _make_fake_env(
    wrist_cmd: WristComplianceCommand,
    *,
    num_envs: int = 2,
    num_bodies: int = 30,
    headless: bool = False,
) -> WholeBodyTrackingForceInjected:
    """Construct a WholeBodyTrackingForceInjected *without* running __init__
    and wire just enough attributes for the behaviours under test."""
    env = WholeBodyTrackingForceInjected.__new__(WholeBodyTrackingForceInjected)

    # Simulator mock.
    sim = MagicMock()
    sim.sim_device = torch.device("cpu")
    # body_pos_w / body_quat_w of shape (num_envs, num_bodies, 3 or 4)
    body_pos_w = torch.zeros(num_envs, num_bodies, 3)
    body_pos_w[:, LEFT_ID] = torch.tensor([1.0, 0.2, 0.9])
    body_pos_w[:, RIGHT_ID] = torch.tensor([1.0, -0.2, 0.9])
    body_quat_w = torch.zeros(num_envs, num_bodies, 4)
    body_quat_w[..., 0] = 1.0  # wxyz identity
    sim._robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=body_pos_w, body_quat_w=body_quat_w),
        set_external_force_and_torque=MagicMock(),
    )
    sim.find_rigid_body_indice = MagicMock(
        side_effect=lambda name: {
            "left_wrist_yaw_link": LEFT_ID,
            "right_wrist_yaw_link": RIGHT_ID,
        }[name]
    )
    sim.draw = None if headless else MagicMock()
    sim.draw_debug_viz = MagicMock()  # baseline no-op

    env.simulator = sim  # type: ignore[attr-defined]
    env.num_envs = num_envs
    env.device = "cpu"
    env.base_quat = torch.zeros(num_envs, 4)
    env.base_quat[:, 3] = 1.0  # xyzw identity (w at index 3) so body-yaw == world

    env._left_wrist_isaac_id = LEFT_ID
    env._right_wrist_isaac_id = RIGHT_ID
    env._wrist_body_ids_t = torch.tensor([LEFT_ID, RIGHT_ID], dtype=torch.long)
    env.last_applied_force_w_by_body_id = {}
    env.log_dict = {}

    # Command manager exposing our wrist command.
    env.command_manager = SimpleNamespace(  # type: ignore[assignment]
        get_state=MagicMock(side_effect=lambda name: wrist_cmd if name == "wrist_compliance_command" else None),
    )
    return env


# ---------------------------------------------------------------------------
# draw_debug_viz
# ---------------------------------------------------------------------------
def _inject_force(wrist_cmd: WristComplianceCommand, *, cmd_vec: torch.Tensor, ext_vec: torch.Tensor) -> None:
    """Pokes F_cmd and F_ext for env 0 / left and right wrists."""
    # F_cmd is body-yaw frame; F_ext is world frame.
    wrist_cmd._cmd_channel.force[0, 0] = cmd_vec
    wrist_cmd._cmd_channel.force[0, 1] = cmd_vec
    wrist_cmd._ext_channel.force[0, 0] = ext_vec
    wrist_cmd._ext_channel.force[0, 1] = ext_vec


def test_draw_debug_viz_calls_draw_line_for_each_wrist_by_default() -> None:
    wrist_cmd = _make_wrist_command()
    env = _make_fake_env(wrist_cmd)

    _inject_force(
        wrist_cmd,
        cmd_vec=torch.tensor([10.0, 0.0, 0.0]),
        ext_vec=torch.tensor([0.0, 10.0, 0.0]),
    )

    # Mock both drawer functions from the draw adapter path.
    draw_line = _fake_draw.draw_line
    draw_sphere = _fake_draw.draw_sphere
    draw_line.reset_mock()
    draw_sphere.reset_mock()
    if True:
        env.draw_debug_viz()

    # Draw 2 arrows per wrist (F_ext + F_cmd) => 4 calls for 2 wrists.
    assert draw_line.call_count == 4
    assert draw_sphere.call_count == 4
    colors = [call.args[3] for call in draw_line.call_args_list]
    # Each wrist contributes exactly one F_ext and one F_cmd arrow.
    assert colors.count(COLOR_F_EXT) == 2
    assert colors.count(COLOR_F_CMD) == 2


def test_draw_debug_viz_skips_zero_force_arrows() -> None:
    wrist_cmd = _make_wrist_command()
    env = _make_fake_env(wrist_cmd)
    # All zero forces.
    _inject_force(wrist_cmd, cmd_vec=torch.zeros(3), ext_vec=torch.zeros(3))

    draw_line = _fake_draw.draw_line
    _fake_draw.draw_sphere.reset_mock()
    draw_line.reset_mock()
    if True:
        env.draw_debug_viz()

    assert draw_line.call_count == 0


def test_draw_debug_viz_early_returns_when_headless() -> None:
    wrist_cmd = _make_wrist_command()
    env = _make_fake_env(wrist_cmd, headless=True)  # sim.draw is None
    _inject_force(
        wrist_cmd,
        cmd_vec=torch.tensor([10.0, 0.0, 0.0]),
        ext_vec=torch.tensor([0.0, 10.0, 0.0]),
    )
    draw_line = _fake_draw.draw_line
    draw_line.reset_mock()
    env.draw_debug_viz()
    assert draw_line.call_count == 0


# ---------------------------------------------------------------------------
# _apply_force_in_physics_step
# ---------------------------------------------------------------------------
def test_apply_force_calls_set_external_with_wrist_only_body_ids() -> None:
    wrist_cmd = _make_wrist_command()
    env = _make_fake_env(wrist_cmd)

    # F_ext world = (1,0,0) on left, (0,2,0) on right.
    wrist_cmd._ext_channel.force[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    wrist_cmd._ext_channel.force[0, 1] = torch.tensor([0.0, 2.0, 0.0])

    # Stub out the base action_manager path (super() calls action_manager).
    env.action_manager = None  # type: ignore[assignment]

    # Patch isaaclab.utils.math so we don't actually need isaaclab.
    fake_math = MagicMock()
    fake_math.quat_apply_inverse.side_effect = lambda _q, v: v  # identity
    with patch.dict(
        "sys.modules", {"isaaclab": MagicMock(), "isaaclab.utils": MagicMock(), "isaaclab.utils.math": fake_math}
    ):
        env._apply_force_in_physics_step()

    set_fn = env.simulator._robot.set_external_force_and_torque  # type: ignore[attr-defined]
    assert set_fn.call_count == 1
    kwargs = set_fn.call_args.kwargs
    torch.testing.assert_close(kwargs["body_ids"], torch.tensor([LEFT_ID, RIGHT_ID], dtype=torch.long))
    forces = kwargs["forces"]
    assert forces.shape == (2, NUM_WRISTS, 3)
    assert kwargs["env_ids"] is None
    # Debug snapshot recorded.
    assert LEFT_ID in env.last_applied_force_w_by_body_id
    assert RIGHT_ID in env.last_applied_force_w_by_body_id


def test_apply_force_does_not_call_draw_line() -> None:
    # Regression: debug arrows must NEVER be drawn from physics step
    # (the physics loop runs N times per control step → would overdraw).
    wrist_cmd = _make_wrist_command()
    env = _make_fake_env(wrist_cmd)
    env.action_manager = None  # type: ignore[assignment]

    fake_math = MagicMock()
    fake_math.quat_apply_inverse.side_effect = lambda _q, v: v
    draw_line = _fake_draw.draw_line
    draw_line.reset_mock()
    with patch.dict(
        "sys.modules",
        {
            "isaaclab": MagicMock(),
            "isaaclab.utils": MagicMock(),
            "isaaclab.utils.math": fake_math,
        },
    ):
        for _ in range(4):  # simulate 4 physics substeps per control step
            env._apply_force_in_physics_step()

    assert draw_line.call_count == 0


# ---------------------------------------------------------------------------
# _update_log_dict
# ---------------------------------------------------------------------------
def test_update_log_dict_merges_command_and_env_owned_keys() -> None:
    wrist_cmd = _make_wrist_command()
    env = _make_fake_env(wrist_cmd)

    # Stub super()._update_log_dict (WholeBodyTrackingManager / BaseTask).
    # We stub via monkey-patching the bound method from the parent class.
    with patch.object(WholeBodyTrackingForceInjected.__mro__[1], "_update_log_dict", lambda _self: None):
        env._update_log_dict()

    log = env.log_dict
    # Command-owned keys.
    for key in (
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
    ):
        assert key in log, key
        assert isinstance(log[key], torch.Tensor)
        assert log[key].dtype == torch.float32

    # Env-owned keys.
    for key in (
        "force/wrist_target_shift_l",
        "force/wrist_target_shift_r",
        "force/wrist_pos_error_l",
        "force/wrist_pos_error_r",
        "force/applied_f_body_l",
        "force/applied_f_body_r",
    ):
        assert key in log, key
        assert isinstance(log[key], torch.Tensor)
        assert log[key].shape == (2,)


def test_wrap_simulator_draw_hook_chains_correctly() -> None:
    wrist_cmd = _make_wrist_command()
    env = _make_fake_env(wrist_cmd)

    # Set F_cmd non-zero so draw_debug_viz will actually draw.
    _inject_force(
        wrist_cmd,
        cmd_vec=torch.tensor([10.0, 0.0, 0.0]),
        ext_vec=torch.tensor([0.0, 10.0, 0.0]),
    )

    original = MagicMock()
    env.simulator.draw_debug_viz = original  # type: ignore[method-assign]

    env._wrap_simulator_draw_hook()

    draw_line = _fake_draw.draw_line
    _fake_draw.draw_sphere.reset_mock()
    draw_line.reset_mock()
    if True:
        env.simulator.draw_debug_viz()

    assert original.called
    assert draw_line.call_count == 4


# ---------------------------------------------------------------------------
# Defensive path: missing wrist command term -> no crash
# ---------------------------------------------------------------------------
def test_apply_force_noop_when_command_missing() -> None:
    env = WholeBodyTrackingForceInjected.__new__(WholeBodyTrackingForceInjected)
    env.simulator = MagicMock()
    env.simulator._robot = SimpleNamespace(  # type: ignore[attr-defined]
        set_external_force_and_torque=MagicMock(),
    )
    env.simulator.sim_device = torch.device("cpu")
    env.action_manager = None  # type: ignore[assignment]
    env._left_wrist_isaac_id = LEFT_ID
    env._right_wrist_isaac_id = RIGHT_ID
    env._wrist_body_ids_t = torch.tensor([LEFT_ID, RIGHT_ID], dtype=torch.long)
    env.last_applied_force_w_by_body_id = {}
    env.command_manager = SimpleNamespace(  # type: ignore[assignment]
        get_state=MagicMock(return_value=None),
    )

    # Should not raise.
    env._apply_force_in_physics_step()
    env.simulator._robot.set_external_force_and_torque.assert_not_called()
