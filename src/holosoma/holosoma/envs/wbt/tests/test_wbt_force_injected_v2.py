"""CPU structural tests for WholeBodyTrackingForceInjectedV2 (no IsaacSim)."""

from __future__ import annotations

import inspect

import torch
from holosoma.envs.wbt.wbt_force_injected import WholeBodyTrackingForceInjected
from holosoma.envs.wbt.wbt_force_injected_v2 import WholeBodyTrackingForceInjectedV2
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def test_v2_env_class_imports() -> None:
    assert inspect.isclass(WholeBodyTrackingForceInjectedV2)


def test_v2_inherits_from_baseline_not_v10() -> None:
    """V14 env must inherit WholeBodyTrackingManager directly, not V10 WholeBodyTrackingForceInjected."""
    assert issubclass(WholeBodyTrackingForceInjectedV2, WholeBodyTrackingManager)
    assert not issubclass(WholeBodyTrackingForceInjectedV2, WholeBodyTrackingForceInjected), (
        "V14 env must be sibling of V10, not subclass"
    )


def test_v2_uses_v14_command_key() -> None:
    src = inspect.getsource(WholeBodyTrackingForceInjectedV2)
    assert "wrist_force_tracking_command" in src
    assert "wrist_compliance_command" not in src, "V14 env must not reference V10 key"


def test_v2_env_module_does_not_import_v10() -> None:
    """V14 env module must not import V10 symbols."""
    from holosoma.envs.wbt import wbt_force_injected_v2
    src = inspect.getsource(wbt_force_injected_v2)
    forbidden = [
        "from holosoma.envs.wbt.wbt_force_injected import",
        "from holosoma.managers.command.terms.wbt_force import",
        "from holosoma.managers.reward.terms.wbt_force import",
        "from holosoma.managers.observation.terms.wbt_force import",
    ]
    for f in forbidden:
        assert f not in src, f"V14 env must not import V10 symbols: {f!r}"


def test_v2_defines_its_own_hooks() -> None:
    """V14 env must define all injection/rotation/viz hooks itself (no MRO fallback to V10)."""
    own_attrs = set(vars(WholeBodyTrackingForceInjectedV2))
    for name in (
        "_apply_force_in_physics_step",
        "_rotate_force_world_to_body",
        "draw_debug_viz",
        "_wrap_simulator_draw_hook",
        "_env_owned_force_metrics",
    ):
        assert name in own_attrs, (
            f"V14 env must define {name} itself (found only in parent MRO)"
        )


def test_v2_injects_only_on_wrist_bodies() -> None:
    """F_ext must only be applied to left + right wrist bodies.

    Mock simulator._robot.set_external_force_and_torque and verify body_ids is
    exactly the two wrist isaac indices, forces shape [N, 2, 3], and
    last_applied is snapshotted by wrist id.
    """
    import types

    env = WholeBodyTrackingForceInjectedV2.__new__(WholeBodyTrackingForceInjectedV2)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._left_wrist_isaac_id = 17
    env._right_wrist_isaac_id = 23
    env._wrist_body_ids_t = torch.tensor([17, 23], dtype=torch.long)
    env.last_applied_force_w_by_body_id = {}

    force_w = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]],
    ])

    class _MockCmd:
        force_ext_w = force_w

    env._get_wrist_command = lambda *, require=True: _MockCmd()  # type: ignore[assignment]
    env._rotate_force_world_to_body = lambda f_w: f_w  # identity for this test

    calls: list[dict] = []
    sim = types.SimpleNamespace()
    robot = types.SimpleNamespace()
    def _spy(forces, torques, env_ids, body_ids):
        calls.append({"forces": forces, "body_ids": body_ids})
    robot.set_external_force_and_torque = _spy
    sim._robot = robot
    env.simulator = sim

    import holosoma.envs.wbt.wbt_force_injected_v2 as _v2_mod
    original_super_apply = _v2_mod.WholeBodyTrackingManager._apply_force_in_physics_step
    _v2_mod.WholeBodyTrackingManager._apply_force_in_physics_step = lambda self: None  # type: ignore[assignment]
    try:
        WholeBodyTrackingForceInjectedV2._apply_force_in_physics_step(env)
    finally:
        _v2_mod.WholeBodyTrackingManager._apply_force_in_physics_step = original_super_apply

    assert len(calls) == 1
    call = calls[0]
    assert torch.equal(call["body_ids"], torch.tensor([17, 23], dtype=torch.long))
    assert call["forces"].shape == (2, 2, 3)
    assert set(env.last_applied_force_w_by_body_id.keys()) == {17, 23}
