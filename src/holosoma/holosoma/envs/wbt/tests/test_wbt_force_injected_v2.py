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
    env.last_applied_force_b_by_body_id = {}

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
    assert set(env.last_applied_force_b_by_body_id.keys()) == {17, 23}


def test_applied_force_snapshot_is_body_frame_not_world() -> None:
    """``last_applied_force_b_by_body_id`` must store the **body-frame** vector
    that was actually pushed to ``set_external_force_and_torque`` — not the
    world-frame F_ext. Use a non-identity body quat so world != body.
    """
    import types

    env = WholeBodyTrackingForceInjectedV2.__new__(WholeBodyTrackingForceInjectedV2)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env._left_wrist_isaac_id = 3
    env._right_wrist_isaac_id = 5
    env._wrist_body_ids_t = torch.tensor([3, 5], dtype=torch.long)
    env.last_applied_force_b_by_body_id = {}

    force_w = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])

    class _MockCmd:
        force_ext_w = force_w

    env._get_wrist_command = lambda *, require=True: _MockCmd()  # type: ignore[assignment]
    # Deterministic non-identity rotation: both wrists apply a fixed per-axis
    # flip so body-frame != world-frame.
    env._rotate_force_world_to_body = lambda f_w: -f_w  # flip all components

    sim = types.SimpleNamespace()
    robot = types.SimpleNamespace()
    robot.set_external_force_and_torque = lambda forces, torques, env_ids, body_ids: None
    sim._robot = robot
    env.simulator = sim

    import holosoma.envs.wbt.wbt_force_injected_v2 as _v2_mod
    original = _v2_mod.WholeBodyTrackingManager._apply_force_in_physics_step
    _v2_mod.WholeBodyTrackingManager._apply_force_in_physics_step = lambda self: None  # type: ignore[assignment]
    try:
        WholeBodyTrackingForceInjectedV2._apply_force_in_physics_step(env)
    finally:
        _v2_mod.WholeBodyTrackingManager._apply_force_in_physics_step = original

    left_b = env.last_applied_force_b_by_body_id[3]
    right_b = env.last_applied_force_b_by_body_id[5]
    # Snapshot must equal the body-frame vector (which we forced = -world).
    assert torch.equal(left_b, torch.tensor([[-10.0, 0.0, 0.0]]))
    assert torch.equal(right_b, torch.tensor([[0.0, -10.0, 0.0]]))
    # Sanity: it must NOT equal the world-frame vector.
    assert not torch.equal(left_b, force_w[:, 0])


def test_wrist_reaction_diagnostic_reads_correct_isaaclab_attribute() -> None:
    """``_try_compute_wrist_reaction_metrics`` must read
    ``body_incoming_joint_wrench_b`` — the actual IsaacLab property —
    **not** the fictitious ``body_incoming_wrench_b``. Regression: previous
    version used the wrong attribute and silently emitted nothing.
    """
    import types

    env = WholeBodyTrackingForceInjectedV2.__new__(WholeBodyTrackingForceInjectedV2)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env._left_wrist_isaac_id = 3
    env._right_wrist_isaac_id = 5

    N, NB = 1, 10
    # wrench[:, body, :3] is force; [:, body, 3:6] is torque.
    wrench_b_buf = torch.zeros(N, NB, 6)
    wrench_b_buf[0, 3, :3] = torch.tensor([0.0, 0.0, -10.0])  # left reaction
    wrench_b_buf[0, 5, :3] = torch.tensor([5.0, 0.0, 0.0])    # right reaction
    body_quat_w = torch.zeros(N, NB, 4)
    body_quat_w[..., 0] = 1.0  # wxyz identity -> reaction_w == reaction_b

    data = types.SimpleNamespace(
        body_incoming_joint_wrench_b=wrench_b_buf,
        body_quat_w=body_quat_w,
    )
    env.simulator = types.SimpleNamespace(_robot=types.SimpleNamespace(data=data))

    # Sibling import check: the wrong-name attribute is NOT used.
    assert not hasattr(data, "body_incoming_wrench_b")

    # Mock wrist_cmd.
    f_cmd_b = torch.tensor([[[0.0, 0.0, 10.0], [-5.0, 0.0, 0.0]]])
    wrist_cmd = types.SimpleNamespace(
        force_cmd_b=f_cmd_b,
        _rotate_cmd_obs_to_world=lambda f: f,  # identity under identity quat
    )

    # The isaaclab import at call time may fail under CPU — patch it.
    import sys
    fake = types.ModuleType("isaaclab.utils.math")
    fake.quat_apply = lambda q, v: v  # type: ignore[attr-defined]
    sys.modules["isaaclab.utils.math"] = fake
    try:
        metrics = WholeBodyTrackingForceInjectedV2._try_compute_wrist_reaction_metrics(  # type: ignore[arg-type]
            env, wrist_cmd,
        )
    finally:
        del sys.modules["isaaclab.utils.math"]

    for k in (
        "force/wrist_reaction_vs_cmd_cos_l",
        "force/wrist_reaction_vs_cmd_cos_r",
        "force/wrist_reaction_vs_cmd_mag_ratio_l",
        "force/wrist_reaction_vs_cmd_mag_ratio_r",
    ):
        assert k in metrics, f"missing: {k}"
    # Both wrists: reaction anti-parallel to F_cmd -> cos ~ -1.
    assert metrics["force/wrist_reaction_vs_cmd_cos_l"].item() < -0.99
    assert metrics["force/wrist_reaction_vs_cmd_cos_r"].item() < -0.99
    # |reaction|/|cmd| = 1.0 (same magnitudes chosen above).
    assert abs(metrics["force/wrist_reaction_vs_cmd_mag_ratio_l"].item() - 1.0) < 1e-5
    assert abs(metrics["force/wrist_reaction_vs_cmd_mag_ratio_r"].item() - 1.0) < 1e-5


def test_wrist_reaction_diagnostic_skips_when_attribute_missing() -> None:
    """If the sim backend has no ``body_incoming_joint_wrench_b``, the
    diagnostic must silently emit nothing (not raise)."""
    import types

    env = WholeBodyTrackingForceInjectedV2.__new__(WholeBodyTrackingForceInjectedV2)
    env._left_wrist_isaac_id = 3
    env._right_wrist_isaac_id = 5
    env.simulator = types.SimpleNamespace(
        _robot=types.SimpleNamespace(data=types.SimpleNamespace()),
    )
    wrist_cmd = types.SimpleNamespace()
    out = WholeBodyTrackingForceInjectedV2._try_compute_wrist_reaction_metrics(  # type: ignore[arg-type]
        env, wrist_cmd,
    )
    assert out == {}


def test_env_owned_metrics_emit_term_rate_when_reset_manager_exposes_last_reset_ids() -> None:
    """term_rate_{active,inactive}_ext must appear once ResetEventManager
    stores ``last_reset_ids`` and we have a prev-tick active_window_flag
    snapshot (i.e. the second metric pass).
    """
    import types

    env = WholeBodyTrackingForceInjectedV2.__new__(WholeBodyTrackingForceInjectedV2)
    env.num_envs = 4
    env.device = torch.device("cpu")
    env._left_wrist_isaac_id = 7
    env._right_wrist_isaac_id = 9
    env.last_applied_force_b_by_body_id = {}

    # prev tick: envs 0,2 had active ext force.
    env._prev_active_window_flag = torch.tensor([1.0, 0.0, 1.0, 0.0])

    reset_mgr = types.SimpleNamespace(last_reset_ids=torch.tensor([0, 3], dtype=torch.long))
    env.reset_manager = reset_mgr

    # Simulate current-tick wrist_cmd._ext_channel.state so active_window_flag
    # recomputes to something.
    wrist_cmd = types.SimpleNamespace(
        _ext_channel=types.SimpleNamespace(
            state=torch.tensor([[0, 0], [1, 0], [0, 0], [1, 1]], dtype=torch.long),
        ),
    )
    # Disable the optional wrench path — no body_incoming_wrench_b on mock.
    env.simulator = types.SimpleNamespace(_robot=types.SimpleNamespace(data=types.SimpleNamespace()))

    wcfg = types.SimpleNamespace()
    metrics = WholeBodyTrackingForceInjectedV2._env_owned_force_metrics(env, wrist_cmd, wcfg)  # type: ignore[arg-type]

    assert "force/term_rate_active_ext" in metrics
    assert "force/term_rate_inactive_ext" in metrics
    # envs reset = {0, 3}; prev_active at those envs = {1.0, 0.0}.
    # -> active_ext[0]=1, active_ext[3]=0; inactive_ext[0]=0, inactive_ext[3]=1.
    expected_active = torch.tensor([1.0, 0.0, 0.0, 0.0])
    expected_inactive = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(metrics["force/term_rate_active_ext"], expected_active)
    assert torch.allclose(metrics["force/term_rate_inactive_ext"], expected_inactive)


def test_env_owned_metrics_skip_term_rate_when_reset_manager_missing() -> None:
    """If reset_manager or last_reset_ids is absent, the metric is silently
    skipped (legacy fallback). This keeps the V14 env usable in unit-test mocks.
    """
    import types

    env = WholeBodyTrackingForceInjectedV2.__new__(WholeBodyTrackingForceInjectedV2)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._left_wrist_isaac_id = 7
    env._right_wrist_isaac_id = 9
    env.last_applied_force_b_by_body_id = {}
    env._prev_active_window_flag = torch.zeros(2)

    wrist_cmd = types.SimpleNamespace(
        _ext_channel=types.SimpleNamespace(state=torch.zeros(2, 2, dtype=torch.long)),
    )
    env.simulator = types.SimpleNamespace(_robot=types.SimpleNamespace(data=types.SimpleNamespace()))
    wcfg = types.SimpleNamespace()
    metrics = WholeBodyTrackingForceInjectedV2._env_owned_force_metrics(env, wrist_cmd, wcfg)  # type: ignore[arg-type]
    assert "force/term_rate_active_ext" not in metrics
    assert "force/term_rate_inactive_ext" not in metrics
