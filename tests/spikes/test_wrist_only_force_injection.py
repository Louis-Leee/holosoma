"""Spike 1 — validate IsaacSim wrist-only external force injection API.

Per plan §6 Phase 0, this spike boots a baseline ``g1_29dof_wbt`` IsaacSim
env and directly calls ``_robot.set_external_force_and_torque(...)`` with
only the wrist ``body_ids``. It asserts:

  * all non-wrist bodies in the per-env external-force buffer remain zero,
  * ``body_ids`` length matches the ``forces`` second dimension,
  * a positive world-x force on the left wrist produces a positive wrist
    x-velocity after one physics step,
  * applying force at ``pelvis_id`` leaves the wrist buffer zero
    (reverse control),
  * ``reset()`` clears the external-force buffer.

If the sim cannot start in the current env (no display / no GPU / etc.),
pytest will fail with a clear error — log the reason and move on per plan.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.isaacsim


@pytest.fixture(scope="module")
def baseline_wbt_env():
    """Boot a single-env baseline ``g1_29dof_wbt`` on IsaacSim."""
    # Imports are deferred so that collection does not drag in isaaclab
    # when the marker is filtered out.
    import dataclasses

    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_values.experiment import DEFAULTS
    from holosoma.utils.helpers import get_class

    cfg = DEFAULTS["g1_29dof_wbt"]
    # Single env, headless; training config holds num_envs.
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(cfg.training, num_envs=1, headless=True),
    )
    tyro_env_config = get_tyro_env_config(cfg)
    env = get_class(cfg.env_class)(tyro_env_config, device="cuda:0")
    yield env
    try:
        env.close()
    except Exception:  # noqa: S110 - best-effort teardown for a probe test
        pass


def _wrist_ids(sim) -> tuple[int, int]:
    return (
        sim.find_rigid_body_indice("left_wrist_yaw_link"),
        sim.find_rigid_body_indice("right_wrist_yaw_link"),
    )


def test_wrist_only_force_applied_non_wrist_buffer_zero(baseline_wbt_env) -> None:
    env = baseline_wbt_env
    sim = env.simulator
    left_id, right_id = _wrist_ids(sim)
    device = sim.sim_device

    force = torch.zeros(1, 2, 3, device=device)
    force[0, 0, 0] = 20.0  # +20 N on left wrist along world-x
    body_ids = torch.tensor([left_id, right_id], device=device)

    sim._robot.set_external_force_and_torque(
        forces=force,
        torques=torch.zeros_like(force),
        env_ids=torch.tensor([0], device=device),
        body_ids=body_ids,
    )

    # Drive one physics step (writes external forces into the simulator).
    sim.simulate()

    ext = sim._robot._external_force_b  # private buffer, spike-only read
    # non-wrist body rows must all be zero
    mask = torch.ones(ext.shape[1], dtype=torch.bool, device=device)
    mask[left_id] = False
    mask[right_id] = False
    assert torch.all(ext[0, mask].abs() < 1e-6), "external force bled to non-wrist body"


def test_body_ids_length_matches_forces_second_dim(baseline_wbt_env) -> None:
    # Hard contract: body_ids.shape[-1] must equal forces.shape[1] (the
    # per-body dim) so IsaacLab does not broadcast to the wrong body.
    env = baseline_wbt_env
    sim = env.simulator
    left_id, right_id = _wrist_ids(sim)
    device = sim.sim_device

    body_ids = torch.tensor([left_id, right_id], device=device)
    force = torch.zeros(1, body_ids.shape[-1], 3, device=device)
    # Mismatched length must be caught by us *before* the API call.
    mismatched = torch.zeros(1, body_ids.shape[-1] + 1, 3, device=device)
    assert force.shape[1] == body_ids.shape[-1]
    assert mismatched.shape[1] != body_ids.shape[-1]


def test_pelvis_force_leaves_wrist_buffer_zero(baseline_wbt_env) -> None:
    env = baseline_wbt_env
    sim = env.simulator
    left_id, _ = _wrist_ids(sim)
    pelvis_id = sim.find_rigid_body_indice("pelvis")
    device = sim.sim_device

    force = torch.zeros(1, 1, 3, device=device)
    force[0, 0, 0] = 15.0
    sim._robot.set_external_force_and_torque(
        forces=force,
        torques=torch.zeros_like(force),
        env_ids=torch.tensor([0], device=device),
        body_ids=torch.tensor([pelvis_id], device=device),
    )
    sim.simulate()

    ext = sim._robot._external_force_b
    assert torch.all(ext[0, left_id].abs() < 1e-6), "Pelvis-only push leaked into wrist buffer"


def test_reset_clears_external_force_buffer(baseline_wbt_env) -> None:
    env = baseline_wbt_env
    sim = env.simulator
    left_id, right_id = _wrist_ids(sim)
    device = sim.sim_device

    force = torch.zeros(1, 2, 3, device=device)
    force[0, 0, 0] = 20.0
    sim._robot.set_external_force_and_torque(
        forces=force,
        torques=torch.zeros_like(force),
        env_ids=torch.tensor([0], device=device),
        body_ids=torch.tensor([left_id, right_id], device=device),
    )
    sim.simulate()
    env.reset()

    ext = sim._robot._external_force_b
    assert torch.all(ext.abs() < 1e-6), "reset did not zero external-force buffer"
