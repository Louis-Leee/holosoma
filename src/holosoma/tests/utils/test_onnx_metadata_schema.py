"""Training-side ONNX metadata schema test (Task 13.5).

Exports a tiny policy, attaches the three new Task 13.5 schema fields
via :func:`attach_onnx_metadata`, and confirms they round-trip through
``onnx.load``.

Also asserts the exact same dict shape that ``ppo.py`` /
``fast_sac_agent.py`` build at export time (pure dict construction, no
training agent required).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import onnx
import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
)


class _ActorWrapper(nn.Module):
    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": actor_obs})


def _export_minimal_onnx(tmpdir: Path, obs_dim: int = 10, act_dim: int = 5) -> Path:
    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[act_dim],
        layer_config=LayerConfig(hidden_dims=[64], activation="ReLU", dropout_prob=0.0),
        min_noise_std=None,
        min_mean_noise_std=None,
    )
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": obs_dim},
        module_config=module_config,
        num_actions=act_dim,
        init_noise_std=0.1,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    wrapper = _ActorWrapper(actor)
    wrapper.eval()
    onnx_path = tmpdir / "policy.onnx"
    export_policy_as_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict={"actor_obs": torch.zeros(1, obs_dim)},
    )
    return onnx_path


def _load_metadata_dict(onnx_path: Path) -> dict:
    model = onnx.load(str(onnx_path))
    return {p.key: json.loads(p.value) for p in model.metadata_props}


def test_schema_fields_round_trip() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _export_minimal_onnx(Path(d))
        attach_onnx_metadata(
            str(path),
            metadata={
                "history_length": 10,
                "obs_term_names_sorted": [
                    "actions",
                    "base_ang_vel",
                    "dof_pos",
                    "dof_vel",
                    "motion_command",
                    "motion_ref_ori_b",
                    "wrist_force_command",
                    "wrist_virtual_stiffness_command",
                ],
                "obs_group_dims": {"actor_obs": 1620, "critic_obs": 3000},
            },
        )
        meta = _load_metadata_dict(path)

    assert meta["history_length"] == 10
    assert meta["obs_term_names_sorted"] == sorted(meta["obs_term_names_sorted"])
    assert "wrist_force_command" in meta["obs_term_names_sorted"]
    assert meta["obs_group_dims"] == {"actor_obs": 1620, "critic_obs": 3000}


def test_attach_on_baseline_style_metadata_does_not_break_onnx_load() -> None:
    """Regression: baseline g1-29dof-wbt style metadata (without the three
    new keys) must still load. We attach only the baseline keys and
    verify onnx.load + check_model succeed."""
    with tempfile.TemporaryDirectory() as d:
        path = _export_minimal_onnx(Path(d))
        attach_onnx_metadata(
            str(path),
            metadata={
                "dof_names": ["dof_a", "dof_b"],
                "kp": [100.0],
                "kd": [1.0],
                "action_scale": 0.25,
                "command_ranges": {},
                "robot_urdf": "<urdf/>",
                "robot_urdf_path": "/tmp/robot.urdf",
            },
        )
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        # Old keys present; new keys absent → inference-side must tolerate.
        keys = {p.key for p in model.metadata_props}
        assert "history_length" not in keys
        assert "obs_term_names_sorted" not in keys
        assert "obs_group_dims" not in keys
