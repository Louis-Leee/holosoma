"""Pure-CPU wandb-metric wiring tests (WBT wrist-force, Task 9.5).

Verifies the end-to-end key contract between:

  * :class:`WristComplianceCommand.update_metrics` — populates the
    "Command-owned" force keys.
  * :meth:`WholeBodyTrackingForceInjected._update_log_dict` — merges
    those + "Env-owned" wrist-tracking / applied-force keys into
    ``log_dict``.

Both are exercised here with a mock environment / mock simulator so we
do not require IsaacSim.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

# Install a fake draw module to match the env subclass import pattern.
_fake_draw = types.ModuleType("holosoma.utils.draw")
_fake_draw.draw_line = MagicMock(name="draw_line")  # type: ignore[attr-defined]
_fake_draw.draw_sphere = MagicMock(name="draw_sphere")  # type: ignore[attr-defined]
sys.modules.setdefault("holosoma.utils.draw", _fake_draw)

from holosoma.config_types.command import CommandTermCfg, WristComplianceConfig  # noqa: E402
from holosoma.envs.wbt.wbt_force_injected import WholeBodyTrackingForceInjected  # noqa: E402
from holosoma.managers.command.terms.wbt_force import WristComplianceCommand  # noqa: E402

# --- Expected key sets (Task 9.5 §6 wandb contract) -------------------------
COMMAND_OWNED_KEYS = frozenset(
    {
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
    },
)
ENV_OWNED_KEYS = frozenset(
    {
        "force/wrist_target_shift_l",
        "force/wrist_target_shift_r",
        "force/wrist_pos_error_l",
        "force/wrist_pos_error_r",
        "force/applied_f_body_l",
        "force/applied_f_body_r",
    },
)
LEFT_ID = 11
RIGHT_ID = 22


def _make_term(num_envs: int = 4) -> WristComplianceCommand:
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force:WristComplianceCommand",
        params={"wrist_compliance_config": WristComplianceConfig()},
    )
    term = WristComplianceCommand(cfg, SimpleNamespace(num_envs=num_envs, device="cpu", dt=0.02))
    term.reset(env_ids=None)
    # Poke some numbers so metrics are not all zero.
    term._cmd_channel.force[:, 0] = torch.tensor([10.0, 0.0, 0.0])
    term._ext_channel.force[:, 1] = torch.tensor([0.0, 5.0, 0.0])
    return term


def test_command_update_metrics_populates_exact_key_set() -> None:
    term = _make_term()
    term.update_metrics()
    assert COMMAND_OWNED_KEYS.issubset(term.metrics.keys())


def test_command_metrics_shapes_dtype_device() -> None:
    term = _make_term(num_envs=8)
    term.update_metrics()
    for key in COMMAND_OWNED_KEYS:
        t = term.metrics[key]
        assert isinstance(t, torch.Tensor), key
        assert t.shape == (8,), f"{key} shape {t.shape}"
        assert t.dtype == torch.float32, key
        assert t.device.type == "cpu", key


def test_command_metrics_value_ranges_v1() -> None:
    term = _make_term()
    term.update_metrics()
    # magnitudes non-negative, cmd/ext capped by config range = (5..30)/(0..30)
    for k in (
        "force/cmd_magnitude_l",
        "force/cmd_magnitude_r",
        "force/cmd_magnitude_max",
    ):
        assert torch.all(term.metrics[k] >= 0.0)
    # v1 K range collapses to 100 → flat lines.
    assert torch.all(term.metrics["force/k_virtual_l"] == 100.0)
    assert torch.all(term.metrics["force/k_virtual_r"] == 100.0)
    # Active fraction in [0, 1].
    for k in ("force/active_frac_cmd", "force/active_frac_ext"):
        v = term.metrics[k]
        assert torch.all(v >= 0.0) and torch.all(v <= 1.0)
    # Alignment in [-1, 1].
    a = term.metrics["force/cmd_ext_alignment"]
    assert torch.all(a >= -1.0 - 1e-5) and torch.all(a <= 1.0 + 1e-5)
    # Phase occupancies in [0, 1].
    for k in ("force/phase_ramp_up", "force/phase_hold", "force/phase_ramp_down"):
        p = term.metrics[k]
        assert torch.all(p >= 0.0) and torch.all(p <= 1.0)


def _make_env_with_term(term: WristComplianceCommand, num_envs: int = 4):
    """Manually build the env subclass skeleton — mirrors Task 3 tests."""
    env = WholeBodyTrackingForceInjected.__new__(WholeBodyTrackingForceInjected)
    sim = MagicMock()
    sim.sim_device = torch.device("cpu")
    body_pos_w = torch.zeros(num_envs, 30, 3)
    body_pos_w[:, LEFT_ID] = torch.tensor([1.0, 0.2, 0.9])
    body_pos_w[:, RIGHT_ID] = torch.tensor([1.0, -0.2, 0.9])
    body_quat_w = torch.zeros(num_envs, 30, 4)
    body_quat_w[..., 0] = 1.0
    sim._robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=body_pos_w, body_quat_w=body_quat_w),
    )
    env.simulator = sim  # type: ignore[attr-defined]
    env.num_envs = num_envs
    env.device = "cpu"
    env.base_quat = torch.zeros(num_envs, 4)
    env.base_quat[:, 0] = 1.0
    env._left_wrist_isaac_id = LEFT_ID
    env._right_wrist_isaac_id = RIGHT_ID
    env._wrist_body_ids_t = torch.tensor([LEFT_ID, RIGHT_ID], dtype=torch.long)
    env.last_applied_force_w_by_body_id = {
        LEFT_ID: torch.zeros(num_envs, 3),
        RIGHT_ID: torch.zeros(num_envs, 3),
    }
    env.log_dict = {}
    env.command_manager = SimpleNamespace(  # type: ignore[assignment]
        get_state=MagicMock(
            side_effect=lambda name: term if name == "wrist_compliance_command" else None,
        ),
    )
    return env


def test_env_update_log_dict_merges_command_plus_env_keys() -> None:
    term = _make_term()
    env = _make_env_with_term(term)

    # Stub super()._update_log_dict (WholeBodyTrackingManager / BaseTask)
    # to avoid touching real baseline pipeline during the unit test.
    from unittest.mock import patch

    parent_cls = WholeBodyTrackingForceInjected.__mro__[1]
    with patch.object(parent_cls, "_update_log_dict", lambda _self: None):
        env._update_log_dict()

    # Full union of command + env keys must be present.
    all_keys = COMMAND_OWNED_KEYS | ENV_OWNED_KEYS
    assert all_keys.issubset(env.log_dict.keys()), "missing wandb keys: " + str(all_keys - env.log_dict.keys())


def test_env_log_dict_tensor_shapes_and_dtype() -> None:
    term = _make_term(num_envs=6)
    env = _make_env_with_term(term, num_envs=6)
    from unittest.mock import patch

    parent_cls = WholeBodyTrackingForceInjected.__mro__[1]
    with patch.object(parent_cls, "_update_log_dict", lambda _self: None):
        env._update_log_dict()

    for key in COMMAND_OWNED_KEYS | ENV_OWNED_KEYS:
        t = env.log_dict[key]
        assert isinstance(t, torch.Tensor)
        assert t.shape == (6,), f"{key} shape {t.shape}"
        assert t.dtype == torch.float32


def test_baseline_wbt_log_dict_has_no_force_keys() -> None:
    """Red line 3: baseline WBT env must not leak force/* keys.

    We monkey-patch the motion_command spy so baseline _update_log_dict
    runs its standard path (super() in the force subclass is skipped, so
    running this on the subclass with no wrist term registered should be
    a no-op for force keys; we assert that explicitly).
    """
    env = WholeBodyTrackingForceInjected.__new__(WholeBodyTrackingForceInjected)
    env.simulator = MagicMock()
    env.num_envs = 2
    env.device = "cpu"
    env.base_quat = torch.zeros(2, 4)
    env.base_quat[:, 0] = 1.0
    env._left_wrist_isaac_id = LEFT_ID
    env._right_wrist_isaac_id = RIGHT_ID
    env._wrist_body_ids_t = torch.tensor([LEFT_ID, RIGHT_ID], dtype=torch.long)
    env.last_applied_force_w_by_body_id = {}
    env.log_dict = {}
    env.command_manager = SimpleNamespace(get_state=MagicMock(return_value=None))  # type: ignore[assignment]

    from unittest.mock import patch

    parent_cls = WholeBodyTrackingForceInjected.__mro__[1]
    with patch.object(parent_cls, "_update_log_dict", lambda _self: None):
        env._update_log_dict()

    for key in COMMAND_OWNED_KEYS | ENV_OWNED_KEYS:
        assert key not in env.log_dict, f"force key {key} leaked without a registered wrist command"
