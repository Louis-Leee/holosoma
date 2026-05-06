# WBT Wrist-Force v14 — Force Tracking via Opposing F_ext Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-scope the WBT wrist-force experiment from a "virtual-spring compliance controller" (v10–v13) to a clean **force tracking** task: reward reverts to the baseline motion-tracking formula; every step the command term samples F_cmd on the wrist, injects `F_ext = -F_cmd` (world-frame) into the simulator on the same wrist, and the robot has to produce F_cmd to cancel F_ext while still tracking the reference motion.

**Architecture:**
- **Reward** — delete the wrist-force reward term; reuse the baseline WBT `motion_relative_body_position_error_exp` etc. verbatim. No `target_shifted`, no `K_virtual`, no `shift`.
- **Command term** — keep the trapezoidal FSM in `WristComplianceCommand` that owns F_cmd, but drop the independent F_ext channel and `k_virtual` buffer. Expose `force_ext_w = -F_cmd_w` (body-yaw → world-rotated) so the env subclass can inject it.
- **Actor obs** — observes `F_cmd` (body-yaw frame, 6 dims) only. No K.
- **Critic obs** — observes `F_cmd` (body-yaw, 6) and `F_ext` (world, 6). No K.
- **Env subclass** — unchanged injection path (`_apply_force_in_physics_step` rotates world F_ext into body frame and calls `set_external_force_and_torque`). What changes: F_ext is now deterministic given F_cmd + base yaw.
- **Naming** — register a new experiment `exp:g1-29dof-wbt-force-v2`, keep v10 (`exp:g1-29dof-wbt-force`) intact so both policies can be evaluated side-by-side.

**Tech Stack:** IsaacLab + IsaacSim, holosoma `WholeBodyTrackingForceInjected` env, PPO, torchrun multi-GPU, tyro CLI.

---

## Rationale (why this re-scope is correct)

The v10–v13 formulation was a virtual-spring **compliance** controller:

```
target_shifted = p_motion + (F_cmd_w + F_ext_w) / K_virtual
reward = exp(-||p_wrist - target_shifted||² / σ²)
```

That pushed the robot to chase a shifted reference point whose shift magnitude was set by a reward hyperparameter K. The policy could "cheat" by driving the wrist toward any point that satisfied `p_wrist = target_shifted` — **it never had to physically produce F_cmd**.

**v14 replaces the reward formulation with a physics-grounded one.** We keep the baseline reward (track the motion reference exactly), and we put the robot in a situation where producing F_cmd is the only way to hold the motion reference:

1. Sample a force profile `F_cmd(t)` on the wrist (same trapezoidal schedule as v10).
2. Apply `F_ext = -F_cmd` (world-frame) to the same wrist body via PhysX.
3. Tell the policy about `F_cmd` through the actor obs; tell the critic about both `F_cmd` and `F_ext` (F_ext is privileged sim-only info).
4. Reward the robot for tracking the **unmodified** motion reference.

Mechanically: the wrist experiences net external force `F_net = F_cmd_robot + F_ext = F_cmd_robot - F_cmd_target`. To keep the wrist on the reference motion trajectory (Newton's 2nd law), the robot must produce `F_cmd_robot = F_cmd_target`. The reward term becomes a direct proxy for "did the robot produce the commanded force", measured through motion deviation.

**What we drop vs v10:**
- `wrist_force_position_tracking_exp` reward term (→ baseline reward is unchanged)
- `K_virtual` field on `WristComplianceConfig`
- `k_virtual_range` on config
- `k_virtual` buffer on `WristComplianceCommand`
- `wrist_virtual_stiffness_command` obs term
- F_ext as an independent random process (→ now deterministic: `-F_cmd` rotated to world)
- All the env-owned "shift"/"wrist_pos_error" metrics (they measured the shifted target, not the motion target)

**What we keep from v10:**
- Trapezoidal F_cmd schedule with magnitude / duration / cooldown ranges
- Left/right wrist independent channels with per-wrist enable flags
- Debug arrow viz (F_cmd orange, F_ext green — now exactly anti-parallel)
- Env-level injection into IsaacLab via `set_external_force_and_torque(is_global=False)` after world→body rotation
- Multi-GPU via torchrun (`NUM_GPUS`)
- Motion file defaults, tyro CLI layout, wandb logger

---

## File Structure

### Files to CREATE

| Path | Responsibility |
|------|----------------|
| `docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md` | **this plan** |
| `demo_scripts/demo_wbt_wrist_force_training_v2.sh` | v2 launcher — selects `exp:g1-29dof-wbt-force-v2` and updates the Phase 7 gate checklist |
| `src/holosoma/holosoma/config_types/command_v2.py` | Frozen `WristForceTrackingConfig` dataclass (F_cmd-only schedule, no K) |
| `src/holosoma/holosoma/managers/command/terms/wbt_force_v2.py` | `WristForceTrackingCommand` — F_cmd trapezoidal FSM; `force_ext_w = R_yaw(base_quat) · (-F_cmd_b)` |
| `src/holosoma/holosoma/managers/observation/terms/wbt_force_v2.py` | `wrist_force_command_v2` (actor+critic), `wrist_force_ext_privileged_v2` (critic-only) |
| `src/holosoma/holosoma/envs/wbt/wbt_force_injected_v2.py` | `WholeBodyTrackingForceInjectedV2` — inherits v10 env, overrides `_env_owned_force_metrics` to the v2 metric set |
| `src/holosoma/holosoma/config_values/wbt/g1/command_force_v2.py` | `g1_29dof_wbt_force_v2_command` preset |
| `src/holosoma/holosoma/config_values/wbt/g1/observation_force_v2.py` | `g1_29dof_wbt_force_v2_observation` preset (actor: baseline+F_cmd; critic: baseline+F_cmd+F_ext) |
| `src/holosoma/holosoma/config_values/wbt/g1/reward_force_v2.py` | `g1_29dof_wbt_force_v2_reward` — **alias** for baseline `g1_29dof_wbt_reward` (no wrist reward) |
| `src/holosoma/holosoma/config_types/tests/test_wrist_force_tracking_config.py` | validates `WristForceTrackingConfig` `__post_init__` |
| `src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force_v2_command.py` | unit tests for the v2 command term |
| `src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force_v2_obs.py` | unit tests for v2 obs terms |
| `src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force_v2_reward.py` | regression test: v2 reward set is identical to baseline (no wrist reward) |
| `src/holosoma/tests/config_values/wbt/g1/test_command_force_v2_preset.py` | preset regression tests |
| `src/holosoma/tests/config_values/wbt/g1/test_observation_force_v2_preset.py` | dim assertions for v2 obs preset |
| `src/holosoma/tests/config_values/wbt/g1/test_reward_force_v2_preset.py` | asserts v2 reward preset ≡ baseline reward preset |
| `src/holosoma/tests/test_wbt_force_v2_wandb_metrics.py` | env-owned wandb metric key contract |

### Files to MODIFY

| Path | What changes |
|------|--------------|
| `src/holosoma/holosoma/config_values/experiment.py` | register `g1_29dof_wbt_force_v2` |
| `src/holosoma/holosoma/config_values/command.py` | register `g1_29dof_wbt_force_v2_command` |
| `src/holosoma/holosoma/config_values/observation.py` | register `g1_29dof_wbt_force_v2_observation` |
| `src/holosoma/holosoma/config_values/reward.py` | register `g1_29dof_wbt_force_v2_reward` |
| `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` | add `g1_29dof_wbt_force_v2 = replace(g1_29dof_wbt_force, ...)` |

### Files DELIBERATELY untouched

| Path | Why not touched |
|------|-----------------|
| `src/holosoma/holosoma/managers/reward/terms/wbt_force.py` | v10 reward stays reachable via `exp:g1-29dof-wbt-force` |
| `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py` | v10 env stays |
| `src/holosoma/holosoma/managers/command/terms/wbt_force.py` | v10 command stays |
| `demo_scripts/demo_wbt_wrist_force_training.sh` | v10 launcher stays; v2 is a **sibling** file, not a replacement |

v10 and v2 coexist as independent experiments so we can eval both against the same motion clip.

---

## Task 1: Frozen `WristForceTrackingConfig` dataclass

**Files:**
- Create: `src/holosoma/holosoma/config_types/command_v2.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/holosoma/config_types/tests/test_wrist_force_tracking_config.py`:

```python
"""Regression tests for WristForceTrackingConfig (v14)."""

from __future__ import annotations

import pytest

from holosoma.config_types.command_v2 import WristForceTrackingConfig


def test_defaults_construct_without_error() -> None:
    cfg = WristForceTrackingConfig()
    assert cfg.force_cmd_magnitude_range == (5.0, 30.0)
    assert cfg.force_cmd_duration_range_s == (1.0, 3.0)
    assert cfg.force_cmd_cooldown_range_s == (0.5, 2.0)
    assert cfg.force_cmd_ramp_frac == 0.25
    assert cfg.force_cmd_activation_prob_per_step == 0.01
    assert cfg.enable_left is True
    assert cfg.enable_right is True
    assert cfg.left_wrist_body_name == "left_wrist_yaw_link"
    assert cfg.right_wrist_body_name == "right_wrist_yaw_link"
    assert cfg.debug_arrow_scale_n_per_m == 50.0


def test_magnitude_lo_cannot_exceed_hi() -> None:
    with pytest.raises(AssertionError):
        WristForceTrackingConfig(force_cmd_magnitude_range=(30.0, 5.0))


def test_magnitude_lo_must_be_non_negative() -> None:
    with pytest.raises(AssertionError):
        WristForceTrackingConfig(force_cmd_magnitude_range=(-1.0, 30.0))


def test_ramp_frac_must_be_in_0_half() -> None:
    with pytest.raises(AssertionError):
        WristForceTrackingConfig(force_cmd_ramp_frac=0.6)


def test_debug_arrow_scale_must_be_positive() -> None:
    with pytest.raises(AssertionError):
        WristForceTrackingConfig(debug_arrow_scale_n_per_m=0.0)


def test_no_k_virtual_attribute() -> None:
    """v14 explicitly drops K_virtual."""
    cfg = WristForceTrackingConfig()
    assert not hasattr(cfg, "k_virtual_range")
    assert not hasattr(cfg, "k_virtual")


def test_no_independent_f_ext_schedule() -> None:
    """v14 derives F_ext from F_cmd, so no F_ext schedule fields."""
    cfg = WristForceTrackingConfig()
    for field_name in (
        "force_ext_magnitude_range",
        "force_ext_duration_range_s",
        "force_ext_cooldown_range_s",
        "force_ext_ramp_frac",
        "force_ext_activation_prob_per_step",
    ):
        assert not hasattr(cfg, field_name), f"v14 must not expose {field_name}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/holosoma/config_types/tests/test_wrist_force_tracking_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'holosoma.config_types.command_v2'`

- [ ] **Step 3: Write minimal implementation**

Create `src/holosoma/holosoma/config_types/command_v2.py`:

```python
"""v14 WBT wrist-force configuration — force tracking formulation.

Drops ``k_virtual`` and the independent F_ext schedule. F_cmd is the only
stochastic process; F_ext is derived deterministically as ``-F_cmd_world``
inside the command term.
"""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class WristForceTrackingConfig:
    """Configuration for the v14 ``WristForceTrackingCommand`` term."""

    # ---- F_cmd (policy-visible, body-yaw frame; F_ext = -F_cmd in world) ----
    force_cmd_magnitude_range: tuple[float, float] = (5.0, 30.0)
    force_cmd_duration_range_s: tuple[float, float] = (1.0, 3.0)
    force_cmd_cooldown_range_s: tuple[float, float] = (0.5, 2.0)
    force_cmd_ramp_frac: float = 0.25
    force_cmd_activation_prob_per_step: float = 0.01

    # ---- Debug-draw visual scale ----
    debug_arrow_scale_n_per_m: float = 50.0

    # ---- Enable flags + wrist body names ----
    enable_left: bool = True
    enable_right: bool = True
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"

    def __post_init__(self) -> None:
        rng = self.force_cmd_magnitude_range
        assert rng[0] >= 0.0, f"force_cmd_magnitude_range lower bound must be >= 0, got {rng}"
        assert rng[0] <= rng[1], f"force_cmd_magnitude_range must satisfy lo <= hi, got {rng}"

        for name, r in (
            ("force_cmd_duration_range_s", self.force_cmd_duration_range_s),
            ("force_cmd_cooldown_range_s", self.force_cmd_cooldown_range_s),
        ):
            assert r[0] >= 0.0, f"{name} lower bound must be >= 0, got {r}"
            assert r[0] <= r[1], f"{name} must satisfy lo <= hi, got {r}"

        assert 0.0 <= self.force_cmd_ramp_frac <= 0.5, (
            f"force_cmd_ramp_frac must be in [0, 0.5], got {self.force_cmd_ramp_frac}"
        )
        assert self.debug_arrow_scale_n_per_m > 0.0, (
            f"debug_arrow_scale_n_per_m must be > 0, got {self.debug_arrow_scale_n_per_m}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/holosoma/config_types/tests/test_wrist_force_tracking_config.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_types/command_v2.py src/holosoma/holosoma/config_types/tests/test_wrist_force_tracking_config.py
git commit -m "feat(wbt-force-v2): add WristForceTrackingConfig (drop K_virtual, single-channel F_cmd)"
```

---

## Task 2: `WristForceTrackingCommand` — single-channel F_cmd with derived F_ext

**Files:**
- Create: `src/holosoma/holosoma/managers/command/terms/wbt_force_v2.py`
- Test: `src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force_v2_command.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force_v2_command.py`:

```python
"""Unit tests for WristForceTrackingCommand (v14)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.command_v2 import WristForceTrackingConfig
from holosoma.managers.command.terms.wbt_force_v2 import (
    NUM_WRISTS,
    STATE_COOLDOWN,
    STATE_RAMP_UP,
    WristForceTrackingCommand,
)


def _mock_env(num_envs: int = 4, dt: float = 0.02) -> Any:
    """Minimal env stub with .num_envs, .device, .dt, and .base_quat (xyzw)."""
    device = torch.device("cpu")
    base_quat = torch.zeros(num_envs, 4, device=device)
    base_quat[:, 3] = 1.0  # identity xyzw
    return SimpleNamespace(
        num_envs=num_envs,
        device=device,
        dt=dt,
        base_quat=base_quat,
    )


def _cfg(**overrides: Any) -> CommandTermCfg:
    params = {"wrist_force_tracking_config": WristForceTrackingConfig(**overrides)}
    return CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand",
        params=params,
    )


def test_initial_state_is_cooldown_zero_force() -> None:
    term = WristForceTrackingCommand(_cfg(), _mock_env())
    term.reset(None)
    assert torch.all(term.force_cmd_b == 0.0)
    assert torch.all(term.force_ext_w == 0.0)
    assert torch.all(term._cmd_channel.state == STATE_COOLDOWN)


def test_force_ext_is_negative_of_rotated_force_cmd() -> None:
    """F_ext (world) must equal -R_yaw(base_quat) · F_cmd_b for all steps."""
    torch.manual_seed(0)
    env = _mock_env(num_envs=8)
    term = WristForceTrackingCommand(
        _cfg(force_cmd_activation_prob_per_step=1.0),  # trigger immediately
        env,
    )
    term.reset(None)
    # Drive the FSM a few steps to accumulate magnitude.
    for _ in range(5):
        term.step()

    f_cmd_b = term.force_cmd_b
    f_ext_w = term.force_ext_w

    # Identity base_quat ⇒ yaw = identity ⇒ world = body.
    expected = -f_cmd_b
    torch.testing.assert_close(f_ext_w, expected, atol=1e-5, rtol=1e-5)


def test_force_ext_rotates_with_base_yaw() -> None:
    """Rotate base by +90° yaw, then F_ext_w should be -R · F_cmd_b."""
    torch.manual_seed(42)
    env = _mock_env(num_envs=2)
    # +90° yaw quat xyzw.
    import math
    half = math.pi / 4
    env.base_quat[:] = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)])

    term = WristForceTrackingCommand(
        _cfg(force_cmd_activation_prob_per_step=1.0),
        env,
    )
    term.reset(None)
    for _ in range(3):
        term.step()

    f_cmd_b = term.force_cmd_b  # (N, 2, 3)
    f_ext_w = term.force_ext_w

    # By hand: +90° yaw sends (x,y,z) → (-y, x, z). Then F_ext = -that.
    rotated = torch.stack(
        [-f_cmd_b[..., 1], f_cmd_b[..., 0], f_cmd_b[..., 2]], dim=-1
    )
    torch.testing.assert_close(f_ext_w, -rotated, atol=1e-5, rtol=1e-5)


def test_no_k_virtual_buffer() -> None:
    term = WristForceTrackingCommand(_cfg(), _mock_env())
    term.reset(None)
    assert not hasattr(term, "k_virtual")


def test_reset_zeros_force() -> None:
    term = WristForceTrackingCommand(
        _cfg(force_cmd_activation_prob_per_step=1.0),
        _mock_env(num_envs=4),
    )
    term.reset(None)
    for _ in range(5):
        term.step()
    assert term.force_cmd_b.abs().sum().item() > 0.0

    term.reset(torch.tensor([0, 1, 2, 3]))
    assert torch.all(term.force_cmd_b == 0.0)
    assert torch.all(term.force_ext_w == 0.0)


def test_metrics_contain_no_k_key() -> None:
    term = WristForceTrackingCommand(_cfg(), _mock_env())
    term.reset(None)
    term.step()
    term.update_metrics()
    for key in term.metrics:
        assert "k_virtual" not in key, f"v14 must not log {key}"


def test_metrics_expose_cmd_ext_anti_parallel_alignment() -> None:
    """cmd_ext_alignment should be exactly -1 whenever both non-zero."""
    torch.manual_seed(0)
    term = WristForceTrackingCommand(
        _cfg(force_cmd_activation_prob_per_step=1.0),
        _mock_env(num_envs=8),
    )
    term.reset(None)
    for _ in range(5):
        term.step()
    term.update_metrics()

    both_active = (term.force_cmd_b.norm(dim=-1) > 1e-3)
    if both_active.any():
        align = term.metrics["force/cmd_ext_alignment"]
        # Should be close to -1 on envs where any wrist is active.
        mean = align.mean().item()
        assert mean < -0.9, f"cmd and ext should be anti-parallel, mean cos = {mean}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force_v2_command.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'holosoma.managers.command.terms.wbt_force_v2'`

- [ ] **Step 3: Write minimal implementation**

Create `src/holosoma/holosoma/managers/command/terms/wbt_force_v2.py`:

```python
"""v14 wrist force-tracking command — F_cmd drives both actor obs and F_ext.

One trapezoidal FSM (``COOLDOWN -> RAMP_UP -> HOLD -> RAMP_DOWN``) generates
F_cmd per (env, wrist). F_ext is **derived**: ``F_ext_w = -R_yaw(base_quat) ·
F_cmd_b`` every step, so the physics wrench on the wrist exactly opposes the
commanded force and the policy must produce +F_cmd to stay on the motion
reference. No K_virtual. No shift. No independent F_ext channel.

See docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.command.terms._gh_utils import (
    TemporalLerp,
    clamp_norm,
    random_uniform,
)

if TYPE_CHECKING:
    from holosoma.config_types.command import CommandTermCfg
    from holosoma.config_types.command_v2 import WristForceTrackingConfig

# ---- FSM state codes -------------------------------------------------------
STATE_COOLDOWN: int = 0
STATE_RAMP_UP: int = 1
STATE_HOLD: int = 2
STATE_RAMP_DOWN: int = 3

NUM_WRISTS: int = 2
DIM: int = 3


def _duration_to_steps(duration_s: torch.Tensor, control_rate_hz: float) -> torch.Tensor:
    steps = torch.round(duration_s * control_rate_hz).to(dtype=torch.int32)
    return torch.clamp(steps, min=1)


class _ForceChannel:
    """One trapezoidal F_cmd channel; identical to v10 ``_ForceChannel`` semantics."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        *,
        magnitude_range: tuple[float, float],
        duration_range_s: tuple[float, float],
        cooldown_range_s: tuple[float, float],
        ramp_frac: float,
        activation_prob_per_step: float,
        enable_left: bool,
        enable_right: bool,
        control_rate_hz: float,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.magnitude_range = magnitude_range
        self.duration_range_s = duration_range_s
        self.cooldown_range_s = cooldown_range_s
        self.ramp_frac = ramp_frac
        self.activation_prob = activation_prob_per_step
        self.control_rate_hz = control_rate_hz

        self.force = torch.zeros(num_envs, NUM_WRISTS, DIM, device=device)
        self._lerp = TemporalLerp(
            (num_envs * NUM_WRISTS,),
            device=device,
            default=0.0,
            easing="linear",
        )
        self.state = torch.full((num_envs, NUM_WRISTS), STATE_COOLDOWN, dtype=torch.int8, device=device)
        self.cooldown_remaining = torch.zeros((num_envs, NUM_WRISTS), dtype=torch.int32, device=device)
        self.hold_remaining = torch.zeros_like(self.cooldown_remaining)
        self.ramp_up_steps = torch.ones_like(self.cooldown_remaining)
        self.ramp_down_steps = torch.ones_like(self.cooldown_remaining)
        self.hold_steps = torch.zeros_like(self.cooldown_remaining)

        self.peak_magnitude = torch.zeros((num_envs, NUM_WRISTS), dtype=torch.float32, device=device)
        self.direction = torch.zeros((num_envs, NUM_WRISTS, DIM), dtype=torch.float32, device=device)

        mask = torch.tensor(
            [float(enable_left), float(enable_right)], dtype=torch.float32, device=device
        )
        self._wrist_enable_mask = mask.view(1, NUM_WRISTS, 1)
        self._wrist_enable_2d = mask.view(1, NUM_WRISTS)

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.force[env_ids] = 0.0
        self.state[env_ids] = STATE_COOLDOWN
        self.cooldown_remaining[env_ids] = 0
        self.hold_remaining[env_ids] = 0
        self.ramp_up_steps[env_ids] = 1
        self.ramp_down_steps[env_ids] = 1
        self.hold_steps[env_ids] = 0
        self.peak_magnitude[env_ids] = 0.0
        self.direction[env_ids] = 0.0
        self._lerp.reset(self._flatten_ids(env_ids))

    def _flatten_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        base = env_ids.view(-1, 1) * NUM_WRISTS
        offsets = torch.arange(NUM_WRISTS, device=self.device, dtype=torch.long)
        return (base + offsets.view(1, -1)).reshape(-1)

    def _pair_to_flat(self, env_idx: torch.Tensor, w_idx: torch.Tensor) -> torch.Tensor:
        return (env_idx * NUM_WRISTS + w_idx).to(dtype=torch.long)

    @property
    def lerp_magnitude_nw(self) -> torch.Tensor:
        return self._lerp.value.view(self.num_envs, NUM_WRISTS)

    @property
    def lerp_active_nw(self) -> torch.Tensor:
        return self._lerp._active.view(self.num_envs, NUM_WRISTS)

    @torch.no_grad()
    def step(self) -> None:
        device = self.device
        hold_mask = self.state == STATE_HOLD
        self.hold_remaining[hold_mask] = self.hold_remaining[hold_mask] - 1
        cool_mask = self.state == STATE_COOLDOWN
        self.cooldown_remaining[cool_mask] = self.cooldown_remaining[cool_mask].clamp(min=0) - 1
        self.cooldown_remaining = self.cooldown_remaining.clamp(min=0)

        self._lerp.update_time(1)

        ramp_up_done = (self.state == STATE_RAMP_UP) & (~self.lerp_active_nw)
        if ramp_up_done.any():
            self.state[ramp_up_done] = STATE_HOLD
            self.hold_remaining[ramp_up_done] = self.hold_steps[ramp_up_done]

        hold_done = (self.state == STATE_HOLD) & (self.hold_remaining <= 0)
        if hold_done.any():
            self.state[hold_done] = STATE_RAMP_DOWN
            idx = hold_done.nonzero(as_tuple=False)
            env_idx = idx[:, 0]
            w_idx = idx[:, 1]
            flat = self._pair_to_flat(env_idx, w_idx)
            lerp = self._lerp
            lerp._start[flat] = lerp.value[flat]
            lerp._end[flat] = 0.0
            lerp._t[flat] = 0
            lerp._T[flat] = self.ramp_down_steps[env_idx, w_idx]
            lerp._active[flat] = True

        ramp_down_done = (self.state == STATE_RAMP_DOWN) & (~self.lerp_active_nw)
        if ramp_down_done.any():
            self.state[ramp_down_done] = STATE_COOLDOWN
            n = int(ramp_down_done.sum().item())
            cool_s = random_uniform(
                (n,), self.cooldown_range_s[0], self.cooldown_range_s[1], device=device
            )
            self.cooldown_remaining[ramp_down_done] = _duration_to_steps(
                cool_s, self.control_rate_hz
            )
            self.peak_magnitude[ramp_down_done] = 0.0
            self.direction[ramp_down_done] = 0.0

        ready = (self.state == STATE_COOLDOWN) & (self.cooldown_remaining == 0)
        ready = ready & self._wrist_enable_2d.expand_as(ready).bool()
        if ready.any():
            triggered = ready & (torch.rand(ready.shape, device=device) < self.activation_prob)
            if triggered.any():
                self._start_episode(triggered)

        magnitude_nw = self.lerp_magnitude_nw.unsqueeze(-1)
        self.force = magnitude_nw * self.direction * self._wrist_enable_mask

    @torch.no_grad()
    def _start_episode(self, triggered: torch.Tensor) -> None:
        device = self.device
        n = int(triggered.sum().item())
        if n == 0:
            return
        idx = triggered.nonzero(as_tuple=False)
        env_idx = idx[:, 0]
        w_idx = idx[:, 1]

        d = torch.randn(n, DIM, device=device)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mag_lo, mag_hi = self.magnitude_range
        mag = random_uniform((n,), mag_lo, mag_hi, device=device)
        peak_vec = clamp_norm(d * mag.unsqueeze(-1), max_norm=float(mag_hi))
        mag = peak_vec.norm(dim=-1)
        d = peak_vec / mag.clamp(min=1e-6).unsqueeze(-1)

        self.direction[env_idx, w_idx] = d
        self.peak_magnitude[env_idx, w_idx] = mag

        dur_s = random_uniform(
            (n,), self.duration_range_s[0], self.duration_range_s[1], device=device
        )
        ramp_up_steps_t = torch.clamp(
            torch.round(dur_s * self.ramp_frac * self.control_rate_hz).to(torch.int32),
            min=1,
        )
        ramp_down_steps_t = ramp_up_steps_t.clone()
        total_steps_t = _duration_to_steps(dur_s, self.control_rate_hz)
        hold_steps_t = torch.clamp(total_steps_t - ramp_up_steps_t - ramp_down_steps_t, min=0)

        self.ramp_up_steps[env_idx, w_idx] = ramp_up_steps_t
        self.ramp_down_steps[env_idx, w_idx] = ramp_down_steps_t
        self.hold_steps[env_idx, w_idx] = hold_steps_t

        flat = self._pair_to_flat(env_idx, w_idx)
        lerp = self._lerp
        lerp._start[flat] = 0.0
        lerp._end[flat] = mag
        lerp._t[flat] = 0
        lerp._T[flat] = ramp_up_steps_t
        lerp._active[flat] = True
        lerp.value[flat] = 0.0

        self.state[env_idx, w_idx] = STATE_RAMP_UP


def _rotate_body_yaw_to_world(base_quat_xyzw: torch.Tensor, force_b: torch.Tensor) -> torch.Tensor:
    """Rotate (N, 2, 3) body-yaw force to world frame via base yaw quat."""
    from holosoma.utils.rotations import quat_apply, yaw_quat

    yaw_q = yaw_quat(base_quat_xyzw, w_last=True)  # (N, 4) xyzw
    yaw_q_nw = yaw_q.unsqueeze(1).expand(-1, force_b.shape[1], -1).reshape(-1, 4)
    flat = force_b.reshape(-1, 3)
    out = quat_apply(yaw_q_nw, flat, w_last=True)
    return out.view(force_b.shape)


class WristForceTrackingCommand(CommandTermBase):
    """Command term for v14 force-tracking: F_cmd sampled, F_ext = -F_cmd (world)."""

    def __init__(self, cfg: "CommandTermCfg", env: Any) -> None:
        super().__init__(cfg, env)
        params: dict[str, Any] = cfg.params or {}
        wcfg = params.get("wrist_force_tracking_config")
        if wcfg is None:
            raise ValueError(
                "WristForceTrackingCommand requires 'wrist_force_tracking_config' in params."
            )
        self.wrist_cfg: WristForceTrackingConfig = wcfg

        self._env = env
        self.num_envs: int = env.num_envs
        self.device: torch.device = torch.device(env.device)

        dt = float(getattr(env, "dt", 0.0))
        if dt <= 0.0:
            raise RuntimeError(f"env.dt must be > 0, got {dt!r}.")
        self.control_rate_hz: float = 1.0 / dt

        self._cmd_channel = _ForceChannel(
            self.num_envs,
            self.device,
            magnitude_range=self.wrist_cfg.force_cmd_magnitude_range,
            duration_range_s=self.wrist_cfg.force_cmd_duration_range_s,
            cooldown_range_s=self.wrist_cfg.force_cmd_cooldown_range_s,
            ramp_frac=self.wrist_cfg.force_cmd_ramp_frac,
            activation_prob_per_step=self.wrist_cfg.force_cmd_activation_prob_per_step,
            enable_left=self.wrist_cfg.enable_left,
            enable_right=self.wrist_cfg.enable_right,
            control_rate_hz=self.control_rate_hz,
        )

        self.metrics: dict[str, torch.Tensor] = {}

    def setup(self) -> None:
        return

    def reset(self, env_ids: torch.Tensor | None) -> None:
        idx = self._as_index_tensor(env_ids)
        if idx.numel() == 0:
            return
        self._cmd_channel.reset(idx)

    def step(self) -> None:
        self._cmd_channel.step()

    # ---- observable buffers ------------------------------------------------
    @property
    def force_cmd_b(self) -> torch.Tensor:
        """Body-yaw frame F_cmd, shape [N, 2, 3]."""
        return self._cmd_channel.force

    @property
    def force_ext_w(self) -> torch.Tensor:
        """World-frame F_ext = -R_yaw(base_quat) · F_cmd_b, shape [N, 2, 3]."""
        return -_rotate_body_yaw_to_world(self._env.base_quat, self._cmd_channel.force)

    # ---- metrics -----------------------------------------------------------
    def update_metrics(self) -> None:
        metrics: dict[str, torch.Tensor] = {}
        f_cmd = self.force_cmd_b
        f_ext = self.force_ext_w
        cmd_mag = f_cmd.norm(dim=-1)
        ext_mag = f_ext.norm(dim=-1)

        metrics["force/cmd_magnitude_l"] = cmd_mag[:, 0].float()
        metrics["force/cmd_magnitude_r"] = cmd_mag[:, 1].float()
        metrics["force/cmd_magnitude_max"] = cmd_mag.max(dim=-1).values.float()
        metrics["force/ext_magnitude_l"] = ext_mag[:, 0].float()
        metrics["force/ext_magnitude_r"] = ext_mag[:, 1].float()
        metrics["force/ext_magnitude_max"] = ext_mag.max(dim=-1).values.float()

        # Sanity: cos(F_cmd_world, F_ext_world) should be -1 when active.
        from holosoma.managers.command.terms.wbt_force_v2 import _rotate_body_yaw_to_world
        f_cmd_w = _rotate_body_yaw_to_world(self._env.base_quat, f_cmd)
        cmd_safe = f_cmd_w.norm(dim=-1).clamp(min=1e-6)
        ext_safe = ext_mag.clamp(min=1e-6)
        cos_per_wrist = (f_cmd_w * f_ext).sum(dim=-1) / (cmd_safe * ext_safe)
        both_active = (cmd_mag > 1e-3) & (ext_mag > 1e-3)
        cos_per_wrist = torch.where(both_active, cos_per_wrist, torch.zeros_like(cos_per_wrist))
        denom = both_active.float().sum(dim=-1).clamp(min=1.0)
        metrics["force/cmd_ext_alignment"] = (cos_per_wrist.sum(dim=-1) / denom).float()

        cmd_active = (self._cmd_channel.state != STATE_COOLDOWN).float()
        metrics["force/active_frac_cmd"] = cmd_active.mean(dim=-1)

        cmd_state = self._cmd_channel.state
        metrics["force/phase_ramp_up"] = (cmd_state == STATE_RAMP_UP).float().mean(dim=-1)
        metrics["force/phase_hold"] = (cmd_state == STATE_HOLD).float().mean(dim=-1)
        metrics["force/phase_ramp_down"] = (cmd_state == STATE_RAMP_DOWN).float().mean(dim=-1)

        self.metrics = metrics

    def _as_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force_v2_command.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/managers/command/terms/wbt_force_v2.py src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force_v2_command.py
git commit -m "feat(wbt-force-v2): WristForceTrackingCommand with derived F_ext = -F_cmd"
```

---

## Task 3: Observation terms (v2)

**Files:**
- Create: `src/holosoma/holosoma/managers/observation/terms/wbt_force_v2.py`
- Test: `src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force_v2_obs.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force_v2_obs.py`:

```python
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
    def __init__(self, term: WristForceTrackingCommand) -> None:
        self._term = term

    def get_state(self, name: str) -> Any:
        if name == "wrist_force_tracking_command":
            return self._term
        return None


def _make_env(num_envs: int = 4) -> Any:
    device = torch.device("cpu")
    base_quat = torch.zeros(num_envs, 4)
    base_quat[:, 3] = 1.0
    env = SimpleNamespace(
        num_envs=num_envs, device=device, dt=0.02, base_quat=base_quat,
    )
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand",
        params={"wrist_force_tracking_config": WristForceTrackingConfig(
            force_cmd_activation_prob_per_step=1.0,
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
    # No NaN / Inf.
    assert torch.isfinite(out).all()


def test_wrist_force_ext_privileged_v2_shape() -> None:
    env = _make_env()
    out = wrist_force_ext_privileged_v2(env)
    assert out.shape == (4, 6)


def test_obs_is_a_clone_not_a_view() -> None:
    env = _make_env()
    cmd = wrist_force_command_v2(env)
    cmd.fill_(123.0)
    # Term's underlying buffer must be unaffected.
    term = env.command_manager.get_state("wrist_force_tracking_command")
    assert not torch.all(term.force_cmd_b == 123.0)


def test_missing_command_raises() -> None:
    env = SimpleNamespace(num_envs=1, command_manager=_CmdMgr(None))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="wrist_force_tracking_command"):
        wrist_force_command_v2(env)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force_v2_obs.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

Create `src/holosoma/holosoma/managers/observation/terms/wbt_force_v2.py`:

```python
"""v14 wrist-force observation terms.

* ``wrist_force_command_v2``: actor+critic, F_cmd body-yaw, [N, 6].
* ``wrist_force_ext_privileged_v2``: critic-only, F_ext world, [N, 6].

The term reads ``wrist_force_tracking_command`` (v14 key) — **not**
``wrist_compliance_command`` (v10 key) — so v10 and v14 experiments can
coexist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt_force_v2 import WristForceTrackingCommand

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


_TERM_KEY = "wrist_force_tracking_command"


def _get_cmd(env: WholeBodyTrackingManager) -> WristForceTrackingCommand:
    term = env.command_manager.get_state(_TERM_KEY)
    if term is None:
        raise RuntimeError(
            f"v14 wrist-force obs terms require the '{_TERM_KEY}' command term to be registered."
        )
    if not isinstance(term, WristForceTrackingCommand):
        raise TypeError(
            f"'{_TERM_KEY}' must be a WristForceTrackingCommand, got {type(term).__name__}."
        )
    return term


def wrist_force_command_v2(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Flatten F_cmd [N, 2, 3] → [N, 6]. Returns a clone."""
    return _get_cmd(env).force_cmd_b.reshape(env.num_envs, 6).clone()


def wrist_force_ext_privileged_v2(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Flatten F_ext world [N, 2, 3] → [N, 6]. Critic-only."""
    return _get_cmd(env).force_ext_w.reshape(env.num_envs, 6).clone()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force_v2_obs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/managers/observation/terms/wbt_force_v2.py src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force_v2_obs.py
git commit -m "feat(wbt-force-v2): obs terms (actor: F_cmd; critic: F_cmd + F_ext priv)"
```

---

## Task 4: Reward preset (v2)

**Files:**
- Create: `src/holosoma/holosoma/config_values/wbt/g1/reward_force_v2.py`
- Test: `src/holosoma/tests/config_values/wbt/g1/test_reward_force_v2_preset.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/tests/config_values/wbt/g1/test_reward_force_v2_preset.py`:

```python
"""v14 reward preset — must equal baseline reward (no wrist reward term)."""

from __future__ import annotations

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward


def test_v2_reward_preset_is_baseline() -> None:
    from holosoma.config_values.wbt.g1.reward_force_v2 import g1_29dof_wbt_force_v2_reward

    assert g1_29dof_wbt_force_v2_reward is g1_29dof_wbt_reward or (
        set(g1_29dof_wbt_force_v2_reward.terms.keys())
        == set(g1_29dof_wbt_reward.terms.keys())
    )


def test_v2_has_no_wrist_force_reward_term() -> None:
    from holosoma.config_values.wbt.g1.reward_force_v2 import g1_29dof_wbt_force_v2_reward

    for name in g1_29dof_wbt_force_v2_reward.terms:
        assert "wrist_force" not in name, (
            f"v14 must NOT carry a wrist_force reward term; found {name}"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/tests/config_values/wbt/g1/test_reward_force_v2_preset.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

Create `src/holosoma/holosoma/config_values/wbt/g1/reward_force_v2.py`:

```python
"""v14 reward preset — alias for the baseline WBT reward preset.

The v14 formulation does not add any wrist-force reward term: reward is
the unmodified motion-tracking reward, and the F_cmd/F_ext mechanism
shows up only through the policy's observations + the sim-injected
wrench on the wrist bodies.
"""

from __future__ import annotations

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

g1_29dof_wbt_force_v2_reward = g1_29dof_wbt_reward

__all__ = ["g1_29dof_wbt_force_v2_reward"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/tests/config_values/wbt/g1/test_reward_force_v2_preset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/reward_force_v2.py src/holosoma/tests/config_values/wbt/g1/test_reward_force_v2_preset.py
git commit -m "feat(wbt-force-v2): reward preset aliases baseline (no wrist reward)"
```

---

## Task 5: Command preset (v2)

**Files:**
- Create: `src/holosoma/holosoma/config_values/wbt/g1/command_force_v2.py`
- Test: `src/holosoma/tests/config_values/wbt/g1/test_command_force_v2_preset.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/tests/config_values/wbt/g1/test_command_force_v2_preset.py`:

```python
"""v14 command preset — adds wrist_force_tracking_command on top of baseline."""

from __future__ import annotations

from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command


def test_preset_adds_v14_term() -> None:
    from holosoma.config_values.wbt.g1.command_force_v2 import g1_29dof_wbt_force_v2_command

    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_command.setup_terms
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_command.reset_terms
    assert "wrist_force_tracking_command" in g1_29dof_wbt_force_v2_command.step_terms


def test_preset_does_not_carry_v10_term() -> None:
    from holosoma.config_values.wbt.g1.command_force_v2 import g1_29dof_wbt_force_v2_command

    assert "wrist_compliance_command" not in g1_29dof_wbt_force_v2_command.setup_terms


def test_preset_keeps_baseline_motion_command() -> None:
    from holosoma.config_values.wbt.g1.command_force_v2 import g1_29dof_wbt_force_v2_command

    for key in g1_29dof_wbt_command.setup_terms:
        assert key in g1_29dof_wbt_force_v2_command.setup_terms
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/tests/config_values/wbt/g1/test_command_force_v2_preset.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

Create `src/holosoma/holosoma/config_values/wbt/g1/command_force_v2.py`:

```python
"""v14 command preset: baseline WBT command + wrist_force_tracking_command."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.command_v2 import WristForceTrackingConfig
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command

_WRIST_FORCE_FUNC = (
    "holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand"
)
_DEFAULT_CFG = WristForceTrackingConfig()


def _term() -> CommandTermCfg:
    return CommandTermCfg(
        func=_WRIST_FORCE_FUNC,
        params={"wrist_force_tracking_config": _DEFAULT_CFG},
    )


g1_29dof_wbt_force_v2_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        **g1_29dof_wbt_command.setup_terms,
        "wrist_force_tracking_command": _term(),
    },
    reset_terms={
        **g1_29dof_wbt_command.reset_terms,
        "wrist_force_tracking_command": _term(),
    },
    step_terms={
        **g1_29dof_wbt_command.step_terms,
        "wrist_force_tracking_command": _term(),
    },
)

__all__ = ["g1_29dof_wbt_force_v2_command"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/tests/config_values/wbt/g1/test_command_force_v2_preset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/command_force_v2.py src/holosoma/tests/config_values/wbt/g1/test_command_force_v2_preset.py
git commit -m "feat(wbt-force-v2): command preset registers wrist_force_tracking_command"
```

---

## Task 6: Observation preset (v2)

**Files:**
- Create: `src/holosoma/holosoma/config_values/wbt/g1/observation_force_v2.py`
- Test: `src/holosoma/tests/config_values/wbt/g1/test_observation_force_v2_preset.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/tests/config_values/wbt/g1/test_observation_force_v2_preset.py`:

```python
"""v14 observation preset — actor: baseline + F_cmd; critic: baseline + F_cmd + F_ext priv."""

from __future__ import annotations

from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    critic_obs_shared_terms,
)


def test_preset_adds_f_cmd_to_both_groups() -> None:
    from holosoma.config_values.wbt.g1.observation_force_v2 import (
        g1_29dof_wbt_force_v2_observation,
    )

    actor_terms = g1_29dof_wbt_force_v2_observation.groups["actor_obs"].terms
    critic_terms = g1_29dof_wbt_force_v2_observation.groups["critic_obs"].terms

    assert "wrist_force_command_v2" in actor_terms
    assert "wrist_force_command_v2" in critic_terms


def test_preset_adds_f_ext_priv_only_to_critic() -> None:
    from holosoma.config_values.wbt.g1.observation_force_v2 import (
        g1_29dof_wbt_force_v2_observation,
    )

    actor_terms = g1_29dof_wbt_force_v2_observation.groups["actor_obs"].terms
    critic_terms = g1_29dof_wbt_force_v2_observation.groups["critic_obs"].terms

    assert "wrist_force_ext_privileged_v2" not in actor_terms
    assert "wrist_force_ext_privileged_v2" in critic_terms


def test_preset_has_no_k_virtual_term() -> None:
    from holosoma.config_values.wbt.g1.observation_force_v2 import (
        g1_29dof_wbt_force_v2_observation,
    )

    for group in g1_29dof_wbt_force_v2_observation.groups.values():
        for name in group.terms:
            assert "virtual_stiffness" not in name
            assert "k_virtual" not in name


def test_preset_preserves_baseline_actor_terms() -> None:
    from holosoma.config_values.wbt.g1.observation_force_v2 import (
        g1_29dof_wbt_force_v2_observation,
    )

    actor_terms = g1_29dof_wbt_force_v2_observation.groups["actor_obs"].terms
    for key in actor_obs_shared.terms:
        assert key in actor_terms


def test_preset_history_length_matches_v10() -> None:
    from holosoma.config_values.wbt.g1.observation_force_v2 import (
        g1_29dof_wbt_force_v2_observation,
    )

    assert g1_29dof_wbt_force_v2_observation.groups["actor_obs"].history_length == 10
    assert g1_29dof_wbt_force_v2_observation.groups["critic_obs"].history_length == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/tests/config_values/wbt/g1/test_observation_force_v2_preset.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

Create `src/holosoma/holosoma/config_values/wbt/g1/observation_force_v2.py`:

```python
"""v14 observation preset.

Actor group (per step): baseline 6 actor terms + wrist_force_command_v2.
Critic group (per step): baseline 10 critic terms + wrist_force_command_v2 + wrist_force_ext_privileged_v2.

history_length=10 on both groups (CHIP-style).
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    critic_obs_shared_terms,
)

_WRIST_FORCE_V2_MOD = "holosoma.managers.observation.terms.wbt_force_v2"

wrist_force_command_v2_term = ObsTermCfg(
    func=f"{_WRIST_FORCE_V2_MOD}:wrist_force_command_v2",
    scale=1.0,
    noise=0.0,
)

wrist_force_ext_privileged_v2_term = ObsTermCfg(
    func=f"{_WRIST_FORCE_V2_MOD}:wrist_force_ext_privileged_v2",
    scale=1.0,
    noise=0.0,
)

_actor_terms = {
    **actor_obs_shared.terms,
    "wrist_force_command_v2": wrist_force_command_v2_term,
}

actor_obs_force_v2 = replace(
    actor_obs_shared,
    history_length=10,
    terms=_actor_terms,
)

_critic_terms = {
    **critic_obs_shared_terms,
    "wrist_force_command_v2": wrist_force_command_v2_term,
    "wrist_force_ext_privileged_v2": wrist_force_ext_privileged_v2_term,
}

critic_obs_force_v2 = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=10,
    terms=_critic_terms,
)

g1_29dof_wbt_force_v2_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_force_v2,
        "critic_obs": critic_obs_force_v2,
    },
)

__all__ = ["g1_29dof_wbt_force_v2_observation"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/tests/config_values/wbt/g1/test_observation_force_v2_preset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/observation_force_v2.py src/holosoma/tests/config_values/wbt/g1/test_observation_force_v2_preset.py
git commit -m "feat(wbt-force-v2): obs preset (actor: F_cmd; critic: F_cmd + F_ext priv)"
```

---

## Task 7: Env subclass (v2)

**Files:**
- Create: `src/holosoma/holosoma/envs/wbt/wbt_force_injected_v2.py`
- Test: extend `src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected.py` is v10-only; create new:
  `src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected_v2.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected_v2.py`:

```python
"""CPU structural tests for WholeBodyTrackingForceInjectedV2 (no IsaacSim)."""

from __future__ import annotations

import inspect

import pytest


def test_v2_env_class_imports() -> None:
    from holosoma.envs.wbt.wbt_force_injected_v2 import WholeBodyTrackingForceInjectedV2

    assert inspect.isclass(WholeBodyTrackingForceInjectedV2)


def test_v2_inherits_from_v10() -> None:
    from holosoma.envs.wbt.wbt_force_injected import WholeBodyTrackingForceInjected
    from holosoma.envs.wbt.wbt_force_injected_v2 import WholeBodyTrackingForceInjectedV2

    assert issubclass(WholeBodyTrackingForceInjectedV2, WholeBodyTrackingForceInjected)


def test_v2_uses_v14_command_key_in_getter() -> None:
    """The env must look up the v14 command key, not the v10 key."""
    from holosoma.envs.wbt.wbt_force_injected_v2 import WholeBodyTrackingForceInjectedV2

    src = inspect.getsource(WholeBodyTrackingForceInjectedV2)
    assert "wrist_force_tracking_command" in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected_v2.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

Create `src/holosoma/holosoma/envs/wbt/wbt_force_injected_v2.py`:

```python
"""v14 WBT env subclass — injects F_ext=-F_cmd on wrists + env-owned wandb metrics.

Inherits from :class:`WholeBodyTrackingForceInjected` (v10) to reuse all of:

* body-id resolution for the two wrists
* ``_apply_force_in_physics_step`` that rotates world F_ext into body frame
  and calls IsaacLab ``set_external_force_and_torque``
* debug arrow viz hooked into the simulator's draw hook
* simulator-draw chaining

What we override:

* ``_get_wrist_command`` — looks up the v14 key ``wrist_force_tracking_command``
  (not v10's ``wrist_compliance_command``).
* ``_env_owned_force_metrics`` — drops the shift / wrist_pos_error /
  k_virtual metrics (those were v10-reward-specific). Keeps only the
  applied-body-force sanity metric.

The v14 command term exposes ``force_ext_w`` as a property that returns
``-R_yaw(base_quat) · F_cmd_b``; the inherited ``_apply_force_in_physics_step``
reads that property, rotates world→body, and pushes to PhysX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.envs.wbt.wbt_force_injected import WholeBodyTrackingForceInjected
from holosoma.managers.command.terms.wbt_force_v2 import WristForceTrackingCommand

if TYPE_CHECKING:
    from holosoma.config_types.command_v2 import WristForceTrackingConfig


_TERM_KEY = "wrist_force_tracking_command"


class WholeBodyTrackingForceInjectedV2(WholeBodyTrackingForceInjected):
    """v14 env. Same physics pipeline as v10 env; different command key + metrics."""

    # Override the command-resolution helper used by the parent's
    # _apply_force_in_physics_step, draw_debug_viz, and _update_log_dict.
    def _get_wrist_command(self, *, require: bool = True):  # type: ignore[override]
        term = self.command_manager.get_state(_TERM_KEY)
        if term is None:
            if require:
                raise RuntimeError(
                    f"WholeBodyTrackingForceInjectedV2 requires the '{_TERM_KEY}' command term."
                )
            return None  # type: ignore[return-value]
        if not isinstance(term, WristForceTrackingCommand):
            raise TypeError(
                f"'{_TERM_KEY}' must be a WristForceTrackingCommand, got {type(term).__name__}."
            )
        return term

    def _env_owned_force_metrics(  # type: ignore[override]
        self,
        wrist_cmd: WristForceTrackingCommand,
        wcfg: "WristForceTrackingConfig",
    ) -> dict[str, torch.Tensor]:
        """v14 only needs the applied-body-force sanity metric.

        No shift, no k_virtual, no wrist_pos_error vs shifted target
        (the reward now tracks the motion reference directly — the
        baseline Episode/rew_motion_tracking_* metrics already cover
        residual tracking quality).
        """
        metrics: dict[str, torch.Tensor] = {}
        device = self.device

        applied_l = self.last_applied_force_w_by_body_id.get(self._left_wrist_isaac_id)
        applied_r = self.last_applied_force_w_by_body_id.get(self._right_wrist_isaac_id)

        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        metrics["force/applied_f_body_l"] = (
            applied_l.norm(dim=-1).float().to(device) if applied_l is not None else zeros
        )
        metrics["force/applied_f_body_r"] = (
            applied_r.norm(dim=-1).float().to(device) if applied_r is not None else zeros
        )

        return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected_v2.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/envs/wbt/wbt_force_injected_v2.py src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected_v2.py
git commit -m "feat(wbt-force-v2): env subclass (override command key + simplify env metrics)"
```

---

## Task 8: Experiment registration + CLI wiring

**Files:**
- Modify: `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
- Modify: `src/holosoma/holosoma/config_values/experiment.py`
- Modify: `src/holosoma/holosoma/config_values/command.py`
- Modify: `src/holosoma/holosoma/config_values/observation.py`
- Modify: `src/holosoma/holosoma/config_values/reward.py`
- Test: `src/holosoma/tests/test_tyro_cli.py` (extend; do not overwrite v10)

- [ ] **Step 1: Write the failing test**

Extend `src/holosoma/tests/test_tyro_cli.py` with (append at the end):

```python
def test_exp_g1_29dof_wbt_force_v2_registered() -> None:
    """Smoke test: new v14 experiment is resolvable via CLI registry."""
    from holosoma.config_values.experiment import EXPERIMENTS

    assert "g1_29dof_wbt_force_v2" in EXPERIMENTS


def test_v2_preset_uses_v14_env_class() -> None:
    from holosoma.config_values.experiment import EXPERIMENTS

    preset = EXPERIMENTS["g1_29dof_wbt_force_v2"]
    assert preset.env_class.endswith("WholeBodyTrackingForceInjectedV2")


def test_v2_preset_reward_is_baseline() -> None:
    from holosoma.config_values.experiment import EXPERIMENTS
    from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

    preset = EXPERIMENTS["g1_29dof_wbt_force_v2"]
    assert set(preset.reward.terms.keys()) == set(g1_29dof_wbt_reward.terms.keys())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/tests/test_tyro_cli.py -v -k v2`
Expected: FAIL (KeyError / AttributeError).

- [ ] **Step 3: Modify the registries**

In `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`, after the existing `g1_29dof_wbt_force = replace(...)` block, append:

```python
g1_29dof_wbt_force_v2 = replace(
    g1_29dof_wbt_force,
    env_class="holosoma.envs.wbt.wbt_force_injected_v2.WholeBodyTrackingForceInjectedV2",
    training=replace(
        g1_29dof_wbt_force.training,
        name="g1_29dof_wbt_force_v2_manager",
    ),
    command=command.g1_29dof_wbt_force_v2_command,
    observation=observation.g1_29dof_wbt_force_v2_observation,
    reward=reward.g1_29dof_wbt_force_v2_reward,
    # Inherit the bigger actor/critic hidden dims from v10 via `replace(g1_29dof_wbt_force, ...)`.
)
```

And extend `__all__`:

```python
__all__ = [
    "g1_29dof_wbt",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_force",
    "g1_29dof_wbt_force_v2",
    "g1_29dof_wbt_w_object",
]
```

In `src/holosoma/holosoma/config_values/command.py`, import and register:

```python
# Add near the existing import:
from holosoma.config_values.wbt.g1.command_force_v2 import g1_29dof_wbt_force_v2_command

# Add to the dict:
    "g1_29dof_wbt_force_v2": g1_29dof_wbt_force_v2_command,
```

In `src/holosoma/holosoma/config_values/observation.py`:

```python
from holosoma.config_values.wbt.g1.observation_force_v2 import g1_29dof_wbt_force_v2_observation
    "g1_29dof_wbt_force_v2": g1_29dof_wbt_force_v2_observation,
```

In `src/holosoma/holosoma/config_values/reward.py`:

```python
from holosoma.config_values.wbt.g1.reward_force_v2 import g1_29dof_wbt_force_v2_reward
    "g1_29dof_wbt_force_v2": g1_29dof_wbt_force_v2_reward,
```

In `src/holosoma/holosoma/config_values/experiment.py`:

```python
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_force,
    g1_29dof_wbt_force_v2,  # NEW
)

EXPERIMENTS = {
    ...,
    "g1_29dof_wbt_force_v2": g1_29dof_wbt_force_v2,
}
```

(Preserve existing entries exactly; only ADD.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s src/holosoma/tests/test_tyro_cli.py -v`
Expected: PASS (existing v10 tests still pass, new v2 tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/experiment.py src/holosoma/holosoma/config_values/experiment.py src/holosoma/holosoma/config_values/command.py src/holosoma/holosoma/config_values/observation.py src/holosoma/holosoma/config_values/reward.py src/holosoma/tests/test_tyro_cli.py
git commit -m "feat(wbt-force-v2): register exp:g1-29dof-wbt-force-v2 across tyro registries"
```

---

## Task 9: Wandb metrics contract test

**Files:**
- Create: `src/holosoma/tests/test_wbt_force_v2_wandb_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `src/holosoma/tests/test_wbt_force_v2_wandb_metrics.py`:

```python
"""Wandb metric-key contract for v14 wrist-force.

The v14 env-owned / command-owned metric keys are the minimal set needed
to audit:

1. Was F_cmd generated? (force/cmd_magnitude_{l,r,max})
2. Was F_ext injected? (force/ext_magnitude_{l,r,max}, force/applied_f_body_{l,r})
3. Are F_cmd and F_ext anti-parallel? (force/cmd_ext_alignment ~ -1)
4. Did the trapezoidal schedule run? (force/phase_*, force/active_frac_cmd)

The policy-learning signal comes from baseline Episode/rew_motion_tracking_*
(unchanged from baseline WBT).
"""

from __future__ import annotations

EXPECTED_COMMAND_METRIC_KEYS = {
    "force/cmd_magnitude_l",
    "force/cmd_magnitude_r",
    "force/cmd_magnitude_max",
    "force/ext_magnitude_l",
    "force/ext_magnitude_r",
    "force/ext_magnitude_max",
    "force/cmd_ext_alignment",
    "force/active_frac_cmd",
    "force/phase_ramp_up",
    "force/phase_hold",
    "force/phase_ramp_down",
}

EXPECTED_ENV_METRIC_KEYS = {
    "force/applied_f_body_l",
    "force/applied_f_body_r",
}

FORBIDDEN_V10_METRIC_KEYS = {
    "force/k_virtual_l",
    "force/k_virtual_r",
    "force/wrist_target_shift_l",
    "force/wrist_target_shift_r",
    "force/wrist_pos_error_l",
    "force/wrist_pos_error_r",
    "force/active_frac_ext",  # v10 had 2 channels; v14 only has cmd
}


def test_command_metrics_match_contract() -> None:
    from types import SimpleNamespace

    import torch

    from holosoma.config_types.command import CommandTermCfg
    from holosoma.config_types.command_v2 import WristForceTrackingConfig
    from holosoma.managers.command.terms.wbt_force_v2 import WristForceTrackingCommand

    device = torch.device("cpu")
    base_quat = torch.zeros(4, 4, device=device)
    base_quat[:, 3] = 1.0
    env = SimpleNamespace(num_envs=4, device=device, dt=0.02, base_quat=base_quat)

    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand",
        params={"wrist_force_tracking_config": WristForceTrackingConfig()},
    )
    term = WristForceTrackingCommand(cfg, env)
    term.reset(None)
    term.step()
    term.update_metrics()

    assert set(term.metrics.keys()) == EXPECTED_COMMAND_METRIC_KEYS

    for forbidden in FORBIDDEN_V10_METRIC_KEYS:
        assert forbidden not in term.metrics, f"v14 must not emit {forbidden}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s src/holosoma/tests/test_wbt_force_v2_wandb_metrics.py -v`
Expected: PASS if Task 2 was done correctly — otherwise this surfaces drift in the metric key set.

- [ ] **Step 3: No new implementation**

This task is a contract test that pins down the metric surface. If the set differs, update Task 2's implementation to match this canonical set.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -s src/holosoma/tests/test_wbt_force_v2_wandb_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/tests/test_wbt_force_v2_wandb_metrics.py
git commit -m "test(wbt-force-v2): pin wandb metric key contract (cmd-ext anti-parallel + applied_f)"
```

---

## Task 10: `demo_wbt_wrist_force_training_v2.sh`

**Files:**
- Create: `demo_scripts/demo_wbt_wrist_force_training_v2.sh`

- [ ] **Step 1: Copy v1 launcher as starting point**

```bash
cp demo_scripts/demo_wbt_wrist_force_training.sh demo_scripts/demo_wbt_wrist_force_training_v2.sh
```

- [ ] **Step 2: Edit the new script**

Replace `exp:g1-29dof-wbt-force` → `exp:g1-29dof-wbt-force-v2`. Replace the Phase 7 checklist with:

```
-----------------------------------------------------------------------
v14 TRAIN GATE — wandb indicators to eyeball during training
  (docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md):

  1. Train/mean_reward monotonically rises in first ~200 iter
  2. Baseline Episode/rew_motion_tracking_* holds / rises
     (this is the ONLY reward — no wrist_force reward term in v14)
  3. Env/force/active_frac_cmd steady-state in [0.2, 0.5]
  4. Env/force/cmd_magnitude_max <= 30 N
  5. Env/force/cmd_ext_alignment ~ -1.0  (anti-parallel sanity)
  6. Env/force/ext_magnitude_l ~ Env/force/cmd_magnitude_l   (|F_ext|=|F_cmd|)
     Env/force/ext_magnitude_r ~ Env/force/cmd_magnitude_r
  7. Env/force/applied_f_body_{l,r} ~ Env/force/ext_magnitude_{l,r}
     (sanity on world->body quat rotation at injection time)
  8. Actor/critic loss not diverging; Episode/rew_survival stable
-----------------------------------------------------------------------
```

Also update the header comments:

```bash
# v14 launcher for WBT wrist-force force-tracking training.
#
# Starts `train_agent.py exp:g1-29dof-wbt-force-v2`. This is the v14
# formulation (docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md):
# reward is the baseline WBT motion-tracking reward; F_cmd is sampled
# on the wrist; F_ext = -F_cmd (world) is injected into the sim; the
# actor observes F_cmd and the critic observes F_cmd + F_ext.
```

Update the override example in the header:

```bash
#     bash demo_scripts/demo_wbt_wrist_force_training_v2.sh \
#         --command.setup_terms.wrist_force_tracking_command.params.wrist_force_tracking_config.force_cmd_magnitude_range "[10.0, 40.0]"
```

- [ ] **Step 3: Run it in SKIP_REINSTALL dry mode on workstation**

This plan is TDD at the Python layer; the shell launcher is validated by a smoke test that will be run on the workstation:

```bash
SKIP_REINSTALL=1 LOGGER=stdout NUM_GPUS=1 bash demo_scripts/demo_wbt_wrist_force_training_v2.sh --training.total_iterations 2 --training.num_envs 16
```

(Only run this step if your workstation IsaacSim env is active. It is not a CI step — the CI equivalent is the tyro CLI smoke test in Task 8.)

- [ ] **Step 4: Commit**

```bash
git add demo_scripts/demo_wbt_wrist_force_training_v2.sh
git commit -m "feat(wbt-force-v2): v2 launcher selecting exp:g1-29dof-wbt-force-v2"
```

---

## Task 11: Push

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/wbt-wrist-force-v10
```

- [ ] **Step 2: Verify CI**

```bash
gh pr list --head feat/wbt-wrist-force-v10 --state open
# If a PR exists, gh pr checks will show current status:
gh pr checks
```

If there's no open PR for this branch yet, you can either:
- keep pushing v14 work into the same branch (the branch's long-term home), or
- open a new PR targeting `main` once the v14 task set is green locally.

No commit here — step is verification only.

---

## Open questions / deferred

1. **F_cmd signal encoding.** v14 gives the actor the raw body-yaw force vector. CHIP-style alternatives (magnitude + unit direction) are left as a follow-up ablation.
2. **Pretraining from v10 policy.** Whether to warm-start v14 from a v10 checkpoint is deferred; first run the fresh v14 training and compare Episode/rew_motion_tracking_* curves against the baseline `exp:g1-29dof-wbt` run.
3. **Magnitude curriculum.** v14 uses the fixed range [5, 30] N from day one. A ramp curriculum (e.g. start at [5, 10] for 1000 iter, then widen) can be added later if early training shows the robot throwing itself over from the full 30 N disturbance.
4. **Eval pipeline.** `eval_agent_force.py` was v10-specific (it reads the compliance command + K). A v14 equivalent (`eval_agent_force_v2.py`) will be a follow-up plan.

---

## Diff summary (expected after all tasks)

```
New files  : 15  (plan, demo, 6 src, 8 tests)
Modified   : 6   (5 config registries + experiment.py v2 block)
Deleted    : 0   (v10 code untouched — v14 ships alongside)
```
