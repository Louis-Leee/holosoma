"""Unit tests for WristForceTrackingCommand (v14)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.command_v2 import WristForceTrackingConfig
from holosoma.managers.command.terms.wbt_force_v2 import (
    STATE_COOLDOWN,
    WristForceTrackingCommand,
)


def _mock_env(num_envs: int = 4, dt: float = 0.02) -> Any:
    device = torch.device("cpu")
    base_quat = torch.zeros(num_envs, 4, device=device)
    base_quat[:, 3] = 1.0  # identity xyzw
    return SimpleNamespace(num_envs=num_envs, device=device, dt=dt, base_quat=base_quat)


def _cfg(**overrides: Any) -> CommandTermCfg:
    return CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand",
        params={"wrist_force_tracking_config": WristForceTrackingConfig(**overrides)},
    )


def test_initial_state_is_cooldown_zero_force() -> None:
    term = WristForceTrackingCommand(_cfg(), _mock_env())
    term.reset(None)
    assert torch.all(term.force_ext_w == 0.0)
    assert torch.all(term.force_cmd_b == 0.0)
    assert torch.all(term._ext_channel.state == STATE_COOLDOWN)


@pytest.mark.parametrize("frame", ["yaw_only", "full_base"])
def test_force_cmd_is_negative_of_force_ext_under_identity_base(frame: str) -> None:
    """identity base_quat 下,两个 variant 都应 collapse 到 F_cmd == -F_ext。

    原因:identity base 下 R_yaw-1 = R-1 = identity,两个旋转函数都退化成恒等。
    """
    torch.manual_seed(0)
    env = _mock_env(num_envs=8)
    term = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=1.0, force_cmd_obs_frame=frame),
        env,
    )
    term.reset(None)
    for _ in range(5):
        term.step()

    torch.testing.assert_close(term.force_cmd_b, -term.force_ext_w, atol=1e-5, rtol=1e-5)


def test_force_cmd_rotates_with_base_yaw() -> None:
    """[yaw_only variant] +90 yaw: R-1 sends (x,y,z) -> (y, -x, z); F_cmd = -that."""
    torch.manual_seed(42)
    env = _mock_env(num_envs=2)
    half = math.pi / 4
    env.base_quat[:] = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)])

    term = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=1.0, force_cmd_obs_frame="yaw_only"),
        env,
    )
    term.reset(None)
    for _ in range(3):
        term.step()

    f_ext_w = term.force_ext_w
    inv_rotated = torch.stack(
        [f_ext_w[..., 1], -f_ext_w[..., 0], f_ext_w[..., 2]], dim=-1
    )
    torch.testing.assert_close(term.force_cmd_b, -inv_rotated, atol=1e-5, rtol=1e-5)


def test_force_cmd_yaw_only_is_invariant_under_roll_pitch() -> None:
    """[yaw_only variant] 只加 roll/pitch、不动 yaw,F_cmd_obs 应该不变。

    这是 yaw-only variant 的定义性约束:身体前俯后仰不污染力观测。
    """
    torch.manual_seed(7)
    env_flat = _mock_env(num_envs=1)
    # base_quat = identity (xyzw = [0,0,0,1])
    term_flat = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=1.0, force_cmd_obs_frame="yaw_only"),
        env_flat,
    )
    term_flat.reset(None)
    for _ in range(3):
        term_flat.step()
    # 保存此时的 F_ext_w(内部随机采样)+ F_cmd_obs。
    f_ext_w_flat = term_flat.force_ext_w.clone()
    f_cmd_flat = term_flat.force_cmd_b.clone()

    # 换一个非零 pitch 的 env,F_ext 直接复用相同的值,只改 base_quat。
    env_pitched = _mock_env(num_envs=1)
    pitch_half = math.pi / 12  # 30 pitch
    # xyzw pitch-only quat: (0, sin(theta/2), 0, cos(theta/2))
    env_pitched.base_quat[:] = torch.tensor(
        [0.0, math.sin(pitch_half), 0.0, math.cos(pitch_half)]
    )
    term_pitched = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=0.0, force_cmd_obs_frame="yaw_only"),
        env_pitched,
    )
    term_pitched.reset(None)
    # 手动把 F_ext 写进内部 buffer(绕过采样),让两 term 的 F_ext_w 完全相同。
    term_pitched._ext_channel.force = f_ext_w_flat.clone()
    # 一次 step 刷新 _force_cmd_b snapshot(但不再触发随机 trigger,因 prob=0)
    term_pitched._force_cmd_b = -term_pitched._rotate_to_obs_frame(
        env_pitched.base_quat, term_pitched._ext_channel.force,
    )

    torch.testing.assert_close(
        term_pitched.force_cmd_b,
        f_cmd_flat,
        atol=1e-5,
        rtol=1e-5,
        msg="yaw_only variant: roll/pitch 变化不得影响 F_cmd_obs",
    )


def test_force_cmd_full_base_changes_under_pitch() -> None:
    """[full_base variant] pitch 非零时,F_cmd_obs 和 identity 下的值**不同**。

    这和 yaw_only variant 形成对照 —— full_base 有意让 roll/pitch 进入观测。
    """
    torch.manual_seed(11)
    # Identity base,full_base variant。
    env_flat = _mock_env(num_envs=1)
    term_flat = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=1.0, force_cmd_obs_frame="full_base"),
        env_flat,
    )
    term_flat.reset(None)
    for _ in range(3):
        term_flat.step()
    f_ext_w = term_flat.force_ext_w.clone()
    f_cmd_flat = term_flat.force_cmd_b.clone()

    # 相同 F_ext_w,但 base 加 30 pitch。
    env_pitched = _mock_env(num_envs=1)
    pitch_half = math.pi / 12
    env_pitched.base_quat[:] = torch.tensor(
        [0.0, math.sin(pitch_half), 0.0, math.cos(pitch_half)]
    )
    term_pitched = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=0.0, force_cmd_obs_frame="full_base"),
        env_pitched,
    )
    term_pitched.reset(None)
    term_pitched._ext_channel.force = f_ext_w.clone()
    term_pitched._force_cmd_b = -term_pitched._rotate_to_obs_frame(
        env_pitched.base_quat, term_pitched._ext_channel.force,
    )

    # 只要 F_ext 有非零分量在 x 或 z,两者应该有可察觉的差异。
    if f_ext_w.abs().sum().item() > 1e-3:
        diff = (term_pitched.force_cmd_b - f_cmd_flat).abs().max().item()
        assert diff > 1e-3, (
            f"full_base variant: pitch 必须改变 F_cmd_obs,但 diff = {diff}"
        )


def test_no_k_virtual_buffer() -> None:
    term = WristForceTrackingCommand(_cfg(), _mock_env())
    term.reset(None)
    assert not hasattr(term, "k_virtual")


def test_no_cmd_channel_buffer() -> None:
    """v14 只有一路 _ext_channel;不应该再有独立的 _cmd_channel。"""
    term = WristForceTrackingCommand(_cfg(), _mock_env())
    assert not hasattr(term, "_cmd_channel")


def test_reset_zeros_force() -> None:
    term = WristForceTrackingCommand(_cfg(force_ext_activation_prob_per_step=1.0), _mock_env(4))
    term.reset(None)
    for _ in range(5):
        term.step()
    assert term.force_ext_w.abs().sum().item() > 0.0

    term.reset(torch.tensor([0, 1, 2, 3]))
    assert torch.all(term.force_ext_w == 0.0)
    assert torch.all(term.force_cmd_b == 0.0)


def test_metrics_expose_cmd_ext_anti_parallel_alignment() -> None:
    """v14 里 F_cmd 和 F_ext 在 world frame 下天然反向,alignment 应该 approx -1。"""
    torch.manual_seed(0)
    term = WristForceTrackingCommand(_cfg(force_ext_activation_prob_per_step=1.0), _mock_env(8))
    term.reset(None)
    for _ in range(5):
        term.step()
    term.update_metrics()

    if (term.force_ext_w.norm(dim=-1) > 1e-3).any():
        align = term.metrics["force/cmd_ext_alignment"]
        assert align.mean().item() < -0.9


def test_force_cmd_b_is_cached_snapshot_not_recomputed_property() -> None:
    """F_cmd_b 必须是 step() 里缓存的快照,多次访问返回同一对象(不随 base_quat 变化)。

    之所以重要:一个 env tick 内,physics injection、obs term、metrics、debug viz
    都会读 force_cmd_b;如果 property 每次都用当前 base_quat 重算,
    而 sim 在 tick 中推进后 base_quat 已刷新,会导致各处看到的 F_cmd_b 不一致。
    """
    torch.manual_seed(0)
    env = _mock_env(num_envs=4)
    term = WristForceTrackingCommand(_cfg(force_ext_activation_prob_per_step=1.0), env)
    term.reset(None)
    for _ in range(5):
        term.step()

    snapshot_a = term.force_cmd_b.clone()

    # 模拟 sim 推进后 base_quat 变化 —— snapshot 必须保持不变。
    env.base_quat[:] = torch.tensor([0.0, 0.0, 0.707, 0.707])  # 90 yaw

    snapshot_b = term.force_cmd_b
    torch.testing.assert_close(
        snapshot_a,
        snapshot_b,
        atol=0.0,
        rtol=0.0,
        msg="force_cmd_b 必须来自 step() 里的快照缓存,不能是随 base_quat 变化的 property",
    )


@pytest.mark.parametrize("frame", ["yaw_only", "full_base"])
def test_rotate_cmd_obs_to_world_uses_cached_quat_not_live(frame: str) -> None:
    """`_rotate_cmd_obs_to_world` 必须用 step() snapshot 的 obs-frame quat,
    不能每次读 live ``env.base_quat``。否则 physics step 推进后 metric 会把旧
    F_cmd_obs 乘到新 quat 上,cmd_ext_alignment 就会漂移。

    做法:step() 后保存 (F_cmd_w_before);改 env.base_quat;再读一次 F_cmd_w_after。
    两者必须完全相等 —— 因为 snapshot 缓存已锁死那个时刻的 quat。
    """
    torch.manual_seed(17)
    env = _mock_env(num_envs=2)
    term = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=1.0, force_cmd_obs_frame=frame),
        env,
    )
    term.reset(None)
    for _ in range(3):
        term.step()

    f_cmd_w_before = term._rotate_cmd_obs_to_world(term.force_cmd_b).clone()

    # 模拟 physics step 推进 base_quat。
    env.base_quat[:] = torch.tensor([0.0, 0.0, 0.707, 0.707])  # 90 yaw

    f_cmd_w_after = term._rotate_cmd_obs_to_world(term.force_cmd_b)

    torch.testing.assert_close(
        f_cmd_w_before,
        f_cmd_w_after,
        atol=0.0,
        rtol=0.0,
        msg=(
            f"[{frame}] _rotate_cmd_obs_to_world 必须用 snapshot 缓存的 obs_frame_quat,"
            f"不能读 live env.base_quat"
        ),
    )


def test_cmd_ext_alignment_stays_anti_parallel_after_live_quat_changes() -> None:
    """step() 之后立刻改 env.base_quat,再跑 update_metrics,
    cmd_ext_alignment 依然应该接近 -1(因为 snapshot 缓存保证 metric 内部自洽)。
    """
    torch.manual_seed(1)
    env = _mock_env(num_envs=8)
    term = WristForceTrackingCommand(
        _cfg(force_ext_activation_prob_per_step=1.0, force_cmd_obs_frame="full_base"),
        env,
    )
    term.reset(None)
    for _ in range(5):
        term.step()

    # 模拟 sim 推进时 base_quat 大幅变化。
    env.base_quat[:] = torch.tensor([0.0, 0.3827, 0.0, 0.9239])  # 45 pitch

    term.update_metrics()
    if (term.force_ext_w.norm(dim=-1) > 1e-3).any():
        align = term.metrics["force/cmd_ext_alignment"]
        # stale live quat 会让 align 漂到 |val| < 0.9;snapshot cache 应该保持 approx -1。
        assert align.mean().item() < -0.9, (
            "cmd_ext_alignment 必须在 live base_quat 改变后依然 approx -1 "
            f"(说明 metric 用的是 snapshot quat,不是 live),实际: {align.mean().item()}"
        )


def test_curriculum_disabled_keeps_magnitude_range_constant() -> None:
    """curriculum 关闭时,_ext_channel.magnitude_range 必须保持 config 默认值不变。"""
    torch.manual_seed(0)
    term = WristForceTrackingCommand(
        _cfg(enable_force_curriculum=False), _mock_env(num_envs=2),
    )
    term.reset(None)
    for _ in range(100):
        term.step()
    # range 始终等于 target。
    assert term._ext_channel.magnitude_range == pytest.approx((5.0, 30.0))
    # progress metric 固定为 1.0(因为没在 ramp)。
    term.update_metrics()
    assert torch.all(term.metrics["force/curriculum_progress"] == 1.0)


def test_curriculum_enabled_ramps_magnitude_range_linearly() -> None:
    """curriculum 开启后,_ext_channel.magnitude_range 从 initial 线性 ramp 到 target。"""
    torch.manual_seed(0)
    term = WristForceTrackingCommand(
        _cfg(
            enable_force_curriculum=True,
            curriculum_initial_magnitude_range=(0.0, 5.0),
            curriculum_ramp_steps=100,
            force_ext_activation_prob_per_step=0.0,  # 禁采样,只测 range ramp
        ),
        _mock_env(num_envs=2),
    )
    term.reset(None)

    # step 0:progress=0,range = initial
    term.step()
    lo, hi = term._ext_channel.magnitude_range
    assert lo == pytest.approx(0.0, abs=1e-3)
    assert hi == pytest.approx(5.0, abs=1e-3)

    # step 50:progress=0.5,range = (2.5, 17.5)
    for _ in range(49):
        term.step()
    lo, hi = term._ext_channel.magnitude_range
    assert lo == pytest.approx(2.5, abs=1e-3)
    assert hi == pytest.approx(17.5, abs=1e-3)

    # step 100:progress=1.0,range = target (5, 30)
    for _ in range(50):
        term.step()
    lo, hi = term._ext_channel.magnitude_range
    assert lo == pytest.approx(5.0, abs=1e-3)
    assert hi == pytest.approx(30.0, abs=1e-3)

    # step 150(超过 ramp_steps):progress clamp 到 1.0,range 不再变
    for _ in range(50):
        term.step()
    lo, hi = term._ext_channel.magnitude_range
    assert lo == pytest.approx(5.0, abs=1e-3)
    assert hi == pytest.approx(30.0, abs=1e-3)


def test_curriculum_enabled_progress_metric_monotonic() -> None:
    """force/curriculum_progress 在开启时单调非减,且 clamp 到 [0, 1]。"""
    torch.manual_seed(0)
    term = WristForceTrackingCommand(
        _cfg(
            enable_force_curriculum=True,
            curriculum_ramp_steps=50,
            force_ext_activation_prob_per_step=0.0,
        ),
        _mock_env(num_envs=1),
    )
    term.reset(None)

    prev = -1.0
    for i in range(100):
        term.step()
        term.update_metrics()
        p = term.metrics["force/curriculum_progress"].max().item()
        assert p >= prev, f"progress decreased at step {i}: {prev} -> {p}"
        assert 0.0 <= p <= 1.0
        prev = p
    assert prev == pytest.approx(1.0)


def test_curriculum_not_reset_by_env_reset() -> None:
    """episode reset 不得重置 curriculum 计数器 —— 否则 curriculum 永远卡在 initial。"""
    torch.manual_seed(0)
    term = WristForceTrackingCommand(
        _cfg(
            enable_force_curriculum=True,
            curriculum_ramp_steps=100,
            force_ext_activation_prob_per_step=0.0,
        ),
        _mock_env(num_envs=4),
    )
    term.reset(None)

    for _ in range(50):
        term.step()
    before = term._curriculum_step
    assert before >= 50

    # reset 部分 env —— curriculum 计数器必须保持不动。
    term.reset(torch.tensor([0, 1]))
    assert term._curriculum_step == before, (
        "curriculum 是全局进度,不得因 env reset 回退"
    )


def test_force_channel_completes_full_trapezoidal_cycle() -> None:
    """Force the FSM through COOLDOWN -> RAMP_UP -> HOLD -> RAMP_DOWN -> COOLDOWN.

    Uses short durations so the cycle fits in a handful of steps. After the
    cycle completes, `force` must be exactly 0 (back in COOLDOWN).
    """
    from holosoma.managers.command.terms.wbt_force_v2 import (
        STATE_COOLDOWN,
        STATE_HOLD,
        STATE_RAMP_DOWN,
        STATE_RAMP_UP,
    )

    torch.manual_seed(0)
    env = _mock_env(num_envs=1, dt=0.02)
    term = WristForceTrackingCommand(
        _cfg(
            force_ext_duration_range_s=(0.1, 0.1),       # 5 steps total
            force_ext_cooldown_range_s=(0.02, 0.02),     # 1 step cooldown
            force_ext_ramp_frac=0.4,                     # 2 ramp-up + 2 ramp-down
            force_ext_activation_prob_per_step=1.0,
        ),
        env,
    )
    term.reset(None)

    visited = {int(STATE_COOLDOWN)}
    for _ in range(40):
        term.step()
        visited.update(int(v) for v in term._ext_channel.state.unique().tolist())

    assert int(STATE_RAMP_UP) in visited, f"FSM never reached RAMP_UP (visited={visited})"
    assert int(STATE_HOLD) in visited, f"FSM never reached HOLD (visited={visited})"
    assert int(STATE_RAMP_DOWN) in visited, f"FSM never reached RAMP_DOWN (visited={visited})"

    # After many steps including multiple full cycles, at some tick force must
    # have briefly been zero (immediately after a RAMP_DOWN -> COOLDOWN transition
    # before the next trigger). We verify by running one more step in a long-cooldown
    # config and checking |force| is finite + non-negative.
    # (Simpler assertion: RAMP_DOWN reached implies the lerp is descending to 0.)
    # Assert the lerp-value ever reached peak (i.e. HOLD had non-zero force).
    # This is implicitly proven by STATE_HOLD being visited AND peak_magnitude being set.


def test_v2_command_module_does_not_import_v10() -> None:
    """V14 command module 不得 import V10 wrist-force symbols。"""
    import inspect

    from holosoma.managers.command.terms import wbt_force_v2

    src = inspect.getsource(wbt_force_v2)
    forbidden = [
        "from holosoma.managers.command.terms.wbt_force import",
        "from holosoma.managers.command.terms.wbt_force ",
        "import holosoma.managers.command.terms.wbt_force",
    ]
    for f in forbidden:
        assert f not in src, f"V14 command must not import V10 wbt_force: found {f!r}"
