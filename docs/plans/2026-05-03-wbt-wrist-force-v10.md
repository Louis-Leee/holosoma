# WBT Wrist Force Training — 实施计划 (v10 — 精简执行版)

> **对 agentic worker 的要求**：按 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 执行；任务用 `- [ ]` 跟踪。本文档**只保留动手写代码需要的信息**，v9.x 的历史推导/rewind 说明一律不在此重复。v9.4 是本 plan 的"为什么这么做"的参考档，v10 是"怎么做"的执行档。

---

## 1. 目标（TL;DR）

在 Holosoma 新增一个 WBT 变体 `exp:g1-29dof-wbt-force`，让 G1 在做 motion tracking 的同时学会**通过 6-D wrist force command 主动发力**。

**核心 idea**：reward 里加虚拟弹簧 `target_shifted = motion_target + (F_ext + F_cmd) / K_virtual`。训练时 K 是 per-env per-wrist 的 buffer（v1 恒 100，v2 可改 range 随机化），actor 和 critic 都能看到 K。

**落地产物**：
- 训练侧：1 个新 experiment `g1_29dof_wbt_force`，actor obs 162/step（stacked 1620），critic 300/step（stacked 3000）
- 部署侧：1 个新 inference config `g1-29dof-wbt-force` + 新 policy 子类 `WholeBodyTrackingForcePolicy` + F_cmd input channel
- 训练监控：wandb 新增 `Env/force/*` 11 个键 + `Episode/rew_wrist_force_position_tracking_exp`（通过复用 Holosoma 既有 `update_metrics()` + `log_dict` 通路，不加新 pipeline）

**执行顺序（两段式 — 先训后部署）**：
1. **先跑训练**：Phase 0 → 1 → 2 → 3 → 4 → 5 → **5.5（训练侧 metadata）** → 6 → **7 TRAIN GATE（跑真实训练 + wandb 8 条收敛指标）**
2. **训练绿后才做部署**：Phase 8（inference metadata 校验 + policy 子类 + F_cmd 输入 + sim-to-sim / 真机 workflow + schema 一致性）

---

## 2. 硬约束（5 条红线，违反即返工）

1. **力只施加在左右 wrist** (`left_wrist_yaw_link`, `right_wrist_yaw_link`)；所有其他 body external force buffer 保持 0
2. **K_virtual 由 `WristComplianceConfig.k_virtual_range` 控制**；v1 range=(100.0, 100.0) 退化 deterministic 100；不做物理 calibration
3. **只追加，不改 Holosoma 现有逻辑**：新 reward 只追加到 preset；新 obs term 只追加；新 experiment 只追加
4. **不 import `third_party/`**（运行时零依赖 third_party）；**允许 copy / 借用** GH / UniFP 代码到 Holosoma 仓库内作为独立文件（见 §3）——license 不用管
5. **允许修改的现有文件仅限追加到 DEFAULTS dict / 末尾追加 instance**（详见 §5.2）

---

## 3. Third-party 代码复用策略（v10 最终）

**v10 决定：放开 copy / 借用——用户原话"完全不用管 license 的问题，想复制就复制，想借用就用，不要禁止"。**

唯一硬限制：**运行时 `import` 绝不能指向 `third_party/`**（否则 Holosoma 对外分发会丢失依赖）。但从 `third_party/` 把代码 **copy 进 Holosoma 仓库内**作为独立文件，爱怎么用怎么用——文件头可加一行 attribution 注释（比如 "Adapted from third_party/gentle-humanoid-training/...") 方便后续回溯来源，license note 不必须。

### 推荐 copy 清单（实现 Task 2 时直接用）

| 源路径 | Holosoma 目标路径 | 用途 |
|---|---|---|
| `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/utils.py`（整个文件，含 `TemporalLerp` L5-133 + `random_uniform` L135-162） | `src/holosoma/holosoma/managers/command/terms/_gh_utils.py` | ramp state + 均匀采样 helper |
| `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/admittance.py` 的 `clamp_norm` (L7-11) + `_norm` (L4-5) | 合并进 `_gh_utils.py` | force magnitude clamp |

**依赖核查**（codex 确认）：`TemporalLerp` / `random_uniform` / `clamp_norm` 只依赖 `torch` + `typing.Optional`，没有 GH 其他 module 的隐式 import——copy 一份就能独立工作。`AdmittanceMassChain` 我们不用，可以不 copy。

### copy 时的工程建议（不是强制）

- 文件头加一行：`"""Adapted from gentle-humanoid-training (third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/utils.py, commit <sha>)."""`——方便 debug 时找到原实现对齐行为
- 删 copy 进来代码里用不到的部分（减小维护面积）
- 不改函数行为；加 type hints / docstring / rename 私有变量都可以

### 禁令（保留）

- ❌ `from third_party... import ...`（运行时硬依赖 third_party 目录）
- ❌ `sys.path.append('third_party/...')`
- ❌ 任何 `pyproject.toml` 依赖指向 `third_party/`

### UniFP / CHIP

- UniFP：reward 公式 `target_shifted = g + F/K` 已 embed 在 §4，其 IsaacGym quadruped 代码不适用 humanoid 场景；copy 价值低，不强制
- CHIP：没开源，无代码可 copy；设计思想（10-step history + K 进 actor obs）已内化到本 plan

---

## 4. Reward 公式（v10）

```python
# per-env per-wrist (body-yaw frame → world)
force_cmd_w_left  = quat_apply(yaw_quat(base_quat), force_cmd_b[:, 0])
force_cmd_w_right = quat_apply(yaw_quat(base_quat), force_cmd_b[:, 1])
force_cmd_w = torch.stack([force_cmd_w_left, force_cmd_w_right], dim=1)  # [N, 2, 3]
F_total_w = force_ext_w + force_cmd_w                                     # [N, 2, 3]

# per-wrist K
k_virtual = wrist_compliance_command.k_virtual                            # [N, 2]
wrist_target_shifted_w = motion_target_w + F_total_w / k_virtual.unsqueeze(-1).clamp(min=1e-3)

error = torch.sum(torch.square(wrist_target_shifted_w - wrist_actual_w), dim=-1)
reward = torch.exp(-error.mean(-1) / (sigma ** 2))
```

**F_ext / F_cmd 采样**：梯形剖面（ramp_up 0.25 × duration → hold 0.5 × duration → ramp_down 0.25 × duration → cooldown）；方向单位球面均匀采样。具体参数见 §6 Task 1 的 `WristComplianceConfig`。

**frame 转换**（必须）：
```python
from isaaclab.utils.math import quat_apply_inverse  # IsaacLab body_quat_w 是 wxyz
```
F_ext 以 world frame 进 `WristComplianceCommand`，env 子类 override `_apply_force_in_physics_step` 做 world→body 转换后传给 `simulator._robot.set_external_force_and_torque(..., body_ids=wrist_only)`。F_cmd 存 body-yaw frame（reward 里用 yaw_quat 转 world）。

---

## 5. 文件清单

### 5.1 新建

| 路径 | 作用 |
|---|---|
| `src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py` | `K_VIRTUAL_RANGE_N_PER_M: tuple[float, float] = (100.0, 100.0)` + alias `G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M = 100.0` |
| `src/holosoma/holosoma/managers/command/terms/_gh_utils.py` | copy 自 GH（见 §3 清单）：`TemporalLerp`、`random_uniform`、`clamp_norm`、`_norm`。文件头加一行 "Adapted from ..." attribution 注释 |
| `src/holosoma/holosoma/managers/command/terms/wbt_force.py` | `WristComplianceCommand(CommandTermBase)`：F_cmd + F_ext ramp state machine（用 `_gh_utils.TemporalLerp` 管 ramp 插值）；暴露 `force_cmd_b[N,2,3]` / `force_ext_w[N,2,3]` / `k_virtual[N,2]` + `update_metrics()`（填 Task 9.5 command-owned wandb 键） |
| `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py` | `WholeBodyTrackingForceInjected(WholeBodyTrackingManager)` override `_apply_force_in_physics_step`，world→body 转换后写 sim；暴露 `last_applied_force_w_by_body_id: dict[int, Tensor]` debug accessor；override `draw_debug_viz()`（箭头）和 `_update_log_dict()`（填 Task 9.5 env-owned wandb 键） |
| `src/holosoma/holosoma/managers/observation/terms/wbt_force.py` | `wrist_force_command`（actor+critic）、`wrist_force_ext_privileged`（critic-only）、`wrist_virtual_stiffness_command`（actor+critic，scale=0.01） |
| `src/holosoma/holosoma/managers/reward/terms/wbt_force.py` | `wrist_force_position_tracking_exp(env, sigma, left_wrist_body_name, right_wrist_body_name)` —— K 从 command buffer 读 |
| `src/holosoma/holosoma/config_values/wbt/g1/command_force.py` | preset `g1_29dof_wbt_force_command` |
| `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py` | preset `g1_29dof_wbt_force_observation`：两个 group `history_length=10`；actor 追加 `wrist_force_command`(6) + `wrist_virtual_stiffness_command`(2, scale=0.01)；critic 追加同上 + `wrist_force_ext_privileged`(6) |
| `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py` | preset `g1_29dof_wbt_force_reward`：继承原 9 个 reward term + 追加 `wrist_force_position_tracking_exp`（weight=2.0, σ=0.3） |
| `src/holosoma_inference/holosoma_inference/policies/wbt_force.py` | `WholeBodyTrackingForcePolicy(WholeBodyTrackingPolicy)` |
| `src/holosoma_inference/docs/workflows/sim-to-sim-wbt-force.md` + `real-robot-wbt-force.md` | 部署 workflow |

### 5.2 修改现有文件（仅追加，不改已有行）

| 路径 | 如何改 |
|---|---|
| `src/holosoma/holosoma/config_types/command.py` | 末尾追加 `WristComplianceConfig` frozen dataclass（字段见 Task 1） |
| `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` | 末尾追加 `g1_29dof_wbt_force` experiment，`env_class="holosoma.envs.wbt.wbt_force_injected:WholeBodyTrackingForceInjected"` |
| `src/holosoma/holosoma/config_values/{command,observation,reward,experiment}.py` | DEFAULTS 追加 `"g1_29dof_wbt_force"` |
| `src/holosoma/holosoma/utils/inference_helpers.py` | `ppo.py:731-742` + `fast_sac_agent.py:986-997` 的 metadata 构造 dict 追加 `history_length` / `obs_term_names_sorted` / `obs_group_dims` 三个 key |
| `src/holosoma_inference/holosoma_inference/config/config_values/observation.py` | 末尾追加 `wbt_force` ObservationConfig（见 Task 15） |
| `src/holosoma_inference/holosoma_inference/config/config_values/inference.py` | 末尾追加 `g1_29dof_wbt_force` InferenceConfig；DEFAULTS 注册 |
| `src/holosoma_inference/holosoma_inference/config/config_types/task.py` | 扩 `InputSource` Literal + `TaskConfig` 追加 `wrist_force_input`, `wrist_force_magnitude_cap: float = 30.0`, `ros_wrist_force_topic` |
| `src/holosoma_inference/holosoma_inference/inputs/api/commands.py` | 追加 `WristForceCmd` frozen dataclass |
| `src/holosoma_inference/holosoma_inference/inputs/api/base.py` | 追加 `WristForceCmdProvider` Protocol |
| `src/holosoma_inference/holosoma_inference/inputs/__init__.py` | `create_input` factory 扩 `"wrist_force"` role dispatch |
| `src/holosoma_inference/holosoma_inference/inputs/impl/keyboard.py` | 追加 F_cmd 键位（左 wrist `u/j/h/k/y/n`，右 wrist numpad `8/2/4/6/9/3`，`/`=归零，`,`/`.`=magnitude 步长；magnitude 受 cap clamp；避开现有 `o/i/]/m/=` 键位） |
| `src/holosoma_inference/holosoma_inference/inputs/impl/ros2.py` | 追加 subscriber，topic 名从 `task.ros_wrist_force_topic` 读 |
| `src/holosoma_inference/holosoma_inference/policies/base.py` | `run()` loop 里 `policy_action()` 之前插入 `self._poll_extra_inputs()`；基类默认 pass（避免子类 override 整个 run） |
| `src/holosoma_inference/holosoma_inference/policies/dual_mode.py` | `_select_policy_class`：若 `"wrist_force_command" in actor_obs` → entry point 优先，兜底 `WholeBodyTrackingForcePolicy` |
| `src/holosoma_inference/holosoma_inference/README.md` | "Whole-Body Tracking" controls 表下追加 F_cmd 键位 + ROS2 topic 小节 |
| `src/holosoma_inference/setup.py` | `holosoma.config.inference` 追加 `g1-29dof-wbt-force`；新 group `holosoma.policies.wbt` 注册 `g1-29dof-force = ...:WholeBodyTrackingForcePolicy` |

### 5.3 必须保持原样

- `envs/wbt/wbt_manager.py`（基类）
- `managers/command/terms/wbt.py`、`managers/observation/terms/wbt.py`、`managers/reward/terms/wbt.py`
- `managers/observation/manager.py`（history 已内置）
- `agents/callbacks/push.py`、`simulator/shared/virtual_gantry.py`（只参考不改）
- 所有现有 `g1_29dof_wbt*` preset

---

## 6. 任务清单（25 task + 1 spike = 26 steps）

**总体执行顺序（v10 两段式）**：
1. **先跑训练**（Phase 0 → 6）：Task 0-13
   - Phase 0 Spike + 常量
   - Phase 1 command 基础设施（Task 1-3）
   - Phase 2 observation（Task 4）
   - Phase 3 reward（Task 5）
   - Phase 4 presets & experiment（Task 6-9）+ wandb 监控（Task 9.5）
   - Phase 5 集成测试 & 训练文档（Task 10-12）
   - **Phase 5.5 训练侧 ONNX schema 扩字段（Task 13.5）**← 为将来部署预埋
   - Phase 6 全局 sanity（Task 13）
2. **TRAIN GATE**（Phase 7）：Task 13.8 — 跑真实训练到收敛，wandb 看 8 条指标
3. **部署**（Phase 8）：Task 14-21 — inference metadata 校验 + policy 子类 + F_cmd 输入链 + sim-to-sim / 真机 workflow

**硬门**：Phase 7 Task 13.8 不绿 → 不进 Phase 8。ONNX 是训练产物，训练不稳导致 obs schema/dim 抖动 → 部署测试全部白跑。



### Phase 0：spike + 常量

- [ ] **Spike 1：IsaacSim wrist-only 外力注入 API 验证**（`tests/spikes/test_wrist_only_force_injection.py`, `@pytest.mark.isaacsim`）
  - 启 baseline `g1_29dof_wbt` env，调 `_robot.set_external_force_and_torque(forces=[[[20,0,0],[0,0,0]]], torques=0, body_ids=[left_id, right_id])`
  - 跑 1 physics step，断言：
    - `simulator._robot._external_force_b`（spike-only 读私有 buffer 打开黑盒）：除两个 wrist 外其余 body 的 buffer 全 0
    - 传入 `body_ids` 长度 = forces 第二维度（避免 broadcast 错 body）
    - 左 wrist world x 速度有变化；反向测试传 `[pelvis_id]` → wrist buffer 保持 0
    - reset 后 external force buffer 全部清零
  - **Gate**：Spike 绿 → 继续 Task 0

- [ ] **Task 0：`_k_virtual.py` 常量**
  - 写 `K_VIRTUAL_RANGE_N_PER_M: tuple[float, float] = (100.0, 100.0)` + alias
  - 测试：range 合法性、alias 等于 range[0]

### Phase 1：command 基础设施

- [ ] **Task 1：`WristComplianceConfig` dataclass**（`config_types/command.py` 末尾）
  - `@dataclass(frozen=True)`
  - 字段：
    - F_cmd: `force_cmd_magnitude_range=(5,30)`, `force_cmd_duration_range_s=(1.0,3.0)`, `force_cmd_cooldown_range_s=(0.5,2.0)`, `force_cmd_ramp_frac=0.25`, `force_cmd_activation_prob_per_step=0.01`
    - F_ext: 同上字段名 `force_ext_*`，`force_ext_magnitude_range=(0,30)`
    - `k_virtual_range: tuple[float, float] = K_VIRTUAL_RANGE_N_PER_M`
    - **v10 新增**：`debug_arrow_scale_n_per_m: float = 50.0`（debug_draw 箭头视觉缩放：30N 力 → 0.6m 箭头长度，约 wrist reach 边缘）+ `debug_draw_total_arrow: bool = True`（是否画 F_total=F_ext+F_cmd 紫色合力箭头）
    - 共用: `enable_left=True`, `enable_right=True`, `left_wrist_body_name="left_wrist_yaw_link"`, `right_wrist_body_name="right_wrist_yaw_link"`
  - `__post_init__`：assert `k_virtual_range[0] > 0`、`range[0] <= range[1]`、`magnitude_range` 下限 ≤ 上限、`ramp_frac ∈ [0, 0.5]`、**`debug_arrow_scale_n_per_m > 0`**
  - 测试：默认构造、frozen、`(0,100)`/`(200,50)` raise、`(50,300)` 成功、`debug_arrow_scale_n_per_m=0` raise

- [ ] **Task 2：copy `_gh_utils.py` + 写 `WristComplianceCommand`**
  - (a) copy `TemporalLerp` / `random_uniform` / `clamp_norm` / `_norm` 到 `managers/command/terms/_gh_utils.py`（见 §3 清单），文件头加 "Adapted from gentle-humanoid-training" 注释；删 `AdmittanceMassChain`（不用）
  - (b) 写 `WristComplianceCommand(CommandTermBase)` in `managers/command/terms/wbt_force.py`：
    - per-env per-wrist state ∈ {COOLDOWN, RAMP_UP, HOLD, RAMP_DOWN}，int counter 推进
    - F_cmd 和 F_ext 两路独立 buffer（用相同 state machine 逻辑，各自的 config 参数）
    - 用 `_gh_utils.TemporalLerp` 管 ramp 插值（省掉自己写 ramp 系数公式的空间）
    - 方向采样：`torch.randn(n, 3); d /= d.norm(dim=-1, keepdim=True).clamp(min=1e-6)`
    - 幅度 clamp：`_gh_utils.clamp_norm(force, max=magnitude)`
    - 均匀采样：`_gh_utils.random_uniform(shape, lo, hi, device)`
    - **duration-to-steps 换算（统一规则）**：`steps = max(1, int(round(duration_s * control_rate_hz)))`；**control rate 从 `1.0 / env.dt` 算**（`base_task.py:112` 定义 `self.dt = control_decimation * sim_dt`，即 control step 秒数；倒数即 control rate Hz）。`duration_s=0.0` 允许并映射到 `steps=1`（最短可能，避免除零）。统一用 `round` 不用 `floor/ceil`（消除单向偏置）
    - **ramp steps `>=1` 硬保证**：`ramp_up_steps = max(1, round(duration_s * ramp_frac * control_rate))`、`ramp_down_steps` 同。即便 `ramp_frac=0`，ramp_up/down_steps 也至少是 1（避免 `TemporalLerp` 内部 `total_steps=0` 除零）
    - **Counter init**（`reset(env_ids)`）：`state = COOLDOWN`、`cooldown_remaining = 0`（允许 env 起步立刻触发 Bernoulli）；`TemporalLerp.reset(env_ids)` 同时清
    - `reset(env_ids)`: 清 state + 从 `cfg.k_virtual_range` per-env per-wrist 独立用 `_gh_utils.random_uniform` 采样 K
    - `step()`: 推进状态机、触发新 episode（`state == COOLDOWN` + `cooldown_remaining==0` + Bernoulli(`activation_prob_per_step`) 为真时采 direction + magnitude + duration + `ramp_up_steps` / `ramp_down_steps` / `hold_steps`，调 `TemporalLerp.set(env_ids, end=peak_magnitude, total_steps=ramp_up_steps)` 等）、**不重采 K**（K 在整个 episode 恒定，对标 GH `motion_tracking.py:781-782`）
  - 暴露 buffer：`force_cmd_b[N,2,3]`、`force_ext_w[N,2,3]`、`k_virtual[N,2]`
  - **wandb 钩子**：对齐 Holosoma `motion_command` 约定（`managers/command/terms/wbt.py:445, 1000-1034`）：
    - `self.metrics: dict[str, torch.Tensor] = {}` 在 `__init__` 初始化
    - `def update_metrics(self)`：每 control step 在 env `_update_log_dict` 里被调，填入 §6.Task 22 的 "command-owned" 小节键（F_cmd / F_ext / K / activation / state buffer）；返回每张量 shape `[N]` 或 `[N, 2]`（两个 wrist），标量直接进 `log_dict` 由 `TensorAverageMeterDict` 自动取均值
  - 测试（pure CPU）：
    - reset 后 force=0、K ∈ range
    - F_cmd / F_ext 独立激活
    - 梯形剖面形状 + 峰值等于采样值
    - `force_ext_w` norm 在 `force_ext_magnitude_range` 内
    - `enable_left=False` 时左 wrist 永远 0
    - 时序：连调两次 `.force_ext_w`（不 `.step()`）同张量
    - v1 K 恒 100；mock range=(50,300) → 左右分布 ∈ [50,300] 且允许不同
    - episode 中途 `k_virtual` 不变、reset 后重新采样
    - `update_metrics()` 填满 §6.Task 22 command-owned 键（dtype float, device=env.device）；键缺失抛 KeyError

- [ ] **Task 3：`WholeBodyTrackingForceInjected` env 子类 + debug-draw arrows**（`envs/wbt/wbt_force_injected.py`, `@pytest.mark.isaacsim`）
  - 继承 `WholeBodyTrackingManager`
  - `__init__`: 从 `command_manager.get_term_cfg("wrist_compliance_command").params` 读 cfg；用 `simulator.find_rigid_body_indice(name)` → `simulator.body_ids[idx]` 解 wrist isaac IDs；`self._wrist_body_ids_t = torch.tensor([left_id, right_id], device)`；初始化 `self.last_applied_force_w_by_body_id: dict[int, Tensor] = {}`
  - override `_apply_force_in_physics_step`:
    - `super()._apply_force_in_physics_step()`（保留原 action apply）
    - `term = command_manager.get_state("wrist_compliance_command")`; `force_w = term.force_ext_w`
    - 用 `body_quat_w[:, wrist_id]` + `quat_apply_inverse` (wxyz) → body frame
    - `simulator._robot.set_external_force_and_torque(forces=forces_body, torques=0, env_ids=None, body_ids=self._wrist_body_ids_t)`
    - 更新 debug snapshot `last_applied_force_w_by_body_id`（detach + clone）
  - **override `draw_debug_viz()` 画 F_cmd / F_ext 箭头**（仅 `debug_viz_enabled=True` 时画；无头不画）：
    - **导入 API**（和 `virtual_gantry.py:280` 同来源）：`from holosoma.utils.draw import draw_line, draw_sphere`。**不**直接 import `isaacsim_draw_adapter`——`holosoma.utils.draw` 是面向所有 simulator 的统一入口
    - **API 行为注意**（从 `utils/adapters/isaacsim_draw_adapter.py` 读源确认）：
      - `draw_line(sim, start, end, color, env_id)`：`env_id` **实际被 adapter 忽略**（IsaacSim 全局 draw），传 0 仅作接口约定；`color` 是 RGB tuple，adapter 自动 append alpha=1.0；start/end 要求 cpu list/tuple，传 torch tensor 需先 `.cpu()`
      - `draw_sphere(sim, pos, radius, color, env_id, pos_id=None)`：`radius`/`pos_id` 也**被 adapter 忽略**（固定 size=20）——不要指望 sphere 大小可调
    - **Call site / 频率（关键）**：`draw_debug_viz` **只能从 `BaseTask`/`BaseSimulator` 的 per-control-step render 路径被调用**（IsaacSim 里是 `render()` → `draw_debug_viz()`，见 `isaacsim.py:906` 类处，对应 `debug_viz_enabled=True`）。**绝不能**在 `_apply_force_in_physics_step` (physics substep loop，每 control step 跑 4 次) 里调——会 overdraw 4x + 可能被 sim 内部 clear 清掉
    - **实现建议**：直接 override `WholeBodyTrackingForceInjected` 的 `draw_debug_viz()`（父类有 `base_simulator.py:415` 默认 no-op），在这里做所有绘制；不要在 physics callback 里画
    - 对 `env_id=0` 的左右两 wrist：
      - 读取 wrist world position：`wrist_pos_w = simulator._robot.data.body_pos_w[0, wrist_isaac_id].cpu()`
      - 读取 F_ext world：`F_ext_w = term.force_ext_w[0, wrist_idx]`（已 world frame）
      - 读取 F_cmd world：`F_cmd_w = yaw_quat(base_quat[0]) · term.force_cmd_b[0, wrist_idx]`（body-yaw → world）
      - **scale**：箭头可视长度 = `||F|| / arrow_scale_N_per_m`，默认 `50.0 N/m` → 30N 画 0.6m（约 wrist reach 边缘，容易看见）；可通过 `WristComplianceConfig.debug_arrow_scale_n_per_m` 覆盖（Task 1 新增字段）
      - 画 F_ext 箭头：**红色** `(1.0, 0.0, 0.0)`，`draw_line(sim, wrist_pos_w, (wrist_pos_w + F_ext_w / scale).cpu(), red, env_id=0)` + 端点 `draw_sphere(sim, end, 0.015, red, env_id=0)`（radius 参数 adapter 忽略但接口保留）
      - 画 F_cmd 箭头：**蓝色** `(0.0, 0.0, 1.0)`，同上逻辑
      - 画 `F_total = F_ext + F_cmd` 箭头：**紫色** `(0.8, 0.0, 0.8)`，方便看合成效果（由 `WristComplianceConfig.debug_draw_total_arrow` 控制，默认 True）
    - **norm 阈值**：若 `F.norm() < 1e-3 N` 跳过画该箭头（避免零长度箭头视觉噪声）
    - **颜色图例**：在 README/workflow 文档里加一行"红=F_ext(sim injected), 蓝=F_cmd(actor receives), 紫=F_total"
  - **override `_update_log_dict()`**（对齐 `wbt_manager.py:71-80` 既有 pattern，**不改父类**）：
    - `super()._update_log_dict()`（保留父类 motion_command metrics）
    - `cmd = self.command_manager.get_state("wrist_compliance_command")`
    - `cmd.update_metrics()`
    - `self.log_dict.update(cmd.metrics)` → 下游 `base_task.py:469` 自动把 `log_dict` 塞进 `extras["to_log"]` → `LoggingHelper.episode_env_tensors.add` → wandb `Env/*` namespace
    - 额外塞 "env-owned" 小节（reward residual / wrist tracking error / applied F 读 sim buffer）——具体键列表见 §6.Task 22
  - 测试：
    - Spike 1 断言 + reset 清零 + debug accessor 正确
    - debug draw smoke（mock `simulator.draw`）：调用 `draw_debug_viz()` 不 raise；传一个 fake `force_ext_w = [[10,0,0],[0,10,0]]` 验证 `draw_line` 被调用 >= 2 次（F_ext 双 wrist）+ F_cmd 双 wrist + 可选 F_total → 至少 4 次（cfg `debug_draw_total_arrow=False`）或 6 次（cfg True）
    - `debug_viz_enabled=False` 时由 `base_simulator.py:906` (IsaacSim) / `isaacgym.py:906` 既有 gate 保证 `draw_debug_viz` 不被调
    - headless 时 `simulator.draw is None`（见 `isaacsim.py:488`），override 内先 `if not getattr(simulator, "draw", None): return` 早返回
    - `F.norm() < 1e-3` 时 `draw_line` 对这条 wrist 不被调（避免零箭头）
    - 回归：**不**在 `_apply_force_in_physics_step` 内画（防 overdraw）——单测 mock physics_step 调用 4 次但 `draw_line` 在这 4 次内**零**调用

### Phase 2：observation

- [ ] **Task 4：三个 obs term**（`managers/observation/terms/wbt_force.py`）
  - `wrist_force_command(env) -> [N, 6]` = `force_cmd_b.reshape(N, 6)`
  - `wrist_force_ext_privileged(env) -> [N, 6]` = `force_ext_w.reshape(N, 6)`
  - `wrist_virtual_stiffness_command(env) -> [N, 2]` = `k_virtual.clone()`
  - 测试：fake command manager → 三 term shape/value 正确；clone 语义不被下游污染

### Phase 3：reward

- [ ] **Task 5：`wrist_force_position_tracking_exp` reward term**（`managers/reward/terms/wbt_force.py`）
  - 签名：`(env, sigma, left_wrist_body_name, right_wrist_body_name) -> [N]`（**不**带 K 参数）
  - 按 §4 公式实现，K 从 `wrist_compliance_command.k_virtual` 读
  - 测试（pure CPU）：
    - 4 个稳态 case（F_cmd=0/F_ext=0；F_cmd=20N/F_ext=0；F_cmd=0/F_ext=20N；F_cmd=20N/F_ext=-20N）
    - yaw rotation 验证
    - per-wrist K：左 K=50 + 右 K=200 + 同 F_cmd=20N → 左 Δx=0.4m、右 Δx=0.1m
    - K → 0 → `clamp(min=1e-3)` 不 NaN（防御性，主防线是 Task 1 `__post_init__` assert）

### Phase 4：presets & experiment

- [ ] **Task 6：command preset `g1_29dof_wbt_force_command`**
  - `config_values/wbt/g1/command_force.py`: 继承 baseline `motion_command` + 追加 `wrist_compliance_command`
  - `config_values/command.py` DEFAULTS 追加
  - 测试：baseline `g1_29dof_wbt_command` identity 不变

- [ ] **Task 7：observation preset `g1_29dof_wbt_force_observation`**
  - `config_values/wbt/g1/observation_force.py`: 新建 ObservationManagerCfg + 两个 ObsGroupCfg（不改 baseline object）
  - actor_obs (`history_length=10`): baseline 6 term + `wrist_force_command` (scale=1.0) + `wrist_virtual_stiffness_command` (scale=0.01)
  - critic_obs (`history_length=10`): actor 所有 + critic privileged 4 term + `wrist_force_ext_privileged` (scale=1.0)
  - DEFAULTS 注册
  - 测试：
    - baseline `g1_29dof_wbt_observation` identity 不变
    - 两个 group `history_length==10`
    - term 集合正确（alphabetical sort 后可推出排序）
    - `wrist_virtual_stiffness_command.scale == 0.01`（两个 group 都是）
    - 维度：actor 162 single / 1620 stacked；critic 300 single / 3000 stacked

- [ ] **Task 8：reward preset `g1_29dof_wbt_force_reward`**
  - 继承原 9 个 reward term + 追加 `wrist_force_position_tracking_exp` (weight=2.0, σ=0.3，**不传 K**)
  - DEFAULTS 注册
  - 测试：baseline `g1_29dof_wbt_reward` identity 不变；新 term params dict 不含 `K_virtual` key

- [ ] **Task 9：experiment `g1_29dof_wbt_force`**
  - `config_values/wbt/g1/experiment.py` 末尾追加
  - `env_class="holosoma.envs.wbt.wbt_force_injected:WholeBodyTrackingForceInjected"`
  - command/obs/reward 用新 preset；其他字段继承 baseline
  - **algo 沿用 baseline**：`actor.input_dim=["actor_obs"]`、`critic.input_dim=["critic_obs"]`
  - **MLP 建议**：首次跑 `[1024, 512, 256]`（input 1620 比 baseline 154 大 10x）；若 overfit 再回到 `[512, 256, 128]`
  - DEFAULTS 追加
  - 测试：baseline `g1_29dof_wbt.algo is ORIGINAL_ALGO`；新 preset input_dim 和 baseline 一样

- [ ] **Task 9.5：wandb 训练监控指标（force-centric）**  _(前置：Task 2 + Task 3 + Task 5；不加新 pipeline，只填键)_
  - **Pipeline 对齐（不新增机制）**：`WristComplianceCommand.update_metrics()` → `WholeBodyTrackingForceInjected._update_log_dict()` → `log_dict` → `extras["to_log"]` → `LoggingHelper.episode_env_tensors.add` → wandb `Env/*`（`agents/modules/logging_utils.py:148-149, 188-189, 339-342`）。另外 `managers/reward/manager.py:200-250` 已自动把每 reward term 汇进 `Episode/rew_*`，故 `Episode/rew_wrist_force_position_tracking_exp` 是**免费**的，不需要新代码，只需在测试里断言该 key 存在
  - **Command-owned 键**（在 `WristComplianceCommand.update_metrics()` 里填，dtype=float, device=env.device）：
    - `force/cmd_magnitude_l` / `force/cmd_magnitude_r` — `force_cmd_b.norm(dim=-1)` 左右分别 → `[N]`，空间均值由 `TensorAverageMeterDict` 自动算
    - `force/cmd_magnitude_max` — `force_cmd_b.norm(dim=-1).max(dim=-1).values` → `[N]`
    - `force/ext_magnitude_l` / `force/ext_magnitude_r` / `force/ext_magnitude_max` — 同上但 `force_ext_w`
    - `force/cmd_ext_alignment` — `cos(F_cmd_w, F_ext_w)` 双 wrist 平均（同向 vs 对抗）→ `[N]`，只在 `||F_cmd||·||F_ext|| > 1e-6` 的 env 上算，否则置 0
    - `force/k_virtual_l` / `force/k_virtual_r` — `k_virtual[:, 0/1]` → `[N]`（v1 全 100 变平线；range 打开后看分布漂移）
    - `force/active_frac_cmd` — `(state != COOLDOWN).float().mean(dim=-1)` → `[N]`（双 wrist 平均的激活占比）
    - `force/active_frac_ext` — 同上但 ext 通道
    - `force/phase_ramp_up` / `force/phase_hold` / `force/phase_ramp_down` — F_cmd state ∈ {RAMP_UP, HOLD, RAMP_DOWN} 各自 `float().mean(dim=-1)` → `[N]`
  - **Env-owned 键**（在 `WholeBodyTrackingForceInjected._update_log_dict` 里填，因为要读 sim buffer + reward 分解）：
    - `force/wrist_target_shift_l` / `force/wrist_target_shift_r` — `||F_total_w / k_virtual||` 左右分别（单位 m，反映"motion_target 被虚拟弹簧拉走多少"）→ `[N]`
    - `force/wrist_pos_error_l` / `force/wrist_pos_error_r` — `||wrist_actual_w - wrist_target_shifted_w||`（reward 对应的位置残差，单位 m）→ `[N]`
    - `force/applied_f_body_l` / `force/applied_f_body_r` — `||last_applied_force_by_body_id[wrist_isaac_id]||`（sanity：sim 真正注入的力大小，应等于 `F_ext` world→body 转完后的 norm）→ `[N]`
  - **v1 值域 sanity**（训练时 wandb 上肉眼可查）：
    - `force/cmd_magnitude_*` ∈ [0, 30]（受 `force_cmd_magnitude_range` 上限 clamp）
    - `force/ext_magnitude_*` ∈ [0, 30]
    - `force/k_virtual_*` ≡ 100.0（v1 range=(100,100)；平线是"配置对了"的信号）
    - `force/active_frac_*` 稳态 ≈ `mean_duration / (mean_duration + mean_cooldown + mean_wait_for_trigger)`；默认 config (activation_prob=0.01 + dt=0.02s + duration~2s + cooldown~1s) 估算稳态 ≈ 0.25-0.45；**< 0.05 或 > 0.9 → config 有问题**
    - `force/cmd_ext_alignment` 随机采样 steady-state ≈ 0（偏正 → 两路 force 有相关；偏负 → 有对抗）
    - `Episode/rew_wrist_force_position_tracking_exp` 训练初期应稳定上升到 0.4-0.7 左右
  - **wandb 面板建议**（抄到 Task 12 文档）：
    - "Force Magnitude"（6 线：cmd/ext × l/r/max）
    - "Force Activity"（`active_frac_cmd/ext` + 3 条 phase 占比）
    - "Virtual Spring Effect"（`wrist_target_shift_l/r` + `wrist_pos_error_l/r`）
    - "K virtual"（`k_virtual_l/r`——v1 平线；v2 打开 range 后看分布）
    - "Reward" 面板把 `Episode/rew_wrist_force_position_tracking_exp` 单拉一条
  - **测试**（`tests/test_wbt_force_wandb_metrics.py`，pure CPU，mock command + env）：
    - 断言 `command.metrics` 含所有 "Command-owned" 键 + `log_dict` 含所有 "Env-owned" 键；每个 value 是 `torch.Tensor`, shape `[N]`, dtype float
    - 值域 assert：magnitude ≥ 0；active_frac ∈ [0, 1]；alignment ∈ [-1, 1]；k == 100.0（v1）
    - 回归：baseline `WholeBodyTrackingManager._update_log_dict()` 的 `log_dict` 不含任何 `force/*` 键（红线 3：不影响 baseline）
    - Task 11 e2e 上 assert `infos["to_log"]` 的 keys 是上述 union 的超集；mock wandb 断言 `wandb.log` 调用的 dict 含 `Env/force/cmd_magnitude_l` 这类带 `Env/` 前缀 key

### Phase 5：集成测试 & 文档

- [ ] **Task 10：tyro CLI smoke**（`tests/test_wbt_force_cli.py`）
  - subprocess `train_agent.py exp:g1-29dof-wbt-force --help` 秒级 exit 0
  - baseline `exp:g1-29dof-wbt --help` 不坏
  - help 输出含 `wrist_compliance_command`

- [ ] **Task 11：IsaacSim e2e**（`tests/e2e/test_wbt_wrist_force_e2e.py`, `@pytest.mark.isaacsim`）
  - fixture：`replace(cfg.wrist_compliance_command.params, force_ext_activation_prob_per_step=1.0, force_cmd_activation_prob_per_step=1.0, force_ext_cooldown_range_s=(0.0, 0.0), force_cmd_cooldown_range_s=(0.0, 0.0))`
  - 跑 20 control step；断言：
    - `last_applied_force_w_by_body_id.keys() == {left_id, right_id}`（红线 1）
    - `force_ext_w.shape == (N, 2, 3)` 且连续 20 step 非零
    - `force_cmd_b` ramp 剖面平滑
    - reward ∈ [0, 1] 无 NaN
    - `obs_dict["actor_obs"].shape[1] == 1620`
    - `obs_dict["critic_obs"].shape[1] == 3000`
    - obs_dict 只有 2 个 key
    - `k_virtual.shape == (N, 2)` 且 v1 全 100.0
  - 独立 smoke：200 step 默认 config，断言 `force_ext_w.abs().sum() > 0` 至少命中一次

- [ ] **Task 12：训练用户文档 `docs/wbt-wrist-force-training.md`**（**只写训练相关**——部署文档在 Task 20 另写，避免未验证的部署流程写进来误导）
  - 启动命令 + 关键参数表（`--exp:g1-29dof-wbt-force`, `--command.setup_terms.wrist_compliance_command.params.k_virtual_range ...`）——canonical 命令见 Task 13.8
  - 红线 1 声明（force 只在 wrist）
  - K_virtual 是 reward hyperparameter 不是测得刚度
  - **部署端 API 留到 Task 20 workflow 文档写**（Phase 8；Task 12 不提前假设部署流程已验证）
  - **debug-draw arrow 使用说明**（Task 3 的产物）：
    - 启用方式：`--simulator.config.debug-viz True --training.headless False`（IsaacSim viewer 必须开）
    - 颜色图例：**红 = F_ext**（sim 注入的外力）、**蓝 = F_cmd**（policy 接收的力指令）、**紫 = F_total = F_ext + F_cmd**（虚拟弹簧合力）
    - 箭头长度：`length_m = ||F|| / debug_arrow_scale_n_per_m`（默认 50 N/m → 30N 画 0.6m）；要画得更小把 scale 调大
    - 只画 env 0；箭头从 wrist 起点出发，端点有小球标注
    - 训练时肉眼验证 F_cmd 蓝箭头方向是否和 motion clip 预期发力方向一致
  - **wandb 监控小节**（Task 9.5 的产物）：
    - 启用方式：`logger:wandb`；`Env/force/*`、`Episode/rew_wrist_force_position_tracking_exp` 自动有
    - 关键键集合（`Env/` 前缀）：`force/cmd_magnitude_{l,r,max}`、`force/ext_magnitude_{l,r,max}`、`force/cmd_ext_alignment`、`force/k_virtual_{l,r}`、`force/active_frac_{cmd,ext}`、`force/phase_{ramp_up,hold,ramp_down}`、`force/wrist_target_shift_{l,r}`、`force/wrist_pos_error_{l,r}`、`force/applied_f_body_{l,r}`
    - 推荐 wandb 面板分组：
      - Panel "Force Magnitude"（cmd/ext × l/r/max）
      - Panel "Force Activity"（active_frac + phase 占比）
      - Panel "Virtual Spring"（wrist_target_shift + wrist_pos_error）
      - Panel "K virtual"（k_virtual_l/r；v1 是平线）
      - Panel "Reward"（单独拉 `Episode/rew_wrist_force_position_tracking_exp`）
    - 健康度自检：`force/active_frac_cmd` 稳态应 ≈ 0.25-0.45（<0.05 = 活动率太低，config 坏；>0.9 = cooldown 没起作用）；`force/k_virtual_*` v1 必须 ≡ 100.0
  - v2 扩展：改 `K_VIRTUAL_RANGE_N_PER_M` 或加 RMA-style F_ext estimator

### Phase 5.5：训练侧 ONNX metadata 扩字段（为 Phase 8 部署准备，训练产物自带 schema）

- [ ] **Task 13.5：训练侧 ONNX metadata schema 扩字段**  _(Phase 5 已有 e2e 跑通后执行；Phase 7 真实训练开始前必须完成，否则训出来的 ckpt 不带 schema 部署要重训)_
  - 修改位置：`ppo.py:731-742` + `fast_sac_agent.py:986-997` metadata dict 构造处（**已列在 §5.2 修改清单**：`src/holosoma/holosoma/utils/inference_helpers.py`）
  - 追加 key：
    - `"history_length": cfg.observation.groups.actor_obs.history_length`
    - `"obs_term_names_sorted": sorted(cfg.observation.groups.actor_obs.terms.keys())`
    - `"obs_group_dims": {"actor_obs": 1620, "critic_obs": 3000}`
  - **只写不读**：这个 task **只负责训练侧写**；inference 侧 runtime 校验留到 Task 14（Phase 8）——这里解耦的原因是训练要先有 ckpt，部署才有校验对象
  - 测试（训练侧，pure CPU + `@pytest.mark.isaacsim` 二选一）：
    - export → onnx.load → 三个新 key 存在且值正确（`history_length==10`、`obs_term_names_sorted` 是 alphabetical list、`obs_group_dims["actor_obs"]==1620`）
    - baseline `g1-29dof-wbt` export 不坏（新 key 也写进 baseline metadata，但 inference 侧向后兼容处理在 Task 14）

### Phase 6：sanity（训练前最后一关）

- [ ] **Task 13：全局 sanity**
  - pre-commit 全绿
  - mypy 全绿
  - `pytest -s --ignore=thirdparty --ignore=src/holosoma_inference -m "not isaacsim and not requires_inference"` 全绿
  - `pytest -m isaacsim` Spike 1 + Task 3 + Task 11 全绿
  - baseline `exp:g1-29dof-wbt --help` 不坏
  - `demo_scripts/demo_omomo_wb_tracking.sh` 第 4 步能启动（5s SIGINT）

### Phase 7：TRAIN GATE —— 跑真实训练到收敛

**这是"训练做完再做部署"的硬门。本 Phase 不绿 → 不进 Phase 8。**

- [ ] **Task 13.8：真实训练跑起来 + 看 wandb 收敛**
  - 启动命令（作为"首次跑"的 canonical 命令写进 Task 12 doc）：
    ```bash
    python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-force \
        simulator:isaacsim \
        logger:wandb \
        --training.seed 1
    ```
    注：WBT 系列 preset 在 `config_values/wbt/g1/experiment.py` 里默认绑定 `simulator.isaacsim`，所以 `simulator:isaacsim` 和省略该 flag 等价（参考 `demo_scripts/demo_omomo_wb_tracking.sh` 的做法）。也可以改用 `simulator:mjwarp` 跑 GPU MuJoCo warp；`simulator:isaacgym` 需要额外装 IsaacGym 才能跑。
  - **wandb 必看指标（Phase 4.5 的 Task 9.5 产物）**——以下**同时满足**才算 gate 过：
    1. `Train/mean_reward` 前 200 iter 单调上升 → 稳定在正区间（不崩）
    2. `Episode/rew_wrist_force_position_tracking_exp` 随训练上升到 **≥ 0.4**（reward 定义是 `exp(-error/σ²)`，0.4 对应稳态 wrist 位置跟踪误差 ~σ=0.3m 的一半）
    3. `Env/force/active_frac_cmd` + `Env/force/active_frac_ext` 稳态 ∈ [0.2, 0.5]（激活采样率对）
    4. `Env/force/k_virtual_l` / `Env/force/k_virtual_r` ≡ 100.0（v1 配置对了，非平线说明 reset 采样 broken）
    5. `Env/force/cmd_magnitude_max` 不爆（≤ 30N）
    6. `Env/force/applied_f_body_l/r` 和 `ext_magnitude_l/r` **相等**（sanity：world→body 转换对了；不等说明 Task 3 的 `quat_apply_inverse` 有 bug）
    7. baseline `Episode/rew_motion_tracking_*` 不塌陷（虚拟弹簧 reward 不能把 motion tracking 挤掉）
    8. actor/critic loss 不 diverge；`Episode/rew_survival` 不塌（G1 没频繁摔）
  - **Gate 不过时的调参触发点**（写进 Task 12 的 debug 小节）：
    - rew 停滞：调 MLP `[1024,512,256]` → `[2048,1024,512]`；或 `history_length=10` → `15`
    - active_frac 过低：`force_ext_activation_prob_per_step` 0.01 → 0.02
    - applied_f_body ≠ ext_magnitude：检查 Task 3 `body_quat_w` wxyz 顺序 + `quat_apply_inverse` 用对
    - motion tracking 塌：降 `wrist_force_position_tracking_exp.weight` 2.0 → 1.0；或调 σ 0.3 → 0.5
  - **产物**：训练 wandb run URL + 一份 ONNX ckpt (wandb artifact) → 把 URL 和 artifact 名记到 Task 12 doc 的"已验证训练 run"小节
  - **Gate 绿**：上面 8 条都过 + 训练至少跑 2000 iter（或 achieve 目标 reward，哪个先到）
  - **Gate 不绿**：**禁止开始 Phase 8**。先回 Phase 0-6 找 bug（大概率是 Task 2 state machine / Task 3 frame 转换 / Task 5 reward 公式）

### Phase 8：inference & deployment（TRAIN GATE 过后）

- [ ] **Task 14：inference 侧 ONNX metadata runtime 校验**
  - 前置：Phase 7 TRAIN GATE 绿 + Task 13.5 已把 schema 写进 ckpt
  - inference 侧 `WholeBodyTrackingPolicy.setup_policy` (`wbt.py:131-142`) 读 metadata 后 strict compare `self.obs_terms_sorted["actor_obs"]` / `self.history_length_dict["actor_obs"]`；不匹配 raise
  - 向后兼容：metadata key 缺失（老 ckpt）→ logger.warning 不 raise
  - 测试：
    - 用 Task 13.8 产出的真实 ckpt 跑 setup → 通过
    - fake metadata mismatch（history / term order）→ `setup_policy` raise
    - 缺 key → warning 不 raise；baseline `g1-29dof-wbt` smoke 不受影响

- [ ] **Task 15：inference obs preset `wbt_force`**
  - `config_values/observation.py` 末尾追加
  - `obs_dict["actor_obs"]` = baseline 6 + `wrist_force_command` + `wrist_virtual_stiffness_command`
  - `obs_dims["wrist_force_command"]=6`、`obs_dims["wrist_virtual_stiffness_command"]=2`
  - `obs_scales["wrist_force_command"]=1.0`、`obs_scales["wrist_virtual_stiffness_command"]=0.01`
  - `history_length_dict = {"actor_obs": 10}`
  - 测试：alphabetical sort 后 term list 和 `g1_29dof_wbt_force_observation.actor_obs.terms.keys()` 一致（用 hardcode 常量断言，不依赖 isaacsim）

- [ ] **Task 16：inference preset `g1_29dof_wbt_force` + entry point**
  - `config_values/inference.py` 追加 `g1_29dof_wbt_force`：复用 `_g1_29dof_wbt_robot`；`observation=wbt_force`；`task=task.wbt`；`secondary=_g1_safety_secondary`
  - DEFAULTS 注册 `"g1-29dof-wbt-force"`
  - `setup.py` entry point `holosoma.config.inference` 追加
  - 测试：`run_policy.py inference:g1-29dof-wbt-force --help` exit 0；重装后 `entry_points(group="holosoma.config.inference")` 发现新 key

- [ ] **Task 17：`WristForceCmd` + Provider + impl + task config + factory wiring**（Task 18 的前置，必须整体落地）
  - `inputs/api/commands.py`: `WristForceCmd(force_left: tuple[float,float,float], force_right: tuple[float,float,float])` frozen
  - `inputs/api/base.py`: `WristForceCmdProvider` Protocol（`poll_wrist_force() -> WristForceCmd | None`、`start()`、`zero()`）
  - `config_types/task.py`: `InputSource` Literal 扩新值；`TaskConfig` 追加 `wrist_force_input`、`wrist_force_magnitude_cap: float = 30.0`、`ros_wrist_force_topic: str = "holosoma/wrist_force_cmd"`
  - `inputs/__init__.py`: `create_input` factory 扩 `"wrist_force"` role dispatch
  - `inputs/impl/keyboard.py`: 左 wrist `u/j/h/k/y/n`，右 wrist numpad `8/2/4/6/9/3`，`/` 归零，`,`/`.` 调 magnitude；cap clamp
  - `inputs/impl/ros2.py`: subscribe `Float32MultiArray[6]`
  - 测试：
    - keystroke → WristForceCmd 正确
    - magnitude clamp：输入 [100,0,0] + cap=30 → 输出 norm ≤ 30
    - `zero()` 后 poll 返回全 0 WristForceCmd
    - ROS2 mock 导入校验
    - `create_input(policy, "keyboard", "wrist_force")` 返回实现 `WristForceCmdProvider` 的对象

- [ ] **Task 18：`BasePolicy._poll_extra_inputs()` hook + `WholeBodyTrackingForcePolicy`**
  - `base.py:816` `run()` loop 里 `policy_action()` 之前插入 `self._poll_extra_inputs()`；基类默认 pass
  - 子类 `policies/wbt_force.py`：
    - `__init__`: `self._wrist_force_cmd = np.zeros((1, 6), dtype=np.float32)`；`self._wrist_force_provider = create_input(self, config.task.wrist_force_input, "wrist_force")`；`.start()`；`self._magnitude_cap = config.task.wrist_force_magnitude_cap`
    - override `_poll_extra_inputs()`: `cmd = self._wrist_force_provider.poll_wrist_force()`; 非 None → clip norm ≤ cap → 写 `_wrist_force_cmd`
    - override `get_current_obs_buffer_dict`: super() + `d["wrist_force_command"] = self._wrist_force_cmd.copy()`
    - override `_handle_stop_policy`: super() + `self._wrist_force_cmd.fill(0.0)` + **`self._wrist_force_provider.zero()`**（防 provider buffer repopulate）
  - 测试：
    - fake provider → obs dict 含 `wrist_force_command` shape (1,6)
    - stop policy → `_wrist_force_cmd` 全 0 且 `provider.zero()` 被调一次
    - magnitude clamp：provider norm=50 + cap=30 → `_wrist_force_cmd` norm ≤ 30
    - hook 回归：baseline Locomotion/WBT policy 下 `_poll_extra_inputs` = no-op

- [ ] **Task 19：policy dispatch**
  - `dual_mode.py:_select_policy_class`：`"wrist_force_command" in actor_obs` → entry point 优先，兜底 `WholeBodyTrackingForcePolicy`
  - `setup.py` 新 group `holosoma.policies.wbt` 注册 `g1-29dof-force = ...:WholeBodyTrackingForcePolicy`
  - 测试：
    - fake config with `wrist_force_command` → 路由到 force subclass
    - baseline WBT config → 仍 `WholeBodyTrackingPolicy`
    - mock entry_points 返回 subclass → 优先用 entry point

- [ ] **Task 20：sim-to-sim + workflow docs**（前置：Task 15-19 + `pip install -e src/holosoma_inference` 重装）
  - 新建 `docs/workflows/sim-to-sim-wbt-force.md`:
    - 启动命令（两 terminal：`run_sim.py` + `run_policy.py inference:g1-29dof-wbt-force --task.wrist-force-input keyboard`）
    - body-yaw frame 约定：+X 前、+Y 左、+Z 上，单位 N
    - MuJoCo 无接触面 → 只能看 Δx，看不到真实 F_real
    - 验证：F_cmd=0 → wrist 跟 motion；F_cmd=[5,0,0] 左 wrist → Δx≈0.05m（K=100 稳态）
    - 键位表 + ROS2 topic 格式
  - 新建 `real-robot-wbt-force.md`:
    - 首次部署 F_cmd=0 → 小幅 1-5N → 逐步加码
    - 紧急停止：`o` → stop + `_wrist_force_cmd=0` + `provider.zero()`
    - 安全：magnitude cap=30N 默认；CLI `--task.wrist-force-magnitude-cap` 只能调低
  - `README.md` "WBT Controls" 下追加 F_cmd 键位小节
  - e2e smoke（marker `requires_inference`）：`run_sim.py` + `run_policy.py` 5s SIGINT；断言 provider 初始化成功、`actor_obs.shape[1]==1620`、policy instance 是 `WholeBodyTrackingForcePolicy`、`_wrist_force_cmd` 初值全 0

- [ ] **Task 21：跨-package schema 一致性 + runtime metadata validation 测试**
  - `tests/e2e/test_wbt_force_inference_schema.py`：import 训练 preset + inference preset 做静态 compare（不依赖 isaacsim）
    - term 名集合相等、alphabetical sort 一致
    - dim：actor 162/1620、critic `obs_dict` 只含 actor_obs（inference 端 critic obs 不参与部署）
    - history_length=10
    - obs_scales 一致（特别 `wrist_force_command=1.0`、`wrist_virtual_stiffness_command=0.01`）
  - 配合 Task 14 metadata 校验：fake ONNX history=5 → raise；fake term 顺序错 → raise；老 ckpt 缺 key → warning
  - 回归：baseline `wbt` config 不变；baseline ONNX 仍能 MuJoCo smoke

---

## 7. Self-Review

| 要求 | 覆盖 |
|---|---|
| F_ext / F_cmd 只加 wrist | 红线 1；Task 3 注入 `left/right_wrist_yaw_link`；Task 11 debug accessor 断言 |
| K_virtual per-env + actor obs | Task 1 `k_virtual_range` + Task 2 `k_virtual` buffer + Task 4 obs term + Task 5 reward 从 buffer 读 |
| F_cmd 给 actor、F_ext 只给 critic | Task 4 term visibility + Task 7 group 配置 |
| 10 步 history（CHIP-style） | Task 7 `history_length=10`；actor 1620 / critic 3000 |
| 以 Holosoma 原始 code 为主 | baseline 2-group 架构；`input_dim=["actor_obs"]` 不 override；无 shim |
| reward 只追加不改 baseline | Task 8 identity 断言；Task 13 demo script smoke |
| 不 import third_party；允许 copy / 借用（license 不管） | §3：运行时不 import `third_party/` 但允许 copy 进仓库（`_gh_utils.py`），爱怎么用怎么用；文件头加 "Adapted from" attribution 方便回溯 |
| 可视化 F_cmd / F_ext 箭头在 wrist 上（训练调试用） | Task 3 `draw_debug_viz`：红=F_ext、蓝=F_cmd、紫=F_total；`debug_arrow_scale_n_per_m=50` 默认；Task 12 文档含颜色图例 |
| wandb 力相关训练监控 | Task 2 `update_metrics()` + Task 3 `_update_log_dict()` override（对齐 `wbt_manager.py:71-80` pattern）+ Task 9.5 定义键集合 + reward episode_sums 自动进 `Episode/rew_*`；pipeline 沿用 `LoggingHelper.episode_env_tensors` → `wandb.log` 既有通路（`logging_utils.py:148, 189, 339-342`）不新增机制 |
| 不影响 Holosoma 已有 code | §5.3 必须保持原样清单；每 preset identity 断言；Task 13 sanity |
| 训练完 eval + deploy 不回头 | Phase 5.5 Task 13.5（训练侧 ONNX schema 扩字段）先下预埋 → Phase 7 TRAIN GATE 跑训练到收敛 → Phase 8 Task 14-21（inference 侧 runtime 校验 + policy 子类 + F_cmd 输入链 + workflow docs + schema 一致性）|

---

## 8. 已知未决项（训练后再决定）

1. **MLP 层宽**：Task 9 smoke training 判断 `[512,256,128]` vs `[1024,512,256]`
2. **history_length=10**：不收敛可试 15-20；overfit 可降 5
3. **F_ext activation prob**：默认 0.01/step，不够就调到 0.02-0.05
4. **F_cmd range saturation**：若 `||F_ext+F_cmd|| > 45N` 占比 > 20% 且 tracking 退化，把 `force_ext_magnitude_range` 上限降到 15-20
5. **K_virtual range 何时拓宽**：v1 range=(100,100) 收敛 → v2 改 `(50,300)`（只改 `_k_virtual.py` 一行，不动其他代码）
6. **critic 是否看更多 K-related 信息**：v9.4 选 CHIP 派系（actor+critic 都看 K）；若 value estimator 方差大，v3 考虑加 GH-style `wrist_force_applied_priv` critic-only term
7. **F_cmd 键位 dogfood**：初版 `u/j/h/k/y/n` + numpad，真机上操作员反馈不好可切 joystick multiplex

---

## 9. 参考（查代码时用）

- Holosoma 外力注入 API: `src/holosoma/holosoma/agents/callbacks/push.py:222-252`, `simulator/shared/virtual_gantry.py:425`
- Holosoma callback 顺序 (WBT): `src/holosoma/holosoma/envs/base_task/base_task.py:405-460`
- Holosoma baseline WBT preset: `src/holosoma/holosoma/config_values/wbt/g1/observation.py`
- Holosoma ONNX exporter: `src/holosoma/holosoma/utils/inference_helpers.py:40-112, 142-217`
- Holosoma wandb pipeline（Task 9.5 参考）:
  - `_update_log_dict` hook 定义：`src/holosoma/holosoma/envs/base_task/base_task.py:444, 469, 549-551`
  - WBT 既有 `_update_log_dict` 参考实现：`src/holosoma/holosoma/envs/wbt/wbt_manager.py:71-80`
  - `MotionCommand.update_metrics()` 参考实现：`src/holosoma/holosoma/managers/command/terms/wbt.py:445, 1000-1034`
  - `TensorAverageMeterDict` + wandb flush：`src/holosoma/holosoma/agents/modules/logging_utils.py:104, 148-149, 188-189, 339-342`
  - reward episode_sums 自动 log（`Episode/rew_*`）：`src/holosoma/holosoma/managers/reward/manager.py:200-250`
- Inference BasePolicy: `src/holosoma_inference/holosoma_inference/policies/base.py:120-160, 588-620, 816`
- Inference WBT policy: `src/holosoma_inference/holosoma_inference/policies/wbt.py:131-174, 225-284`
- Inference dual-mode dispatch: `src/holosoma_inference/holosoma_inference/policies/dual_mode.py:12-37`
- GH `TemporalLerp` 源（Task 2 copy 进 `_gh_utils.py`）: `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/utils.py:5-133`
- GH `random_uniform` 源（Task 2 copy 进 `_gh_utils.py`）: 同文件 `:135-162`
- GH `clamp_norm` + `_norm` 源（Task 2 copy 进 `_gh_utils.py`）: `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/admittance.py:4-11`
- GH 仓库 license 状态：无 LICENSE 文件（v10 不作为障碍——按用户指令 license 不管）
- GH kp 采样模式参考（per-episode resample）: `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py:781-782`
- **设计推导档（为什么这么做）**：`docs/plans/2026-05-03-wbt-wrist-force-v9.md`（v9.4）
