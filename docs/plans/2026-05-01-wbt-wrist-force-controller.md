# WBT 带力追踪 Low-Level Controller Implementation Plan (v9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **重要：实际执行范围是下面的 "★ Path A' (RECOMMENDED)" 小节（§A'.0 起）。**
> 本文件前半段（L132 起的 "PATH A: UniFP 式"）以及末尾的 "PATH B: CHIP 式" 都是 **历史参考，禁止实施**——它们的 task 编号、常量（K=200 等）、history 策略等与 v9 矛盾，follow 上去会实现错的变体。

**状态 (v9)：** 基于 demo_scripts/demo_omomo_wb_tracking.sh 跑通验证 + K_virtual 分析，再简化一步：

1. **K_virtual 固定 100 N/m（training-time hyperparameter）**（§A'.4）—— 选 100 的理由：同向叠加 F_ext+F_cmd 最大 60N，÷ 100 → Δx=0.6m 在 G1 arm reach 边缘而非超出；仍落在 GH `kp_range=[5,250]` 区间内。**不是**测得的机器人 Cartesian 刚度，只是 reward 里的 F↔Δx exchange rate
2. **不加 observation history**（§A'.7b.5 延续）—— v1 的定位是 "domain randomization against unobserved wrist F_ext"，不是真正的 learned compliance（真 compliance 需要 proprio history 让 actor 隐式估 F_ext，留给 v2）
3. **不从 GH / UniFP copy 任何代码**（§A'.5.1 延续）
4. **Observation 极简**（§A'.7b 延续）—— actor 160 dim / critic 298 dim
5. **Task 数 14**（§A'.7 延续；Task A'-0 = 手写常量文件）
6. **保持不变的硬约束**：F_ext/F_cmd wrist-only；K_virtual 固定 100 N/m；F_cmd 给 actor，F_ext 只给 critic；red line 1-5 全部生效

**Goal:** 在不影响任何现有训练/测试代码的前提下，新增 WBT + wrist force tracking 实验变体 `exp:g1-29dof-wbt-force`。

**实施方案只有一个：★ Path A' (RECOMMENDED, v9)。** Path A / Path B 是历史参考，详见文件中段 "PATH A / PATH B" 小节顶部的 ⛔ banner。

- **★ Path A' v9** ——reward 用 UniFP 双力公式 `g̃ = g + (F_ext + F_cmd)/K`（F_ext 来自 sim, F_cmd 来自 actor input）。F_ext 梯形剖面**只注入左右 wrist**；其他 body 零力。**`K_virtual = 100 N/m` 固定 training hyperparameter**（见 §A'.4 "从 Δx 工作空间反推"推导，不是 Jacobian/PD 校准结果）。Critic privileged F_ext。**v1 不做 history，不 port 任何 third-party code**（ramp state machine 自己写）。满足主动施力 + F_cmd as input 硬约束，同时解决"zero-shot 接触未训练过"短板。
- Path A（UniFP 纯版，仅 F_cmd）：**历史参考。** 见下方 "PATH A" 小节 banner。
- Path B（CHIP 纯版，Hindsight）：**历史参考。** 与 "F_cmd as input" 硬约束冲突。

**Tech Stack:** Python 3.10, PyTorch, Isaac Sim 4.5 / Isaac Lab（WBT 唯一支持的 simulator），tyro 配置系统，pytest。

---

## 方法路径对比（UniFP vs CHIP vs GentleHumanoid）

**论文出处：**
- UniFP：`third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891-1900` (`_reward_tracking_ee_force_world`)
- CHIP：`third_party/CHIP.pdf`（NVIDIA / Stanford / UT Austin，NeurIPS/ICRA 2026 投稿）
- GentleHumanoid：`third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py:547-1150`

### 一张表对齐差异

| 维度 | Path A: UniFP | **★ Path A' (推荐)** | Path B: CHIP | GentleHumanoid |
|---|---|---|---|---|
| **语义** | Active force production | **Active force + DR against unobserved wrist F_ext** (真正的 learned compliance 留给 v2 加 history) | Adaptive compliance | Passive compliance |
| **Policy 里的力相关输入** | `F_cmd` 6-D body-yaw | **`F_cmd` 6-D body-yaw** | `1/k` 2-D | 无 |
| **Sim 里注入 F_ext** | ✗ | **✅ 只到左右 wrist**（梯形剖面 [0,30] N × [1,3] s）| ✅ wrist | ✅ 弹簧式 |
| **Reward `g̃` 公式** | `g + F_cmd/K` | **`g + (F_ext + F_cmd)/K`** | `g` 不改 | admittance 输出 |
| **Observation motion goal** | 原 goal 不变 | **原 goal 不变** | Hindsight 偏移 | admittance 输出 |
| **Actor 辅助输入** | F_cmd | **F_cmd（v1 不加 history）** | `1/k` + 10 步 proprio + 11 步 actions | — |
| **Critic privileged** | 无特别设计 | **Ground-truth F_ext (6-D, wrist only)** | Ground-truth F_ext | F_applied / F_expected |
| **是否需要 env 子类** | 否 | **是**（F_ext 注入 wrist） | 是 | 是 |
| **K_virtual 选取** | 写死 200（UniFP 默认） | **固定 100 N/m（training hyperparameter，见 §A'.4）** | N/A | N/A |
| **部署施力机制** | F_cmd 接触产生力 | **F_cmd 接触产生力，更鲁棒** | target 放墙里 + 1/k | 同 CHIP |
| **stiff ↔ compliant 可调** | ✗ | ✗（固定 K） | ✅ 连续 | ✗ |
| **代码改动量** | 小 (12 task) | **中 (14 task + 1 spike，无校准脚本)** | 中大 (13 task + 3 spike) | — |

**硬约束（v9 权威清单；本节前任何早期版本描述以此为准）：**
- **F_ext / F_cmd 只施加在左右 wrist**，其他 body（head、torso、root、elbow、hip 等）力严格为 0
- **K_virtual = 100 N/m，固定**（training-time hyperparameter，不是测得的机器人刚度；§A'.4）。不随机化，不进 obs
- **F_cmd 是 actor 的 input**（部署时上层通过这个接口施力，6-D body-yaw frame）
- **不加 observation history**（`history_length=1`，跟 baseline 一致；§A'.7b.5）
- **不加 adaptive compliance `1/k`**（Path B 的能力，留给 future extension）
- **不 copy / import / port 任何 third-party 代码**（GH / UniFP / CHIP 全部只读参考，Holosoma 运行代码里一行都不出现）

### CHIP 的核心洞察（一句话）

> **不要改 reward 的 reference motion，改 observation 的 goal。**
>
> reference motion（full body pose, link vel, joint state）又密又难改还得保证物理可行；
> tracking goal（只有 head + 2 wrists 的 pose）又稀疏又好改。CHIP 只在 goal 上做 hindsight perturbation：
>
> - 观察 `g_hind = g − (1/k)·f` 给 actor
> - reward 用**原 g** 算 `r(s, g)`
>
> 训练时 policy 学到：看到被扰动的 g_hind → 产出的动作回到原 g → 对外力呈现 1/k 的 compliance 行为。

### CHIP 的训练流程（按 CHIP 论文 Fig. 2）

```
每个 control step（50Hz）：
  1. 采样 / 持续 per-wrist F_ext  (梯形剖面，方向均匀, ∈ [0,40] N, duration ∈ [1,3]s)
  2. sim 注入 F_ext 到 left/right_wrist_yaw_link 的 rigid body（world frame）
  3. actor_obs:
       - motion_command 里 wrist 两个 body 的 position target 被替换为 g - (1/k)·f_ext （hindsight）
       - 其他 12 个 body 的 target 不变
       - (1/k) 作为新 obs 追加
       - 10-step proprio + 11-step past action history
  4. critic_obs: 同 actor, 另外加 ground-truth F_ext 作为 privileged
  5. reward: Holosoma 现有全部 motion tracking reward, **不变**
```

### CHIP 的部署流程（按 CHIP 论文 §III 结尾）

```
上层（遥操作 / planner）给出：
  - head + wrist 的 kinematic target：g_t
  - per-wrist compliance coefficient：1/k

Policy 输入: (g_t, 1/k, proprio_history)
Policy 输出: 关节 position target 残差（和 Path A 完全一样）

可选的 damper model（论文 §III）：
  g_t ← α · x_eef + (1 - α) · g_{t-1}   # inference 端做，不影响训练
```

### 为什么 CHIP 就能"主动施力"

关键：**没有 F_cmd 这个 input**。上层要施力时：

1. 把 wrist target 放到**表面"里面"** Δx = f_target · (1/k)，比如想推门 30N + `1/k=0.05` → Δx = 1.5 m（夸张的例子，实际 `1/k=0.02` Δx=0.6 m 更合理）
2. 没接触时，policy 按 compliance 行为，wrist 自然沿 target 方向推进（空中就是滑向 target）
3. 接触后，表面挡住 wrist，policy 继续按 "yield 1/k" 行为产生关节 torque 去抵住 Δx → 产生 F_contact ≈ Δx / (1/k) = f_target

**所以 CHIP 的"施力 API"就是 "把 target 放在该去的地方 + 调 1/k"**。比 UniFP 的 `F_cmd` 直给更间接，但**换来了 stiff↔compliant 可调性**。

### 两种路径在 Holosoma 下的代码改动量对比（**历史参考，v9 已决定走 Path A'**）

> 下表是早期 v1-v3 在 Path A（纯 UniFP）vs Path B（纯 CHIP）之间的评估。v9 最终走 Path A'（两者结合），不走 A 也不走 B。保留此表仅作为为什么选 A' 的背景。

| 要素 | Path A (UniFP) | Path B (CHIP) |
|---|---|---|
| 新增 dataclass | `WristForceConfig` | `WristComplianceConfig`（含 F_ext 采样 + 1/k 采样） |
| 新增 command term | `WristForceCommand`（采样 F_cmd，不注入 sim） | `WristComplianceCommand`（采样 1/k + F_ext，**注入 sim**，维护 hindsight 偏移量供 obs 用） |
| 新增 env 子类 | 不需要 | 需要 |
| 预估 task 数 | 12 | 16–18 |

### v9 结论

只走 Path A'。Path A 和 Path B 都是历史参考，**不实施**。见 §A'.0 起的 v9 硬约束和任务分解。

---

## ═══════════════════════════════════════════════════════════════════
## PATH A: UniFP 式（原 plan） — **历史参考，DO NOT IMPLEMENT**
## ═══════════════════════════════════════════════════════════════════

> ⛔ **STOP — 这一节（从本小节开始到 "PATH A' (RECOMMENDED)" 前）是 v1 原始方案，已被 Path A' v9 取代。**
>
> 保留是为了追溯"为何选双力公式"的设计演进，不是为了实施。请勿按本节的 task 1-12、文件清单、或 K=200 N/m 等参数写代码。
>
> **执行范围从 §A'.0（下面带"★ Path A'"前缀的小节）开始。**

## 背景与关键参考

> "我要能通过 low level controller 主动施力" ×
> "要把 force command 作为输入输入到 actor 里面" ×
> "参考 UniFP" ×
> "对比 gentle-humanoid-training 和 holosoma 本身的 loss，还是以 holosoma 为主"

**这不是扰动训练，是主动力追踪训练。** policy 的任务是：**在给定 F_cmd 下，产生对应的末端力**；部署时通过与真实表面接触，位置偏移经由接触刚度转化为真实力。

### UniFP 的 "virtual impedance" reward —— 本 plan 的核心参考

UniFP (`third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891-1900`)：

```python
def _reward_tracking_ee_force_world(self):
    forces_global = self.forces[:, self.gripper_idx, 0:3]            # F_ext 真正注入 sim
    forces_cmd = self.current_Fxyz_gripper_cmd                        # F_cmd body-yaw frame
    forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
    forces_offset = (forces_global + forces_cmd_global)
    curr_ee_goal_cart_world_offset = (
        forces_offset / self.gripper_force_kps + self.curr_ee_goal_cart_world
    )
    ee_pos_error = torch.sum(torch.abs(self.ee_pos - curr_ee_goal_cart_world_offset), dim=1)
    return torch.exp(-ee_pos_error / self.cfg.rewards.tracking_ee_sigma * 2)
```

**关键洞察：**

- `K_virtual = gripper_force_kps` 是**纯训练信号用的虚拟刚度**，UniFP 里固定为 200 N/m（`b2z1_pos_force_config.py:151` 的 `gripper_force_kp ∈ [200, 200]`）。**sim 里没有真正建这个弹簧**。
- 公式语义：如果存在一个刚度 K 的虚拟弹簧，policy 把 wrist 偏离原 target 的距离 `F_cmd / K`，这个弹簧就会产生大小为 F_cmd 方向相反的力。
- **训练阶段**：sim 里 wrist 空气中自由移动，没有任何真实接触；policy 学到的是"给我 F_cmd，我就把 wrist 偏移 F_cmd/K 的距离"。
- **部署阶段**：wrist 接触真实表面时，同样的位置偏移 × 表面真实刚度 ≈ F_cmd。这就是 **impedance control** 思想：policy 学出来的是阻抗行为，力通过接触自然产生。
- UniFP 同时有 `forces_global`（F_ext：sim 实际注入的 perturbation）和 `forces_cmd`（F_cmd：给 policy 看的指令）。**本 plan v1 只做 F_cmd 这一半**（active force），F_ext 留给 v2 作为 robustness 扩展。

### GentleHumanoid 的 force reward —— 对比参考但**不**采用

GentleHumanoid (`third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py:1135-1150`)：

```python
@reward
def force_reward(self):
    force_norm = self.force_applied_w.norm(dim=-1, keepdim=False)
    force_norm_diff = (self.force_applied_w - self.force_expected_w).norm(dim=-1).mean(...)
    force_reward = _calc_exp_sigma(force_norm_diff, self.reward_sigma["force"])
    force_max_limit = self.force_safe_limit_tl.current + self.force_penalty_offset
    force_exd = (force_norm > force_max_limit).any(dim=-1, keepdim=True)
    return (force_reward * (~force_exd)).mean(...)
```

**GentleHumanoid 做的是 passive compliance：**

- sim 里有一个真的弹簧注入 `F_applied`（`motion_tracking.py:1001`：`self.force_applied_w[:] = clamp_norm(force_kp_scaled * project_pos_diff(force_origin_w - pos_w, ...), max_force)`）。
- admittance 动力学模型算出 `F_expected`（`motion_tracking.py:914-922`）。
- reward 奖励二者一致：policy 要学会让 wrist 的实际位置 / 速度符合 admittance 模型预期，使得**弹簧注入的力**恰好等于 admittance **预期的力**。

这是"感觉到外力后柔性响应"——**顺应**，不是**主动施力**，方向与用户需求相反。因此本 plan **不采用** GentleHumanoid 的 reward 公式 / admittance 模型 / τ_safe / net_wrench_limiter。

### Holosoma 当前 WBT 架构（只读，不动）

- **实验注册：** `src/holosoma/holosoma/config_values/experiment.py:19`，tyro subcommand 名 `exp:g1-29dof-wbt`。
- **Env class：** `src/holosoma/holosoma/envs/wbt/wbt_manager.py:13`（`WholeBodyTrackingManager`，强制 IsaacSim only）。本 plan **不新建 env 子类**，因为不往 sim 注入力。
- **MotionCommand：** `src/holosoma/holosoma/managers/command/terms/wbt.py:511`。它是 class-based term 的参考样板；其 `body_pos_relative_w`（形状 `[num_envs, 14, 3]`）是"motion 目标适配到机器人当前 xy + yaw 后的目标 body 位置，世界系"，也就是 Holosoma 里 motion tracking reward 追踪的 target，**正是我们要给 wrist 做偏移的那个 target**。
- **Motion command 跟踪的 body 顺序：** `src/holosoma/holosoma/config_values/wbt/g1/command.py:19-34`。两个手腕在这个列表里是 `left_wrist_yaw_link`（index 10）和 `right_wrist_yaw_link`（index 13）。本 plan 会通过 body name 解析而不是硬编码 index。
- **现有 wrist 相关 reward：** `src/holosoma/holosoma/managers/reward/terms/wbt.py:87` 的 `motion_relative_body_position_error_exp` 对 14 个 body 取 `mean`。wrist 只占 2/14。新 reward 会给 wrist 一个专项 2.0 的权重（和 UniFP 对齐）。
- **Step 顺序：** `src/holosoma/holosoma/envs/base_task/base_task.py:417-444` —— `_physics_step`（循环 `decimation` 次，每次 `_apply_force_in_physics_step`）→ `_post_physics_step` 里 `_update_tasks_callback()`（调用 `command_manager.step()`、`curriculum_manager.step()`、`randomization_manager.step()`，**然后**计算 reward 和 obs）。所以 `WristForceCommand.step()` 会在每个 control step 开始时更新 `force_cmd_w`，之后同一个 step 的 reward/obs 都能读到最新值。这一顺序符合我们的设计。
- **Reward manager 支持 class-based 和 function-based 两种 term**（`src/holosoma/holosoma/managers/reward/base.py:14`：`RewardTermBase`）。本 plan 用 function-based 即可，因为新 reward 是无状态函数。

### G1 URDF 关键 link

- `left_wrist_yaw_link`、`right_wrist_yaw_link` 是两手腕最末端 link（已在 `src/holosoma/holosoma/config_values/robot.py:328-337` 的 `body_names` 中；已在 `body_names_to_track` 中）。

---

## Scope 检查

单一子系统（WBT 训练管线的力追踪变体），不跨越多个独立子系统。无需再拆分。

---

## 文件结构

### 新建文件（全部只增）

| 路径 | 职责 |
|---|---|
| `src/holosoma/holosoma/config_types/command.py` **[修改]** | 在文件末尾追加 `WristForceConfig` dataclass |
| `src/holosoma/holosoma/managers/command/terms/wbt_force.py` | 新增 `WristForceCommand(CommandTermBase)`：采样 + body-yaw/world 双视角 |
| `src/holosoma/holosoma/managers/observation/terms/wbt_force.py` | 新增 `wrist_force_command` obs fn（body-yaw frame 6-D） |
| `src/holosoma/holosoma/managers/reward/terms/wbt_force.py` | 新增 `wrist_force_position_tracking_exp` reward fn |
| `src/holosoma/holosoma/config_values/wbt/g1/command_force.py` | 新增 `g1_29dof_wbt_force_command`（motion command + wrist_force_command） |
| `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py` | 新增 `g1_29dof_wbt_force_observation`（加 wrist_force_command obs 条目） |
| `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py` | 新增 `g1_29dof_wbt_force_reward`（原 reward + `wrist_force_position_tracking_exp`，weight=2.0） |
| `src/holosoma/holosoma/config_values/command.py` **[修改]** | `DEFAULTS` dict 注册新 command config |
| `src/holosoma/holosoma/config_values/observation.py` **[修改]** | `DEFAULTS` dict 注册新 observation config |
| `src/holosoma/holosoma/config_values/reward.py` **[修改]** | `DEFAULTS` dict 注册新 reward config |
| `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` **[修改]** | 新增 `g1_29dof_wbt_force` ExperimentConfig（继承 g1_29dof_wbt，只换 command/observation/reward） |
| `src/holosoma/holosoma/config_values/experiment.py` **[修改]** | `DEFAULTS` dict 注册 `"g1_29dof_wbt_force": g1_29dof_wbt_force` |
| `src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py` | `WristForceConfig` 单元测试 |
| `src/holosoma/holosoma/managers/command/terms/tests/__init__.py` | 空 |
| `src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force.py` | `WristForceCommand` 单元测试（fake sim） |
| `src/holosoma/holosoma/managers/observation/terms/tests/__init__.py` | 空 |
| `src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force.py` | obs term 单元测试 |
| `src/holosoma/holosoma/managers/reward/terms/tests/__init__.py` | 空 |
| `src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force.py` | reward term 单元测试（含 F=0 → motion tracking 等价、F>0 → target 正确偏移） |
| `src/holosoma/holosoma/config_values/wbt/g1/tests/__init__.py` | 空 |
| `src/holosoma/holosoma/config_values/wbt/g1/tests/test_command_force.py` | command preset 回归测试 |
| `src/holosoma/holosoma/config_values/wbt/g1/tests/test_observation_force.py` | obs preset 回归测试 |
| `src/holosoma/holosoma/config_values/wbt/g1/tests/test_reward_force.py` | reward preset 回归测试 |
| `src/holosoma/holosoma/config_values/wbt/g1/tests/test_experiment_force.py` | experiment 回归测试（含"base exp 未被改动"断言） |
| `src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py` | tyro CLI smoke test（subprocess `--help`） |
| `src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py` | IsaacSim e2e（`@pytest.mark.isaacsim`） |
| `docs/wbt-wrist-force-training.md` | 用户文档 |

### 必须保持原样的文件（一行都不改）

- `src/holosoma/holosoma/envs/wbt/wbt_manager.py`
- `src/holosoma/holosoma/managers/command/terms/wbt.py`（motion command 实现）
- `src/holosoma/holosoma/managers/observation/terms/wbt.py`
- `src/holosoma/holosoma/managers/reward/terms/wbt.py`（现有 motion tracking 公式）
- `src/holosoma/holosoma/config_values/wbt/g1/command.py`（`g1_29dof_wbt_command`、`g1_29dof_wbt_command_w_object`）
- `src/holosoma/holosoma/config_values/wbt/g1/observation.py`（`g1_29dof_wbt_observation`、`g1_29dof_wbt_observation_w_object`）
- `src/holosoma/holosoma/config_values/wbt/g1/reward.py`（`g1_29dof_wbt_reward` 及其他现有 preset）
- `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` 里的现有 4 个 dataclass 实例（只能在末尾**追加**新 entry）
- 所有现有测试文件

---

## Controller 规格（训练 + 部署的输入输出、物理含义、reward）

这一节是给下游使用者看的，不涉及实现细节；想看实现看"设计要点"+"任务分解"。

### 输入（policy 看到什么）

每个 control step（50Hz），actor 看到：

| 分组 | 内容 | 维度 | 说明 |
|---|---|---|---|
| 原 WBT obs | `motion_command`, `motion_ref_ori_b`, `base_ang_vel`, `dof_pos`, `dof_vel`, `actions` | ~继承 | 不变 |
| **新增** | `wrist_force_command` | **6** | body-yaw frame，左右手腕各 3D，单位：N × 0.01 scale |

Critic 额外看特权 obs（`base_lin_vel` 等，也含 `wrist_force_command`），用于价值函数。

**`wrist_force_command` 的物理含义：** 当前 control step，上层系统希望机器人的左（右）手腕在 **body-yaw 方向** 对环境施加多大的力。body-yaw frame 意思是：坐标系原点固定在机器人 base，X 轴指向机器人当前朝向（yaw=0 的方向），Z 垂直向上。不随机器人俯仰/翻滚旋转，只跟随 yaw。

部署时由**上层任务规划器或遥操作系统**提供。F_cmd = 0 表示"不需要施力，按 motion tracking 走"。

### 输出（policy 产生什么）

与原 WBT 完全相同：

| 输出 | 维度 | 含义 |
|---|---|---|
| 关节 position target 残差 | 29（G1 全身 DoF）| `action_scale=0.25`，最终 `q_target = a × 0.25 + q_default` |

**注意：policy 不直接输出力**。它输出关节位置指令，交由低层 PD 控制器（Kp, Kd 固化在 robot config）产生 torque：

```
torque = Kp · (q_target - q_actual) - Kd · q̇_actual
```

### "force control" 的物理含义（最关键的部分）

这个 controller 产生力的机制是 **"virtual impedance"（虚拟阻抗）**，**不是**闭环力控制：

**训练阶段（sim 里，手腕悬空自由运动）：**
- policy 收到 `F_cmd`
- reward 让 policy 把 wrist 驱到 `motion_target + F_cmd / K_virtual` 位置
- 整个过程**没有任何真实力**产生或测量 —— 纯位置任务
- policy 学到一个映射：`F_cmd → 关节构型偏移 → 关节 target → 关节 torque`

**部署阶段（真机 / sim 接触场景）：**

- **自由空间（没接触任何东西）：** 手腕悬在比原 motion target 偏移 `F_cmd / K_virtual` 的位置上。**不产生任何力**，因为没有接触。这很重要 —— 这个 controller **不能在空中"发力"**。
- **接触刚性表面（比如按门、推按钮）：** 手腕想偏到 `motion_target + F_cmd/K_virtual`，但被表面挡住。关节 PD 为了维持 q_target 持续产生 torque，经由 Jacobian 映射到末端，产生真实力：

  ```
  F_real ≈ (K_arm · K_surface) / (K_arm + K_surface) · (F_cmd / K_virtual)
  ```

  **当 K_arm ≈ K_virtual 且 K_surface 足够大（刚性表面）时：`F_real ≈ F_cmd`** ✓
- **接触软表面（海绵、软垫）：** K_surface 小 → 软表面吸收了一部分位置偏移 → F_real < F_cmd。方向仍然正确，幅度偏小。

### 这是 stiff impedance controller，**不是** compliant controller

重要澄清：

- **Stiff（本 plan）：** policy 积极抵抗**任何** wrist 偏离预期位置的外力。人推它它推回来。适合主动施力场景（推门、按按钮、擦黑板、维持接触力）。
- **Compliant（非本 plan）：** policy 对外力柔性让步。人推它它让开。适合人机协作、被动接触场景。

两者物理上互斥，实现路径也完全不同。Plan v1 专门做 stiff。Compliant 需要另起 plan（会用到 GentleHumanoid 的 admittance reference + 外力感知，且很难和 active force tracking 同时达成）。

### 可以做什么、不能做什么

| 任务 | 是否适用 | 用法 |
|---|---|---|
| 机器人推门 | ✓ | motion 让手放门上；F_cmd 指门方向 20N |
| 按按钮 | ✓ | motion 让手停按钮位；F_cmd 指按钮方向 5N 持续 0.5s |
| 擦黑板/桌面 | ✓ | motion 给擦动轨迹；F_cmd 垂直表面 8N |
| 握/抓物体（手腕层面）| 部分 | F_cmd 可提供大方向，但爪子本身需要独立控制 |
| 空中产生力（没接触） | ✗ | 物理上不可能，力必须通过接触存在 |
| 被动吸收外力（人碰机器人）| ✗ | 这个 controller 会抵抗，不会让开 |
| 精确的绝对力追踪（<5% 误差）| ✗ | 依赖 K_virtual 与 K_arm、K_surface 的匹配；一般 20-50% 精度 |
| 已知方向的接触力控制（方向准）| ✓ | 方向永远准确 |

### Reward 公式（训练信号的精确定义）

**新增且唯一的新 reward term：** `wrist_force_position_tracking_exp`（weight=2.0）

```
force_w = yaw_quat(base_quat) · force_cmd_b              [把 body-yaw frame 的 F_cmd 旋到 world frame]
target_shifted_w[wrist] = motion_target_w[wrist] + force_w / K_virtual
error_per_wrist = Σ_xyz (target_shifted_w - robot_actual_w)²
reward = exp( - mean_over_2_wrists(error_per_wrist) / σ² )
```

其中：
- `K_virtual = 200 N/m`（固定）
- `σ = 0.3 m`
- 当 `F_cmd = 0` → 完全等价纯 wrist motion tracking（wrist 对齐 motion target 时 reward = 1.0）
- 当 `F_cmd ≠ 0` → 目标向 F_cmd 方向偏移 `F_cmd / K_virtual` 米

**现有所有 reward term 保持不变**（per 用户 "以 holosoma 为主"）：
- `motion_global_ref_position_error_exp` (weight 0.5)
- `motion_global_ref_orientation_error_exp` (weight 0.5)
- `motion_relative_body_position_error_exp` (weight 1.0)  ← 这个仍对 14 个 body 取 mean 跟踪，包括两个 wrist
- `motion_relative_body_orientation_error_exp` (weight 1.0)
- `motion_global_body_lin_vel` (weight 1.0)
- `motion_global_body_ang_vel` (weight 1.0)
- `action_rate_l2` (weight -0.1)
- `limits_dof_pos` (weight -10)
- `undesired_contacts` (weight -0.1)

**多 reward 共存的含义：** 两个 wrist 同时被**旧 reward**（拉向 motion target）和**新 reward**（拉向 shifted target）约束。新 reward 权重 2.0 远大于旧 reward 对 wrist 的隐含贡献（weight 1.0 / 14 body ≈ 0.07），所以 policy 主导行为是 force tracking。但 F_cmd=0 时两者一致，不会冲突。**这正是 UniFP 的多 reward 共存设计**（UniFP 同时保留 `tracking_ee_world` 和 `tracking_ee_force_world`）。

### 简明流程图

```
Training (sim, no contact, no real forces):

   F_cmd (bc yaw frame) ─┐
                         ▼
   motion_target + yaw_rot(F_cmd) / K_virtual = shifted target
                         ▼
   reward = exp(-‖wrist - shifted_target‖² / σ²)
                         ▼
   policy 学到  F_cmd → 关节构型偏移

Inference (real robot, real contact):

   F_cmd ───┐
            ▼
   policy ─→ 关节 q_target  ─→  PD ─→  torque
                                          │
                                     (与接触面对抗)
                                          │
                                          ▼
                                  F_real ≈ F_cmd  (当 K 匹配 + 刚性接触)
                                  F_real = 0       (无接触)
                                  F_real < F_cmd  (软接触)
```

### 部署时调用方的职责

因为 controller 本身**不感知接触**，上层系统需要：

1. 通过视觉 / 触觉 / 动作脚本判断"当前应该接触到东西了吗"，据此设置 F_cmd。
2. 想"松力"时把 F_cmd 设回 0，让 wrist 回到纯 motion tracking。
3. 如果需要精确力值，部署后做一次 sim 校准（见 §2 "部署端 sanity"），给 F_cmd 乘一个常数补偿系数即可。

---

## 设计要点（关键决策）

### 1. Force 的参考坐标系：body-yaw frame（对齐 UniFP）

- **采样/存储/obs 暴露：** body-yaw frame（即 world frame 绕 Z 轴旋转到"机器人当前朝向 +X"的坐标系）。这样 policy 看到的力指令与机器人当前朝向相关，平移不变。
- **Reward 计算：** 需要与 `body_pos_relative_w`（world frame）同框架，所以先把 `force_cmd_b` 旋转到 world，再加到 wrist target。
- **`base_yaw_quat` 获取：** Holosoma 没有现成的 yaw-only quat 工具函数暴露在 env 级别，但 `MotionCommand` 用了 `yaw_quat`（`holosoma.utils.rotations.yaw_quat`），且 env 有 `base_quat`（`WholeBodyTrackingManager._init_buffers` 设置）。我们在 reward 里实时调用 `yaw_quat(env.base_quat)` 就能得到当前 yaw-only quat，再用 `quat_apply` 旋转 force 到 world。

### 2. Virtual K 的选取（**固定 K = 200 N/m，不做随机化**）

**决策：** 本 plan **固定 `K_virtual = 200 N/m`**，训练 **不** 对 K 做 domain randomization。K 不进入 policy 的 observation。

**K 是什么：** "汇率"参数，把 policy 看到的力指令换算成它要执行的位置偏移：`Δx = F_cmd / K_virtual`。训练时 policy 学到的是"给我 F_cmd，我就让 wrist 偏离 motion target 这么多米"；部署时这个位置偏移通过真实表面接触被动转化为真实力（`F_real ≈ K_arm × Δx`，当 K_arm ≈ K_virtual 时 F_real ≈ F_cmd）。

**为什么选 200：**
- UniFP 用 `gripper_force_kps = 200 N/m`（`b2z1_pos_force_config.py:151`），对应 F_cmd=60N 时偏移 0.3m，UniFP 已在 B2Z1 quadruped+arm 和 G1 humanoid 两种 embodiment 上验证过。
- G1 humanoid 手臂短（0.6-0.8m 可达），本 plan 把 `force_magnitude_range=[5, 30]` N 配 K=200 → 30N 对应 0.15m 偏移，在可达范围内。
- 与 G1 臂的物理有效 Cartesian 刚度 `K_arm ≈ (J^T Kp J)^{-1}` 在同一量级，部署时 `K_arm ≈ K_virtual` 是大概率事件，力追踪精度合理。

**为什么不做随机化（决策记录）：**
- 随机化 K 并不解决"训练 K 与真实物理 K 失配"的问题，只是给部署一个"stiffness 旋钮"。
- 当前 plan 的下游需求是固定场景主动施力（推/按/压），不需要运行时调 stiffness。
- 固定 K 训练最稳、样本效率最高。若后续确实有运行时调 stiffness 需求，再扩展 v2（把 K 加入 observation、用 log-uniform per-episode 采样、σ 自适应等）。

**位置：** K_virtual 是 **reward term** 的参数（Task 4 的 `wrist_force_position_tracking_exp(..., K_virtual: float = 200.0)`），**不** 在 `WristForceConfig` 里。命令采样的 force magnitude 与 K 是正交的。

**部署端 sanity（训练后做一次即可，不改训练）：** 对训好的 policy 在 sim 里让 wrist 接触刚性墙，施加 `F_cmd = 20N (+X)`，测量实际产生的接触力 `F_real`。算比例 `α = F_real / F_cmd`：
- `α ≈ 1`：K_virtual 匹配 OK，直接部署。
- `α ≈ 0.5`：实际 K_arm 偏大，部署时上层给 policy 的 F_cmd 乘 `1/α` 补偿即可（**不用重训**）。
- `α ≤ 0.3` 或 `α ≥ 3`：失配严重，考虑换 K_virtual 重训（分别往 400 或 100 调）。

**单位约定（严格）：** Force = Newton (N), Position = meter (m), K = N/m. obs 里 `wrist_force_command` 再乘 `obs_scale = 0.01`。Reward 里直接用 N 和 m，不额外 scale。

### 3. Force 剖面：ramp up → hold → ramp down（对齐 UniFP `settling_time_force_gripper_s`）

- 每个 env 的每个手腕独立采样：peak magnitude、duration、ramp 时长、cooldown
- UniFP 用 1.0s settling_time；我们默认 ramp_up=ramp_down=0.2s（50Hz env 下即 10 control steps）
- F_cmd 为 0 的间歇期（cooldown）也很重要：policy 需要学会"收力"（放下爪子/停止推动）以应对 F_cmd=0 的目标状态

### 4. F_cmd 的方向采样：单位球面均匀

- 不用 body frame 的 xyz 独立均匀（这会让 magnitudes 集中在 √3× 的立方体上，不均匀）
- 用 `torch.randn + normalize`：球面均匀。magnitude 单独从 uniform 采样。

### 5. Reward 设计：不替换，只新增

**新 reward：** `wrist_force_position_tracking_exp`（weight=2.0，参考 UniFP 同名 reward）

```python
def wrist_force_position_tracking_exp(env, sigma: float) -> torch.Tensor:
    force_cmd_b = env.command_manager.get_state("wrist_force_command").force_cmd_b  # [N, 6] body-yaw frame
    force_cmd_w = ...rotate force_cmd_b to world using yaw_quat(env.base_quat)...   # [N, 6] world frame
    motion_cmd = env.command_manager.get_state("motion_command")
    wrist_target_w = motion_cmd.body_pos_relative_w[:, wrist_idx_in_track_list]     # [N, 2, 3]
    wrist_actual_w = motion_cmd.robot_body_pos_w[:, wrist_idx_in_track_list]        # [N, 2, 3]
    K_virtual = reward cfg param
    shifted_target_w = wrist_target_w + force_cmd_w.view(N, 2, 3) / K_virtual
    error = torch.sum(torch.square(shifted_target_w - wrist_actual_w), dim=-1)      # [N, 2]
    return torch.exp(-error.mean(-1) / sigma**2)                                    # [N]
```

**F_cmd=0 时：** `shifted_target_w == wrist_target_w`，新 reward 完全等价于 wrist 的纯 motion tracking，不会干扰现有行为 ✓

**F_cmd ≠ 0 时：** wrist 的新 target 向 F_cmd 方向偏移。现有 `motion_relative_body_position_error_exp` 会惩罚 wrist 偏离 motion target，新 reward 会奖励 wrist 偏到 shifted target。两者在 wrist 上是"拉扯"关系。**这是有意的**——K_virtual 决定平衡点：新 reward 的 weight 2.0 比现有 per-body 贡献（`weight=1.0 / 14 ≈ 0.07`）大 30 倍，policy 会偏向新 target，但不完全抛弃 motion。这精确对齐 UniFP 的多-reward 共存设计。

现有 `motion_relative_body_position_error_exp` **不改** ——"以 holosoma 为主"。

### 6. K_virtual 作为 reward 参数而非 command 参数

设计上 K_virtual 属于 reward 的 hyperparameter（决定训练信号的尺度），所以放在 `RewardTermCfg.params` 里，不是 `WristForceConfig`。这样后期调参更清晰：改 K_virtual 不影响 F_cmd 的采样。

### 7. 不往 sim 注入力（v1 关键决策）

- **理由：** UniFP 的 F_cmd 本身就是纯训练信号，sim 里不需要对应的真实接触。policy 学到"对于指令 F_cmd，wrist 应该偏移 F_cmd/K"，部署时触接真实表面自然产生力。
- **好处：** 无需新 env 子类，代码最小侵入。
- **代价：** policy 只能训练到"位置偏移"的语义；部署表面刚度与 K_virtual 差异会导致实际力幅度偏差。这是 UniFP 论文已经讨论并接受的。
- **后续扩展（不在本 plan）：** 如果要加 F_ext 扰动（UniFP 的 ext stream）做 robustness 训练，届时再加 env 子类和 `_apply_force_in_physics_step` 覆写。

---

## 任务分解

### Task 1: `WristForceConfig` dataclass

**Files:**
- Modify: `src/holosoma/holosoma/config_types/command.py`（文件末尾追加）
- Create: `src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py`

- [ ] **Step 1: 写失败测试**

Create: `src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py`

```python
"""Tests for WristForceConfig dataclass."""

from __future__ import annotations

import pytest

from holosoma.config_types.command import WristForceConfig


def test_wrist_force_config_defaults_instantiate():
    cfg = WristForceConfig()
    assert cfg.force_magnitude_range == [5.0, 30.0]
    assert cfg.duration_range_s == [0.5, 2.0]
    assert cfg.cooldown_range_s == [0.5, 3.0]
    assert cfg.ramp_up_s == 0.2
    assert cfg.ramp_down_s == 0.2
    assert cfg.activation_prob_per_step == 0.005
    assert cfg.enable_left is True
    assert cfg.enable_right is True
    assert cfg.left_wrist_body_name == "left_wrist_yaw_link"
    assert cfg.right_wrist_body_name == "right_wrist_yaw_link"


def test_wrist_force_config_override():
    cfg = WristForceConfig(
        force_magnitude_range=[10.0, 40.0],
        enable_left=False,
        left_wrist_body_name="custom_left",
    )
    assert cfg.force_magnitude_range == [10.0, 40.0]
    assert cfg.enable_left is False
    assert cfg.left_wrist_body_name == "custom_left"


def test_wrist_force_config_is_frozen():
    cfg = WristForceConfig()
    with pytest.raises(Exception):
        cfg.force_magnitude_range = [0.0, 1.0]  # type: ignore[misc]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py -v`
Expected: `ImportError: cannot import name 'WristForceConfig' from 'holosoma.config_types.command'`

- [ ] **Step 3: 追加 dataclass 到 `src/holosoma/holosoma/config_types/command.py` 末尾**

不要碰前面的类。在文件末尾（`MotionConfig` 之后）追加：

```python
########################################################################################################################
# Wrist force command configuration (UniFP-style active force tracking)
########################################################################################################################
@dataclass(frozen=True)
class WristForceConfig:
    """Configuration for the wrist force command term.

    Samples per-env 3-D force commands at the left and right wrist in a yaw-aligned
    body frame. Follows UniFP's "virtual impedance" paradigm: no force is injected
    into simulation — the command modulates the reward target such that the policy
    learns a position offset F_cmd / K_virtual from the motion target. At deployment,
    contacting a real surface converts this position offset into real force.
    """

    force_magnitude_range: list[float] = field(default_factory=lambda: [5.0, 30.0])
    """Uniform sampling range for peak |F| magnitude (Newtons) of each force episode."""

    duration_range_s: list[float] = field(default_factory=lambda: [0.5, 2.0])
    """Uniform sampling range for total duration of one force episode (seconds)."""

    cooldown_range_s: list[float] = field(default_factory=lambda: [0.5, 3.0])
    """Uniform sampling range for the quiet interval between consecutive force episodes (seconds)."""

    ramp_up_s: float = 0.2
    """Linear ramp-up duration at the start of each force episode (seconds). Aligned to UniFP settling_time_force_gripper_s."""

    ramp_down_s: float = 0.2
    """Linear ramp-down duration at the end of each force episode (seconds)."""

    activation_prob_per_step: float = 0.005
    """Probability per control step that a wrist in cooldown activates a new force episode."""

    enable_left: bool = True
    """If True, sample force episodes on the left wrist."""

    enable_right: bool = True
    """If True, sample force episodes on the right wrist."""

    left_wrist_body_name: str = "left_wrist_yaw_link"
    """Body name (must exist in the robot's body_names) for the left wrist."""

    right_wrist_body_name: str = "right_wrist_yaw_link"
    """Body name for the right wrist."""
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py -v`
Expected: `3 passed`

- [ ] **Step 5: 回归测试**

Run: `pytest src/holosoma/holosoma/config_types/tests/ -v`
Expected: 所有现有测试仍然通过

- [ ] **Step 6: Commit**

```bash
git add src/holosoma/holosoma/config_types/command.py src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py
git commit -m "feat(wbt-force): add WristForceConfig dataclass"
```

---

### Task 2: `WristForceCommand` — 力剖面采样

**Files:**
- Create: `src/holosoma/holosoma/managers/command/terms/wbt_force.py`
- Create: `src/holosoma/holosoma/managers/command/terms/tests/__init__.py`（空）
- Create: `src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force.py`

- [ ] **Step 1: 写失败测试**

Create: `src/holosoma/holosoma/managers/command/terms/tests/__init__.py` (空文件)

Create: `src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force.py`

```python
"""Unit tests for WristForceCommand (pure CPU, no IsaacSim)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.command import CommandTermCfg, WristForceConfig
from holosoma.managers.command.terms.wbt_force import WristForceCommand


def _make_env(num_envs=4, device="cpu"):
    body_names = [
        "pelvis",
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    sim = SimpleNamespace(_body_list=body_names)
    env = SimpleNamespace(
        num_envs=num_envs,
        device=device,
        dt=0.02,
        simulator=sim,
    )
    return env


def _make_term(**force_kwargs):
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force:WristForceCommand",
        params={"wrist_force_config": WristForceConfig(**force_kwargs)},
    )
    return cfg


def test_setup_initializes_buffers():
    env = _make_env(num_envs=4)
    term = WristForceCommand(_make_term(), env)
    term.setup()
    assert term.force_cmd_b.shape == (4, 6)
    assert torch.all(term.force_cmd_b == 0)
    assert term.left_wrist_body_id == 2
    assert term.right_wrist_body_id == 3


def test_step_with_zero_activation_keeps_force_at_zero():
    env = _make_env(num_envs=4)
    term = WristForceCommand(_make_term(activation_prob_per_step=0.0), env)
    term.setup()
    for _ in range(50):
        term.step()
    assert torch.all(term.force_cmd_b == 0)


def test_ramp_profile_monotonic_peak_at_middle():
    env = _make_env(num_envs=1)
    term = WristForceCommand(
        _make_term(
            force_magnitude_range=[10.0, 10.0],
            duration_range_s=[1.0, 1.0],
            cooldown_range_s=[100.0, 100.0],
            ramp_up_s=0.2,
            ramp_down_s=0.2,
            activation_prob_per_step=1.0,
            enable_right=False,
        ),
        env,
    )
    term.setup()
    magnitudes = []
    for _ in range(60):  # 60 * 0.02s = 1.2s
        term.step()
        magnitudes.append(torch.linalg.norm(term.force_cmd_b[0, :3]).item())
    # Peak 10.0 somewhere in the middle of the episode.
    assert max(magnitudes) == pytest.approx(10.0, abs=1e-4)
    # Ramp-down ends the episode at or near zero, and cooldown keeps it at zero.
    assert magnitudes[-1] == pytest.approx(0.0, abs=1e-4)


def test_sampled_force_magnitude_in_range():
    """Peak |F| must lie within the configured magnitude range."""
    env = _make_env(num_envs=64)
    term = WristForceCommand(
        _make_term(
            force_magnitude_range=[8.0, 12.0],
            duration_range_s=[1.0, 1.0],
            cooldown_range_s=[100.0, 100.0],
            ramp_up_s=0.0,
            ramp_down_s=0.0,
            activation_prob_per_step=1.0,
        ),
        env,
    )
    term.setup()
    # Step once to activate + produce peak force.
    term.step()
    mags = torch.linalg.norm(term.force_cmd_b.view(64, 2, 3), dim=-1)
    active = mags > 1e-6
    assert active.any(), "Expected some envs to have active force after guaranteed activation"
    active_mags = mags[active]
    assert (active_mags >= 7.99).all()
    assert (active_mags <= 12.01).all()


def test_reset_clears_only_specified_envs():
    env = _make_env(num_envs=4)
    term = WristForceCommand(
        _make_term(
            force_magnitude_range=[20.0, 20.0],
            activation_prob_per_step=1.0,
            ramp_up_s=0.2,
            duration_range_s=[2.0, 2.0],
        ),
        env,
    )
    term.setup()
    for _ in range(20):
        term.step()
    pre = term.force_cmd_b.clone()
    assert (pre != 0).any()
    term.reset(torch.tensor([0, 1], device=env.device))
    assert torch.all(term.force_cmd_b[0] == 0)
    assert torch.all(term.force_cmd_b[1] == 0)
    assert torch.equal(term.force_cmd_b[2:], pre[2:])


def test_enable_left_false_keeps_left_zero():
    env = _make_env(num_envs=2)
    term = WristForceCommand(
        _make_term(
            enable_left=False,
            activation_prob_per_step=1.0,
            force_magnitude_range=[20.0, 20.0],
        ),
        env,
    )
    term.setup()
    for _ in range(30):
        term.step()
    assert torch.all(term.force_cmd_b[:, :3] == 0)
    assert (term.force_cmd_b[:, 3:] != 0).any()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force.py -v`
Expected: `ModuleNotFoundError: No module named 'holosoma.managers.command.terms.wbt_force'`

- [ ] **Step 3: 实现 `WristForceCommand`**

Create: `src/holosoma/holosoma/managers/command/terms/wbt_force.py`

```python
"""Wrist force command term for UniFP-style active force tracking.

This term samples per-env force episodes on the left and right wrist and exposes
the commanded force as state in a yaw-aligned body frame (``force_cmd_b``, 6-D:
``[fx_left, fy_left, fz_left, fx_right, fy_right, fz_right]``).

The force command is a *training signal only* — it is NOT injected into the
simulator. The reward term uses it to shift the motion tracking target by
``F_cmd / K_virtual``, implementing UniFP's virtual impedance formulation
(paper Eq. 2 / code ``_reward_tracking_ee_force_world`` at
``third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891``).
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from holosoma.config_types.command import CommandTermCfg, WristForceConfig
from holosoma.managers.command.base import CommandTermBase


def _ensure_config(raw: Any) -> WristForceConfig:
    """Recover a WristForceConfig from either a dataclass instance or a dict.

    tyro may deliver the value as a dict after CLI round-trip. Mirrors the
    pattern used in MotionCommand.__init__ at
    ``src/holosoma/holosoma/managers/command/terms/wbt.py:517-521``.
    """
    if isinstance(raw, WristForceConfig):
        return raw
    return WristForceConfig(**raw)


class WristForceCommand(CommandTermBase):
    """Samples random wrist force episodes and exposes a yaw-body-frame 6-D command.

    Profile per wrist per episode: linear ramp-up -> hold -> linear ramp-down, then
    a uniform-sampled cooldown before the next Bernoulli-gated activation.
    """

    def __init__(self, cfg: CommandTermCfg, env: Any):
        super().__init__(cfg, env)
        self._env = env
        self.cfg_force: WristForceConfig = _ensure_config(cfg.params["wrist_force_config"])

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        env = self._env
        self.num_envs = env.num_envs
        self.device = env.device

        body_list = list(env.simulator._body_list)
        try:
            self.left_wrist_body_id = body_list.index(self.cfg_force.left_wrist_body_name)
        except ValueError as exc:  # pragma: no cover - config error
            raise ValueError(
                f"Left wrist body '{self.cfg_force.left_wrist_body_name}' not found in robot body_list: {body_list}"
            ) from exc
        try:
            self.right_wrist_body_id = body_list.index(self.cfg_force.right_wrist_body_name)
        except ValueError as exc:  # pragma: no cover - config error
            raise ValueError(
                f"Right wrist body '{self.cfg_force.right_wrist_body_name}' not found in robot body_list: {body_list}"
            ) from exc

        self.dt = float(env.dt)
        self.ramp_up_steps = max(int(round(self.cfg_force.ramp_up_s / self.dt)), 0)
        self.ramp_down_steps = max(int(round(self.cfg_force.ramp_down_s / self.dt)), 0)

        num_envs = self.num_envs
        dev = self.device

        # 2 channels: 0=left, 1=right.
        self.active = torch.zeros(num_envs, 2, dtype=torch.bool, device=dev)
        self.phase_counter = torch.zeros(num_envs, 2, dtype=torch.long, device=dev)
        self.total_steps = torch.zeros(num_envs, 2, dtype=torch.long, device=dev)
        self.cooldown_counter = torch.zeros(num_envs, 2, dtype=torch.long, device=dev)
        self.peak_force_b = torch.zeros(num_envs, 2, 3, dtype=torch.float32, device=dev)

        # Public: current per-step force command in body-yaw frame (6-D flat).
        self.force_cmd_b = torch.zeros(num_envs, 6, dtype=torch.float32, device=dev)

        self.enable_mask = torch.tensor(
            [self.cfg_force.enable_left, self.cfg_force.enable_right],
            dtype=torch.bool,
            device=dev,
        )

        logger.info(
            f"[WristForceCommand] num_envs={num_envs} left_body_id={self.left_wrist_body_id} "
            f"right_body_id={self.right_wrist_body_id} ramp_up_steps={self.ramp_up_steps} "
            f"ramp_down_steps={self.ramp_down_steps} enable_left={self.cfg_force.enable_left} "
            f"enable_right={self.cfg_force.enable_right}"
        )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        idx = self._ensure_env_ids(env_ids)
        if idx.numel() == 0:
            return
        self.active[idx] = False
        self.phase_counter[idx] = 0
        self.total_steps[idx] = 0
        self.cooldown_counter[idx] = 0
        self.peak_force_b[idx] = 0.0
        self.force_cmd_b[idx] = 0.0

    def step(self) -> None:
        self._maybe_activate_new_episodes()
        self._handle_channel_terminations()
        self._advance_active_channels()

    # ------------------------------------------------------------------ #
    # Helpers (step internals)
    # ------------------------------------------------------------------ #
    def _maybe_activate_new_episodes(self) -> None:
        ready = (~self.active) & (self.cooldown_counter <= 0) & self.enable_mask.unsqueeze(0)
        if not ready.any():
            return
        rand = torch.rand_like(self.cooldown_counter, dtype=torch.float32)
        activate = ready & (rand < float(self.cfg_force.activation_prob_per_step))
        if not activate.any():
            return

        n_act = int(activate.sum().item())
        mag_lo, mag_hi = self.cfg_force.force_magnitude_range
        dur_lo, dur_hi = self.cfg_force.duration_range_s

        magnitudes = torch.empty(n_act, device=self.device).uniform_(float(mag_lo), float(mag_hi))
        durations_s = torch.empty(n_act, device=self.device).uniform_(float(dur_lo), float(dur_hi))
        min_duration_steps = max(self.ramp_up_steps + self.ramp_down_steps + 1, 1)
        durations_steps = (durations_s / self.dt).round().clamp(min=float(min_duration_steps)).to(torch.long)

        # Uniform direction on the unit sphere, then scale by magnitude.
        dirs = torch.randn(n_act, 3, device=self.device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        peak = dirs * magnitudes.unsqueeze(-1)

        self.active[activate] = True
        self.phase_counter[activate] = 0
        self.total_steps[activate] = durations_steps
        self.peak_force_b[activate] = peak

    def _handle_channel_terminations(self) -> None:
        done = self.active & (self.phase_counter >= self.total_steps)
        if done.any():
            self.active = self.active & ~done
            self.phase_counter = torch.where(done, torch.zeros_like(self.phase_counter), self.phase_counter)
            cd_lo, cd_hi = self.cfg_force.cooldown_range_s
            cd_lo_steps = max(int(round(cd_lo / self.dt)), 1)
            cd_hi_steps = max(int(round(cd_hi / self.dt)), cd_lo_steps)
            n_done = int(done.sum().item())
            new_cd = torch.randint(cd_lo_steps, cd_hi_steps + 1, (n_done,), device=self.device, dtype=torch.long)
            self.cooldown_counter[done] = new_cd
        inactive = ~self.active
        self.cooldown_counter = torch.where(
            inactive & (self.cooldown_counter > 0), self.cooldown_counter - 1, self.cooldown_counter
        )

    def _advance_active_channels(self) -> None:
        active = self.active
        phase = self.phase_counter
        total = self.total_steps
        ru = self.ramp_up_steps
        rd = self.ramp_down_steps

        scale = torch.zeros_like(self.phase_counter, dtype=torch.float32)
        active_idx = active.nonzero(as_tuple=False)
        if active_idx.numel() > 0:
            e = active_idx[:, 0]
            w = active_idx[:, 1]
            s = phase[e, w].float()
            t = total[e, w].float()
            hold_end = (t - rd).clamp(min=float(ru))

            ramp_up_val = s / float(max(ru, 1))
            hold_val = torch.ones_like(s)
            ramp_down_val = (t - s).clamp(min=0.0) / float(max(rd, 1))

            if ru > 0:
                in_ramp_up = s < ru
            else:
                in_ramp_up = torch.zeros_like(s, dtype=torch.bool)
            in_hold = (~in_ramp_up) & (s < hold_end)

            scaled = torch.where(
                in_ramp_up,
                ramp_up_val,
                torch.where(in_hold, hold_val, ramp_down_val),
            ).clamp(0.0, 1.0)
            scale[e, w] = scaled

        scale = scale * self.enable_mask.unsqueeze(0).to(scale.dtype)
        force = self.peak_force_b * scale.unsqueeze(-1)
        self.force_cmd_b[:] = force.reshape(self.num_envs, 6)

        self.phase_counter = torch.where(active, self.phase_counter + 1, self.phase_counter)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _ensure_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Accessors (consumed by observation term and reward term)
    # ------------------------------------------------------------------ #
    @property
    def force_left_b(self) -> torch.Tensor:
        """Left wrist force in yaw-aligned body frame, [num_envs, 3]."""
        return self.force_cmd_b[:, :3]

    @property
    def force_right_b(self) -> torch.Tensor:
        """Right wrist force in yaw-aligned body frame, [num_envs, 3]."""
        return self.force_cmd_b[:, 3:]
```

- [ ] **Step 4: 运行全部单元测试确认通过**

Run: `pytest src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/managers/command/terms/wbt_force.py src/holosoma/holosoma/managers/command/terms/tests/
git commit -m "feat(wbt-force): add WristForceCommand with ramp force profile"
```

---

### Task 3: `wrist_force_command` observation term

**Files:**
- Create: `src/holosoma/holosoma/managers/observation/terms/wbt_force.py`
- Create: `src/holosoma/holosoma/managers/observation/terms/tests/__init__.py`（空）
- Create: `src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force.py`

- [ ] **Step 1: 写失败测试**

Create: `src/holosoma/holosoma/managers/observation/terms/tests/__init__.py` (空)

Create: `src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force.py`

```python
"""Unit tests for wrist_force_command observation term."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.observation.terms.wbt_force import wrist_force_command


class _FakeWristCommand:
    def __init__(self, force: torch.Tensor):
        self.force_cmd_b = force


class _FakeCommandManager:
    def __init__(self, state):
        self._state = state

    def get_state(self, name):
        return self._state


def test_wrist_force_command_returns_6d_tensor():
    force = torch.tensor([[1.0, 2.0, 3.0, -1.0, -2.0, -3.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    env = SimpleNamespace(num_envs=2, command_manager=_FakeCommandManager(_FakeWristCommand(force)))
    out = wrist_force_command(env)
    assert out.shape == (2, 6)
    assert torch.allclose(out, force)


def test_wrist_force_command_missing_term_raises():
    env = SimpleNamespace(num_envs=1, command_manager=_FakeCommandManager(None))
    with pytest.raises(AssertionError):
        wrist_force_command(env)
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force.py -v`
Expected: `ModuleNotFoundError: No module named 'holosoma.managers.observation.terms.wbt_force'`

- [ ] **Step 3: 实现 obs term**

Create: `src/holosoma/holosoma/managers/observation/terms/wbt_force.py`

```python
"""Observation terms for WBT-with-force training."""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.wbt_force import WristForceCommand


def _get_wrist_force_command(env) -> WristForceCommand:
    term = env.command_manager.get_state("wrist_force_command")
    assert term is not None, "wrist_force_command state not found in command manager"
    assert isinstance(term, WristForceCommand), f"Expected WristForceCommand, got {type(term)}"
    return term


def wrist_force_command(env) -> torch.Tensor:
    """Current wrist force command in yaw-aligned body frame.

    Shape ``[num_envs, 6]``: ``[fx_left, fy_left, fz_left, fx_right, fy_right, fz_right]``.
    This is the same frame used by UniFP's ``current_Fxyz_gripper_cmd`` before
    rotation to the world frame via ``base_yaw_quat``.
    """
    return _get_wrist_force_command(env).force_cmd_b
```

- [ ] **Step 4: 运行**

Run: `pytest src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/managers/observation/terms/wbt_force.py src/holosoma/holosoma/managers/observation/terms/tests/
git commit -m "feat(wbt-force): add wrist_force_command observation term"
```

---

### Task 4: `wrist_force_position_tracking_exp` reward term

**Files:**
- Create: `src/holosoma/holosoma/managers/reward/terms/wbt_force.py`
- Create: `src/holosoma/holosoma/managers/reward/terms/tests/__init__.py`（空）
- Create: `src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force.py`

- [ ] **Step 1: 写失败测试**

Create: `src/holosoma/holosoma/managers/reward/terms/tests/__init__.py` (空)

Create: `src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force.py`

```python
"""Unit tests for wrist_force_position_tracking_exp reward.

We mock MotionCommand and WristForceCommand so the test is pure CPU and does not
require IsaacSim. The key invariants:

- With F_cmd = 0, the reward reduces to pure wrist position tracking (exp(0) = 1
  when wrist matches motion target).
- With F_cmd != 0, the "target" for the wrist shifts by F_cmd_w / K_virtual, and
  a robot whose actual wrist is at that shifted target still gets reward exp(0) = 1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.reward.terms.wbt_force import wrist_force_position_tracking_exp


class _FakeWristCommand:
    def __init__(self, force_cmd_b: torch.Tensor):
        self.force_cmd_b = force_cmd_b


class _FakeMotionCommand:
    def __init__(self, body_names_to_track, body_pos_relative_w, robot_body_pos_w):
        # The fake only needs these attributes; motion_cfg with body_names_to_track.
        self.motion_cfg = SimpleNamespace(body_names_to_track=body_names_to_track)
        self.body_pos_relative_w = body_pos_relative_w
        self.robot_body_pos_w = robot_body_pos_w


class _FakeCommandManager:
    def __init__(self, motion=None, force=None):
        self._states = {"motion_command": motion, "wrist_force_command": force}

    def get_state(self, name):
        return self._states.get(name)


def _make_env(N, body_names, body_relative_w, robot_w, force_b, base_quat_xyzw):
    motion = _FakeMotionCommand(body_names, body_relative_w, robot_w)
    force = _FakeWristCommand(force_b)
    return SimpleNamespace(
        num_envs=N,
        device="cpu",
        base_quat=base_quat_xyzw,
        command_manager=_FakeCommandManager(motion=motion, force=force),
    )


def _identity_quat(N):
    # xyzw identity quaternion
    q = torch.zeros(N, 4)
    q[:, 3] = 1.0
    return q


BODY_NAMES = ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]


def test_reward_one_when_wrist_matches_target_and_force_zero():
    N = 4
    body_relative = torch.zeros(N, 3, 3)
    body_relative[:, 1] = torch.tensor([0.3, 0.1, 0.8])    # left wrist target
    body_relative[:, 2] = torch.tensor([-0.3, 0.1, 0.8])   # right wrist target
    robot = body_relative.clone()                          # perfect tracking
    force_b = torch.zeros(N, 6)                            # no force command
    env = _make_env(N, BODY_NAMES, body_relative, robot, force_b, _identity_quat(N))
    reward = wrist_force_position_tracking_exp(
        env, sigma=0.3, K_virtual=200.0,
        left_wrist_body_name="left_wrist_yaw_link",
        right_wrist_body_name="right_wrist_yaw_link",
    )
    assert reward.shape == (N,)
    assert torch.allclose(reward, torch.ones(N), atol=1e-5)


def test_reward_one_when_wrist_at_shifted_target_under_force():
    N = 1
    body_relative = torch.zeros(N, 3, 3)
    body_relative[:, 1] = torch.tensor([0.3, 0.1, 0.8])
    body_relative[:, 2] = torch.tensor([-0.3, 0.1, 0.8])
    # F_cmd = (10, 0, 0) at left wrist, (0, 10, 0) at right wrist; K=200.
    # Expected wrist offset: left (0.05, 0, 0); right (0, 0.05, 0).
    force_b = torch.tensor([[10.0, 0.0, 0.0, 0.0, 10.0, 0.0]])
    robot = body_relative.clone()
    robot[:, 1] = body_relative[:, 1] + torch.tensor([0.05, 0.0, 0.0])
    robot[:, 2] = body_relative[:, 2] + torch.tensor([0.0, 0.05, 0.0])
    env = _make_env(N, BODY_NAMES, body_relative, robot, force_b, _identity_quat(N))
    reward = wrist_force_position_tracking_exp(
        env, sigma=0.3, K_virtual=200.0,
        left_wrist_body_name="left_wrist_yaw_link",
        right_wrist_body_name="right_wrist_yaw_link",
    )
    assert torch.allclose(reward, torch.ones(N), atol=1e-5)


def test_reward_decays_when_wrist_off_target():
    N = 1
    body_relative = torch.zeros(N, 3, 3)
    body_relative[:, 1] = torch.tensor([0.3, 0.1, 0.8])
    body_relative[:, 2] = torch.tensor([-0.3, 0.1, 0.8])
    # Wrist is 0.15 m off (large error) with zero force command.
    robot = body_relative.clone()
    robot[:, 1] += torch.tensor([0.15, 0.0, 0.0])
    force_b = torch.zeros(N, 6)
    env = _make_env(N, BODY_NAMES, body_relative, robot, force_b, _identity_quat(N))
    reward = wrist_force_position_tracking_exp(
        env, sigma=0.3, K_virtual=200.0,
        left_wrist_body_name="left_wrist_yaw_link",
        right_wrist_body_name="right_wrist_yaw_link",
    )
    assert reward.item() < 0.95
    assert reward.item() > 0.0


def test_reward_yaw_quat_rotation_applied_to_force():
    """A body-yaw-frame force along +X must become world +Y when robot yaw is +pi/2."""
    N = 1
    body_relative = torch.zeros(N, 3, 3)
    body_relative[:, 1] = torch.tensor([0.0, 0.0, 1.0])
    body_relative[:, 2] = torch.tensor([0.0, 0.0, 1.0])
    # xyzw quat for yaw=pi/2: (0, 0, sin(pi/4), cos(pi/4)).
    import math
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    base_quat = torch.tensor([[0.0, 0.0, s, c]])
    # Force only on left wrist: body frame (+X, 0, 0), magnitude 10.
    force_b = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # With yaw=pi/2, body +X -> world +Y. Offset at left wrist should be (0, 10/200, 0)=(0, 0.05, 0).
    robot = body_relative.clone()
    robot[:, 1] = body_relative[:, 1] + torch.tensor([0.0, 0.05, 0.0])
    env = _make_env(N, BODY_NAMES, body_relative, robot, force_b, base_quat)
    reward = wrist_force_position_tracking_exp(
        env, sigma=0.3, K_virtual=200.0,
        left_wrist_body_name="left_wrist_yaw_link",
        right_wrist_body_name="right_wrist_yaw_link",
    )
    assert torch.allclose(reward, torch.ones(N), atol=1e-4)
```

- [ ] **Step 2: 运行验证失败**

Run: `pytest src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force.py -v`
Expected: `ModuleNotFoundError: No module named 'holosoma.managers.reward.terms.wbt_force'`

- [ ] **Step 3: 实现 reward term**

Create: `src/holosoma/holosoma/managers/reward/terms/wbt_force.py`

```python
"""Reward term for UniFP-style wrist force tracking via virtual impedance.

Formula (mirrors UniFP ``_reward_tracking_ee_force_world`` at
``third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891``):

    target_w_shifted = body_pos_relative_w[wrist] + rotate_to_world(F_cmd_b) / K_virtual
    error = sum_xyz((target_w_shifted - robot_body_pos_w[wrist])^2)
    reward = exp(-mean_over_2_wrists(error) / sigma**2)

With F_cmd = 0, this reduces exactly to pure wrist motion tracking (reward=1 when
wrist matches motion target).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.command.terms.wbt_force import WristForceCommand
from holosoma.utils.rotations import quat_apply, yaw_quat

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command(env) -> MotionCommand:
    term = env.command_manager.get_state("motion_command")
    assert term is not None, "motion_command not found in command manager"
    assert isinstance(term, MotionCommand), f"Expected MotionCommand, got {type(term)}"
    return term


def _get_wrist_force_command(env) -> WristForceCommand:
    term = env.command_manager.get_state("wrist_force_command")
    assert term is not None, "wrist_force_command not found in command manager"
    assert isinstance(term, WristForceCommand), f"Expected WristForceCommand, got {type(term)}"
    return term


def _resolve_wrist_indexes_in_tracked(env, left_name: str, right_name: str) -> tuple[int, int]:
    motion = _get_motion_command(env)
    tracked = list(motion.motion_cfg.body_names_to_track)
    try:
        left_idx = tracked.index(left_name)
    except ValueError as exc:
        raise ValueError(
            f"'{left_name}' must appear in motion_cfg.body_names_to_track; got {tracked}"
        ) from exc
    try:
        right_idx = tracked.index(right_name)
    except ValueError as exc:
        raise ValueError(
            f"'{right_name}' must appear in motion_cfg.body_names_to_track; got {tracked}"
        ) from exc
    return left_idx, right_idx


def wrist_force_position_tracking_exp(
    env,
    sigma: float,
    K_virtual: float = 200.0,
    left_wrist_body_name: str = "left_wrist_yaw_link",
    right_wrist_body_name: str = "right_wrist_yaw_link",
) -> torch.Tensor:
    """Virtual-impedance wrist force tracking reward.

    Args:
        env: The environment (must have a motion_command state and a
            wrist_force_command state registered on its command_manager).
        sigma: Exponential kernel bandwidth (meters). Same units as UniFP's
            ``tracking_ee_sigma``.
        K_virtual: Virtual stiffness (N/m) used to convert force command into a
            position offset. UniFP default is 200 N/m.
        left_wrist_body_name: Body name to resolve within
            ``motion_command.motion_cfg.body_names_to_track``.
        right_wrist_body_name: Same for right wrist.

    Returns:
        Reward tensor of shape ``[num_envs]``.
    """
    motion = _get_motion_command(env)
    wrist_cmd = _get_wrist_force_command(env)

    left_tracked_idx, right_tracked_idx = _resolve_wrist_indexes_in_tracked(
        env, left_wrist_body_name, right_wrist_body_name
    )

    # Body-yaw frame commands -> world frame.
    force_b = wrist_cmd.force_cmd_b.view(-1, 2, 3)  # [N, 2, 3]
    yq = yaw_quat(env.base_quat, w_last=True)       # xyzw yaw-only quat
    # quat_apply broadcasts over the body axis by repeating the quaternion.
    force_w_left = quat_apply(yq, force_b[:, 0], w_last=True)
    force_w_right = quat_apply(yq, force_b[:, 1], w_last=True)
    force_w = torch.stack([force_w_left, force_w_right], dim=1)  # [N, 2, 3]

    # Target and actual wrist positions (world frame).
    wrist_target_w = motion.body_pos_relative_w[:, [left_tracked_idx, right_tracked_idx], :]  # [N, 2, 3]
    wrist_actual_w = motion.robot_body_pos_w[:, [left_tracked_idx, right_tracked_idx], :]     # [N, 2, 3]

    # UniFP virtual impedance: target shifted by F / K.
    wrist_target_shifted_w = wrist_target_w + force_w / float(K_virtual)

    error = torch.sum(torch.square(wrist_target_shifted_w - wrist_actual_w), dim=-1)  # [N, 2]
    return torch.exp(-error.mean(-1) / (sigma ** 2))
```

- [ ] **Step 4: 运行**

Run: `pytest src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force.py -v`
Expected: `4 passed`

- [ ] **Step 5: 整个 managers 子树测试没回归**

Run: `pytest src/holosoma/holosoma/managers/ -v`
Expected: 所有现有测试仍通过

- [ ] **Step 6: Commit**

```bash
git add src/holosoma/holosoma/managers/reward/terms/wbt_force.py src/holosoma/holosoma/managers/reward/terms/tests/
git commit -m "feat(wbt-force): add wrist_force_position_tracking_exp reward (UniFP virtual impedance)"
```

---

### Task 5: `g1_29dof_wbt_force_command` preset

**Files:**
- Create: `src/holosoma/holosoma/config_values/wbt/g1/command_force.py`
- Modify: `src/holosoma/holosoma/config_values/command.py`
- Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/__init__.py`（空）
- Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_command_force.py`

- [ ] **Step 1: 创建 preset 文件**

Create: `src/holosoma/holosoma/config_values/wbt/g1/command_force.py`

```python
"""WBT-with-force command preset for the G1 robot."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, WristForceConfig
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command

wrist_force_config = WristForceConfig()  # defaults: ±[5, 30] N, 0.5-2.0s, ramp 0.2s

g1_29dof_wbt_force_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        **g1_29dof_wbt_command.setup_terms,
        "wrist_force_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_force:WristForceCommand",
            params={"wrist_force_config": wrist_force_config},
        ),
    },
    reset_terms={
        **g1_29dof_wbt_command.reset_terms,
        "wrist_force_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_force:WristForceCommand",
        ),
    },
    step_terms={
        **g1_29dof_wbt_command.step_terms,
        "wrist_force_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_force:WristForceCommand",
        ),
    },
)

__all__ = ["g1_29dof_wbt_force_command"]
```

- [ ] **Step 2: 在 `src/holosoma/holosoma/config_values/command.py` 的 `DEFAULTS` 注册**

先 read 现有文件：

```bash
cat src/holosoma/holosoma/config_values/command.py
```

在导入块追加：

```python
from holosoma.config_values.wbt.g1.command_force import g1_29dof_wbt_force_command
```

在 `DEFAULTS` dict 里追加条目（放在 `"g1_29dof_wbt_w_object"` 后面、末尾 `}` 前）：

```python
    "g1_29dof_wbt_force": g1_29dof_wbt_force_command,
```

- [ ] **Step 3: 写 preset 回归测试**

Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/__init__.py` (空)

Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_command_force.py`

```python
"""Regression tests for g1_29dof_wbt_force_command preset."""

from holosoma.config_types.command import WristForceConfig
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command
from holosoma.config_values.wbt.g1.command_force import g1_29dof_wbt_force_command


def test_force_command_preserves_motion_command():
    assert "motion_command" in g1_29dof_wbt_force_command.setup_terms
    assert "motion_command" in g1_29dof_wbt_force_command.reset_terms
    assert "motion_command" in g1_29dof_wbt_force_command.step_terms


def test_force_command_adds_wrist_force_command():
    for grp in (
        g1_29dof_wbt_force_command.setup_terms,
        g1_29dof_wbt_force_command.reset_terms,
        g1_29dof_wbt_force_command.step_terms,
    ):
        assert "wrist_force_command" in grp

    setup = g1_29dof_wbt_force_command.setup_terms["wrist_force_command"]
    assert setup.func == "holosoma.managers.command.terms.wbt_force:WristForceCommand"
    assert isinstance(setup.params["wrist_force_config"], WristForceConfig)


def test_original_wbt_command_unchanged():
    assert "wrist_force_command" not in g1_29dof_wbt_command.setup_terms
    assert "wrist_force_command" not in g1_29dof_wbt_command.reset_terms
    assert "wrist_force_command" not in g1_29dof_wbt_command.step_terms
```

- [ ] **Step 4: 运行**

Run: `pytest src/holosoma/holosoma/config_values/wbt/g1/tests/test_command_force.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/command_force.py src/holosoma/holosoma/config_values/command.py src/holosoma/holosoma/config_values/wbt/g1/tests/__init__.py src/holosoma/holosoma/config_values/wbt/g1/tests/test_command_force.py
git commit -m "feat(wbt-force): add g1_29dof_wbt_force_command preset"
```

---

### Task 6: `g1_29dof_wbt_force_observation` preset

**Files:**
- Create: `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py`
- Modify: `src/holosoma/holosoma/config_values/observation.py`
- Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_observation_force.py`

- [ ] **Step 1: 创建 preset**

Create: `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py`

```python
"""Observation preset for WBT-with-force training (G1)."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    critic_obs_shared_terms,
)

# 0.01 maps +/-50 N to roughly +/-0.5 units, keeping the signal in a similar
# range to dof_pos (~+/-1). Matches UniFP's ee_force/base_force obs scale.
_WRIST_FORCE_OBS_SCALE = 0.01

actor_obs_force_terms = {
    **actor_obs_shared.terms,
    "wrist_force_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt_force:wrist_force_command",
        scale=_WRIST_FORCE_OBS_SCALE,
        noise=0.0,
    ),
}

critic_obs_force_terms = {
    **critic_obs_shared_terms,
    "wrist_force_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt_force:wrist_force_command",
        scale=_WRIST_FORCE_OBS_SCALE,
        noise=0.0,
    ),
}

g1_29dof_wbt_force_observation = ObservationManagerCfg(
    groups={
        "actor_obs": replace(actor_obs_shared, terms=actor_obs_force_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_force_terms,
        ),
    },
)

__all__ = ["g1_29dof_wbt_force_observation"]
```

- [ ] **Step 2: 在 `src/holosoma/holosoma/config_values/observation.py` 注册**

Add import:
```python
from holosoma.config_values.wbt.g1.observation_force import g1_29dof_wbt_force_observation
```

Add to `DEFAULTS` dict:
```python
    "g1_29dof_wbt_force": g1_29dof_wbt_force_observation,
```

- [ ] **Step 3: 写回归测试**

Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_observation_force.py`

```python
"""Regression tests for g1_29dof_wbt_force_observation preset."""

from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation
from holosoma.config_values.wbt.g1.observation_force import g1_29dof_wbt_force_observation


def test_force_obs_adds_wrist_force_command_actor_and_critic():
    actor = g1_29dof_wbt_force_observation.groups["actor_obs"]
    critic = g1_29dof_wbt_force_observation.groups["critic_obs"]
    assert "wrist_force_command" in actor.terms
    assert "wrist_force_command" in critic.terms
    assert actor.terms["wrist_force_command"].scale == 0.01
    assert actor.terms["wrist_force_command"].noise == 0.0


def test_force_obs_preserves_base_actor_terms():
    actor = g1_29dof_wbt_force_observation.groups["actor_obs"]
    for base_term in g1_29dof_wbt_observation.groups["actor_obs"].terms:
        assert base_term in actor.terms


def test_original_wbt_observation_unchanged():
    assert "wrist_force_command" not in g1_29dof_wbt_observation.groups["actor_obs"].terms
    assert "wrist_force_command" not in g1_29dof_wbt_observation.groups["critic_obs"].terms
```

- [ ] **Step 4: 运行**

Run: `pytest src/holosoma/holosoma/config_values/wbt/g1/tests/test_observation_force.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/observation_force.py src/holosoma/holosoma/config_values/observation.py src/holosoma/holosoma/config_values/wbt/g1/tests/test_observation_force.py
git commit -m "feat(wbt-force): add g1_29dof_wbt_force observation preset"
```

---

### Task 7: `g1_29dof_wbt_force_reward` preset

**Files:**
- Create: `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py`
- Modify: `src/holosoma/holosoma/config_values/reward.py`
- Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_reward_force.py`

- [ ] **Step 1: 创建 preset**

Create: `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py`

```python
"""Reward preset for WBT-with-force training (G1).

Extends g1_29dof_wbt_reward by appending ``wrist_force_position_tracking_exp`` at
weight 2.0 (matching UniFP's ``tracking_ee_force_world`` weight of +2.0 in
``third_party/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py`` reward block).
All existing reward terms are kept unchanged — per user directive "Holosoma loss
to remain primary".
"""

from __future__ import annotations

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

g1_29dof_wbt_force_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "wrist_force_position_tracking_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt_force:wrist_force_position_tracking_exp",
            params={
                "sigma": 0.3,
                "K_virtual": 200.0,
                "left_wrist_body_name": "left_wrist_yaw_link",
                "right_wrist_body_name": "right_wrist_yaw_link",
            },
            weight=2.0,
        ),
    }
)

__all__ = ["g1_29dof_wbt_force_reward"]
```

- [ ] **Step 2: 在 `src/holosoma/holosoma/config_values/reward.py` 注册**

Add import:
```python
from holosoma.config_values.wbt.g1.reward_force import g1_29dof_wbt_force_reward
```

Add to `DEFAULTS` dict:
```python
    "g1_29dof_wbt_force": g1_29dof_wbt_force_reward,
```

- [ ] **Step 3: 写回归测试**

Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_reward_force.py`

```python
"""Regression tests for g1_29dof_wbt_force_reward preset."""

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward
from holosoma.config_values.wbt.g1.reward_force import g1_29dof_wbt_force_reward


def test_force_reward_preserves_all_base_terms():
    for term_name in g1_29dof_wbt_reward.terms:
        assert term_name in g1_29dof_wbt_force_reward.terms, f"Base term {term_name} lost"


def test_force_reward_adds_wrist_force_position_tracking():
    assert "wrist_force_position_tracking_exp" in g1_29dof_wbt_force_reward.terms
    term = g1_29dof_wbt_force_reward.terms["wrist_force_position_tracking_exp"]
    assert term.func == "holosoma.managers.reward.terms.wbt_force:wrist_force_position_tracking_exp"
    assert term.weight == 2.0
    assert term.params["K_virtual"] == 200.0
    assert term.params["sigma"] == 0.3


def test_original_wbt_reward_unchanged():
    assert "wrist_force_position_tracking_exp" not in g1_29dof_wbt_reward.terms
```

- [ ] **Step 4: 运行**

Run: `pytest src/holosoma/holosoma/config_values/wbt/g1/tests/test_reward_force.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/reward_force.py src/holosoma/holosoma/config_values/reward.py src/holosoma/holosoma/config_values/wbt/g1/tests/test_reward_force.py
git commit -m "feat(wbt-force): add g1_29dof_wbt_force_reward preset (UniFP-style virtual impedance reward, weight 2.0)"
```

---

### Task 8: `g1_29dof_wbt_force` experiment 注册

**Files:**
- Modify: `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
- Modify: `src/holosoma/holosoma/config_values/experiment.py`
- Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_experiment_force.py`

- [ ] **Step 1: 在 `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` 追加 `g1_29dof_wbt_force`**

在文件里的 `g1_29dof_wbt_fast_sac_w_object` 定义之后、`__all__` 之前，追加：

```python
g1_29dof_wbt_force = replace(
    g1_29dof_wbt,
    training=replace(g1_29dof_wbt.training, name="g1_29dof_wbt_force_manager"),
    command=command.g1_29dof_wbt_force_command,
    observation=observation.g1_29dof_wbt_force_observation,
    reward=reward.g1_29dof_wbt_force_reward,
)
```

同时把 `"g1_29dof_wbt_force"` 加入 `__all__`。在文件末尾 docstring 里追加示例：

```python
"""
Example 4: Robot with wrist force tracking (UniFP-style position-force controller):
python src/holosoma/holosoma/train_agent.py \\
  exp:g1-29dof-wbt-force \\
  logger:wandb \\
  --command.setup_terms.motion_command.params.motion_config.motion_file=<path-to-npz>
"""
```

- [ ] **Step 2: 在 `src/holosoma/holosoma/config_values/experiment.py` 注册**

Add to the import block:
```python
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_force,
    g1_29dof_wbt_w_object,
)
```

Add to `DEFAULTS`:
```python
    "g1_29dof_wbt_force": g1_29dof_wbt_force,
```

- [ ] **Step 3: 写回归测试**

Create: `src/holosoma/holosoma/config_values/wbt/g1/tests/test_experiment_force.py`

```python
"""Regression tests for g1_29dof_wbt_force experiment."""

from holosoma.config_values.experiment import DEFAULTS
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt, g1_29dof_wbt_force


def test_force_exp_registered_in_defaults():
    assert "g1_29dof_wbt_force" in DEFAULTS
    assert DEFAULTS["g1_29dof_wbt_force"] is g1_29dof_wbt_force


def test_force_exp_reuses_base_wbt_env_class():
    # We deliberately do NOT create a new env subclass — no sim force injection.
    assert g1_29dof_wbt_force.env_class == g1_29dof_wbt.env_class


def test_force_exp_swaps_command_obs_reward():
    assert "wrist_force_command" in g1_29dof_wbt_force.command.setup_terms
    assert "wrist_force_command" in g1_29dof_wbt_force.observation.groups["actor_obs"].terms
    assert "wrist_force_command" in g1_29dof_wbt_force.observation.groups["critic_obs"].terms
    assert "wrist_force_position_tracking_exp" in g1_29dof_wbt_force.reward.terms


def test_base_exp_is_untouched():
    assert "wrist_force_command" not in g1_29dof_wbt.command.setup_terms
    assert "wrist_force_command" not in g1_29dof_wbt.observation.groups["actor_obs"].terms
    assert "wrist_force_position_tracking_exp" not in g1_29dof_wbt.reward.terms


def test_force_exp_shares_base_algo_and_robot():
    assert g1_29dof_wbt_force.algo is g1_29dof_wbt.algo
    assert g1_29dof_wbt_force.robot is g1_29dof_wbt.robot
    assert g1_29dof_wbt_force.simulator is g1_29dof_wbt.simulator
    assert g1_29dof_wbt_force.action is g1_29dof_wbt.action
    assert g1_29dof_wbt_force.termination is g1_29dof_wbt.termination
    assert g1_29dof_wbt_force.randomization is g1_29dof_wbt.randomization
    assert g1_29dof_wbt_force.curriculum is g1_29dof_wbt.curriculum
```

- [ ] **Step 4: 运行**

Run: `pytest src/holosoma/holosoma/config_values/wbt/g1/tests/test_experiment_force.py -v`
Expected: `5 passed`

- [ ] **Step 5: 现有 tyro CLI 测试没回归**

Run: `pytest src/holosoma/holosoma/config_types/tests/test_tyro_cli.py -v`
Expected: 所有现有测试通过

- [ ] **Step 6: Commit**

```bash
git add src/holosoma/holosoma/config_values/wbt/g1/experiment.py src/holosoma/holosoma/config_values/experiment.py src/holosoma/holosoma/config_values/wbt/g1/tests/test_experiment_force.py
git commit -m "feat(wbt-force): register g1_29dof_wbt_force experiment"
```

---

### Task 9: tyro CLI smoke test（不启动 IsaacSim）

**Files:**
- Create: `src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py`

- [ ] **Step 1: 写 CLI smoke test**

Create: `src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py`

```python
"""CLI smoke tests for exp:g1-29dof-wbt-force (no IsaacSim required)."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_help(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "src/holosoma/holosoma/train_agent.py", *args, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="train_agent.py paths assume Unix layout")
def test_wbt_force_exp_listed_in_cli_help():
    result = _run_help()
    combined = result.stdout + result.stderr
    assert "exp:g1-29dof-wbt-force" in combined, f"Missing in CLI help output:\n{combined[:2000]}"


@pytest.mark.skipif(sys.platform == "win32", reason="train_agent.py paths assume Unix layout")
def test_wbt_force_exp_shows_wrist_force_params():
    result = _run_help("exp:g1-29dof-wbt-force")
    combined = result.stdout + result.stderr
    # tyro normalizes snake_case to kebab-case in CLI help output.
    assert "wrist-force-command" in combined or "wrist_force_command" in combined, (
        f"wrist_force_command params not listed:\n{combined[:2000]}"
    )
```

- [ ] **Step 2: 运行**

Run: `pytest src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py -v`
Expected: `2 passed`（若 `train_agent.py --help` 输出格式有差异，按实际调整 assert 字符串）

- [ ] **Step 3: 运行完整 CI filter**

Run:
```bash
pytest -s --ignore=thirdparty --ignore=src/holosoma_inference \
    -m "not isaacsim and not requires_inference" \
    src/holosoma/holosoma/config_types \
    src/holosoma/holosoma/config_values \
    src/holosoma/holosoma/managers \
    src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py
```
Expected: 没新增失败

- [ ] **Step 4: Commit**

```bash
git add src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py
git commit -m "test(wbt-force): tyro CLI smoke tests"
```

---

### Task 10: IsaacSim e2e 测试（`@pytest.mark.isaacsim` gated）

**Files:**
- Create: `src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py`

- [ ] **Step 1: 写 e2e 测试**

Create: `src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py`

```python
"""E2E integration test for WBT-with-force. Requires IsaacSim.

Gated by ``@pytest.mark.isaacsim``. To run:

    pytest -m isaacsim src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py -v
"""

from __future__ import annotations

import dataclasses

import pytest

from holosoma.config_values import experiment
from holosoma.train_agent import get_tyro_env_config, training_context
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch


@pytest.fixture(scope="module")
def force_env():
    seeding()
    num_envs = 4
    device = "cuda"
    tyro_config = dataclasses.replace(
        experiment.g1_29dof_wbt_force,
        training=dataclasses.replace(experiment.g1_29dof_wbt_force.training, num_envs=num_envs),
    )
    with training_context(tyro_config):
        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(tyro_config.env_class)(tyro_env_config, device=device)
        env.reset_all()
        yield env


@pytest.mark.isaacsim
def test_wrist_force_command_state_registered(force_env):
    term = force_env.command_manager.get_state("wrist_force_command")
    assert term is not None
    assert term.force_cmd_b.shape == (force_env.num_envs, 6)


@pytest.mark.isaacsim
def test_wrist_force_present_in_obs(force_env):
    obs = force_env.obs_buf_dict
    assert "actor_obs" in obs
    assert obs["actor_obs"].shape[0] == force_env.num_envs


@pytest.mark.isaacsim
def test_force_reward_finite_and_bounded(force_env):
    """Reward must stay finite and in [0, 1]."""
    actions = torch.zeros(
        (force_env.num_envs, force_env.dim_actions), device=force_env.device, dtype=torch.float32
    )
    for _ in range(10):
        force_env.step({"actions": actions})
    # Force tracking reward values live on reward_manager.episode_sums_raw.
    raw = force_env.episode_sums_raw if hasattr(force_env, "episode_sums_raw") else None
    # Not all reward managers expose raw per-term; fall back on overall rew_buf.
    assert torch.isfinite(force_env.rew_buf).all()


@pytest.mark.isaacsim
def test_force_reward_term_reduces_to_motion_tracking_when_disabled(force_env):
    """When force is zero (e.g., env just reset + activation_prob=0), the new reward
    term should yield 1.0 if the robot perfectly matches motion target. We only
    check it's finite and not NaN here — the stronger invariants are in the CPU
    unit tests."""
    term = force_env.command_manager.get_state("wrist_force_command")
    assert torch.isfinite(term.force_cmd_b).all()


@pytest.mark.isaacsim
def test_base_wbt_experiment_still_loads():
    """Sanity: baseline g1_29dof_wbt still wired correctly (no spinning up second IsaacSim)."""
    from holosoma.config_values.experiment import DEFAULTS

    assert "g1_29dof_wbt" in DEFAULTS
    base = DEFAULTS["g1_29dof_wbt"]
    assert "wrist_force_command" not in base.command.setup_terms
    assert "wrist_force_position_tracking_exp" not in base.reward.terms
```

- [ ] **Step 2: 在有 IsaacSim 的机器上执行一次 sanity**

Run: `pytest -m isaacsim src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py -v`
Expected: `5 passed`（依赖 IsaacSim 环境；CI filter `not isaacsim` 会自动 skip）

- [ ] **Step 3: Commit**

```bash
git add src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py
git commit -m "test(wbt-force): add isaacsim-gated e2e tests"
```

---

### Task 11: 用户文档

**Files:**
- Create: `docs/wbt-wrist-force-training.md`

- [ ] **Step 1: 写用户文档**

Create: `docs/wbt-wrist-force-training.md`

````markdown
# WBT Wrist Force Tracking 训练指南

用 **UniFP 式虚拟阻抗（virtual impedance）** reward 训练 G1 手腕的 **主动力追踪** low-level controller。

## 是什么，不是什么

- **是：** policy 输入 = motion command + 6-D wrist force command（左右各 3D）。训练目标是让 policy 在给定 F_cmd 下把手腕主动"施力"——具体做法是在 motion 目标位置基础上偏移 `F_cmd / K_virtual`。部署时机器人接触真实表面，这个偏移经由表面刚度 → 真实力。
- **不是：** GentleHumanoid 的 compliance（顺应）训练，不是让 policy 被动吸收外界推力。也不是 perturbation 训练。
- **UniFP 参考：** `third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891` 的 `_reward_tracking_ee_force_world`。

## 快速开始

```bash
CONVERTED_FILE="$RETARGET_DIR/converted_res/robot_only/sub3_largebox_003_mj_fps50.npz"

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-force \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE
```

与 `exp:g1-29dof-wbt` 的区别（**只有三处**）：

1. `command`：加了一个 `wrist_force_command` term，按 ramp 剖面采样 per-env 力指令（body-yaw frame）。
2. `observation`：actor + critic 各加一个 `wrist_force_command` 条目（6 维，scale 0.01）。
3. `reward`：在现有 reward 列表后面追加 `wrist_force_position_tracking_exp`（weight 2.0）：

$$
\text{reward} = \exp\left(-\frac{\| \mathbf{x}_{\text{wrist}}^{\text{actual}} - (\mathbf{x}_{\text{wrist}}^{\text{motion}} + \mathbf{F}^w_{\text{cmd}} / K_{\text{virtual}}) \|^2}{\sigma^2}\right)
$$

所有现有 reward term（motion tracking、regularization、undesired contact）完全不变。env、algo、robot、randomization 一行不动。

## 参数调整

### 力采样参数（`wrist_force_command`）

```bash
# 把最大力从 30N 调到 80N
--command.setup_terms.wrist_force_command.params.wrist_force_config.force_magnitude_range='[10.0, 80.0]'

# 只在右手腕施加力
--command.setup_terms.wrist_force_command.params.wrist_force_config.enable_left=false

# 更高激活频率（每 control step 2% 概率）
--command.setup_terms.wrist_force_command.params.wrist_force_config.activation_prob_per_step=0.02

# 更长持续期
--command.setup_terms.wrist_force_command.params.wrist_force_config.duration_range_s='[1.0, 5.0]'
```

### Reward 参数（`wrist_force_position_tracking_exp`）

```bash
# 调整虚拟刚度（越大 = 位置偏移越小 = policy 越"硬")
--reward.terms.wrist_force_position_tracking_exp.params.K_virtual=400.0

# 调整 reward 权重
--reward.terms.wrist_force_position_tracking_exp.weight=4.0

# 调整 reward 带宽
--reward.terms.wrist_force_position_tracking_exp.params.sigma=0.2
```

### 全部 `wrist_force_config` 字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `force_magnitude_range` | `[5.0, 30.0]` | 每段力峰值 uniform 采样范围（N） |
| `duration_range_s` | `[0.5, 2.0]` | 每段力总时长（s） |
| `cooldown_range_s` | `[0.5, 3.0]` | 两段力之间空窗（s） |
| `ramp_up_s` | `0.2` | 线性 ramp-up 时长（s）|
| `ramp_down_s` | `0.2` | 线性 ramp-down 时长（s） |
| `activation_prob_per_step` | `0.005` | 每 control step cooldown 结束后激活新力的概率 |
| `enable_left` / `enable_right` | `true` | 是否启用对应手腕 |
| `left_wrist_body_name` | `left_wrist_yaw_link` | G1 URDF 的 body name |
| `right_wrist_body_name` | `right_wrist_yaw_link` | 同上 |

### Reward 全部参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `sigma` | `0.3` | 指数核带宽（m） |
| `K_virtual` | `200.0` | 虚拟刚度（N/m）。UniFP 默认。F=30N → 0.15m 偏移。 |
| `left_wrist_body_name` / `right_wrist_body_name` | `*_wrist_yaw_link` | 对齐 `body_names_to_track`（必须在列表中） |

## 设计备注

- **为什么不动原 motion tracking reward？** 按用户指示"以 holosoma 为主"。现有 `motion_relative_body_position_error_exp` 继续对 14 个 body 取 mean 跟踪 motion；wrist 在新旧 reward 之间"拉扯"。当 F_cmd = 0 时两者一致；F_cmd ≠ 0 时新 reward 权重 2.0 远大于原 reward 对 wrist 的隐含贡献（1.0 / 14 ≈ 0.07），policy 会偏向 force-aware target。**这完全对齐 UniFP 的多-reward 共存设计**（UniFP 同时有 `tracking_ee_world` 和 `tracking_ee_force_world`）。
- **为什么不注入 force 到 sim？** UniFP 的 F_cmd 是纯训练信号；sim 里不需要对应的真实接触。policy 学到的是"虚拟阻抗"行为，部署时碰到真实表面自然产生力。不注入 sim 的好处：简洁、无 env 子类、不引入 Newton 3rd-law 物理纠葛。
- **v2 扩展：F_ext 扰动（robustness）。** UniFP 还有个 "ext stream" 作为 hidden 扰动注入 sim，让 policy 学会在未知外力下仍按 F_cmd 施力。这需要加 env 子类 + `_apply_force_in_physics_step` 覆写，**不在本 plan v1 范围**。

## 测试

```bash
# CPU 测试（不需要 IsaacSim）
pytest \
    src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py \
    src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force.py \
    src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force.py \
    src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/tests/ \
    src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py

# IsaacSim e2e
pytest -m isaacsim src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py -v
```

## 参考文件

- `docs/third-party-lowlevel-training-survey.md` — UniFP 与 GentleHumanoid 完整对比
- `third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891` — `_reward_tracking_ee_force_world`
- `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py:1135` — GentleHumanoid `force_reward`（作为对比；本 plan 未采用）
````

- [ ] **Step 2: Commit**

```bash
git add docs/wbt-wrist-force-training.md
git commit -m "docs(wbt-force): user guide for UniFP-style wrist force tracking"
```

---

### Task 12: 全局 sanity

- [ ] **Step 1: mypy**

Run: `mypy src/holosoma/holosoma/managers/command/terms/wbt_force.py src/holosoma/holosoma/managers/observation/terms/wbt_force.py src/holosoma/holosoma/managers/reward/terms/wbt_force.py src/holosoma/holosoma/config_types/command.py`
Expected: no errors（`src/holosoma_inference/` 被 mypy.ini exclude，不影响）

- [ ] **Step 2: pre-commit**

Run:
```bash
pre-commit run --files \
    src/holosoma/holosoma/config_types/command.py \
    src/holosoma/holosoma/managers/command/terms/wbt_force.py \
    src/holosoma/holosoma/managers/observation/terms/wbt_force.py \
    src/holosoma/holosoma/managers/reward/terms/wbt_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/command_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/observation_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/reward_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/experiment.py \
    src/holosoma/holosoma/config_values/experiment.py \
    src/holosoma/holosoma/config_values/command.py \
    src/holosoma/holosoma/config_values/observation.py \
    src/holosoma/holosoma/config_values/reward.py \
    docs/wbt-wrist-force-training.md \
    src/holosoma/holosoma/config_types/tests/test_wrist_force_config.py \
    src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force.py \
    src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force.py \
    src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/tests/test_command_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/tests/test_observation_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/tests/test_reward_force.py \
    src/holosoma/holosoma/config_values/wbt/g1/tests/test_experiment_force.py \
    src/holosoma/holosoma/envs/tests/test_wbt_force_cli.py \
    src/holosoma/holosoma/envs/tests/test_wbt_wrist_force_e2e.py
```
Expected: 无错误（若 ruff auto-fix 了格式则正常）

- [ ] **Step 3: 完整 CI filter（不要 IsaacSim）**

Run:
```bash
pytest -s --ignore=thirdparty --ignore=src/holosoma_inference \
    -m "not isaacsim and not requires_inference"
```
Expected: 没有比改动前新增任何失败

- [ ] **Step 4: `--help` smoke**

```bash
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt --help > /dev/null
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-force --help > /dev/null
```
两条命令都应几秒内 exit 0。

- [ ] **Step 5: 若 pre-commit 自动修改了文件，一并 commit**

```bash
git add -u
git diff --cached --quiet || git commit -m "chore(wbt-force): apply pre-commit auto-fixes"
```

---

## ═══════════════════════════════════════════════════════════════════
## PATH A' (RECOMMENDED): UniFP reward + wrist-only F_ext 注入（极简 v8）
## ═══════════════════════════════════════════════════════════════════

> **v9 更新。** 基于以下已确认的事实：
> 1. Holosoma 的 observation manager **已内置 per-group `history_length`**（`src/holosoma/holosoma/managers/observation/manager.py:71, 213-264`）—— 不需要自己造 history stacker（但 v9 也不使用 history，详见 §A'.7b.5）
> 2. Holosoma 的 IsaacSim simulator **已经用过 `self._robot.set_external_force_and_torque`**（`src/holosoma/holosoma/agents/callbacks/push.py:236`、`simulator/shared/virtual_gantry.py:425`）—— 外力注入 API 现成可用
> 3. **第三方代码只读参考，不 port 任何代码**（详见 §A'.5.1；v9 红线 4）
> 4. **K_virtual 固定为 training-time hyperparameter**，不做 IsaacSim 校准（详见 §A'.4；v9 红线 2）

### A'.0 Scope 硬约束（五条红线）

**红线 1：力只施加在左右 wrist，不施加在 body 其余任何地方。**
- `F_ext` 注入：只往 `left_wrist_yaw_link` 和 `right_wrist_yaw_link` 两个 rigid body 写外力 tensor，其他 12 个 tracked body **一律 0**
- `F_cmd` 指令空间：6-D（左 wrist 3D + 右 wrist 3D），没有 head / torso / root / elbow 等力指令
- reward 只看 wrist 2 body 的误差，其他 12 body 完全走 Holosoma 原 motion tracking reward

**红线 2：`K_virtual = 100.0 N/m`，固定、写死、不随机化、不进 obs。**
- **澄清**：Holosoma 本身是 kinematic motion tracker，**没有一个现成的 "K_arm" 数字** —— 只有 joint-level PD（`config_values/robot.py:498-504`）：shoulder × 3/elbow/wrist_roll `Kp=14.25`, wrist_pitch/yaw `Kp=16.78`，平均 15 Nm/rad
- 真实末端 Cartesian 刚度 `K_cart(q) = J(q)^{-T} · diag(K_q) · J(q)^{-1}` 是构型函数，非常数
- **`K_virtual` 不是测得的机器人 Cartesian 刚度，而是 reward 里的 F↔Δx exchange rate（training-time hyperparameter）。** 详见 §A'.4
- v9 直接选定固定值 **100.0 N/m**，详见 §A'.4 推导（不做 IsaacSim 校准脚本）
- Task A'-0 只负责把这个常量写进 `_k_virtual.py`
- **物理意义**：`K_virtual = 100` 对应 F_cmd=10N → Δx=0.10m（日常 case），F_cmd=30N → Δx=0.30m，同向叠加 F_ext+F_cmd 最大 60N → Δx=0.60m（刚好在 G1 arm reach 边缘，不超出）
- **敏感性**：训练后若 F_real / F_cmd 偏离 1 只有 2x 内，inference 端加常数 α 补偿即可，不用重训

**红线 3：完全不碰 Holosoma 已有 reward 项和 `g1_29dof_wbt` experiment object。**
- 新 reward `wrist_force_position_tracking_exp` 只**追加**到 preset
- 现有 `motion_relative_body_position_error_exp` (weight 1.0) 仍旧 mean 14 body，自动包含 wrist 2 个 → wrist 在 Holosoma 原 reward 里仍被约束
- wrist 两个点会同时受 "原 reward 拉向 motion target" 和 "新 reward 拉向 target_shifted" 的平衡（同 UniFP 多-reward 共存设计）

**红线 4：UniFP / GH / CHIP 代码一律只"读源参考"，不 import、不 copy、不 port。自己写简单版本。**
- Holosoma 运行时**绝不**依赖 `third_party/` 任何代码
- `third_party/` 下的 code 只作为"读取 + 参考" 资源，不加 PYTHONPATH，不加 pyproject 依赖
- 不 copy GH 的 `TemporalLerp` / `rand_points_isotropic` / `clamp_norm` 等 utility；ramp state machine 自己写（~50 行）
- 详细规则见 §A'.5.1

**红线 5：绝不修改 Holosoma 已有的任何现有文件的逻辑（只允许在 DEFAULTS dict 末尾追加 entry，或在 experiment.py 末尾追加 dataclass instance）。**
- 需要修改的现有文件清单（仅限 DEFAULTS dict / 末尾追加）：
  - `src/holosoma/holosoma/config_values/command.py`（DEFAULTS 追加）
  - `src/holosoma/holosoma/config_values/observation.py`（DEFAULTS 追加）
  - `src/holosoma/holosoma/config_values/reward.py`（DEFAULTS 追加）
  - `src/holosoma/holosoma/config_values/experiment.py`（DEFAULTS 追加）
  - `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`（末尾追加 `g1_29dof_wbt_force = ExperimentConfig(...)`）
  - `src/holosoma/holosoma/config_types/command.py`（末尾追加 `WristComplianceConfig` dataclass）
- 所有其他 Holosoma 文件（manager 基类、env manager 基类、观测 term、reward term、push.py、virtual_gantry.py、observation manager、算法、PPO 代码等）**一字不改**
- 每个 preset 回归测试都有 `assert original_obj is module.original_obj` identity check，CI 里跑这个作为 guard

### A'.1 与 Path A 的差异一表（v9）

| 维度 | Path A（UniFP 纯 v1） | **Path A' v9（本节）** |
|---|---|---|
| sim 里注入 F_ext | ✗ | **✅ wrist only**（左右 wrist 梯形剖面） |
| F_cmd 进 actor | ✅ | ✅ |
| obs history | ✗ | **✗（v1 不做）** —— 若训练有问题 v2 再加 |
| critic 看 F_ext ground truth | ✗ | **✅** 新 obs term，critic only |
| Reward 公式 | `g̃ = g + F_cmd/K` | `g̃ = g + (F_ext + F_cmd)/K` ★ |
| `K_virtual` 选取 | 写死 200（UniFP 默认）| **写死 100 N/m**（§A'.4 Training hyperparameter） |
| Env 子类 | 不需要 | **需要**：override `_apply_force_in_physics_step` 调 `self._robot.set_external_force_and_torque` |
| 外力注入 API | N/A | Holosoma 已用过（见 `push.py:236`, `virtual_gantry.py:425`）|
| Ramp profile 工具 | 自己写 | **自己写最简 state machine**（~50 行，不 port GH） |
| GH 代码依赖 | 无 | **无**（只读 GH 源参考，不 copy 也不 import） |
| K_virtual 校准脚本 | 无 | **无（v9 跳过）**；常量 100 N/m 手写 |
| 总 task 数 | 12 | **14**（含常量文件 + 测试 + 文档 + sanity，实际"新代码"task 只有 A'-1 ~ A'-9 共 9 个）|

### A'.2 Reward（Path A' 正式版，替代 Path A 的版本）

公式：

```
force_w_left  = yaw_quat(base_quat) · force_cmd_b[:, 0]        # body-yaw → world
force_w_right = yaw_quat(base_quat) · force_cmd_b[:, 1]
F_total_w_left  = force_ext_w_left  + force_w_left              # UniFP 双力叠加
F_total_w_right = force_ext_w_right + force_w_right
target_shifted_w[left]  = motion_target_w[left]  + F_total_w_left  / K_virtual
target_shifted_w[right] = motion_target_w[right] + F_total_w_right / K_virtual
error = Σ_xyz (target_shifted_w − wrist_actual_w)²
reward = exp(−mean_over_2_wrists(error) / σ²)     # Holosoma-style exp-squared kernel
```

**Kernel 选型说明**（v9 澄清）：**target shift 公式沿用 UniFP，但 exp kernel 用 Holosoma 风格（squared error + σ²），不是 UniFP 的 L1 + `exp(-err/σ * 2)`。** 原因：(a) 与 Holosoma 现有 `motion_relative_body_position_error_exp`（`managers/reward/terms/wbt.py:87`）kernel 一致，便于在同一 reward preset 里与其他 tracking term 加权；(b) squared error 对大偏差惩罚更重，符合"希望 wrist 误差尽量小"的意图。

**注意**：`F_ext` 来自 sim ground truth（`WristComplianceCommand.force_ext_w`，见 A'.5），reward 端**不经过 actor obs**，所以不会破坏"actor 不看 F_ext"的信息边界。

**四个稳态情形**（参见 explainer §5.5；K_virtual=100 N/m）：

| 场景 | F_cmd | F_ext | g̃ 相对 g 的偏移 | policy 学到 |
|---|---|---|---|---|
| A（自由空间，无指令）| 0 | 0 | 0 | 纯 motion tracking |
| B（空中"发力"）| 20N +X | 0 | +0.20 m · x̂ | 把 wrist 偏 Δx，接触墙时产生力 |
| C（被外力扰动）| 0 | +20N +X | +0.20 m · x̂ | (v1 是 DR，不是 learned compliance；见 §A'.9 note) |
| D（稳态施力）| 20N +X | -20N（墙反推） | 0 | wrist 保持原位，靠 PD 发力 |

**Δx 工作空间 bound（对 K=100 N/m）：**

| 合力 `||F_ext + F_cmd||` | Δx | 相对 G1 arm reach (~0.7m) |
|---|---|---|
| 30 N（单路 max） | 0.30 m | 舒适区 |
| 45 N（F_ext=15 + F_cmd=30 同向） | 0.45 m | 舒适区 |
| **60 N（max 叠加，F_ext+F_cmd 都 30 同向）** | **0.60 m** | **arm reach 边缘（kinematic 软上限）** |

> Policy 在 Δx > 0.6m 的采样下无法完全达成 target_shifted —— 这是合理的"上限软截断"，与"wrist 物理可达 ~0.7m"匹配。若训练曲线显示 >20% step 处于 saturation，v2 再考虑把 `force_ext_magnitude_range` 上限从 30 降到 15-20 N（见 §A'.10）。

### A'.3 F_ext 注入规格（只在 wrist）

借用 CHIP Fig. 9 / UniFP 行 1144-1152 的梯形剖面，仅在左右 wrist 两处注入：

| 参数 | 值 | 来源 |
|---|---|---|
| 幅度范围 | `force_magnitude_range = [0, 30] N` | G1 wrist 可承受量级（比 CHIP 40N 保守） |
| 持续时间 | `duration_range_s = [1.0, 3.0]` | CHIP Fig. 9 |
| cooldown | `cooldown_range_s = [0.5, 2.0]` | 与 UniFP 对齐 |
| ramp | 线性 ramp-up 0.25·duration → hold 0.5·duration → ramp-down 0.25·duration | CHIP Fig. 9 trapezoid |
| 方向 | 单位球面均匀采样（`d = torch.randn(n, 3); d /= d.norm(dim=-1, keepdim=True).clamp(min=1e-6)`，自己实现） | UniFP 行 1124 等价 |
| 激活概率 | `activation_prob_per_step = 0.01` | 两 wrist 独立 Bernoulli；与 UniFP 相近 |
| 注入 body | **only** `left_wrist_yaw_link`, `right_wrist_yaw_link` | 红线 1 |
| 频率 | **Force profile 在 control rate (50 Hz) 推进**（command term `.step()` 在 control step 末尾被调用）；cached force value 在每个 physics substep (200 Hz, 4x decimation) 写入 sim | 见下方 "Physics step vs Control step" 澄清 |
| API | `self.simulator._robot.set_external_force_and_torque(forces_body, zeros, env_ids=all, body_ids=...)` | 已在 `push.py:236` / `virtual_gantry.py:425` 验证工作 |
| Frame 转换 | **IsaacLab 2.1 `is_global=False` 硬编码** → 传入前必须 world → body-local。**明确 import**：`from isaaclab.utils.math import quat_apply_inverse`（IsaacLab 的 `body_quat_w` 是 **wxyz**；不要用 `holosoma.utils.rotations` 里假设 xyzw 的 helper）| `push.py:230-231` |
| Body ID resolution | **使用 Holosoma 惯用 pattern**：`body_names_idx = simulator.find_rigid_body_indice(name)` 然后 `isaac_body_id = simulator.body_ids[body_names_idx]`（见 `push.py` / `virtual_gantry.py` 现有调用）| 不直接调 `_robot.find_bodies()` 的返回 list |

**Physics step vs Control step 频率澄清 + 一致性分析（v9 新增，回应 Codex 评审意见 #9 + 补回 off-by-one 疑问）：**

Holosoma 的真实 callback 顺序（WBT 下 `_update_tasks_before_termination = False`，参见 `envs/base_task/base_task.py:405-460`）：

```
step(actions):
  _pre_physics_step:   action_manager.process_actions
  _physics_step:       for 4 substeps:
                         _apply_force_in_physics_step   ← 读 command_manager 缓存 → write sim
                         simulator.simulate_at_each_physics_step
  _post_physics_step:
    _refresh_sim_tensors
    _check_termination                     ← 读 command_manager 缓存
    _compute_reward                        ← 读 command_manager 缓存 ★ 与 physics 同步
    reset envs
    _update_tasks_callback:                ← command_manager.step() 在这里 advance
      command_manager.step()
    _compute_observations                  ← 读 ADVANCED 后的 command_manager
```

**force_ext_w 与 reward 的一致性（关键）**：

- Step N 开始时，`command_manager.wrist_compliance_command.force_ext_w` 是 step N-1 末尾 `command_manager.step()` 算出的值——叫它 `F_N`
- Physics 4 个 substep 都 apply `F_N`（ZOH）
- `_compute_reward` 在 command_manager.step() **之前**，读的也是 `F_N` → **reward 看到的 F_ext = physics 实际施加的 F_ext，完全同步，无 off-by-one**
- 然后 `_update_tasks_callback` 调 `command_manager.step()`，把内部 state 推进到 `F_{N+1}`
- `_compute_observations` 在 step 之后，actor/critic obs 看到的 `force_cmd_b` / `force_ext_w` 是 `F_{N+1}`，即 **"下一 step 将要施加的力"**——这是 Holosoma 所有 command term 的标准 convention（和 `motion_command` 一样：obs 里看到的是下一 step 的 target）

**所以**：
- **Command term 梯形状态机在 50 Hz 推进**（RAMP_UP/HOLD/RAMP_DOWN 用 control-step count）
- **F_ext 写入 sim 在每个 physics substep 都执行**（env 子类的 override），每个 control step 内 4 个 substep 写入同一个值（ZOH，与 action ZOH 对齐）
- **Reward 与 physics 同步读取同一个 cached force**（不存在 off-by-one；见上面 callback trace）
- **Observation 看到"下一步将施加的力"**（标准 command convention；与 `motion_command` 时序一致，actor 无需特殊适配）

**实现要求**：`WristComplianceCommand` 的 `.step()` 必须只更新 state machine + 采样，不能在 `_apply_force_in_physics_step` 或 reward 调用之间修改 `force_ext_w` / `force_cmd_b` buffer。单测 A'-2 加一条：连续调两次 `.force_ext_w`（中间不调 `.step()`）返回同一张量。

**Ramp 实现策略**：在 `WristComplianceCommand` 里用简单 `int8` state + 整数 counter（counter 单位是 control step），自己写 ~50 行，不 port 任何 third-party code。

**F_cmd 剖面**：同样机制，独立实例。F_cmd 也跑梯形剖面是为了让 policy 见过 "指令从 0 升到 X" 的过渡（而不是阶跃），对应真实部署时上层 target 刚移动到表面接触瞬间那段。

**实现**：直接新建 `WristComplianceCommand`（不复用 Path A 的 `WristForceCommand`——后者是历史参考的 Path A 里才有的 class name），同时维护 `force_cmd_b: [N, 2, 3]` + `force_ext_w: [N, 2, 3]`；obs term 用 `.reshape(N, 6)` 展平。

### A'.4 `K_virtual` 的固定值选取（v9：training-time hyperparameter，不是测得的刚度）

**v9 决定：`K_virtual = 100.0 N/m`，写死在 `_k_virtual.py` 里，不做 IsaacSim 校准脚本。**

**重要澄清（修正 Codex 评审意见 #7）：`K_virtual` 不是测得或估算得到的 G1 Cartesian 刚度。它是 reward 里 "F↔Δx exchange rate" 这个 training-time hyperparameter**：

- Policy 看到的世界是："F_cmd 越大 → target 偏离原 motion 越远 → wrist 要跟着走"
- "越多 Newton 等于多少米"这个比例关系由训练时的 `K_virtual` 定义，**policy 学到的是"每单位 F_cmd 把 wrist 偏 `1/K_virtual` 米"的位移映射**
- 部署时 F_cmd → 实际接触力的关系由 **policy 学到的位移映射 × 真实接触表面刚度 `K_surface`** 决定，稳态下 F_real ≈ F_cmd · (K_surface / K_virtual)
- 所以：`K_virtual` 确实影响部署行为（它定义了 policy 内化的 scaling），但**误差可以被 inference 端乘常数 α = K_virtual / K_surface 补偿**，不用重训（这是 UniFP 的 calibration convention）

#### 怎么选 `K_virtual = 100`：从 Δx 工作空间反推

**设计目标**：采样分布内 `||F_ext + F_cmd||` 上限除以 `K_virtual` 得到的 Δx 应落在 wrist "可达但不轻松" 的区间——让 policy 既有足够的"力度区分"训练信号，又不会被 kinematic 极限切掉太多 sample。

- G1 arm reach ≈ 0.7 m（physics plausible；参考 G1 URDF）
- 采样范围：`force_ext_magnitude_range = [0, 30] N`（红线 2），`force_cmd_magnitude_range = [5, 30] N`
- 同向最大叠加：`||F_ext + F_cmd|| ≤ 60 N`

| K_virtual | Δx @ 30N（单路 max） | Δx @ 60N（同向叠加 max） | 问题 |
|---|---|---|---|
| 50 N/m | 0.6 m | **1.2 m** | 叠加 Δx 远超 wrist reach |
| **100 N/m ★** | **0.3 m** | **0.6 m** | **叠加 Δx 刚到 wrist reach 边缘** |
| 200 N/m（UniFP 默认） | 0.15 m | 0.3 m | Δx 太小，policy 信号弱；且 30N F_cmd 只让 wrist 偏 15cm 可能训不出明显 wrist 位置变化 |

**v9 选 `100 N/m`**：保证同向叠加的 Δx 刚好到 arm reach 边缘（合理的 "soft upper bound"），又不过小而让训练信号萎缩。

#### 辅助参考（不是主要判据，仅供 sanity）

Holosoma arm joint-level PD（`config_values/robot.py:498-504`）：
- shoulder × 3 / elbow / wrist_roll: Kp = 14.25 Nm/rad
- wrist_pitch/yaw: Kp = 16.78 Nm/rad；**平均 ~15 Nm/rad**

关节空间 Kp（Nm/rad）不能直接当 Cartesian K（N/m）——真实 `K_cart(q) = J(q)^{-T} · diag(K_q) · J(q)^{-1}` 是构型 q 的函数。做极简 K/L² 估算（L=0.4-0.5m, 4-joint arm）得 15-94 N/m 的数量级；100 N/m 跟这个 range 同量级，不会完全脱离物理直觉。

GH `kp_range = [5, 250]` N/m 是真实弹簧刚度的随机化区间，100 N/m 落在 mid-high 段（几何中值 35 偏硬）——我们固定不随机化，选 mid-high 一点让 Δx 更保守。

#### 代码落地

`src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py`（手写）：

```python
"""Virtual force-to-displacement exchange rate for G1 wrist force training.

This is a REWARD HYPERPARAMETER, not a measured/calibrated robot stiffness.
It defines how many Newton of force command corresponds to how many meters
of wrist target displacement inside the virtual impedance reward
(target_shifted = motion_target + F_total / K_virtual).

Value chosen so that the maximum combined force draw
(||F_ext + F_cmd|| = 60 N under same-direction sampling)
produces a shifted target at ~0.6 m from the motion target, which is at the
edge of G1 arm reach (~0.7 m) — a natural soft upper bound for training.

See docs/plans/2026-05-01-wbt-wrist-force-controller.md §A'.4 for full
rationale. If training reveals systematic F_real / F_cmd mismatch < 2x,
apply an inference-time scalar correction rather than retraining.
"""

G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M: float = 100.0
```

reward preset 从这里 `import`，reward 公式代码和 Task 定义里**不出现具体数字**（只引用这个常量）—— 未来若要改值，只需改这一个文件。

**敏感性分析**：训练后若 F_real / F_cmd 偏 0.5-2x → inference 端乘 α 补偿，不重训。若偏 > 5x → 再考虑调此常量重训（且训练曲线若显示 wrist 长期处于 kinematic saturation，应先把 `force_ext_magnitude_range` 上限从 30 降到 15-20 N，而不是动 K）。

### A'.5 新文件结构

| 路径 | 类型 | 职责 |
|---|---|---|
| `src/holosoma/holosoma/config_types/command.py` | **改** | 末尾追加 `WristComplianceConfig` dataclass（F_cmd 采样 + F_ext 采样 + ramp 参数） |
| `src/holosoma/holosoma/managers/command/terms/wbt_force.py` | **新** | `WristComplianceCommand(CommandTermBase)`：**自己写简单的 per-env 梯形 state machine**（ramp_up → hold → ramp_down → cooldown），方向用 `torch.randn + normalize` 采样，不依赖任何 GH 代码。暴露 `force_cmd_b: [N, 6]`, `force_ext_w: [N, 2, 3]` |
| `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py` | **新** | `WholeBodyTrackingForceInjected(WholeBodyTrackingManager)`：override `_apply_force_in_physics_step`，调 `self.simulator._robot.set_external_force_and_torque(...)`（**world → body frame 转换必需**）；同时暴露 `last_applied_force_w_by_body_id: dict[int, torch.Tensor]` debug accessor 供 e2e 测试断言使用（避免测试依赖 IsaacLab 私有 `_external_force_b` buffer） |
| `src/holosoma/holosoma/managers/observation/terms/wbt_force.py` | **新** | (a) `wrist_force_command(env)` → `force_cmd_b.reshape(N, 6)` 给 actor+critic；(b) `wrist_force_ext_privileged(env)` → `force_ext_w.reshape(N, 6)` 仅 critic |
| `src/holosoma/holosoma/managers/reward/terms/wbt_force.py` | **新** | `wrist_force_position_tracking_exp(env, sigma, K_virtual, left_wrist_body_name, right_wrist_body_name)` 按 §A'.2 双力公式 |
| `src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py` | **新** | 手写常量 `G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M: float = 100.0` + docstring 说明这是 training hyperparameter |
| `src/holosoma/holosoma/config_values/wbt/g1/command_force.py` | **新** | `g1_29dof_wbt_force_command` preset |
| `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py` | **新** | `g1_29dof_wbt_force_observation` preset：actor group 是 baseline 原样 + `wrist_force_command`；critic group 是 baseline 原样 + `wrist_force_command` + `wrist_force_ext_privileged`；**`history_length=1`（跟 baseline 一样）** |
| `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py` | **新** | `g1_29dof_wbt_force_reward` preset：原 reward 全保留，追加 `wrist_force_position_tracking_exp`（weight=2.0，K_virtual 从 `_k_virtual.py` import） |
| `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` | **改**（末尾追加） | `g1_29dof_wbt_force` experiment，`env_class="holosoma.envs.wbt.wbt_force_injected:WholeBodyTrackingForceInjected"` |
| `src/holosoma/holosoma/config_values/command.py` | **改** | DEFAULTS 追加 `"g1_29dof_wbt_force"` |
| `src/holosoma/holosoma/config_values/observation.py` | **改** | 同上 |
| `src/holosoma/holosoma/config_values/reward.py` | **改** | 同上 |
| `src/holosoma/holosoma/config_values/experiment.py` | **改** | 同上 |
| 所有对应的 `tests/test_*.py` | **新** | TDD 覆盖 |

**必须保持原样（一行不改）：**
- `src/holosoma/holosoma/envs/wbt/wbt_manager.py`（基类；`wbt_force_injected.py` 继承它不改源）
- `src/holosoma/holosoma/managers/command/terms/wbt.py`（MotionCommand）
- `src/holosoma/holosoma/managers/observation/terms/wbt.py`
- `src/holosoma/holosoma/managers/reward/terms/wbt.py`
- `src/holosoma/holosoma/managers/observation/manager.py`（history 已内置，无需改）
- Holosoma 现有 `push.py` / `virtual_gantry.py`（我们只是参考 API 用法，不改它们）
- 所有 `g1_29dof_wbt*` 现有 preset

#### A'.5.1 关于 gentle-humanoid-training 代码复用（v8：极简，不 port）

**v8 决定：不从 GH port 任何代码。**

之前版本计划 port `TemporalLerp`、`rand_points_isotropic`、`clamp_norm` 等 utility。v8 砍掉：

- **ramp state machine**：直接在 `WristComplianceCommand` 里用简单的 `int8` 状态 + 整数计数器实现（`cooldown_remaining`, `ramp_up_remaining`, `hold_remaining`, `ramp_down_remaining`），<50 行，不需要 `TemporalLerp` 那种通用 lerp class
- **方向均匀采样**：`d = torch.randn(n, 3); d /= d.norm(dim=-1, keepdim=True)` 一行搞定，不需要 GH 的 `rand_points_isotropic`
- **幅度 clamp**：若需要的话直接 `torch.clamp` 或 `F.normalize(..., dim=-1) * magnitude`，不需要 GH 的 `clamp_norm`

**只从 GH 读源代码（"参考 API 用法"），不 copy 代码**：
- 读 `motion_tracking.py` 看它怎么调 `physx.apply_forces_and_torques_at_position`（已确认 Holosoma 有等价 wrapper `set_external_force_and_torque`，见 `push.py:236`、`virtual_gantry.py:425`）
- 读 `G1_gentle.yaml` 看它的 force perturbation 相关参数配置量级（比如 `max_force`、`force_alpha` 等）
- 这些都是"读一次、做笔记、然后自己重写"，**Holosoma 最终代码里一行 GH 代码都不出现**

**保留禁令（红线 4 依然生效）**：
- ❌ `from gentle_humanoid_training ... import ...`（不 import）
- ❌ `sys.path.append('third_party/gentle-humanoid-training/...')`
- ❌ 在 pyproject.toml 加 GH 依赖
- ❌ 把 `third_party/gentle-humanoid-training/` 加到 PYTHONPATH
- ❌ copy GH 代码片段（v8 决定：连 copy 都不做，改为自己重写逻辑）

**同样原则适用于 UniFP**：只读参考，不 import 不 copy。UniFP 的代码本来也是 IsaacGym 的，不适用 IsaacSim。

### A'.6 前置 spike（开始 task 前必做）

**v9：Spike 2（history）已经被 §A'.1 的调研消解（Holosoma 原生支持 `history_length`）。只剩 1 个 spike。**

**Spike 1: IsaacSim wrist-only 外力注入 API 集成验证**

目的：确认 `set_external_force_and_torque(..., body_ids=wrist_only)` 的行为符合预期——**只给 wrist 两个 body 传 force tensor，其他 body 的 external force buffer 不被污染**。

```python
# tests/spikes/test_pathprime_spike1_wrist_only_force_injection.py
# @pytest.mark.isaacsim
#
# 1. 启动 g1_29dof_wbt baseline env（不走新 experiment）
# 2. 一次性调用：
#      body_ids = [left_wrist_isaac_id, right_wrist_isaac_id]
#      forces = torch.tensor([[[20, 0, 0], [0, 0, 0]]])  # [1, 2, 3]，right wrist 为零以测选择性
#      torques = torch.zeros_like(forces)
#      self._robot.set_external_force_and_torque(forces, torques, env_ids=None, body_ids=body_ids)
# 3. 跑 1 physics step
# 4. 断言（第一层：直接检查 API contract）：
#    - 只传入 body_ids=[left_id, right_id]，其他 body 的 external force buffer 保持 0
#      （SPIKE-ONLY，可通过 simulator._robot._external_force_b 读所有 body 的 external
#       force tensor；这是 IsaacLab 私有 buffer，只在 spike 里用来"打开黑盒"验证 API
#       行为。生产 e2e 测试 Task A'-11 禁止读这个 buffer——改用 env 子类的
#       last_applied_force_w_by_body_id debug accessor。）
#    - 传入的 forces tensor 维度跟 body_ids 长度一致（避免 broadcast 错 body）
# 5. 断言（第二层：1 step 后物理合理性）：
#    - 左 wrist world x 速度有变化（力确实被施加）
#    - 非 wrist body（头、胯、右小臂等）没有被施加外力的痕迹（不是"位移 < 1cm" 的硬阈值——这个在
#      articulated body 里会被关节耦合破坏——而是通过 external force buffer 白名单验证）
# 6. 反向测试：传 body_ids=[wrong_body_id]（比如 pelvis）→ 断言 wrist 的 buffer 保持 0
#    （证明 body_ids 参数真的在做 selection，不是全局广播）
# 7. reset 后断言 external force buffer 全部清零
```

**Codex 评审意见 #10 修正**：原版写"右 wrist 位移 < 1 cm"作为 isolation 判据，这不对——articulated body 里左 wrist 施力会通过关节耦合和 base dynamics 影响右 wrist，位移 < 1 cm 可能 false-fail。v9 改成**直接查 external force buffer + 用 body_ids 白名单断言 API 选择性**，这才是"force injection 只到 wrist"的 sound check。

Gate：Spike 1 绿 → 进入 Task A'-0（写常量文件）→ Task A'-1 起全面开工。

### A'.7 任务分解（v9：14 task + 1 spike，分 6 phase）

格式沿用 Path A 的 TDD 6 步骤（写失败测试 → 确认失败 → 实现 → 通过 → 回归 → commit）。

**Phase 0（前置验证 & 常量）：**

- [ ] **Spike 1**（见 §A'.6）—— 外力注入 wrist-only 集成验证

- [ ] **Task A'-0：创建 `_k_virtual.py` 常量文件**（v9：不做校准脚本，直接写固定值）
  - 手写 `src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py`
  - 内容：`G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M: float = 100.0`（见 §A'.4 推导）
  - 文件头 docstring 明确"this is a reward hyperparameter, not a measured stiffness"
  - 单元测试：import 成功、值等于 100.0、类型是 float
  - Commit

**Phase 1（command 基础设施）：**

- [ ] **Task A'-1：`WristComplianceConfig` dataclass**
  - `src/holosoma/holosoma/config_types/command.py` 末尾追加
  - 字段：
    - F_cmd: `force_cmd_magnitude_range=[5, 30]`, `force_cmd_duration_range_s=[1.0, 3.0]`, `force_cmd_cooldown_range_s=[0.5, 2.0]`, `force_cmd_ramp_frac=0.25`, `force_cmd_activation_prob_per_step=0.01`
    - F_ext: `force_ext_magnitude_range=[0, 30]`, `force_ext_duration_range_s=[1.0, 3.0]`, `force_ext_cooldown_range_s=[0.5, 2.0]`, `force_ext_ramp_frac=0.25`, `force_ext_activation_prob_per_step=0.01`
    - 共用：`enable_left=True`, `enable_right=True`, `left_wrist_body_name="left_wrist_yaw_link"`, `right_wrist_body_name="right_wrist_yaw_link"`
  - `@dataclass(frozen=True)` + unit test

- [ ] **Task A'-2：`WristComplianceCommand` command term**（自己写最简 state machine，不 port GH）
  - `src/holosoma/holosoma/managers/command/terms/wbt_force.py`
  - per-env + per-wrist 状态机（state ∈ {COOLDOWN, RAMP_UP, HOLD, RAMP_DOWN}），用整数 counter 推进 —— 不引入 `TemporalLerp` 或其他通用 lerp class
  - F_cmd 和 F_ext 两路用**相同的**状态机实现，各自独立参数
  - 方向采样：`d = torch.randn(n_active, 3); d /= d.norm(dim=-1, keepdim=True).clamp(min=1e-6)`
  - 每 step 调 `step()`：
    1. 对每 wrist 判断是否需要触发新 episode（`state == COOLDOWN` + `cooldown_remaining == 0` + Bernoulli(`activation_prob_per_step`)）
    2. 新 episode：采方向 + 幅度 + duration + ramp_up_steps + ramp_down_steps；初始化 `state = RAMP_UP`
    3. 状态推进：RAMP_UP count → HOLD count → RAMP_DOWN count → COOLDOWN（cooldown_steps）→ 可触发下一轮
    4. 输出 force = peak_dir × (当前 ramp 插值系数 ∈ [0, 1])
  - 暴露：`force_cmd_b: [N, 6]`（flatten 自 `[N, 2, 3]`）, `force_ext_w: [N, 2, 3]`
  - 单元测试（pure CPU）：
    - reset 后全 0
    - F_cmd 和 F_ext 互相独立（一个激活不影响另一个）
    - 梯形剖面形状正确（ramp up → hold → ramp down，峰值等于采样值）
    - `force_ext_w` norm 在 `force_ext_magnitude_range` 内
    - `enable_left=False` 时左 wrist 永远 0

- [ ] **Task A'-3：`WholeBodyTrackingForceInjected` env 子类**
  - `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py`
  - 继承 `WholeBodyTrackingManager`
  - **Imports（关键，Codex 评审意见 #5 修正）**：
    ```python
    from isaaclab.utils.math import quat_apply_inverse
    # NOTE: IsaacLab 的 body_quat_w 是 wxyz；不要用 holosoma.utils.rotations 里的 helper，
    # 那些 helper 假设 xyzw frame，用错会静默旋转错方向
    ```
  - **Body ID resolution（Codex 评审意见 #6 修正——用 Holosoma 惯用 pattern）**：
    ```python
    def __init__(self, ...):
        super().__init__(...)
        cfg = self.command_manager.get_term_cfg("wrist_compliance_command")
        # Holosoma pattern: find_rigid_body_indice → body_ids (见 push.py, virtual_gantry.py)
        left_holosoma_idx = self.simulator.find_rigid_body_indice(cfg.left_wrist_body_name)
        right_holosoma_idx = self.simulator.find_rigid_body_indice(cfg.right_wrist_body_name)
        # body_ids 把 Holosoma index 映射到 IsaacLab articulation body id
        self._left_wrist_isaac_id = int(self.simulator.body_ids[left_holosoma_idx])
        self._right_wrist_isaac_id = int(self.simulator.body_ids[right_holosoma_idx])
        self._wrist_body_ids_t = torch.tensor(
            [self._left_wrist_isaac_id, self._right_wrist_isaac_id], device=self.device,
        )
        # Debug accessor for e2e test (avoids reading private _external_force_b)
        self.last_applied_force_w_by_body_id: dict[int, torch.Tensor] = {}
    ```
  - **Override `_apply_force_in_physics_step`**：
    ```python
    def _apply_force_in_physics_step(self):
        super()._apply_force_in_physics_step()  # 原本的 action apply
        term = self.command_manager.get_state("wrist_compliance_command")
        if term is None:
            return
        force_w = term.force_ext_w  # [N, 2, 3] world frame (cached from last control step)
        # world → body frame per body; IsaacLab body_quat_w is wxyz
        left_quat_w = self.simulator._robot.data.body_quat_w[:, self._left_wrist_isaac_id]
        right_quat_w = self.simulator._robot.data.body_quat_w[:, self._right_wrist_isaac_id]
        force_b_left = quat_apply_inverse(left_quat_w, force_w[:, 0])
        force_b_right = quat_apply_inverse(right_quat_w, force_w[:, 1])
        forces_body = torch.stack([force_b_left, force_b_right], dim=1)  # [N, 2, 3]
        torques = torch.zeros_like(forces_body)
        self.simulator._robot.set_external_force_and_torque(
            forces=forces_body,
            torques=torques,
            env_ids=None,  # all envs
            body_ids=self._wrist_body_ids_t,
        )
        # Debug snapshot (for e2e test; keep detached copy to avoid graph retention)
        self.last_applied_force_w_by_body_id = {
            self._left_wrist_isaac_id: force_w[:, 0].detach().clone(),
            self._right_wrist_isaac_id: force_w[:, 1].detach().clone(),
        }
    ```
  - 单测（isaacsim gated）：覆盖 Spike 1 的断言，外加 reset 清零、`last_applied_force_w_by_body_id` debug accessor 正确性

**Phase 2（observation）：**

- [ ] **Task A'-4：两个 obs term + 单测**
  - `src/holosoma/holosoma/managers/observation/terms/wbt_force.py`
  - `wrist_force_command(env) -> [N, 6]`：返回 `WristComplianceCommand.force_cmd_b`
  - `wrist_force_ext_privileged(env) -> [N, 6]`：返回 `WristComplianceCommand.force_ext_w.reshape(N, 6)`
  - 单测：用 `SimpleNamespace` + fake `_FakeCommandManager` 验证输出

**Phase 3（reward）：**

- [ ] **Task A'-5：`wrist_force_position_tracking_exp` reward term**
  - `src/holosoma/holosoma/managers/reward/terms/wbt_force.py`
  - 签名：`wrist_force_position_tracking_exp(env, sigma, K_virtual, left_wrist_body_name, right_wrist_body_name) -> [N]`
  - 实现按 §A'.2 双力公式：
    ```python
    motion = env.command_manager.get_state("motion_command")
    wrist_cmd = env.command_manager.get_state("wrist_compliance_command")
    force_cmd_b = wrist_cmd.force_cmd_b.view(-1, 2, 3)       # body-yaw frame
    force_ext_w = wrist_cmd.force_ext_w                       # world frame
    yq = yaw_quat(env.base_quat, w_last=True)
    force_cmd_w_left  = quat_apply(yq, force_cmd_b[:, 0], w_last=True)
    force_cmd_w_right = quat_apply(yq, force_cmd_b[:, 1], w_last=True)
    force_cmd_w = torch.stack([force_cmd_w_left, force_cmd_w_right], dim=1)
    F_total_w = force_ext_w + force_cmd_w                     # [N, 2, 3] ★ 双力叠加
    left_idx, right_idx = resolve_tracked_index(left_wrist_body_name, right_wrist_body_name)
    wrist_target_w = motion.body_pos_relative_w[:, [left_idx, right_idx], :]
    wrist_actual_w = motion.robot_body_pos_w[:, [left_idx, right_idx], :]
    wrist_target_shifted_w = wrist_target_w + F_total_w / float(K_virtual)
    error = torch.sum(torch.square(wrist_target_shifted_w - wrist_actual_w), dim=-1)
    return torch.exp(-error.mean(-1) / (sigma ** 2))
    ```
  - 单测（pure CPU）覆盖 4 个稳态情形（§A'.2 表格）+ yaw rotation 验证

**Phase 4（config presets & experiment）：**

- [ ] **Task A'-6：command preset `g1_29dof_wbt_force_command`**
  - `src/holosoma/holosoma/config_values/wbt/g1/command_force.py`
  - motion_command（原样继承）+ `wrist_compliance_command`
  - `src/holosoma/holosoma/config_values/command.py` DEFAULTS 追加
  - 回归测试：断言 `g1_29dof_wbt_command` object identity 未变

- [ ] **Task A'-7：observation preset `g1_29dof_wbt_force_observation`**（v9 极简，无 history）
  - `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py`
  - **2 个 group，`history_length=1`（跟 baseline 一致）**：
    - `actor_obs`：baseline 原 6 个 term **+ `wrist_force_command`** (6 dim)
    - `critic_obs`：baseline 原 10 个 term **+ `wrist_force_command`** (6 dim) **+ `wrist_force_ext_privileged`** (6 dim, critic only)
  - **不改 baseline** `g1_29dof_wbt_observation` 对象：新 preset 通过**新建 dataclass** 而不用 `replace`/共享引用（避免 aliasing）
  - DEFAULTS 注册 `"g1_29dof_wbt_force"`
  - 回归测试：
    - `assert g1_29dof_wbt_observation is module.g1_29dof_wbt_observation`（identity check）
    - 新 preset actor group term 清单 = baseline 6 term + `wrist_force_command`
    - 新 preset critic group term 清单 = baseline 10 term + 2 个新 term
    - 维度测试：`get_obs_dims()` 算出 actor = 160 dim, critic = 298 dim

- [ ] **Task A'-8：reward preset `g1_29dof_wbt_force_reward`**
  - `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py`
  - 原 9 个 reward term 全部继承（"以 holosoma 为主"）
  - 追加 `wrist_force_position_tracking_exp`（weight=2.0，K_virtual=`G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M`，σ=0.3）
  - DEFAULTS 注册
  - 回归测试：断言 `g1_29dof_wbt_reward` 未变；新 preset 的前 9 项 hash 等于原 preset

- [ ] **Task A'-9：experiment `g1_29dof_wbt_force`**（继承 baseline algo）
  - `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` 末尾追加（不改 `g1_29dof_wbt`）
  - `env_class="holosoma.envs.wbt.wbt_force_injected:WholeBodyTrackingForceInjected"`
  - command/observation/reward 用新 preset
  - **algo 完全继承 baseline**：2 group 结构保持（`actor_obs` / `critic_obs`），`input_dim=["actor_obs"]` / `["critic_obs"]` **一字不动**。MLP 的 input dim 会自动适配（baseline actor 154 → 160；critic 286 → 298，仅 +4%）
  - 层宽保持 baseline `[512, 256, 128]`
  - 其他（robot、simulator、curriculum、randomization、terrain、termination）完全继承
  - `src/holosoma/holosoma/config_values/experiment.py` DEFAULTS 追加
  - 回归测试：baseline `g1_29dof_wbt` object 的 `algo` field identity 未动 (`assert g1_29dof_wbt.algo is ORIGINAL_ALGO`)

**Phase 5（集成测试 & 文档）：**

- [ ] **Task A'-10：tyro CLI smoke test**
  - `tests/test_wbt_force_cli.py`
  - subprocess 跑 `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-force --help` 几秒内 exit 0
  - 断言 baseline `exp:g1-29dof-wbt --help` 依旧 ok
  - 断言 `--help` 输出里出现 `wrist_compliance_command` 可配参数

- [ ] **Task A'-11：IsaacSim e2e（`@pytest.mark.isaacsim`）**
  - `tests/e2e/test_wbt_wrist_force_e2e.py`
  - **测试夹具覆盖 activation_prob = 1.0（Codex 评审意见 HIGH #2 修正）**：config 的默认 `activation_prob_per_step=0.01` 意味着 20 step × 2 wrist 有 ~67% 概率全程零力，e2e 断言会 flaky。测试时用 `replace(cfg.wrist_compliance_command.params, force_ext_activation_prob_per_step=1.0, force_cmd_activation_prob_per_step=1.0, force_ext_cooldown_range_s=(0.0, 0.0), force_cmd_cooldown_range_s=(0.0, 0.0))` override，保证每个 step 都有力激活
  - 初始化新 experiment、跑 20 control step
  - 断言：
    - **红线 1 强约束（Codex 评审意见 #11 修正）**：通过 env 子类暴露的 `last_applied_force_w_by_body_id` debug accessor 验证——它的 keys 只能是 `{left_wrist_isaac_id, right_wrist_isaac_id}`；**不**直接读 IsaacLab 私有 `_external_force_b` buffer（那是 implementation detail）
    - 辅助断言：command term 内部 `force_ext_w` 形状是 `[N, 2, 3]`（只有 2 个 wrist channel），没有其他 body channel
    - `force_ext_w` 连续 20 step 每步都有非零 sample（activation=1.0 下应该确定性非零，flake-free）
    - `force_cmd_b` 的 ramp 剖面可观察到（连续 step 值平滑变化）
    - reward 张量非 NaN、在 `[0, 1]`
    - actor obs dim = 160（baseline 154 + F_cmd 6）
    - critic obs dim = 298（baseline 286 + F_cmd 6 + F_ext privileged 6）
  - **独立的 random-activation smoke 断言（不带 override）**：跑 200 step 默认 config，仅断言 `force_ext_w.abs().sum() > 0` 至少命中一次（200 步下 P(全零) = `0.99^400 ≈ 1.8%`，可接受，但若需要完全 flake-free 就只保留上面 activation=1.0 fork）

- [ ] **Task A'-12：用户文档 `docs/wbt-wrist-force-training.md`**
  - 启动命令 + 关键参数表
  - "F_ext 和 F_cmd 只施加 wrist" 显式声明（红线 1）
  - K_virtual 来源（`_k_virtual.py` 常量 + §A'.4 说明"这是 training hyperparameter，不是测得刚度"）
  - 部署端：上层 API 只需给 F_cmd 6-D body-yaw frame
  - 调参建议（σ、F_cmd 量级）
  - **v1 定位声明**：这是 "domain randomization against unobserved wrist F_ext"，不是 learned compliance；真 compliance 需要 proprio history 让 actor 隐式估 F_ext（v2）
  - v2 扩展说明（若训练不稳，考虑加 history / force_exd_penalty / adaptive compliance）

**Phase 6（sanity）：**

- [ ] **Task A'-13：全局 sanity**
  - pre-commit 全绿
  - mypy 全绿
  - `pytest -s --ignore=thirdparty --ignore=src/holosoma_inference -m "not isaacsim and not requires_inference"` 全绿
  - `pytest -m isaacsim` 新增 Spike 1 + Task A'-3 + Task A'-11 全绿
  - `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt --help` 依旧 ok（baseline 不受污染）
  - `demo_scripts/demo_omomo_wb_tracking.sh` 第 4 步的 baseline 命令仍能启动（不需要真跑完，启动 5s 后 SIGINT 就行）

### A'.7b Observation 结构（v9：不加 history）

> **v8 决定：不做 history。** Baseline Holosoma `g1_29dof_wbt` 本来就 `history_length=1`（无 history）、policy 仍然能学 motion tracking —— 说明当前 proprio `dof_pos / dof_vel / actions` 单步观测已足够。force tracking 加一层 "actor 要学隐式估 F_ext" 的需求，但 v1 先**不**上 history，看训练曲线决定是否需要补。

#### A'.7b.1 Holosoma baseline 的实测 obs 维度

| Term | dim | 来源 |
|---|---|---|
| `motion_command` (`joint_pos + joint_vel`) | **58** | `wbt.py:827` = `cat([joint_pos_29, joint_vel_29])` |
| `motion_ref_pos_b` | 3 | |
| `motion_ref_ori_b` (rot6d) | 6 | |
| `robot_body_pos_b` | 42 (14×3) | |
| `robot_body_ori_b` (rot6d) | 84 (14×6) | |
| `base_lin_vel` | 3 | |
| `base_ang_vel` | 3 | |
| `dof_pos`, `dof_vel`, `actions` | 29 each | |

**Baseline**：actor_obs = **154 dim**，critic_obs = **286 dim**。

#### A'.7b.2 v1 observation 结构（极简：只加 2 个 term）

和 baseline 完全同构，**2 个 obs group**，**`history_length=1`**（跟 baseline 一样），只追加：

**Actor group** 相对 baseline：
- 原有 6 个 term 全保留
- **+ `wrist_force_command` (6 dim)** —— actor 看力指令

**Critic group** 相对 baseline：
- 原有 10 个 term 全保留
- **+ `wrist_force_command` (6 dim)** —— 和 actor 对齐
- **+ `wrist_force_ext_privileged` (6 dim)** —— critic 开挂看 ground-truth F_ext

#### A'.7b.3 维度估算

- actor: 154 + 6 = **160 dim**（baseline 154 × 1.04，几乎没扩）
- critic: 286 + 6 + 6 = **298 dim**（baseline 286 × 1.04）

相比 baseline 几乎没扩（+4%）。层宽 `[512, 256, 128]` 完全够，不改。

#### A'.7b.4 为什么不给 actor F_ext

- F_ext 是 **sim-only ground truth**，部署时 humanoid wrist 无 F/T 传感器 → actor 看不到
- critic 可以开挂（降 value 方差）
- F_cmd 两边都给（部署时 actor 真拿得到）

#### A'.7b.5 关于 history（v1 不做，v2 再考虑）

**v1 不加 history。** 理由：
- Holosoma baseline 证明单步 obs 已能训 motion tracking
- 加 history 会扩维度、增训练成本、引入更多设计选择
- 是否需要 history 是**训练后观察**的问题，不是先验的必需设计

**重要但 partial observability 警告（Codex 评审意见 #4 修正）**：

reward 公式依赖 F_ext，但 actor obs **不**含 F_ext 也**无** history。force 刚发生的 control step，actor 没有任何线索能知道 F_ext 的大小方向——它只能从"wrist 位置相比 motion target 的偏离"间接感知，且这个偏离是下一个 step 才可见（本步的 dof_pos 还是 force 作用前的）。

因此 **v1 的数学本质是 "domain randomization against unobserved wrist perturbations"，不是 "learned compliance"**——policy 学的是一种对未知 wrist 力鲁棒的 tracking 行为，而不是通过观察力调整响应。真正的 learned compliance 需要 actor 能隐式估 F_ext，这需要 proprio 历史。

**若 v1 训练曲线显示 wrist tracking 在 F_ext 事件期间显著退化**，v2 的补救路线是加最简单的 history：单独一个 `dof_pos_history_stateful_term`，连续 5 步 control step。现在不做，但文档和 self-review 需要诚实声明这个 limitation。

**这条意味着**（v9 现状）：
- 不需要 copy GH 的 3 个 history class
- Task A'-8a 不存在（history 不做）
- §A'.5.1 明确所有 third-party code 都只读不 port，包括 ramp profile utility
- 实现工作量最小

---

### A'.8 与 Path B 的对比（HISTORICAL — 为什么选 A' 而非 B）

| 维度 | **Path A'（推荐）** | Path B（CHIP 纯版） |
|---|---|---|
| 满足"F_cmd 作为 actor 输入" | ✅ | ✗ |
| 主动施力接口语义 | F_cmd 直给 Newton | 间接：target 放墙里 + 1/k |
| adaptive compliance (1/k) | ✗ | ✅ |
| 改动 reward | +1 term | 0 |
| 改动 obs（motion goal）| 不动原 motion_command | 要 patch wrist 分量 |
| Spike 数 | 1 | 3 |
| Task 数 | 14 | 13 |
| 风险 | 低-中 | 中 |

### A'.9 Self-Review（Path A' 专属，v9）

| 用户要求 | 覆盖 |
|---|---|
| "F_ext 和 F_cmd 只加 wrist" | §A'.0 红线 1；Task A'-3 注入到 `left/right_wrist_yaw_link`；Task A'-11 断言通过 env debug accessor（`last_applied_force_w_by_body_id` 的 keys ⊆ wrist body ids） |
| "K 保持跟 holosoma 原始一致而固定" | §A'.0 红线 2 + §A'.4 说明这是 **training-time hyperparameter 不是测得刚度**；Task A'-0 手写 `_k_virtual.py = 100.0 N/m`（从 Δx 工作空间反推，让同向叠加合力不超 arm reach）；inference 端可用常数 α 校正 |
| "F_cmd 作为 actor 输入" | Task A'-4 obs term；Task A'-7 actor group；§A'.7b.4 显式论证不给 F_ext 给 actor |
| "v1 不做 history（极简）" | §A'.7b.5 v1 不做，`history_length=1`；actor 160 dim / critic 298 dim，仅 +6 dim over baseline |
| "不加 1/k（compliance 可调）" | §A'.10 已知未决项列为 future extension，v1 不实现 |
| "CHIP 仅作对照参考，没有可 port 的代码" | CHIP 没开源 → 我们没用 CHIP 的任何实现细节，只借鉴"F_ext 注入 + critic privileged"这种 architecture-level 思想 |
| "UniFP 是 IsaacGym + quadruped → 不 port 代码" | 只借用 reward 公式 `g̃ = g + (F_ext + F_cmd) / K`（Task A'-5 实现），**不**抄 UniFP 的 Python 代码 |
| "GH 的 code 只作参考，不 import 不 copy（v8）" | §A'.5.1 明确：只读 GH 源看 API 用法，最终 Holosoma 代码里**一行 GH 代码都不出现**；Task A'-2 的 ramp state machine 自己写简单版；Task A'-3 直接用 Holosoma 已有的 `set_external_force_and_torque` |
| "reward 以 holosoma 为主" | Task A'-8 原 9 reward 全部保留，只追加 1 项；回归测试断言 identity |
| "不影响 Holosoma 已有 training/testing code" | 红线 5 枚举所有允许修改的文件（6 个）+ 禁止所有其他文件；每 preset 都有 `assert original is module.original` identity check；Task A'-13 Sanity 跑 `exp:g1-29dof-wbt --help` 验证 baseline 启动没坏；Task A'-13 跑 `demo_scripts/demo_omomo_wb_tracking.sh` 的第 4 步验证 demo script 没坏 |

**v9 诚实声明（由 Codex 评审触发）：**
- **"Learned compliance" 是 v2 能力，不是 v1。** v1 actor 无 history、无 F_ext obs，force 发生时 actor 无法及时知道——数学上 v1 训练的是对未知 wrist 扰动鲁棒的 tracking（domain randomization），不是"感受力并调整响应"。若需后者（wiping、door opening 等真 compliance 应用），必须在 v2 加 proprio history
- **K_virtual=100 N/m 不是测得的 G1 Cartesian 刚度。** 它是 reward exchange rate hyperparameter，选这个值是为了让同向叠加合力（max 60N）除以它得到的 Δx 刚好到 arm reach 边缘，产生合理的 kinematic "soft upper bound"。若推演发现 F_real/F_cmd 系统性偏离，inference 端用标量 α 校正即可（§A'.4 末尾）

### A'.10 已知未决项（留给执行期/训练后决定）

v1 的"不做、先看"清单：

1. **history**：v1 不加。若 v1 训练发现 policy 对 F_ext 响应慢或 F_real/F_cmd 相关性差 → v2 加单个 stateful `dof_pos_history` term（简单版，连续 5 步）
2. **F_ext 激活概率 0.01/step**：对应平均每 100 step（2s @ 50Hz）一次 episode。若 compliance 行为学不出可升到 0.02-0.05
3. **双 wrist 是否同时采样力**：v1 两路 Bernoulli 独立，允许同时发力。若同时双力 degrade motion tracking 严重 → v2 限制"最多一路激活"
4. **force range saturation 监控**：若训练曲线显示 `||F_ext + F_cmd|| > 45 N` 的 step 占比 > 20% 且 wrist tracking 在这些 step 明显退化，说明 kinematic saturation 吃掉太多 sample → v2 把 `force_ext_magnitude_range` 上限从 30 降到 15-20 N
5. **是否加 `force_exd_penalty`**：v1 不加，先看 baseline 曲线；若 policy 学到 overshoot 发力（F_real >> F_cmd）再补
6. **`1/k` adaptive compliance input**（Path B 的核心）：若未来需要 stiff↔compliant 可调，在 A' 基础上加 `compliance_range=[0, 0.05]` 的 episode-level 采样 + actor obs；reward 改成 `target_shifted = g + (F_ext + F_cmd)·(1/k)`。3-5 task，不推翻现有结构

---

## ═══════════════════════════════════════════════════════════════════
## PATH B: CHIP 式 Hindsight Perturbation — **历史参考，DO NOT IMPLEMENT**
## ═══════════════════════════════════════════════════════════════════

> ⛔ **STOP — Path B 与硬约束 "F_cmd 作为 actor 输入" 冲突（CHIP 没有 F_cmd），已被否决。**
>
> 保留是为了对比"为何选 A' 而非 B"。请勿按本节写代码。

### B 背景与原理

**论文：** CHIP (`third_party/CHIP.pdf`)
- 标题："CHIP: Adaptive Compliance for Humanoid Control through Hindsight Perturbation"
- 机构：NVIDIA / Stanford / UT Austin
- 平台：Unitree G1（与 Holosoma 对齐）

**方法一句话：** 在 RL 训练中对 end-effector 施加**真实扰动力 F_ext**，但**不改 reward**；改的是 actor 看到的 tracking goal：`g_hind = g − (1/k)·f_ext`。policy 为了最大化 reward（追踪**原 g**），被迫对外力呈现 1/k 的 compliance 行为。部署时没有 F_ext，policy 同样按 1/k 比例对接触产生的反力做柔性响应。

**核心公式：**

| 符号 | 含义 | 维度 | frame |
|---|---|---|---|
| `g_wrist` | wrist 的原 motion tracking target | `[N, 2, 3]` | world |
| `f_ext_wrist_w` | 当前步 sim 真实注入 wrist 的扰动力 | `[N, 2, 3]` | world |
| `1/k` | per-wrist compliance coefficient | `[N, 2]` | 标量 (m/N) |
| `g_wrist_hind` | hindsight-perturbed wrist target (actor obs 用) | `[N, 2, 3]` | world |

训练期每个 control step：

```
g_wrist_hind = g_wrist − (1/k) · f_ext_wrist_w      # actor 看到这个
reward.inputs = g_wrist (不改), f_ext (不进入 reward 公式)
critic.obs += f_ext_wrist_w (privileged)
```

### B.1 新文件结构

| 路径 | 职责 |
|---|---|
| `src/holosoma/holosoma/config_types/command.py` **[修改]** | 追加 `WristComplianceConfig` |
| `src/holosoma/holosoma/managers/command/terms/wbt_compliance.py` | `WristComplianceCommand`：per-episode 采样 1/k；per-step 维护 F_ext 梯形剖面（参考 CHIP Fig. 9）；存 `compliance_2d`、`force_ext_w`、`force_ext_b`（暴露给 obs）、`hindsight_wrist_offset_w`（暴露给 obs） |
| `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py` | `WholeBodyTrackingForceInjected(WholeBodyTrackingManager)`：override `_apply_force_in_physics_step` 把 `force_ext_w` 通过 IsaacLab ArticulationView 的 `set_external_force_and_torque` 写到 left/right wrist rigid body |
| `src/holosoma/holosoma/managers/observation/terms/wbt_compliance.py` | 三个 obs fn：`wrist_compliance_command` (2-D)、`motion_command_hindsight` (替换 `motion_command` 给 actor 的 wrist 分量)、`wrist_force_ext_world_privileged` (6-D, critic only) |
| `src/holosoma/holosoma/managers/observation/history_stacker.py` | 10-step proprio / 11-step past-actions rolling buffer（如果 Holosoma 没有现成实现） |
| `src/holosoma/holosoma/managers/reward/terms/wbt_force.py` | **不新增 reward term**（复用 Path A 的文件名但内容为空 noop）；Holosoma 现有 reward 全部保持 |
| `src/holosoma/holosoma/config_values/wbt/g1/command_compliance.py` | 新增 `g1_29dof_wbt_compliance_command` |
| `src/holosoma/holosoma/config_values/wbt/g1/observation_compliance.py` | 新增 `g1_29dof_wbt_compliance_observation`：swap `motion_command` → `motion_command_hindsight` 给 actor；actor 增加 `wrist_compliance_command`；critic 增加 `wrist_force_ext_world_privileged`；actor/critic 都加 history-stacking |
| `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` **[修改]** | 新增 `g1_29dof_wbt_force_chip` 实验注册（env_cls → `WholeBodyTrackingForceInjected`，command/obs 用 CHIP preset，reward 继承 `g1_29dof_wbt_reward` 原样） |
| `src/holosoma/holosoma/config_values/command.py` **[修改]** | 注册 `"g1_29dof_wbt_compliance_command"` |
| `src/holosoma/holosoma/config_values/observation.py` **[修改]** | 注册 `"g1_29dof_wbt_compliance_observation"` |
| `src/holosoma/holosoma/config_values/experiment.py` **[修改]** | 注册 `"g1_29dof_wbt_force_chip"` |
| 各路径对应 `tests/` | 单测 + 集成 |

**必须保持原样（一行不改）：**
- Holosoma 所有现有 reward term
- `WholeBodyTrackingManager` 基类
- 现有 `motion_command` command term 和 `motion_command` obs term（新 obs term 并列存在）

### B.2 关键实现风险与前置 spike

Path B 开执行前，**必须先做三个前置 spike**，任一失败整个方案需要调整设计：

#### Spike 1: IsaacSim 外力注入 API 可用性

**目标：** 确认 `WholeBodyTrackingManager` 的 `_apply_force_in_physics_step` hook 能通过 `simulator.robot` 的 IsaacLab `ArticulationView` 调用 `set_external_force_and_torque([num_envs, num_bodies, 3])` 并在下一 physics step 生效。

**验证：**
```python
# 在一个最小脚本里对 left_wrist_yaw_link 每步注入 world-frame 10N (+X)
# 跑 100 step，读 root state 和 wrist world pos，验证 wrist 确实被力推动
```

**如果 API 不 directly available：** 需要在 `simulator/isaacsim/isaacsim.py` 加 `apply_rigid_body_force_at_body_names(env_ids, body_names, forces_w)` 方法。

#### Spike 2: Observation history stacker 机制

**目标：** 确认 Holosoma `managers/observation/manager.py` 是否支持 "last N steps of term X concatenated"。CHIP 要的是：
- actor obs 里有 10 步 proprio + 11 步 past actions
- critic 同上

**如果没有：** 需要加一个 `HistoryStackedObservationGroup` 或 obs term 装饰器，在每个 step 存入 ring buffer 并在 `compute` 时 concat 返回。

#### Spike 3: 替换 actor 的 motion_command obs 子集而不影响 reward

**目标：** reward 的 `motion_relative_body_position_error_exp` 读 `MotionCommand.body_pos_relative_w`（原 g）。actor 看到的 `motion_command_hindsight` obs 要返回**同一 MotionCommand 里的 body_pos_relative_w，但 wrist 两个分量替换为 hindsight 值**。

**陷阱：** 绝对不能 mutate `MotionCommand.body_pos_relative_w`。正确做法是 obs fn 取一个 clone，patch 掉 wrist 两个 index，返回新 tensor。

**验证：** 加单元测试 `test_hindsight_obs_does_not_mutate_source_motion_command`。

### B.3 任务分解（简版；每个 task 的展开格式与 Path A 相同）

> 下列 13 个任务假定前置 spike 全部通过。开始执行前，先做三个 spike；如果 spike 暴露出设计问题，回到本节修正 plan。

#### Task B-0: 前置三个 spike（**强制**，先于所有其他 task）

- [ ] Spike 1: IsaacSim 外力注入在 WBT env 里能不能每 physics step 生效（写一个 `tests/spikes/test_chip_spike1_force_injection.py`，`@pytest.mark.isaacsim`，跑一次看结果）
- [ ] Spike 2: 观察 `managers/observation/manager.py` 是否有 history buffer 机制；若无，记录所需增补量
- [ ] Spike 3: 写一个 pure-CPU 单元测试验证"patch motion command wrist 分量不 mutate 源"

**Gate：三个 spike 全绿再走 Task B-1。** 任何一个红了，停下找用户讨论设计调整。

#### Task B-1: `WristComplianceConfig` dataclass

新增字段（所有默认值都从 CHIP 论文 §V-A、Tab. III、Fig. 9 抄）：

```python
@dataclass(frozen=True)
class WristComplianceConfig:
    # --- 1/k sampling ---
    compliance_range: list[float] = field(default_factory=lambda: [0.0, 0.05])  # m/N
    compliance_log_uniform: bool = False                                         # CHIP 论文用 uniform
    # --- F_ext sampling (CHIP Fig. 9) ---
    force_magnitude_range: list[float] = field(default_factory=lambda: [0.0, 40.0])
    force_duration_range_s: list[float] = field(default_factory=lambda: [1.0, 3.0])
    force_cooldown_range_s: list[float] = field(default_factory=lambda: [0.5, 2.0])
    # Trapezoidal profile: ramp-up 0.25 · duration, hold 0.5 · duration, ramp-down 0.25 · duration
    ramp_up_frac: float = 0.25
    ramp_down_frac: float = 0.25
    activation_prob_per_step: float = 0.01
    # --- Body names ---
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"
    enable_left: bool = True
    enable_right: bool = True
```

#### Task B-2: `WristComplianceCommand` command term

接口（详细实现留到执行时按 Path A 的 TDD 流程走）：

```python
class WristComplianceCommand(CommandTermBase):
    # Per episode (reset), sample 1/k per-wrist from compliance_range.
    # Per step, maintain ramp-up→hold→ramp-down F_ext per wrist (trapezoid).
    #
    # Public state (read by obs / env):
    compliance_2d: torch.Tensor       # [N, 2]   (1/k)
    force_ext_w: torch.Tensor         # [N, 2, 3]  world-frame
    # For hindsight obs: offset to subtract from wrist goal
    hindsight_offset_w: torch.Tensor  # [N, 2, 3] = compliance_2d.unsqueeze(-1) * force_ext_w
```

单元测试覆盖：
- reset 后 1/k 在 range 内；different envs 不同值
- F_ext 剖面梯形；峰值在中段
- `hindsight_offset_w == compliance_2d.unsqueeze(-1) * force_ext_w`
- `enable_left=False` → compliance_2d[:,0]=0 and F_ext[:,0]=0

#### Task B-3: `WholeBodyTrackingForceInjected` env 子类

```python
class WholeBodyTrackingForceInjected(WholeBodyTrackingManager):
    def _apply_force_in_physics_step(self):
        super()._apply_force_in_physics_step()  # actions
        term = self.command_manager.get_state("wrist_compliance_command")
        if term is None:
            return
        # Write force_ext_w to the two wrist rigid bodies via IsaacLab API
        self.simulator.apply_rigid_body_force_by_body_names(
            env_ids=None,
            body_names=[term.cfg_compliance.left_wrist_body_name,
                        term.cfg_compliance.right_wrist_body_name],
            forces_w=term.force_ext_w,  # [N, 2, 3]
        )
```

单元测试：`@pytest.mark.isaacsim` 注入 (+X, 20N) 到 left wrist，对比有/无注入时的 wrist world xyz 位移方向正确。

#### Task B-4: `wrist_compliance_command` obs term

返回 `compliance_2d`。2-D。actor + critic 都看到。

#### Task B-5: `motion_command_hindsight` obs term (**CHIP 核心**)

```python
def motion_command_hindsight(env) -> torch.Tensor:
    """Return the same flattened tensor as `motion_command`, but the two wrist
    entries' position components have been shifted by -hindsight_offset_w.

    IMPORTANT: does not mutate MotionCommand state; clones then patches."""
    motion = env.command_manager.get_state("motion_command")
    compliance_term = env.command_manager.get_state("wrist_compliance_command")
    # body_pos_relative_w: [N, 14, 3]; find left/right wrist index in tracked bodies.
    pos_rel_w = motion.body_pos_relative_w.clone()
    left_idx = motion.motion_cfg.body_names_to_track.index(
        compliance_term.cfg_compliance.left_wrist_body_name)
    right_idx = motion.motion_cfg.body_names_to_track.index(
        compliance_term.cfg_compliance.right_wrist_body_name)
    offset = compliance_term.hindsight_offset_w  # [N, 2, 3]
    pos_rel_w[:, left_idx]  -= offset[:, 0]
    pos_rel_w[:, right_idx] -= offset[:, 1]
    # ... match flattening/format of the original motion_command obs ...
    return flattened
```

单元测试：
- `test_hindsight_matches_motion_when_force_zero` (F_ext=0 → hindsight == original motion)
- `test_hindsight_does_not_mutate_source_motion_command`
- `test_non_wrist_body_targets_unchanged`

#### Task B-6: `wrist_force_ext_world_privileged` obs term

Critic-only。返回 `force_ext_w.reshape(N, 6)`。

#### Task B-7: observation history stacker

**只有 Spike 2 表明 Holosoma 没有现成实现时才做这个 task。**

新增通用 `HistoryStackedTerm` wrapper 或在 observation manager 加 "stack last N steps" 机制。需要与现有 observation group 框架配合，在 env reset 时清空 history。

单元测试：term 连续 step 3 次，第 4 次请求 `num_history=3` → concat [t-2, t-1, t] 的 obs 值，padding 用 0 或 repeat first 步。

#### Task B-8: command preset `g1_29dof_wbt_compliance_command`

- 原有 `motion_command` 保留（reward 需要用它的原 g）
- 新加 `wrist_compliance_command`（类型 `WristComplianceCommand`）

#### Task B-9: observation preset `g1_29dof_wbt_compliance_observation`

**Actor group** 相对于 `g1_29dof_wbt_observation` 的变化：
1. 把 `motion_command` 换成 `motion_command_hindsight`（或**并列加**，见下面讨论）
2. 新增 `wrist_compliance_command`
3. 把 proprio / actions 相关的 term 套上 history stacker（N=10 for proprio, N=11 for actions）

**Critic group：**
1. 继承 actor 的改动
2. 新增 `wrist_force_ext_world_privileged`

**关于 motion_command vs motion_command_hindsight 二选一还是并列：** CHIP 论文 Fig. 2 显示 actor **只看** hindsight goal，不看原 goal。因此采用**替换**而非并列（不然 policy 可以直接偷看原 g 忽略 compliance）。

#### Task B-10: experiment `g1_29dof_wbt_force_chip`

```python
g1_29dof_wbt_force_chip = ExperimentConfig(
    env_cls="holosoma.envs.wbt.wbt_force_injected:WholeBodyTrackingForceInjected",
    command_config="g1_29dof_wbt_compliance_command",
    observation_config="g1_29dof_wbt_compliance_observation",
    reward_config="g1_29dof_wbt_reward",   # ← CHIP 的 reward 就是 Holosoma 原 reward
    # rest inherited from g1_29dof_wbt
    ...
)
```

回归测试：断言原 `g1_29dof_wbt` experiment object identity / 字段都没变。

#### Task B-11: CLI smoke test

```bash
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-force-chip --help
```

#### Task B-12: IsaacSim e2e test（`@pytest.mark.isaacsim`）

- 初始化 env，跑 10 step
- 断言 `force_ext_w` 非零 at least 一次
- 断言 `hindsight_offset_w` 有 non-trivial 值
- 断言 actor obs dim 比 Path A 多 2 维（1/k）并且 history 长度正确
- 断言 critic obs dim 比 actor 多 6 维（F_ext privileged）
- 断言 reward 张量里 motion_tracking 部分 shape 与 `g1_29dof_wbt` baseline 一致（reward 没变）

#### Task B-13: 文档

`docs/wbt-wrist-force-chip-training.md`：
- 使用方法（启动命令）
- 部署端：上层如何给 wrist kinematic target + 1/k 实现"主动施力"（"把 target 放到表面里面" + 举例）
- 可选 damper model 的说明（inference 端实现，不影响 training）
- 与 Path A 的差异速查表

---

## Self-Review (HISTORICAL — Path A / Path B 讨论，不适用于 v9)

> ⛔ **DO NOT READ AS LIVE GUIDANCE.** 下方 Self-Review 是 v1-v3 时 Path A (UniFP 纯版) 和 Path B (CHIP 纯版) 的 spec 覆盖检查，里面的 task 编号（Task 1-12 / Task B-0~B-13）、`K=200` 示例、`WristForceConfig` 类名等都对应历史方案。
>
> **v9 Path A' 的 Self-Review 在 §A'.9（正文中段）+ §A'.10（已知未决项），那才是权威版本。**

### Path A: Spec 覆盖检查 (HISTORICAL)

| 用户要求 | 覆盖的任务 |
|---|---|
| "带 Force tracking 的 low level controller，不只是 perturbation" | Task 4 的 `wrist_force_position_tracking_exp`：用 UniFP 虚拟阻抗公式，是**主动力追踪**训练信号，非 perturbation。设计要点 §7 专门解释了 v1 不做 sim 注入的理由。 |
| "看一下 UniFP" | 整个 plan 的核心 reward 公式（Task 4、§1、§5）完全对齐 UniFP `_reward_tracking_ee_force_world`（code excerpt 引用在背景段）。 |
| "把 force command 作为输入输入到 actor 里面" | Task 3（obs term）+ Task 6（preset 加到 actor+critic 两个 group） |
| "看一下 gentle humanoid 是怎么处理 force tracking reward 的" | 背景里提供了 GentleHumanoid `force_reward` 的 code excerpt + 解释它是 passive compliance（与需求方向相反），说明**不**采用的原因。 |
| "通过 low level controller 主动施力" | 整个设计围绕主动力生成展开 —— 与 Task 1 的改版 v1 最大的差别就在这里。 |
| "以 holosoma 为主" | 现有 reward term **一行不改**（Task 7 的 `test_original_wbt_reward_unchanged` 验证）；只追加一个新 reward term。env/algo/robot 全部继承。 |
| "不要影响其他任何训练及测试代码" | 所有修改是纯增量（新文件 + DEFAULTS dict 末尾追加）。Task 5-8 的每个 preset 回归测试都有 `test_original_*_unchanged` 断言。 |
| "先写好 plan 放进 /docs 我审阅后执行" | 本文件 at `docs/plans/2026-05-01-wbt-wrist-force-controller.md` ✓ |

### Placeholder Scan

- 所有代码块都是完整实现，无 TBD / "implement later"
- 所有测试都是完整断言
- 所有 CLI 示例可直接运行
- `base_quat` 和 `yaw_quat`/`quat_apply` 的调用已在 Task 4 的具体代码里定位到 `src/holosoma/holosoma/utils/rotations.py`（`w_last=True` 对应 xyzw 格式）

### Type Consistency 检查

- `WristForceConfig.*_wrist_body_name` 字段在 Task 1 定义，Task 2 `WristForceCommand.setup` 读取，Task 5 命令 preset 不传（用默认），Task 7 reward preset 单独传（允许用户通过 reward params 覆盖）—— 两条路径彼此独立但字段名保持一致
- `force_cmd_b` 形状 `[num_envs, 6]` 在 Task 2 声明，Task 3（obs term）返回原样，Task 4（reward）reshape 到 `[num_envs, 2, 3]` 再旋转
- `left_tracked_idx` 和 `right_tracked_idx` 在 Task 4 的 `_resolve_wrist_indexes_in_tracked` 函数里通过 `body_names_to_track.index(name)` 得到，与 `MotionCommand.body_pos_relative_w` / `robot_body_pos_w` 的第二维对齐
- Reward term func 字符串 `"holosoma.managers.reward.terms.wbt_force:wrist_force_position_tracking_exp"` 在 Task 7 preset 与 Task 4 的 fn 定义匹配

### 关键设计一致性 checks

- **F_cmd = 0 时 reward = 1.0**（wrist 完美跟 motion）—— Task 4 的 `test_reward_one_when_wrist_matches_target_and_force_zero` 严格验证 ✓
- **F_cmd > 0 时 target 正确偏移** —— Task 4 的 `test_reward_one_when_wrist_at_shifted_target_under_force` 验证（left_F=(10,0,0), right_F=(0,10,0), K=200 → offsets (0.05,0,0) 和 (0,0.05,0)）✓
- **yaw-frame force 旋转到 world frame 正确** —— Task 4 的 `test_reward_yaw_quat_rotation_applied_to_force`（yaw=pi/2 时 body +X → world +Y）✓
- **ramp profile 无 NaN 边界情况** —— Task 2 的 `test_ramp_profile_monotonic_peak_at_middle` 验证峰值 10.0；实现里对 `ramp_up_s=0 / ramp_down_s=0` 的边界用 `max(ru, 1)` 和 `in_ramp_up` 显式 bool 做了保护

### Path B (CHIP): Spec 覆盖检查（仅在选 Path B 时适用）

| 要求 | 覆盖 |
|---|---|
| 对比 CHIP 与 GentleHumanoid / UniFP | "方法路径对比" 整节 |
| reward 不动（以 holosoma 为主） | Task B-10 reward_config 直接复用 `g1_29dof_wbt_reward`；Task B-12 e2e 断言 reward 形状与 baseline 一致 |
| force input 给 actor | 以 `1/k` 的 compliance command 替代 `F_cmd`；`motion_command_hindsight` 里隐式表达 F_ext（对 wrist target 做 shift）；actor 不直接看 F_ext（符合 CHIP 论文 Fig. 2） |
| 不影响其他训练/测试代码 | Path B 所有修改为增量；Task B-10 的回归 test 断言原 experiment 未改 |
| 前置风险可视化 | Task B-0 三个 spike 独立先跑；任一失败即暂停回到 plan |

### Path B: 已知未决项

- **Spike 2（history stacker）** 若 Holosoma 没有现成机制，需要额外扩展 observation manager，可能新增 50-100 行代码；该改动虽只影响新 obs group 的 term，但因为要接入 manager 的 setup/reset lifecycle，仍属于 global 改动。写 spike 测试时特别注意 reset 时清空 history。
- **F_ext 注入频率：** CHIP 论文是每 physics step（control decimation 为 4 → 每 4 physics step 一次 force 更新？）还是每 control step？Spike 1 需要确认并在 env 子类里选择正确 frequency。建议：per control step 更新 force，per physics step 应用同一 force 值到 sim（consistent with PD 扭矩的应用节奏）。
- **F_ext 采样频率 vs 1/k 采样频率：** 1/k 每 episode reset 时重采；F_ext 按梯形剖面 per episode 内多次激活。两者耦合注意：同一个 1/k 经历不同幅度 F_ext 的 episode 过程，policy 才能学到 `(1/k) × f → 位置偏移` 的 generalization。

### Placeholder Scan

- Path A 已全部实现，完整代码在 Task 1-12 中
- Path B 只给出接口 + 测试意图，**实现细节故意留到执行时按 TDD 展开**（与 Path A 相同格式）。开始执行 Path B 前必须把每个 Task B-N 的 Step 1-6 按 Path A 模板补全

### Path A: Type Consistency 检查

- `WristForceConfig.*_wrist_body_name` 字段在 Task 1 定义，Task 2 `WristForceCommand.setup` 读取，Task 5 命令 preset 不传（用默认），Task 7 reward preset 单独传（允许用户通过 reward params 覆盖）—— 两条路径彼此独立但字段名保持一致
- `force_cmd_b` 形状 `[num_envs, 6]` 在 Task 2 声明，Task 3（obs term）返回原样，Task 4（reward）reshape 到 `[num_envs, 2, 3]` 再旋转
- `left_tracked_idx` 和 `right_tracked_idx` 在 Task 4 的 `_resolve_wrist_indexes_in_tracked` 函数里通过 `body_names_to_track.index(name)` 得到，与 `MotionCommand.body_pos_relative_w` / `robot_body_pos_w` 的第二维对齐
- Reward term func 字符串 `"holosoma.managers.reward.terms.wbt_force:wrist_force_position_tracking_exp"` 在 Task 7 preset 与 Task 4 的 fn 定义匹配

### Path A: 关键设计一致性 checks

（Path A 的关键检查见前文段落 "关键设计一致性 checks"）

---

## Execution Handoff

Plan complete (v9) and saved to `docs/plans/2026-05-01-wbt-wrist-force-controller.md`.

**路径选择（推荐已加 ★）：**

- **★ Path A' v9 (推荐，最简)**：**14 task + 1 spike**（Task A'-0 就是写一个 `K_virtual=100.0 N/m` 常量文件；A'-10~13 测试+文档+sanity；核心新代码 Task A'-1~9 共 9 个）。UniFP 双力 reward + **固定 K_virtual=100 N/m**（training hyperparameter）+ F_ext/F_cmd wrist-only + critic privileged F_ext + **无 history，无 GH code copy**。Actor 160 dim / Critic 298 dim。
- Path A（UniFP 纯版，已过时）：不建议
- Path B（CHIP 纯版，历史参考）：和 "F_cmd as input" 硬约束冲突

**v9 相比 v8 的简化 + Codex 评审修正：**
- **K_virtual = 100 N/m（而非 v8 早期一度提到的 50）**：从 Δx 工作空间反推，让同向叠加合力（max 60N）÷ K = 0.6m 刚到 arm reach 边缘而非超出（v8 的 50 会让 Δx 达 1.2m，无意义）；详见 §A'.4
- 不做 IsaacSim 校准脚本（Task A'-0 只是手写常量 + 单测）
- 声明 K_virtual 是 **training hyperparameter，不是测得刚度**（§A'.4 开头）
- Body ID resolution 用 `simulator.find_rigid_body_indice + simulator.body_ids` 惯用 pattern，明确 `from isaaclab.utils.math import quat_apply_inverse`（wxyz）
- env 子类暴露 `last_applied_force_w_by_body_id` debug accessor，e2e 测试不读 IsaacLab 私有 `_external_force_b` buffer
- Spike 1 断言改用 body_ids 白名单 + 外力 buffer 白名单（不再"右 wrist 位移 < 1cm"——articulated body 耦合会 false-fail）
- 常量文件改名 `_k_virtual_measured.py` → `_k_virtual.py`（v9 不 measure）
- 明确声明 v1 = "DR against unobserved F_ext"，不是 learned compliance（§A'.7b.5 + §A'.9）

**执行模式：**

1. **Subagent-Driven (recommended)** — 逐 task 派独立 agent 实现，task 间 review，快速迭代
2. **Inline Execution** — 在当前 session 顺序执行 task，批量带 checkpoint

**请告诉我：** 确认 Path A' v9 可执行？走 Subagent-Driven 还是 Inline？
