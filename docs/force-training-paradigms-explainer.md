# 力追踪 Low-Level Controller 的四种范式——原理深入浅出

> **目的**：从原理上解释 Holosoma、UniFP、CHIP、GentleHumanoid 四者在"让 humanoid/quadruped 学会与外力/接触力相关的控制"上的差别，既讲直觉也写数学。为 plan `2026-05-01-wbt-wrist-force-controller.md` 提供理论依据。
>
> **读者假设**：了解 RL（PPO、actor-critic）、刚体动力学基础、PD 控制器，不假设读过这四个系统的代码。
>
> **关键一句话**：
>
> | 系统 | 信仰 |
> |---|---|
> | **Holosoma** | 我是 motion tracker，不管力 |
> | **UniFP** | 给我 F_cmd，我偏 F_cmd/K；给我 F_ext，我也偏 F_ext/K；两者叠加 |
> | **CHIP** | 力这事儿，只改 observation 的 goal，reward 完全不碰 |
> | **GentleHumanoid** | 对着 admittance 物理模型拟合，让 sim 力跟 admittance 模型的期望一致 |

---

## 目录

1. [问题设置与统一记号](#1-问题设置与统一记号)
2. [PPO + Asymmetric Actor-Critic 基础](#2-ppo--asymmetric-actor-critic-基础)
3. [四种范式的数学本质](#3-四种范式的数学本质)
4. [Holosoma: 纯 motion tracker（baseline）](#4-holosoma-纯-motion-tracker)
5. [UniFP: F_cmd 直给 + 虚拟阻抗 reward](#5-unifp-f_cmd-直给--虚拟阻抗-reward)
6. [UniFP 里 F_cmd / F_ext 到底给 actor 还是 critic？](#6-unifp-里-f_cmd--f_ext-到底给-actor-还是-critic)
7. [CHIP: Hindsight perturbation + reward 不动](#7-chip-hindsight-perturbation--reward-不动)
8. [GentleHumanoid: Admittance-based compliance](#8-gentlehumanoid-admittance-based-compliance)
9. [四者的几何统一视图](#9-四者的几何统一视图)
10. [联系与互补](#10-联系与互补)
11. [对我们需求的映射](#11-对我们需求的映射)
12. [参考文件索引](#12-参考文件索引)
13. [一页小抄](#13-一页小抄)

---

## 1. 问题设置与统一记号

### 1.1 所有方法共享的 setting

- 一个 humanoid / quadruped，底层 PD 控制器（固定 Kp, Kd）将 `q_target` 转成 joint torque。
- Policy `π_θ` 输出 joint position 残差 `a_t`，即 `q_target = a_t · scale + q_default`。
- 训练环境是 sim（IsaacGym / IsaacSim / MJWarp）。上层任务是 motion tracking：让 body 的某些 link 跟随参考运动 `g_t`（position + 可选 orientation + 可选 velocity）。
- Policy 在时序 `t ∈ [0, T]` 里选择动作最大化折扣累计 reward `∑ γ^t r_t`。

### 1.2 统一记号

| 记号 | 含义 | 单位 | frame |
|---|---|---|---|
| `x_t ∈ ℝ³` | 某个 end-effector（wrist / gripper）当前 world 坐标 | m | world |
| `g_t ∈ ℝ³` | 当前 motion tracking goal（参考位置）| m | world |
| `F_ext ∈ ℝ³` | sim 真实注入 end-effector 的外力 | N | world |
| `F_cmd ∈ ℝ³` | 上层给 policy 的力**指令**（可选；不是所有方法都有） | N | body-yaw / body |
| `1/k ∈ ℝ⁺` | compliance coefficient（柔性系数） | m/N | 标量 |
| `K` | 虚拟刚度（K ≡ 1/(1/k)） | N/m | 标量 |
| `σ` | tracking reward 的 exponential kernel 带宽 | m | — |

### 1.3 分解一般的 reward 形式

所有四家的"位置追踪"reward 骨架都是：

```
r_pos(t) = exp(−‖x_t − g̃_t‖² / σ²)
```

其中 `g̃_t` 是**实际进入 reward 的目标**。四家的区别，**全部在于怎么定义 `g̃_t` 和 sim 里有没有 F_ext**。

把四家浓缩成一条公式：

```
g̃_t = g_t + α · F_cmd / K  +  β · F_ext / K
```

| 系统 | α | β | sim 有 F_ext | policy 看 F_cmd |
|---|---|---|---|---|
| Holosoma | 0 | 0 | ✗ | ✗ |
| UniFP | +1 | +1 | ✅ | ✅（body-yaw frame） |
| CHIP | — | — | ✅ | ✗（policy 看 1/k；g̃_t = g_t） |
| GentleHumanoid | — | — | ✅ | ✗（g̃_t = admittance-filtered goal） |

这里 CHIP 和 GH 不能直接填到同一公式里，因为它们在**其他地方**做文章（CHIP 改 observation 的 goal；GH 用二阶动力学 admittance 模型替 `g̃_t`）。后面会分别展开。

---

## 2. PPO + Asymmetric Actor-Critic 基础

> 这一节是后面讨论"把 F_cmd / F_ext 塞给 actor 还是 critic"的理论前提。已经熟的可跳到第 3 节。

### 2.1 Actor-Critic 是什么

RL 里有两个"网络"要学：

- **Actor** `π_θ(a | o_t)`：看观测 `o_t`，输出动作分布。真正控制机器人的是它。
- **Critic** `V_φ(o_t^critic)`：看观测 `o_t^critic`，输出一个标量 "从这个状态开始未来 reward 的期望"。它**不控制机器人**，只在训练时给 actor 打分。

PPO（Proximal Policy Optimization）的 update 公式核心是 **advantage** `A_t`：

```
A_t = Q(s_t, a_t) − V(s_t)  ≈ r_t + γ V(s_{t+1}) − V(s_t)   (TD(0))
```

用更低方差的 GAE（generalized advantage estimation）时：

```
A_t^GAE = ∑_{l=0}^{∞} (γλ)^l δ_{t+l}     其中 δ_t = r_t + γ V(s_{t+1}) − V(s_t)
```

PPO 的 actor loss（简化）：

```
L^actor = −E[ min(ρ_t · A_t, clip(ρ_t, 1−ε, 1+ε) · A_t) ]
  where ρ_t = π_θ(a_t|o_t) / π_θ_old(a_t|o_t)
```

Critic loss：

```
L^critic = E[ (V_φ(o_t^critic) − V_target_t)² ]
  where V_target_t = A_t^GAE + V_φ_old(o_t^critic)
```

### 2.2 Asymmetric Actor-Critic（非对称观测）

**关键观察**：actor 和 critic 看到的观测**不必相同**。通常：

- Actor 看 `o_t^actor`：**部署时**真正可获得的量（本体感知、命令、IMU 等）
- Critic 看 `o_t^critic` = `o_t^actor` ∪ privileged obs：额外加**只在 sim 里能拿到**的量（ground-truth 外力、地面摩擦系数、mass 误差、接触状态……）

为什么这样做？**critic 只在训练时用**。部署时：
```
a_t = π_θ(o_t^actor)     # actor 上真机
（critic 扔掉）
```

所以给 critic "开挂" 看 sim-only 信息，**不会**让 deploy 时 policy 非法——actor 看不到。

### 2.3 给 critic 开挂的三个数学理由

#### 理由 1：降低价值估计方差（核心）

假设真实的最优 value function 依赖一个隐藏变量 `h`（e.g. 真实外力 F_ext）：

```
V*(s, h) = E[ ∑ γ^t r_t | s_0 = s, h ]
```

若 critic 只看 `s`，它必须从 `s` **推断** `h` 的后验 `p(h|s)`，然后边缘化：

```
V(s) = E_h[ V*(s, h) ] = ∫ V*(s, h) p(h|s) dh
```

这个边缘化会带入**推断误差**和**边缘化方差**。V(s) 的方差下界是 `Var_h[V*(s,h)]`（由全方差定理）。

若 critic 直接看 `h`：

```
V(s, h) = V*(s, h)      # 零边缘化方差
```

**critic 方差小 → V_target 更准 → advantage 更准 → actor 梯度方向更准 → 样本效率提升**。

实测效果：asymmetric 在 domain randomization 严重（mass/friction/force 随机）的 legged RL 里**样本效率常常 2-3x** vs symmetric。

#### 理由 2：actor 的 input 应当匹配 deploy 时的 "信息边界"

如果让 actor 看 `h`，训练时 policy 会学到 `π_θ(a | s, h)`——**依赖** `h`。部署时 `h` 不可得，只能填一个估计 `ĥ` 或零；policy 进入 OOD，性能崩。

**原则：actor observation 集合 = deploy 时真能拿到的集合** (modulo 噪声)。

#### 理由 3：Information bottleneck 视角

让 actor 信息受限，实际上是一种**隐式正则化**。actor 必须通过**本体感知历史** + past actions 隐式地反推隐藏变量（就是"学一个内部 estimator"）。CHIP 和 UniFP 都用 `s_{t−K:t}` history 就是这个原因——给 actor 时间维度的 clue 来估计 F_ext。

### 2.4 IsaacLab / Holosoma 里的对应术语

- Holosoma 的 observation manager 里 group 概念就是 actor / critic 两套：
  - `policy` group → 给 actor
  - `critic` group → 给 critic（通常是 policy + 若干 privileged term）
- privileged obs 常见候选（legged RL 社区）：`base_lin_vel`（IMU 没法准测）、`external_force`、`friction_coef`、`mass_params`、`contact_mask`、`height_scan_around_base`（sim 可直接给） 等

Holosoma 目前 baseline 就是 asymmetric：
- actor 不看 `base_lin_vel`（真机 IMU 积分漂移）
- critic 看 `base_lin_vel`（sim ground truth）

### 2.5 "放 actor 还是 critic" 的决策模板

对任何候选输入 `x`，按以下顺序问：

1. **部署时可获得（或可估）？**
   - 否 → 只能放 critic
   - 是 → 继续
2. **是 policy 执行任务所**必须**的指令？**
   - 是（例如 F_cmd：policy 不知道指令没法执行）→ 放 actor（同时放 critic 也好）
   - 否 → 继续
3. **部署时是噪声很大 / 有偏的估计吗？**
   - 是 → 放 critic（用 ground-truth）；actor 隐式估计
   - 否 → 放 actor + critic 都 OK

四家对各种量的实际选择，都可以用这个模板解释（见 §6）。

---

## 3. 四种范式的数学本质

### 3.1 统一抽象：policy 到底在学什么

Policy 输入 observation `o_t`，输出动作 `a_t`。`a_t` 经过 PD 最终产生 torque，物理仿真到下一个状态。训练目标是最大化累计 reward。

从信号传递的角度看：

```
       ┌───────── F_ext（有或无） ──────────┐
obs ───▶ policy ──▶ a_t ──▶ PD ──▶ torque ──▶ sim ──▶ x_{t+1}
 ▲                                             │
 └─────── g_t（可能被修改）◀──────── reward ◀──┘
```

力参与训练的方式只有三种入口，各自对应四家的不同选择：

1. **sim 物理层**：注入 F_ext，让 robot 动力学真的受到外力影响 → UniFP ✅, CHIP ✅, GH ✅, Holosoma ✗
2. **observation 层**：给 policy 看（F_cmd 或 F_ext 或 1/k）→ UniFP 看 F_cmd, CHIP 看 1/k, GH 不看力
3. **reward 层**：修改 `g̃_t` → UniFP 按 F_cmd+F_ext 偏移, GH 按 admittance 模型偏移, CHIP 不改

注意这三个入口**互相正交**，任何组合都是合法的训练 setup。四家就是挑了不同的组合。

### 3.2 "什么算作'主动施力'"的物理原理

这一点最容易被学习范式的细节淹没。先把物理说清楚：

**humanoid 的 wrist 不是力传感器/力输出设备**。它能产生力**只有一种方式**：通过关节 torque 经由 Jacobian `J` 投影到末端：

```
F_ee = J^{-T} τ_joint
```

而 `τ_joint` 来自 PD:

```
τ_joint = Kp (q_target − q) + Kd (q̇_target − q̇)
```

即最终：

```
F_ee ≈ K_arm · (x_target − x_ee) + damping term
```

其中 `K_arm = J^{-T} Kp J^{-1}` 是**关节空间 PD 投影出的末端 Cartesian 刚度**（关节构型相关，不是常数）。

**结论："主动施力"等价于"让 policy 把 x_target 设在当前 x_ee 朝着施力方向偏移 Δx 的位置上"。** 不管你用 UniFP 还是 CHIP 还是 GH，最终都落到"让 Δx 与目标力 F 成正比"，比例就是 `1/K_arm`。

这也是为什么所有这些方法都绕不开"位置偏移"这件事——**末端的力永远通过 Δx 产生**。

### 3.3 四家的设计选择本质是：Δx 谁决定？怎么决定？

- **Holosoma**：没有 Δx 概念，x_target = g_t。
- **UniFP**：`Δx = (F_cmd + F_ext) / K_virtual`，K_virtual 是**训练 hyperparameter**，不是物理量。
- **CHIP**：`Δx = (1/k) · F_ext`，1/k 是**policy 的 input**，每 episode 采样，policy 学到"看到 1/k 就让 x 偏 (1/k)·F_ext"。
- **GH**：`Δx` 来自**二阶 admittance 动力学** `m ẍ + b ẋ + F_ext = F_drive`，每 step 积分计算。

---

## 4. Holosoma: 纯 motion tracker

### 4.1 训练 setup

- 输入（actor）：`motion_command`（14 bodies × {pos, quat, lin_vel, ang_vel}）, `dof_pos`, `dof_vel`, `base_ang_vel`, `actions`
- 输入（critic）：加 `base_lin_vel`
- F_ext：✗（只有 `_push_robots` 给 base 打 impulse，没到 end-effector）
- F_cmd：✗

### 4.2 Reward（`src/holosoma/holosoma/managers/reward/terms/wbt.py`）

```
r = 0.5 · exp(−‖ref_pos − robot_ref_pos‖² / σ²_root)       # root 位置
  + 0.5 · exp(−quat_error² / σ²_rot)                      # root 姿态
  + 1.0 · exp(−mean_14‖body_pos_rel − robot_body_pos‖² / σ²)  # 14 body 位置
  + 1.0 · exp(−mean_14 quat_error² / σ²)                     # 14 body 姿态
  + 1.0 · exp(−mean_14‖body_lin_vel − robot_body_lin_vel‖² / σ²)
  + 1.0 · exp(−mean_14‖body_ang_vel − robot_body_ang_vel‖² / σ²)
  − 0.1 · ‖a_t − a_{t−1}‖²                                # action rate
  − 10   · joint_limit_violation
  − 0.1  · undesired_contacts
```

全部是位置/姿态/速度追踪，没力的概念。这是 CHIP、UniFP 改造的**起点 baseline**。

### 4.3 数学上它学到什么

Policy 学到的映射：

```
π_θ: (g_t, q, q̇, ...) → a_t
```

最优解接近 kinematic "逆解 + feedforward"，在没有外力的 sim 里对 motion 做精确 tracking。**看到外力时会失败**——它没见过外力，不知道怎么反应。

---

## 5. UniFP: F_cmd 直给 + 虚拟阻抗 reward

### 5.1 核心思想（直觉）

> 给 policy 看一个**力指令 F_cmd**。告诉它："如果有个刚度 K 的虚拟弹簧，你想产生 F_cmd 的力，就得把 wrist 从 g_t 偏离 F_cmd/K 米。你偏得准我就奖励你。"
>
> 同时，sim 里真的施加随机外力 F_ext 来扰动 wrist。让 policy 在 `(F_cmd, F_ext)` 任意组合下都能把 wrist 驱到"对应的偏移位置"。

### 5.2 Sim 层：F_ext 如何注入

`third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:130`：

```python
self.gym.apply_rigid_body_force_tensors(
    self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
```

`self.forces[:, gripper_idx, 0:3]` 是每个 env 在 gripper 上要注入的**世界坐标系**力。每 physics step 调一次，由 `_update_commands` 方法生成梯形剖面（行 1144-1152）：
- 每个 env 独立采样 F_ext 的**目标幅度**（`max_push_force_xyz_gripper_ext`）和**方向**
- 每 env 独立采样 `push_duration` 和 `settling_time_force_gripper`
- 剖面：线性 ramp-up → hold → 线性 ramp-down

### 5.3 F_cmd：policy 的显式输入

`current_Fxyz_gripper_cmd`（body-yaw frame，3-D）被 mix 到 `commands` tensor 里：

```python
# 行 1075-1077
self.commands[env_ids, INDEX_EE_FORCE_X] = self.current_Fxyz_gripper_cmd[env_ids, 0]
self.commands[env_ids, INDEX_EE_FORCE_Y] = self.current_Fxyz_gripper_cmd[env_ids, 1]
self.commands[env_ids, INDEX_EE_FORCE_Z] = self.current_Fxyz_gripper_cmd[env_ids, 2]
```

然后在 obs_buf 里：

```python
# 行 422
(self.commands * self.commands_scale)[:, :15]  # ← policy 看到了 F_cmd (index 9, 10, 11)
```

### 5.4 Reward 公式（**UniFP 的灵魂**）

`third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891`：

```python
def _reward_tracking_ee_force_world(self):
    forces_global = self.forces[:, self.gripper_idx, 0:3]            # F_ext (world)
    forces_cmd = self.current_Fxyz_gripper_cmd                        # F_cmd (body-yaw)
    forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)    # F_cmd (world)
    forces_offset = forces_global + forces_cmd_global                 # ★ 相加
    target_shifted = forces_offset / self.gripper_force_kps + self.curr_ee_goal_cart_world
    error = torch.sum(torch.abs(self.ee_pos - target_shifted), dim=1)
    return torch.exp(-error / σ * 2)
```

数学形式：

```
g̃_t = g_t + (F_ext + F_cmd) / K
r_pos(t) = exp(−‖x_t − g̃_t‖₁ / σ · 2)
```

### 5.5 为什么这个公式漂亮（4 种情形统一）

**情形 A：F_cmd=0, F_ext=0**（自由空间，无指令）
```
g̃_t = g_t  → 纯 motion tracking
```

**情形 B：F_cmd=20N, F_ext=0**（空中要"发力"，但没接触）
```
g̃_t = g_t + 20N/200 · x̂ = g_t + 0.1 m x̂
→ policy 把 wrist 推到比 g_t 偏 10cm 的位置
→ 接触真实墙面时，这 10cm 的 "想偏但被挡" → PD 产生 F_real ≈ 10cm × K_arm
→ 若 K_arm ≈ K_virtual=200, F_real ≈ 20N ✓
```

**情形 C：F_cmd=0, F_ext=人推一把 +X 20N**（sim 扰动，指令为 0）
```
g̃_t = g_t + 20N/200 · x̂ = g_t + 0.1 m x̂
→ policy "应当"把 wrist 随外力偏移 10cm 再回 g_t
→ 学到 compliance 行为（被推时让一下）
```

**情形 D：F_cmd=20N, F_ext=接触反力 −20N**（真实施力稳态）
```
g̃_t = g_t + (−20+20)/200 = g_t
→ policy 保持 wrist 在 g_t, 但 PD 仍需产生 F_cmd 的 torque 来抵消 F_ext
→ 这正是部署时的稳态
```

**统一效果**：一个 reward 同时训 (1) 空中偏移, (2) 对 F_ext 的 compliance, (3) 接触稳态保持位置。

### 5.6 数学警告：K_virtual 的三重含义

`K_virtual`（代码里 `gripper_force_kps`，默认 200 N/m）身兼三职：

1. **训练信号尺度**：把 F（N）翻译成 Δx（m）作为位置误差单位。
2. **部署时的"换算率"**：F_cmd=20N → policy 认为应该偏 0.1m。上层使用者需要知道这个 K。
3. **与 K_arm 的匹配**：部署时实际产生的力 `F_real ≈ K_arm · Δx`。只有 `K_arm ≈ K_virtual` 时 `F_real ≈ F_cmd`。

第 3 点是 UniFP 最脆弱的假设。若部署环境 K_arm 与训练 K_virtual 差 2x，F_real 偏差 50%，上层要自己做校准。

### 5.7 UniFP Observation 全景

**Actor（行 414-422）** = [2]body_ori + [3]base_ang_vel + [17]dof_pos_diff + [17]dof_vel + [17]actions + [1+1]sin_cos_phase + [15]commands（**含 F_cmd**）+ obs_history ×T

**Critic（行 379-411）** = [3]base_lin_vel + [3]ee_pos_sphe + [3]**F_ext_local_gripper** + [3]**F_ext_local_base** + ee_goal_offset + ... ← 多 ground-truth F_ext

**Obs history**：`obs_history` 是 maxlen buffer，每 step 往里 append `obs_now`，最后 `torch.stack` 成 `[N, T, K]` → reshape `[N, T·K]`。是让 policy 从时序推理外力的关键。

### 5.8 UniFP 的 domain randomization

- `K_virtual` 也可以随机化（`randomize_gripper_force_gains=True`，行 759），但文章里默认关闭（`gripper_force_kp_range = [200, 200]`）
- F_cmd 和 F_ext 分别有独立的 `forced_prob_cmd/ext`——某 env 这一 episode 是否启用
- `gripper_force_kd` 可选 critical damping

---

## 6. UniFP 里 F_cmd / F_ext 到底给 actor 还是 critic？

这一节是全文最该盯着的 **deploy-vs-train 区分**。先把 UniFP 源码里的实际答案讲清楚，再用 §2 的模板解释为什么这么做，最后说"如果搬到 holosoma plan 里应该怎么分配"。

### 6.1 UniFP 源码里的实际分配

源文件 `third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`，关键在 `compute_observations`（行 333-424）。

把 UniFP 的观测按 group 解剖：

| 信号 | Actor (`obs_buf`, 行 414-422) | Critic (`privileged_obs_buf`, 行 379-403) | Predictor head (`obs_pred`, 行 405-410) |
|---|---|---|---|
| body orientation | ✅ | ✅ | — |
| base_ang_vel | ✅ | ✅ | — |
| dof_pos_diff, dof_vel, actions | ✅ | ✅ | — |
| sin/cos gait phase | ✅ | ✅ | — |
| `commands[:, :15]`（含 **F_cmd** at `[9:12]`）| ✅ | ✅ | — |
| **base_lin_vel** | ✗ | ✅ | ✅ |
| **ee_pos_sphe**（末端球坐标，sim-only）| ✗ | ✅ | ✅ |
| **`forces_local[:, gripper_idx]`（F_ext_gripper）**| **✗** | **✅** | **✅** |
| **`forces_local[:, base_idx]`（F_ext_base）**| **✗** | **✅** | **✅** |
| mass_params, friction_coef, motor_strength | ✗ | ✅ | — |
| stance_mask, contact_mask, projected_gravity | ✗ | ✅ | — |
| ee_goal_offset_local_sphere（**含 `(F_ext+F_cmd)/K` 偏移后的球坐标**）| ✗ | ✅ | — |
| Obs history stacked × T | ✅（`obs_history`）| ✅（`critic_history`）| — |

一句话结论：

- **F_cmd**：actor ✅ + critic ✅
- **F_ext**：actor ✗，critic ✅（加 predictor head 监督）

### 6.2 为什么 F_cmd 必须给 actor（物理意义）

F_cmd 是**控制指令**。它的物理意义是"上层系统想让末端施加多大的力"。这个量在 deploy 时也可用——**上层就是 deploy 时的 F_cmd 来源**。

**失败案例**：如果 F_cmd 不给 actor，policy 永远不知道"现在上层想要 20N 还是 0N"，就只能当 motion tracker 用（无法根据 F_cmd 产生不同行为）。训练目标（reward）里 F_cmd 会改变 `g̃`，但如果 actor 看不到 F_cmd，它根本无从知道 `g̃` 被偏到哪了——只能"瞎猜"——等价于 partial observability 问题。PPO 在这个情况下会收敛到**"对所有可能的 F_cmd 的平均 behavior"**，基本就是 motion tracker。

按 §2.5 决策模板：
- F_cmd deploy 时可获得？**是**（上层直接给）
- 是执行任务必须的指令？**是**
- → **放 actor**（同时也放 critic 没有坏处——critic 算 value 时 F_cmd 是状态一部分）

### 6.3 为什么 F_ext 只给 critic（物理意义）

F_ext 是 sim 里**真实注入**的外部扰动。它的物理意义是"墙/人/其他东西对末端施加的外力"。关键问题：

- **deploy 时能直接测到吗？** 大多数情况**不能**——G1 humanoid 的 wrist 没有 F/T 传感器（CHIP paper §VI 作为 limitation 明确写了这点）。即使有传感器，也有噪声、偏移、带宽限制。
- **敏感到什么程度？** 末端 F_ext 是 "接触不接触" + "接触了多硬" 的**核心隐藏变量**，value function 估计高度依赖它。

按 §2.5 决策模板：
- deploy 时可获得？**否** → 放 critic
- → actor 不能看 F_ext，但 actor 必须**隐式推断** F_ext 才能做 compliance 行为

### 6.4 Actor 隐式推断 F_ext：obs history 的作用

UniFP 的 actor 不直接看 F_ext，但它必须"知道"当前被多大外力推着，才能做对动作。怎么做到？**通过 obs history**。

数学上：二阶动力学方程

```
M(q) q̈ + C(q, q̇) q̇ + g(q) = τ + J^T F_ext
```

已知 `q, q̇, τ`（actor 都看得到，τ 来自自己的 past action），就能解出 F_ext：

```
F_ext ≈ J^{-T} [ M q̈ + C q̇ + g − τ ]
```

actor 的 stacked history `[o_{t−K}, ..., o_t]` 里隐含了 `q̈ ≈ (q̇_t − q̇_{t−1})/dt`。所以 policy 的 MLP/Transformer 理论上能学到这个"隐式 F_ext 估计器"。

**关键点**：history 越长、越 noisy 的 IMU 越需要历史来滤噪。UniFP 用 `obs_history.maxlen` 的 stack（`third_party/UniFP/.../legged_robot_b2z1_pos_force.py:423-430`）；CHIP 用 10 步 proprio + 11 步 past actions。history 长度是个关键超参。

### 6.5 Predictor Head（`obs_pred`）的作用——UniFP 的小创新

UniFP 不止让 critic 看 F_ext，还额外有一个 `obs_pred`：

```python
# 行 405-410
obs_pred = torch.cat((
    self.base_lin_vel * self.obs_scales.lin_vel,       # 3
    self.ee_pos_sphe_arm[...],                         # 3
    self.forces_local[:, self.gripper_idx] * ...,      # 3  ← F_ext_gripper
    self.forces_local[:, self.robot_base_idx] * ...,   # 3  ← F_ext_base
), dim=-1)
```

这个 `obs_pred` 通常作为 **辅助监督目标**给一个 **asymmetric encoder**（教师-学生结构）：
- **Teacher/Critic encoder** 看 privileged obs，输出 value 和 latent embedding
- **Student/Actor encoder** 看 actor obs，预测 `obs_pred`（即重建 F_ext 等 privileged 量）
- 监督信号：`L_pred = ‖student_pred − obs_pred_truth‖²`

这让 actor 主动学出一个 **"F_ext estimator"**，不只是被 PPO 梯度间接拉动。部署时只用 actor（已经内含 estimator），F_ext 无需真测。

**类比**：这就是 **DAgger / teacher-student distillation** 的思想，在一个 network 内完成。RMA（Rapid Motor Adaptation, Kumar et al. 2021）把这个做得更系统——先训 teacher with privileged，再用 privileged 作为监督 distill 一个 history-only student。

### 6.6 对照：CHIP 怎么分配 F_ext / 1/k

- **1/k**：CHIP 版"compliance 指令"。deploy 时由上层给（类似 F_cmd 角色）→ actor ✅ + critic ✅
- **F_ext**：sim ground truth → critic ✅（privileged），actor ✗
- **proprio history (10 步)、past actions (11 步)**：actor ✅ + critic ✅（帮 actor 隐式估 F_ext）
- **原 goal `g`（未扰动）**：actor 看的是 `g_hind`（已扰动），**critic 也看 `g_hind`？还是看 g？** 论文没明写，但从"critic 应知 ground-truth"角度看，critic 应该同时拿 `g_hind` 和 `g`（或者直接拿 `f_ext` 就够了，因为 `g = g_hind + (1/k)·f_ext` 可推）

### 6.7 对照：GentleHumanoid 的分配

| 信号 | Actor | Critic |
|---|---|---|
| `force_keypoint_b`（admittance-modified goal）| ✅ | ✅ |
| `force_applied_b`（sim 真实注入力）| ✗ | ✅（privileged，名称 `force_priv`）|
| `force_expected_b`（admittance 期望力）| ✗ | ✅ |
| `force_sample_timer` | ✗ | ✅ |
| admittance 状态（admit.x, admit.v）| ✗ | ✅ |

和 UniFP/CHIP 一致原则：sim 注入的 ground-truth 力只给 critic。

### 6.8 决策矩阵（针对我们 plan 用到的信号）

假设我们做 Path A'（UniFP reward + CHIP 基础设施），需要决定以下候选输入的分配：

| 候选信号 | 物理意义 | deploy 可得？ | 放 actor | 放 critic | 备注 |
|---|---|---|---|---|---|
| **F_cmd**（6-D, body-yaw）| 上层的力指令 | ✅ | **✅** | ✅ | 核心任务输入 |
| **F_ext**（6-D, world）| sim 真实注入的扰动 | ✗ | **✗** | **✅** | 降 value 方差 |
| motion_command（14 body）| 参考 motion | ✅（motion clip 已知）| ✅ | ✅ | 原有 |
| dof_pos, dof_vel, base_ang_vel | proprio | ✅ | ✅ | ✅ | 原有 |
| base_lin_vel | velocity | ✗（IMU 漂移）| ✗ | ✅ | Holosoma 已如此 |
| actions history（K 步）| past torque 间接 | ✅（自己的输出）| ✅ | ✅ | 帮 actor 估 F_ext |
| proprio history（K 步）| q, q̇ 的时序 | ✅ | ✅ | ✅ | 同上 |
| `ee_goal_offset`（`(F_ext+F_cmd)/K` 偏移后的 goal）| 训练信号可视化 | 部分 | 可选 | ✅ | UniFP 在 critic 里放，冗余但有帮助 |
| `1/k`（若未来加 adaptive stiffness）| 上层的 stiffness 指令 | ✅ | ✅ | ✅ | 角色同 F_cmd |

### 6.9 常见错误放置会怎么样

| 错误做法 | 后果 |
|---|---|
| F_cmd 只给 critic，不给 actor | actor 无法区分"要发力还是不要发力"，退化成 motion tracker |
| F_ext 给 actor | deploy 时没有 F_ext ground truth；actor 进入 OOD，性能崩 |
| actor 看 base_lin_vel（sim ground truth）| IsaacGym sim2real gap 已知经典 pitfall，deploy 时 IMU 积分漂移 → policy OOD |
| 不给 obs history | actor 无法估计 F_ext，PPO 梯度方向错，训练不稳定或学不到 compliance |
| critic 不看 F_ext | value function 高方差，样本效率大降 |

### 6.10 Asymmetric PPO 的一个数学 "好消息"

这段是给 "为什么这么放是对的" 的一个干净理论解释。

假设真实状态 `s = (s_obs, h)`，其中 `s_obs` 是 actor 可观测部分，`h` 是隐藏变量（F_ext, friction, ...）。

- **Optimal V^***: `V*(s) = V*(s_obs, h)` 依赖 h。
- **Symmetric critic**: `V_sym(s_obs) = E_h[V*(s_obs, h) | s_obs]`。方差分解：`Var[V_sym] = Var[E_h V*] + E_{s_obs}[Var_h V*]` 里**忽略了第二项**（critic 看不到 h，只能吐边缘期望）。但 rollout 里真实 return `R_t = V*(s_obs, h_t)` 是依 `h_t` 变的，bootstrap target `R_t` 的方差 = `Var[V*(s_obs, h)]` 很大。critic 拟合 `R_t` 时残差方差 `≥ E[Var_h V*]`。
- **Asymmetric critic**: `V_asym(s_obs, h) = V*(s_obs, h)`。拟合残差只剩 reward 过程里的内在随机性，方差显著更低。

结论：**给 critic 看 h 不会让 policy 依赖 h，只会让 advantage 估计更准**。这是 PPO + asymmetric actor-critic 的核心 PROvision。

引用：Pinto et al. 2017 (Asymmetric AC for Robotics), Lee et al. 2020 (ANYmal learning in challenging terrain), Kumar et al. 2021 (RMA)。

### 6.11 回到 UniFP：三个 critic-only 信号协同的作用

UniFP 的 critic 拿到了：

1. **F_ext_gripper**：ground-truth 外力 → 降 V 方差
2. **ee_pos_sphe_arm**（末端球坐标）：ground-truth 末端位置 → 降 Q 相关 V 的方差
3. **mass_params, friction_coef, motor_strength**：ground-truth domain randomization 参数 → 让 critic 知道"这个 env 是哪种变体"

三者合力使得在**超大 domain randomization** 下 PPO 仍能收敛。UniFP 原文也强调在 B2Z1 quadruped 上重力+外力+电机力+摩擦全随机，没有 asymmetric critic 根本训不起来。

### 6.12 实战清单（我们 plan Path A' 的 observation group 建议）

**Actor group（`o_t^actor`）：**
- 原 WBT：motion_command (14 body pos/quat/lin/ang_vel), dof_pos_diff, dof_vel, base_ang_vel, actions, projected_gravity
- **新增**：`F_cmd_body_yaw`（6-D），`proprio_history_stack`（例如 last 5-10 步 dof_pos, dof_vel, actions）

**Critic group（`o_t^critic` = actor ∪ privileged）：**
- Actor 全部
- 原 WBT privileged：base_lin_vel
- **新增**：`F_ext_world`（6-D ground truth），optionally `(F_cmd+F_ext)/K` 偏移后的 goal（冗余但 CHIP/UniFP 都放了）

**(可选) Predictor head**：如果想效仿 UniFP 的 `obs_pred` 机制给 actor 一个 F_ext estimator auxiliary loss，需要 holosoma 的 agent 支持这个——暂不建议 v1 加。

---

## 7. CHIP: Hindsight perturbation + reward 不动

### 7.1 核心思想（直觉）

> "前人改 reward 来让 policy 学 compliance 很麻烦（得改 reference motion 或加一堆力 reward）。我**反过来：reward 完全不动，改 observation 的 goal**。sim 里施加 F_ext 时，给 policy 看一个**被扰动过的 goal** `g_hind = g − (1/k)·F_ext`；但 reward 仍用原 g 算位置误差。policy 为了最大化 reward，必须把 wrist 驱到 g（远离 g_hind）——即必须'抵消'那个 -(1/k)F_ext 的偏移——这就学会了 `(1/k)`-compliance 行为。"

### 7.2 Sim 层：F_ext 注入

CHIP paper §III（和 UniFP 相同）：每 end-effector 独立采样 ∈[0,40]N 的 F_ext，duration ∈[1,3]s，梯形剖面（Fig. 9）。通过 IsaacLab 的 rigid body force API 注入。

### 7.3 Observation: hindsight goal

CHIP paper Eq. 2：

```
g_hind(t) = g(t) − (1/k) · f_ext(t)
```

仅对 end-effector（wrist）的 position goal 做这个偏移；其他 body 的 goal 不动。

Policy 输入：
- `g_hind(t)`（而非原 g）
- `1/k`（每个 end-effector 一个标量，episode reset 时采样 ∈ [0, 0.05]）
- `s_{t−10:t}`（10 步 proprio）
- `a_{t−11:t−1}`（11 步 past actions）

Critic 额外看：
- 原 `g(t)`（非 privileged，因为可推）
- `f_ext(t)`（**privileged**）

### 7.4 Reward: 完全不动

用原 `g(t)` 算 tracking reward（paper Table III 列出，本质就是 BeyondMimic / SONIC 的 reward 集）：

```
r(s, g) = exp(−‖x − g‖² / σ²) + ...  # 原 motion tracking reward
```

**不加任何 force-specific reward**。

### 7.5 为什么 policy 会学到 compliance（数学推导）

Policy 要最大化 `r(s, g)` = 让 x ≈ g。但 policy 只看到 `g_hind = g − (1/k)·F_ext`，并且知道 `1/k`。

在训练充分收敛后，policy 相当于学到一个内部估计器 `f̂_ext ≈ F_ext`（通过 obs history 和 past actions 推断；CHIP paper §III 明确说了这一点），然后：

```
x_target_policy = g_hind + (1/k) · f̂_ext
                ≈ g − (1/k) · F_ext + (1/k) · F_ext
                = g
```

即 policy 内部恢复了原 g 作为目标。

**部署时**（F_ext=0，但接触反力 f_contact ≠ 0）：
- 上层把 wrist kinematic target 放在"表面里面"位置 `g + Δ`，即 observation 里 `g_obs = g + Δ`（policy 看到的 goal）
- 上层同时给 `1/k`
- 环境里没有 F_ext，但接触墙面产生反作用 `f_contact`
- Policy 按学到的行为：`x ≈ g_obs + (1/k) · f̂_contact`
  - 自由空间时 `f_contact=0`：wrist 滑向 `g_obs`（墙里面的 target）
  - 接触墙面稳态时 wrist 被挡在墙表面：`x_wall = g_obs + (1/k)·f_contact`
  - → `f_contact = (g_obs − x_wall)/(1/k) = K_virtual · Δ_penetration`

即实际施力 = 由上层指定的 "target 穿透量 × 1/(1/k)"。

### 7.6 CHIP 的 input/output 特征

| 维度 | 详情 |
|---|---|
| Actor input | `g_hind` (wrist 被偏移的 goal) + `1/k` + proprio history (10 steps) + past actions (11 steps) |
| Critic input | actor 全部 + `f_ext` ground truth (privileged) |
| 是否有 F_cmd | ✗ |
| 是否可调 stiffness at inference | ✅（continuous `1/k ∈ [0, 0.05]`）|
| reward 改动 | 无 |

### 7.7 CHIP 的精妙之处

CHIP paper Table I 显示：用 hindsight observation 方式达到的 tracking 精度 ≈ 没加 force perturbation 的 baseline；而 GentleHumanoid（用 reward shaping）精度明显更差。

**原因**：reward shaping 会让 optimal policy 与"纯 tracker"不同；而 observation shaping 把扰动推到输入空间，optimal policy 仍然是 tracker，只是面对更 noisy 的 input。后者对 tracking 精度零损害。

---

## 8. GentleHumanoid: Admittance-based compliance

### 8.1 核心思想（直觉）

> "如果 wrist 是个质量块 `m`，有阻尼 `b`，被一个虚拟弹簧拉向 reference target `g`，同时受外力 `F_ext`，它会按二阶动力学 `m ẍ + b ẋ + K(x − g) = F_ext` 运动。我**用这个动力学模型算出 wrist'应该'在哪里**，reward 让 policy 跟这个'被 admittance 修正过的 goal'。"

这是控制理论里经典的 admittance control 思路。

### 8.2 Sim 层：弹簧式 F_ext 注入

`third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py:1001`：

```python
self.force_applied_w[:] = clamp_norm(
    self.force_kp_scaled * self.project_pos_diff(
        self.force_origin_w - pos_w, force_dir=self.force_dir_w),
    self.max_force * self.force_alpha)
```

注入力不是梯形剖面，而是：
- 采样一个 `force_origin_w`（虚拟弹簧的"固定点"）
- `F_applied = K_spring · (force_origin_w − wrist_pos_w)` projected to `force_dir`
- 加了安全 clamp
- 随 wrist 位置动态变化 —— 像个真实弹簧

### 8.3 Admittance 模型（核心数学）

`active_adaptation/envs/mdp/commands/admittance.py:60`：

```python
def step(self, F_drive_b, F_ext_b):
    F_damp = -self.damping * self.v
    F_total = F_drive_b + F_ext_b + F_damp
    a = F_total / self.mass
    self.v = self.v + a * dt
    self.x = self.x + self.v * dt
```

即：
```
m ẍ = F_drive + F_ext − b ẋ
```

这里 `F_drive` 是一个"把虚拟 mass 拉向 reference point"的弹簧力：

```python
# motion_tracking.py:884-893
K_p_drive = force_limit / 0.05        # such that 5cm → max force
K_d_drive = 2√(K_p · m)               # critical damping
F_drive_b = clamp(K_p_drive · (ref_point_b − admit_x) + K_d_drive · (ref_vel_b − admit_v))
```

Admittance 输出的 `x` 即 "admittance-modified goal"，记为 `g̃_admit`。这个 `g̃_admit` 既是 reward target（policy 要 track 它），又用来算"预期的外力" `F_expected`：

```python
# motion_tracking.py:914
F_expected_b = clamp(force_kp · (force_origin_b − g̃_admit))
```

### 8.4 Reward（多项）

```
r_keypoint_imp = exp(−‖x_wrist − g̃_admit‖² / σ²_kp)          # 追 admittance-modified goal
r_force_target = exp(−‖x_wrist − g̃_admit‖² / σ²_ft)          # 又一个位置 reward
r_force = exp(−‖F_applied − F_expected‖² / σ²_F)              # 力一致性 ★
r_force_exd_penalty = −1 if ‖F_applied‖ > safe_limit else 0    # 安全上限
```

注意 `r_force` 是 GH 的核心创新：**让 sim 注入的真实力和 admittance 模型期望的力相等**。这意味着 policy 必须让 wrist 运动得恰好满足 admittance 动力学。

### 8.5 为什么 policy 会学到 compliance

训练目标要求：
1. `x_wrist ≈ g̃_admit`（跟 admittance 修正后 goal）
2. `F_applied ≈ F_expected`（实际力 = 模型期望力）

两者一起，policy 必须产生**让 wrist 恰好按 admittance 二阶动力学运动**的 torque。在弹簧式 F_ext 下，这就是经典的 compliance 行为。

### 8.6 为什么 CHIP Table I 说 GH 精度差

- GH 的 `r_keypoint_imp` 把**所有**14 body 的 reward target 都在 admittance 影响下漂移
- 导致 policy "motion tracking" 和 "compliance" 两个目标互相竞争
- CHIP 的观察空间修改只影响 wrist 两个点的 goal（且只在 obs 层，reward 用原 g），对 motion tracking 精度零损害

### 8.7 没有 F_cmd，也没有 runtime 可调 stiffness

GH 的 `compliance=True` 是 config 固定的；一个 policy 要么 compliant 要么 stiff，训练时决定。无法在 inference 时切换。

---

## 9. 四者的几何统一视图

### 9.1 `g̃_t` 的位置解构

用"tracking goal 在哪里"这个单一坐标点理解：

```
     g_hind (CHIP 给 policy 看的)
        │          ←── (1/k) · F_ext
        │
        ▼
   ╔═══ g_t ═══╗ ←── 原 motion goal (所有方法的 "真实" reference)
        │
        │          →── (F_cmd + F_ext) / K_virtual
        ▼
     g̃_UniFP (UniFP reward 用的)
        │
        │          →── admittance 动力学积分
        ▼
     g̃_admit (GH reward 用的)
```

- **Holosoma**：reward target = `g_t`，obs goal = `g_t`
- **UniFP**：reward target = `g_t + (F_ext + F_cmd)/K`，obs goal = `g_t`（policy 看原 g_t，额外收 F_cmd）
- **CHIP**：reward target = `g_t`，obs goal = `g_t − (1/k)·F_ext`
- **GH**：reward target = `g̃_admit`（二阶动力学积分的结果），obs goal = `g̃_admit`

### 9.2 核心区别三问

**Q1: F_ext 是否注入 sim？**
- Holosoma ✗, UniFP ✅, CHIP ✅, GH ✅

**Q2: Policy observation 里含什么"力/compliance"量？**
- Holosoma 无
- UniFP 有 **F_cmd**（actor）+ **F_ext_local**（critic privileged）
- CHIP 有 **1/k**（actor）+ **F_ext**（critic privileged）
- GH 有 **F_applied/F_expected/force_keypoint**（privileged）

**Q3: Reward 的 `g̃_t` 是什么？**
- Holosoma: `g_t`
- UniFP: `g_t + (F_cmd + F_ext)/K`（单次函数变换）
- CHIP: `g_t`（**不变**）
- GH: `g̃_admit`（二阶 ODE 积分）

### 9.3 "主动施力" 在四个范式中怎么实现？

| 方法 | 上层怎么指令"施 20N +X 力" | Policy 做什么 | 真实力从哪来 |
|---|---|---|---|
| Holosoma | 无接口 | — | — |
| UniFP | `F_cmd = [20, 0, 0]` | 把 wrist 偏 +X 20/K=0.1m | 接触墙时 K_arm·0.1m ≈ 20N |
| CHIP | `g_obs = g + [0.4, 0, 0]`（穿透 0.4m）+ `1/k = 0.02` | 看到 g_obs, 没外力时滑向 g_obs | 接触墙 `1/k · f = 0.4 → f = 20N` |
| GH | 无直接接口 | — | — |

**只有 UniFP 和 CHIP 能主动施力**。UniFP 接口更自然（直接 N），CHIP 接口更间接（target 位移 + 1/k），但 CHIP 换来 runtime stiffness 可调。

---

## 10. 联系与互补

### 10.1 UniFP 和 CHIP 的隐藏同构

惊人发现：如果把 UniFP 的 `K_virtual` 视为 `1/compliance = 1/(1/k)`，则两者 reward target 可写成统一形式：

UniFP:  `g̃ = g + (1/K_virtual) · (F_ext + F_cmd)`
CHIP:   `g̃ = g`（但 obs 是 `g − (1/k) · F_ext`）

**Policy 内部行为**的稳态：
- UniFP policy：`x ≈ g̃ = g + (1/K_virtual) · (F_ext + F_cmd)` 即 `x − g = (1/K_virtual) · (F_ext + F_cmd)`
- CHIP policy：`x ≈ g = g_obs + (1/k) · f_ext_estimated`，重排 `x − g_obs = (1/k) · f_ext`

**二者都表达的是 Hooke 定律：位置偏移 = compliance × 力**。只是 UniFP 在 reward 明写这个关系，CHIP 在 obs 明写这个关系。

关键差异：
- UniFP 的"力"里含 F_cmd（主动指令），CHIP 里没有
- UniFP 的 compliance `1/K_virtual` 是**训练常数**（policy 不可控），CHIP 的 `1/k` 是**episode 变量 + policy 输入**（可控）

### 10.2 GH 和 CHIP 的隐藏对偶

两者都"修改 goal"，差别在**哪一层修改**：

- **CHIP**：修改 observation 的 g，不改 reward 的 g
- **GH**：修改 reward 的 g（到 `g̃_admit`），修改 observation 的 g（也是 `g̃_admit`）

CHIP paper 的主张："只在 observation 层动 goal，不碰 reward，精度更高"。这一点 Table I 给了实证。

### 10.3 四者的取长补短图谱

```
           有 F_cmd input?
                  ▲
      UniFP  ●────┴────●  (未命名：CHIP + F_cmd)
              \         / 
               \       / 
                \     /
          reward 动? \ / reward 不动?
          ──────●   ╳   ●───────▶ 
               GH  / \  CHIP
                  /   \
                 /     \
       Holosoma●─────────● (未命名：GH + reward 不动)
                  ▼
           没有 F_cmd input
```

四个象限里：
- **UniFP**（有 F_cmd, 改 reward）：直给接口，训练稳定
- **CHIP**（无 F_cmd, 不改 reward）：可调 compliance，tracking 精度高
- **GH**（无 F_cmd, 改 reward）：严格物理模型，但精度折损
- **Holosoma**：baseline

**缺失的第四象限**："有 F_cmd + 不改 reward"：理论上可以实现，实现方式是 obs 里给 F_cmd，但 g 仍用原版，reward 用原版——policy 得靠 critic 的 advantage 信号学到"F_cmd 就是我该偏移的方向"。没人做过（推测因为信号太弱）。

### 10.4 最实际的互补组合

**UniFP reward + CHIP infrastructure** = 我们 plan v3 提到的 "Path A'"：
- 保留 UniFP 的 `(F_ext + F_cmd)/K` reward 公式（满足"F_cmd 作为输入"）
- 借用 CHIP 的 F_ext 注入基础设施（让 policy 见过外力，zero-shot 接触更鲁棒）
- 借用 CHIP 的 obs history stacking（让 policy 能从时序估计 F_ext）
- 借用 CHIP 的 critic privileged F_ext（降价值函数方差）
- 暂不加 CHIP 的 1/k（保持训练简单；未来可加）

这种组合把 UniFP 的"直给 F_cmd"和 CHIP 的"见过外力"两个好处结合。

**GH 唯一值得借鉴的一点**：`force_exd_penalty`（力超阈值小惩罚）可作为安全保险丝，防止 policy 学到 overshooting 发力。

---

## 11. 对我们需求的映射

### 11.1 需求回顾

1. **F_cmd 作为 input**：policy 必须显式接收 "施多少力" 的指令
2. **主动施力**：接触真实表面时能按指令产生力
3. **不影响现有 Holosoma 训练/测试代码**
4. **以 Holosoma 为主**：reward 尽量不动，motion tracking 精度不降

### 11.2 四个方案打分

| 需求 | Holosoma | UniFP | CHIP | GH |
|---|---|---|---|---|
| F_cmd 作为 input | ✗ | ✅ | ✗ | ✗ |
| 主动施力 | ✗ | ✅ | ✅（间接）| ✗ |
| 以 Holosoma 为主（reward 不动）| ✅ | ⚠️（新增 force reward）| ✅ | ✗（重写多个 reward）|
| motion tracking 精度 | ✅ | ✓ | ✅（paper 证明）| ⚠️ |

UniFP 是唯一同时满足"F_cmd input + 主动施力"的方案，这是 Path A 选 UniFP 的原因。

### 11.3 Path A（纯 UniFP）vs Path A'（UniFP + CHIP 基建）vs Path B（纯 CHIP）

| 维度 | Path A | Path A' | Path B |
|---|---|---|---|
| F_cmd as input | ✅ | ✅ | ✗ |
| F_ext 注入 sim（policy 见过外力）| ✗ | ✅ | ✅ |
| Obs history | ✗ | ✅ | ✅ |
| Critic privileged F_ext | ✗ | ✅ | ✅ |
| Reward 改动 | 新加 1 项 | 新加 1 项（含 F_ext）| 不改 |
| Stiffness runtime 可调 | ✗ | ✗ | ✅ |
| Code 改动量 | 小 | 中 | 中大 |
| 部署 zero-shot 接触 | 风险：没见过外力 | 低 | 低 |

### 11.4 推荐 Path A'

**理由**：
1. 满足硬约束 1 和 2（F_cmd + 主动施力）
2. 借用 CHIP 的 F_ext 注入 + history，解决 Path A 的 "zero-shot 接触未训练过"短板
3. reward 改动仍只 1 项（`(F_ext + F_cmd)/K`），遵循 "以 Holosoma 为主"
4. 如果未来需要 adaptive stiffness，在 Path A' 上扩展 `1/k` input 即可（几乎不需要推翻）

### 11.5 可视化决策树

```
需要 F_cmd 作为显式输入？
├─ 是 ──▶ 需要 adaptive stiffness？
│         ├─ 否 ──▶ 需要 zero-shot 接触鲁棒？
│         │         ├─ 否 ──▶ Path A (纯 UniFP)
│         │         └─ 是 ──▶ Path A' (UniFP + CHIP 基建) ★ 推荐
│         └─ 是 ──▶ Path A' + 扩展 1/k input (混合)
└─ 否 ──▶ Path B (纯 CHIP)
```

---

## 12. 参考文件索引

- UniFP 源码：`third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`
  - Reward: 行 1886-1900
  - F_ext 注入: 行 130, 1144-1152
  - F_cmd 进 obs: 行 1075-1077, 422
  - Critic obs: 行 379-411
- CHIP 论文：`third_party/CHIP.pdf`
  - Reward 设计: Tab. III (p.11)
  - F_ext 剖面: Fig. 9 (p.11)
  - Hindsight 公式: §III 的 `g_hind_ptb = g − (1/k) · f`
- GentleHumanoid 源码：`third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py`
  - Admittance 模型: `admittance.py`
  - F_applied/F_expected: 行 994-1026, 914-922
  - Rewards: 行 1111-1145
- Holosoma Reward: `src/holosoma/holosoma/managers/reward/terms/wbt.py:75-107`
- Plan 文件: `docs/plans/2026-05-01-wbt-wrist-force-controller.md`

---

## 13. 一页小抄

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 问题：让 humanoid 学会与力相关的控制                                       │
│                                                                          │
│ 统一公式：g̃ = g + α·F_cmd/K + β·F_ext/K                                  │
│                                                                          │
│ ┌────────────┬────────────┬────────────┬────────────┬────────────┐      │
│ │            │ Holosoma   │ UniFP      │ CHIP       │ GH         │      │
│ ├────────────┼────────────┼────────────┼────────────┼────────────┤      │
│ │ F_ext sim? │     ✗      │     ✅     │     ✅     │     ✅     │      │
│ │ F_cmd obs? │     ✗      │     ✅     │     ✗      │     ✗      │      │
│ │ 1/k obs?   │     ✗      │     ✗      │     ✅     │     ✗      │      │
│ │ g̃ = g    │     ✅     │(+F/K)      │     ✅     │ admittance │      │
│ │ 主要用途   │ 纯 motion  │ 主动施力   │ adaptive   │ passive    │      │
│ │            │  tracker   │ (stiff)    │ compliance │ compliance │      │
│ │ 推荐场景   │ baseline   │ 按按钮、   │ VR 遥操作、│ 协作接触   │      │
│ │            │            │ 推门、擦   │ VLA        │            │      │
│ └────────────┴────────────┴────────────┴────────────┴────────────┘      │
│                                                                          │
│ 我们的选择（对应 plan v3）：                                              │
│   Path A  : 纯 UniFP                                                     │
│   Path A' : UniFP reward + CHIP 基础设施（F_ext 注入 + history + critic）│
│             ★ 推荐：满足 F_cmd 硬约束 + 鲁棒接触                          │
│   Path B  : 纯 CHIP                                                      │
└──────────────────────────────────────────────────────────────────────────┘
```
