# 第三方 Low-Level Controller 训练方法调研

> 目标：为 Holosoma 下一步开发提供参考。重点对比 `third_party/UniFP`（CoRL 2025 Best Paper）和 `third_party/gentle-humanoid-training`（Stanford arXiv 2511.04679）两个 low-level controller 的训练细节。两份代码都已作为 git submodule 固化在 `third_party/` 下。
>
> **关键提醒：** 两个项目的目标、机器人形态、simulator、训练 framework 都完全不同。不要把它们当作"同一类方法的两种实现"来类比；他们解决的是不同问题。

---

## 目录

- [项目 1：UniFP（Unified Force + Position）](#项目-1unifpunified-force--position)
  - [1. 核心 idea](#1-核心-idea)
  - [2. 训练入口与命令](#2-训练入口与命令)
  - [3. 任务 / Env](#3-任务--env)
  - [4. Observation（actor + critic + 估计器 target）](#4-observationactor--critic--估计器-target)
  - [5. Action](#5-action)
  - [6. Reward](#6-reward)
  - [7. Termination](#7-termination)
  - [8. 网络结构](#8-网络结构)
  - [9. 算法与超参数](#9-算法与超参数)
  - [10. 数据集 / 参考轨迹](#10-数据集--参考轨迹)
  - [11. Domain randomization 与 curriculum](#11-domain-randomization-与-curriculum)
  - [12. Simulator 与机器人](#12-simulator-与机器人)
  - [13. 需要注意的 bug / 歧义](#13-需要注意的-bug--歧义)
- [项目 2：GentleHumanoid（上肢 compliance 的 WBT）](#项目-2gentlehumanoid上肢-compliance-的-wbt)
  - [1. 核心 idea](#1-核心-idea-1)
  - [2. 三阶段训练入口](#2-三阶段训练入口)
  - [3. 任务 / Env](#3-任务--env-1)
  - [4. Observation（policy / priv / priv_critic）](#4-observationpolicy--priv--priv_critic)
  - [5. Action](#5-action-1)
  - [6. Reward（四组）](#6-reward四组)
  - [7. Termination](#7-termination-1)
  - [8. 网络结构（Teacher-Student）](#8-网络结构teacher-student)
  - [9. 算法与超参数](#9-算法与超参数-1)
  - [10. 数据集（AMASS + Inter-X + LAFAN）](#10-数据集amass--inter-x--lafan)
  - [11. Domain randomization 与 curriculum](#11-domain-randomization-与-curriculum-1)
  - [12. Simulator 与机器人](#12-simulator-与机器人-1)
  - [13. 需要注意的继承关系 / 歧义](#13-需要注意的继承关系--歧义)
- [横向对比（便于决策下一步）](#横向对比便于决策下一步)
- [对 Holosoma 下一步的启发](#对-holosoma-下一步的启发)

---

# 项目 1：UniFP（Unified Force + Position）

**Repo:** `third_party/UniFP/`
**Paper:** `third_party/UniFP/UniFP.pdf`（arXiv:2505.20829，CoRL 2025 Best Paper；Zhi、Li、Yin、Jia、Huang）
**项目主页：** https://unified-force.github.io/

## 1. 核心 idea

UniFP 训练一个**同时能输出位置和力**的 low-level policy，做的是 **legged loco-manipulation** 场景：四足 + 机械臂 + 机械爪，机器人既要走路又要对环境施力/承受力。

核心贡献一句话概括：**把 impedance control 的方程直接编码进 RL 的 reward 和 estimator 里**。不是学"位置跟踪 reward + 力跟踪 reward"，而是学一个带 force offset 的单一 reward：

$$x_{\text{target}} = x_{\text{cmd}} + \frac{F_{\text{cmd}} + F_{\text{ext}} - F_{\text{react}}}{K}$$

配合一个 **concurrent state estimator (CSE)**：policy 从 32 帧本体感知历史里预测外力，训练时用 ground truth 监督；部署时这个 estimator 就是**虚拟力传感器**。

⚠️ **已发布的代码只覆盖 B2Z1（Unitree B2 四足 + Z1 机械臂）**。Paper 里也训练了 G1 humanoid 变体（把 `x_cmd_ee`/`F_cmd_ee` 去掉，只保留 `v_cmd_base` + `F_cmd_base`），但**该变体未开源**（README 的 TODO 列表明确标注）。

## 2. 训练入口与命令

- **Script:** `legged_gym/scripts/train_b2z1posforce.py`（24 行，强制 `headless=True`）
- **任务注册：** `legged_gym/envs/__init__.py:10` 里只注册了 `b2z1_pos_force` 一个任务
- **命令：**
  ```bash
  cd legged_gym/scripts
  python train_b2z1posforce.py --task=b2z1_pos_force --headless
  ```
- **Runner:** `legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py`（wandb project 硬编码为 `"UniFP"`）
- **Eval:** `play_b2z1posforce.py` 导出 JIT 模块（`adaptation_module.pt`、`adaptation_decoder.pt`、`actor_body.pt`）

## 3. 任务 / Env

- **任务名：** `b2z1_pos_force`
- **Env 类：** `LeggedRobot_b2z1_pos_force`（`legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`，2344 行）
- **Config 类：** `B2Z1PosForceRoughCfg`（`legged_gym/envs/b2/b2z1_pos_force_config.py`）

**这一个 policy 同时学 5 件事：**

1. **Base velocity 跟踪** — 跟随 `(v_x, v_y, ω_yaw)` 指令。`p=0.3` 概率采样"站立不动"；进入 force 阶段后提升到 `p=0.8`（`zero_vel_cmd_prob_after_force=0.8`）
2. **End-effector 位置跟踪** — 跟随 spherical coordinate 的 EE target `(r, θ_pitch, φ_yaw)`
3. **EE 力施加** — 跟踪 `F_ee_cmd ∈ [-60N, 60N]^3`
4. **Base 力施加** — 跟踪 `F_base_cmd ∈ [-50N, 50N]^3`
5. **对外部扰动的 compliance** — 处理未知的 EE/base 外力（up to 60N）

**Two-stage curriculum** 由 `commands.force_start_step=8000`（config:178）控制：前 8000×24 步做纯 locomotion + reaching；之后开启 force commands 和 external disturbances。Paper §3.2 明确说从一开始就加力会破坏平衡学习。

- **Sim 频率：** `dt=0.005s`, `decimation=4` → **control 50 Hz / physics 200 Hz**
- **并行环境数：** 默认 `num_envs=4096`

## 4. Observation（actor + critic + 估计器 target）

Asymmetric actor-critic。

### 4.1 Actor obs（`obs_buf`）

`num_single_obs=73`, `frame_stack=32` → **actor 输入 = 73×32 = 2336 维**

单帧构成（按序）：

| 分量 | 维度 | 内容 |
|---|---|---|
| `roll, pitch` | 2 | base 姿态欧拉角 |
| `base_ang_vel` | 3 | body frame 角速度，scale 0.25 |
| `dof_pos - default` | 17 | 关节位置误差（不含 gripper 2 DoF） |
| `dof_vel` | 17 | 关节速度，scale 0.05 |
| `actions` | 17 | 上一帧 action |
| `sin(2π·phase)` | 1 | gait clock |
| `cos(2π·phase)` | 1 | gait clock |
| `commands * scales` | 15 | 见下 |

**Commands 15 维：**

- `[0:3]` 基座速度指令 `(v_x, v_y, ω_yaw)`
- `[3:6]` EE 球坐标目标 `(r, pitch, yaw)`
- `[6:9]` EE 姿态指令 `(roll, pitch, yaw)` — **代码里从未写入，永远为 0**
- `[9:12]` `F_ee_cmd`
- `[12:15]` `F_base_cmd`

**Noise scales:** gravity 0.05, ang_vel 0.2, dof_pos 0.01, dof_vel 1.5

### 4.2 Critic obs（privileged）

`single_num_privileged_obs=149`, `c_frame_stack=3` → **critic 输入 = 447 维**

除了 actor 能看到的内容外，critic 额外有：

- `base_lin_vel`（ground truth，actor 看不到）
- 真实 EE 球坐标位置
- **真实外力**（EE + base，actor 必须靠 estimator 推断）
- `mass_params_tensor`（22 维：base 质量/CoM + 17 个关节质量扰动）
- `friction_coeffs`, `motor_strength`（domain randomization 参数）
- `stance_mask`（期望步态） + `contact_mask`（实际接触）
- `ee_goal_offset_local_sphere`：impedance 偏移后的目标（`F/K` shift 过的）

### 4.3 估计器 target（`obs_pred`）

12 维，作为 MSE 监督目标：

1. `base_lin_vel`（3）
2. `ee_pos_sphe_arm`（3）
3. `forces_local[gripper]`（3）← EE 外力
4. `forces_local[robot_base]`（3）← base 外力

**Loss weights:** `[0.2, 0.2, 1.0, 1.0]` — **力 loss 权重是位置/速度的 5 倍**。

## 5. Action

- **Shape:** `num_actions=17`（12 腿 + 5 臂；wrist_rotate 和 gripper 由固定内部 PD 控制，不由 policy 输出）
- **含义：** PD 控制器的**关节位置残差**
- **公式：** `q_target = 0.25 × motor_strength × a_t + q_default`（`action_scale=0.25`，`motor_strength` 是 per-env 随机 gain）
- **Torque:** `torques = Kp·(q_target - q) - Kd·q̇`
- **PD gains（刻意偏软，Paper §C.5）：**
  - 腿：hip/thigh `Kp=300`，calf `Kp=500`；`Kd=7.5/12.5`
  - 臂：`Kp=64-128`，`Kd=1.5-3.0`
  - **关键：偏软的 PD 故意让关节在受力时有 overshoot，这样位置误差就携带了力的信息，喂给 estimator 学习**
- **Action clip:** 100.0；`action_delay=3` 在 config 里声明但**代码里没实现**

## 6. Reward

Reward = Σ (scale × term × dt)。带方向："Only positive rewards" 为 **False**（config:256），负 reward 不 clip。

### 6.1 Tracking / shaping（正 reward）

| Term | Weight | 含义 |
|---|---|---|
| `tracking_ee_force_world` | **+2.0** | **核心 unified reward（EE）**：`target = curr_goal + (F_ext + F_cmd)/K_gripper`，`exp(-‖ee_pos - target‖₁ / 1.0 × 2)` |
| `tracking_lin_vel_force_world` | **+2.0** | **核心 unified reward（base）**：`v_target = v_cmd + (F_base + F_base_cmd)/K_base_d`，`exp(-err² / 0.25)` |
| `tracking_ang_vel` | +1.0 | 标准 `exp(-(ω_z_cmd - ω_z)² / 0.25)` |
| `feet_contact_number` | +2.0 | `+1` 如果接触 == stance_mask，否则 `-0.3` |
| `ref_dof_leg` | +1.0 | 跟踪内置正弦 trot 参考姿态 |
| `stand_still` | +0.5 | 不走时鼓励回 default pose |
| `alive` | +1.5 | 每步 +1 存活 reward |
| `feet_air_time` | +1.0 | 奖励 air_time > 0.5s 的着地 |
| `feet_height` | +1.0 | 走路时奖励前脚抬高到 10cm |

### 6.2 Regularization（负 reward）

| Term | Weight |
|---|---|
| `torques` | -5e-6 |
| `lin_vel_z` | -1.5 |
| `ang_vel_xy` | -0.02 |
| `dof_acc` | -2.5e-7 |
| `dof_vel` | -8e-4 |
| `dof_acc_arm` | -4.5e-7 |
| `dof_vel_arm` | -2e-4 |
| `collision` | -5.0 |
| `action_rate` | -0.02 |
| `action_rate_arm` | -0.045 |
| `dof_pos_limits` | -10.0（软阈值 80%） |
| `torque_limits` | -0.005（软阈值 90%） |
| `hip_pos` | -0.5 |
| `feet_drag` | -0.0008 |
| `base_height` | -2.0（target 0.50 m） |
| `feet_pos_xy` | -0.5 |
| `feet_height_high` | -15（脚超过 20cm 大惩罚） |

### 6.3 与 Paper Table A.1 的对应

| Paper 项 | 代码项 | 权重 |
|---|---|---|
| gripper position `exp(-‖x_ee - (x_cmd+(F_ext+F_cmd-F_react)/B)‖/0.5)` | `tracking_ee_force_world` | +2.0 |
| base velocity `exp(-‖v_base - (v_cmd+F_base/D)‖/0.25)` | `tracking_lin_vel_force_world` | +2.0 |

核心两个 reward 完全是 impedance control 方程。

## 7. Termination

- `|pitch| > 1.0 rad` 或 `|roll| > 0.8 rad` → 终止
- Timeout: `episode_length_s=20s`, `dt=0.02s` → 1000 步/episode
- `termination_contact_indices` 配置为空 list，所以接触终止**无效**

## 8. 网络结构

在 `legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py`。

### Adaptation Encoder（RMA 风格）

```
2336 (32 帧历史) → 512 → 256 → 128 → 64 (latent)
```

- Hidden: `[512, 256, 128]`, activation ELU
- latent dim = `num_obs / num_single_obs × 2 = 32 × 2 = 64`

### Adaptation Decoder

```
64 (latent) → 128 → 64 → 12 (obs_pred)
```

- 独立 Adam，`lr=1e-5`
- MSE loss 按 `[0.2, 0.2, 1.0, 1.0]` 加权

### Actor

```
concat(latest 73 frame, 64 latent) → 137 → 512 → 256 → 128 → 17
```

state-independent 可学习 `std`（`init_noise_std=1.0`）。**不是 recurrent**。

### Critic

```
447 (privileged) → 512 → 256 → 128 → 1
```

## 9. 算法与超参数

**算法：PPO + Concurrent State Estimation（CSE）辅助头**

| 超参 | 值 |
|---|---|
| `gamma` | 0.99 |
| `lam` (GAE) | 0.95 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.01 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `lr` | 1e-3 initial，**adaptive KL**（desired_kl=0.01，LR × 1.5 或 /1.5） |
| `max_grad_norm` | 1.0 |
| `num_steps_per_env` | 24 |
| `max_iterations` | **60000** |
| **总 env interactions** | **~5.9B** 步 |
| `save_interval` | 每 200 iter |
| Adaptation LR | 1e-5（独立 optimizer） |
| Timeout bootstrap | 是 |

**更新流程（每 minibatch）：**
1. 标准 PPO update（actor + critic + entropy）
2. 独立 Adam 步更新 adaptation loss（4 个 label × 权重）

## 10. 数据集 / 参考轨迹

**完全不用 mocap 或 retargeting。纯 RL from scratch。**

唯一的"参考"是 **程序化 scripted gait**（`compute_ref_state`）：
- `cycle_time=0.64s` 的 trot
- 通过正弦函数在 thigh/calf 上加关节 deltas
- 只进 `ref_dof_leg` reward，不做 imitation

EE goal 是纯随机球坐标采样：
- `r ∈ [0.35, 0.95]m`, `pitch ∈ [-2π/5, 2π/5]`, `yaw ∈ [-3π/5, 3π/5]`
- Lerp 到目标 1-3s，hold 0.5-2s
- 有一个固定 bbox 的碰撞检查避免与底盘冲突

## 11. Domain randomization 与 curriculum

### 11.1 物理参数 DR

| 参数 | 范围 |
|---|---|
| Ground friction | [0.3, 2.0]（256 buckets） |
| Base mass add | [0, 15] kg |
| Base CoM shift | ±0.15 m per axis |
| Gripper payload | [0, 0.2] kg |
| Leg/arm motor strength | [0.85, 1.15] |
| Leg mass scale | **disabled** |

### 11.2 Push disturbances（关键）

**两条独立的力流**，这是 UniFP 的关键 trick：

1. **cmd 流** — 力指令，**告诉 policy**（通过 `commands[9:12]` 和 `[12:15]`）
2. **ext 流** — 外部扰动，**不告诉 policy**（policy 必须通过 CSE 推断）

**Gripper force push（`_push_gripper`）：**
- 间隔 `[3.5, 9.0]s`，持续 `[1.0, 3.0]s`
- 每轴 `[-60, 60]N`，最大 3D 约 104N
- 概率 0.8（20% 的环境完全不受力）
- **剖面：linear ramp up → hold → ramp down**

**Base force push（`_push_robot_base`）：**
- 同结构，cmd ±50N xy；ext xy ±50N，z ±5N
- ⚠️ **该函数在主循环里被注释掉了（line 129）！** 所以实际训练里 base force disturbance 完全没施加，尽管 base force reward 和 commands 都在。**这是个可能的 bug，或者是刻意 ablation。**

### 11.3 Curriculum

- **Two-stage force curriculum:** `force_start_step=8000`
  - 前 8000×24 步：纯 reaching + locomotion
  - 之后：打开 force commands 和 disturbances
- `update_command_curriculum` 定义了但**没有在 step loop 里调用**
- `commands.curriculum=False`, `terrain.curriculum=False`

## 12. Simulator 与机器人

- **NVIDIA Isaac Gym Preview 4**（PhysX GPU）
- Terrain: `rough flat` proportion=1.0，heights `[0, 0.05]m`
- 机器人：**Unitree B2 四足 + Unitree Z1 机械臂**（B2Z1）
  - 腿 12 DoF + 臂 7 DoF = **19 DoF 总**，policy 控制 17 DoF
  - URDF: `resources/robots/b2z1/b2z1.urdf`
- **Python ≤ 3.8**（Isaac Gym Preview 4 要求）

## 13. 需要注意的 bug / 歧义

1. **`_push_robot_base` 在主循环被注释掉**（line 129），base-force disturbance 实际没施加
2. **EE orientation cmd（commands[6:9]）被观测但从不写入**，所以总是 0；`tracking_ee_orn` 相关 reward 也被设为 0
3. **`gripper` 质量随机范围 [0, 0.2]kg**，paper 写的是 [0, 0.5]kg
4. **Base force z 分量不对称**：ext 是 `[-50, 50]·0.1 = ±5N`，而 paper 广告的是 ±60N
5. **`action_delay=3`** config 声明了但没实现
6. **G1 humanoid 变体没开源**，README TODO 里列着
7. `randomize_gripper_force_gains=True` 但 `kp ∈ [200, 200]`（min=max），实际是常量

---

# 项目 2：GentleHumanoid（上肢 compliance 的 WBT）

**Repo:** `third_party/gentle-humanoid-training/`
**Paper:** `GentleHumanoid_code.pdf`（arXiv:2511.04679，Nov 2025，Stanford，Lu/Feng/Shi/Piseno/Bao/Liu）
**论文标题：** *GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction*

## 1. 核心 idea

训练一个 **universal whole-body motion tracking (WBT) policy**（Unitree G1 29-DoF），额外让 **上肢有 compliance**：可以被人推、被物体阻挡、被引导移动而不硬刚。应用场景包括 sit-to-stand、拥抱、握手、拿气球等 contact-rich interaction。

"Gentle" 的具体技术贡献：

- **Admittance reference dynamics model**：一个并行 sim 的 M-D 弹簧模型，把原始 motion target **柔化**成 compliant reference
- **Unified spring 力模型**：同时建模 resistive contact（被阻挡）和 guiding contact（被引导）
- **One-sided projection**：接触物理的单侧性（力不能"拉"）
- **τ_safe**：可调节的力安全阈值，作为 observation 同时作为 penalty
- **Force curriculum**：训练进度 60% 后才逐步引入外力

⚠️ **关键继承关系：** 这个 repo 实际上是 Botian Xu 的 `active_adaptation` framework 的 renamed fork（源自 FACET paper [ref 10]）。三阶段 teacher-student PPO、LayerNorm-Mish MLP、symmetry loss、locomotion rewards 这些都是继承的，**不是 GentleHumanoid 的原创**。原创的是 compliance 部分。

## 2. 三阶段训练入口

顶层 `train.sh` 对每个任务跑三阶段：

1. **TRAIN**（teacher PPO + student encoder supervised 监督）
2. **ADAPT**（pure supervised student encoder 蒸馏，1B 帧）
3. **FINETUNE**（student PPO，2B 帧，entropy 更软）

```bash
# 每阶段都是：
torchrun --nproc_per_node=4 scripts/train.py \
    task=G1/G1_gentle +exp={train,adapt,finetune} \
    wandb.id=... [checkpoint_path=run:$PROJECT/$PREV_ID]
```

- **Python 入口：** `scripts/train.py`（217 行），Hydra 配置
- **默认规模：** 16384 envs × 4 GPUs × ~5 小时（A100）per phase
- **三个 task 变体：**
  - `G1/G1_gentle` — GentleHumanoid with compliance（正本）
  - `G1/G1_no_force` — baseline，无外力（paper "Vanilla-RL"）
  - `G1/G1_extreme_force` — baseline，30N 随机扰动（paper "Extreme-RL"）

## 3. 任务 / Env

- **Env 类：** `SimpleEnv(_Env)`（`active_adaptation/envs/locomotion.py` — **文件名具有误导性**，实际是 WBT）
- **Env config：** `cfg/task/G1/G1.yaml`（base） + `G1_gentle.yaml`（override）
  - `num_envs: 16384`
  - `max_episode_length: 1000`
  - `sim.step_dt: 0.02`（50 Hz control），`isaac_physics_dt: 0.005`（200 Hz physics），`decimation=4`
  - `terrain: plane`
- **Command manager 是关键：** `MotionTrackingCommand_impedance`（`active_adaptation/envs/mdp/commands/motion_tracking.py:541`），这是 1177 行的核心文件
  - Base 类 `MotionTrackingCommand`（line 71）处理 motion tracking
  - Impedance 子类叠加 compliance

## 4. Observation（policy / priv / priv_critic）

三组 observation（`cfg/task/G1/G1_gentle.yaml:71-108`）：
- **policy**：student（可部署）的输入
- **priv**：teacher 专属的 privileged obs
- **priv_critic**：critic 专属

所有 obs 过 `VecNorm`（running mean/std，`decay=0.9999`）。Noise 在 normalize **之前**加（clamp 到 ±3σ）。

### 4.1 Policy（student actor 输入）

| Obs | 维度 | 说明 |
|---|---|---|
| `boot_indicator_state` | 1 | reset 后 25 步衰减到 0 |
| `command` | 22 | motion 指令打包：5 个 future step 的 root_z (5) + root xy diff (4×2=8) + heading xy (4×2=8) + τ_safe (1) |
| `target_joint_pos_obs` | ~145 | 5 个 future step × 29 关节 |
| `target_projected_gravity_b` | 15 | 5 future step × 3 |
| `root_ang_vel_history[0]` | 3 | σ=0.05 |
| `projected_gravity_history[0]` | 3 | σ=0.01 |
| `joint_pos_history` | 174 | 6 个时间 tap（当前 + 1,2,3,4,8 步前）× 29 关节 |
| `prev_actions[:3]` | 87 | 前 3 步 action |

**Total ≈ 450 维**

### 4.2 Priv（teacher-only）

- Tracking priv: target root pos err (15), target linvel (15), relative quat (15)
- **Impedance priv:** `force_priv` = `[force_keypoint_b, force_applied_b, force_expected_b, force_sample_timer]` ≈ 19 维（admittance 模型状态）
- Feet: `body_height` (4), `contact_forces` (6)
- Root: `root_linvel_b`(EMA 0.2) (3), `*_history[0:8]` (27+27)
- Joints: `joint_pos_history[0:8]` (261)
- Bodies: `current_keypoint_b`, `target_keypoints_diff_b_obs`, velocities
- Action/torque: `applied_action` (29), `applied_torque` (29)

### 4.3 Priv_critic

只有 `cum_error`（3 维归一化 tracking error），同时也是 termination 依据。

### 4.4 Paper 对应

Paper §III-E.1：
- student `o_t = (τ_safe, m_tar, ω, g, q_hist, a_{t-3:t-1})` ← 精确对应 policy 组
- teacher `o_priv = (x_ref, ẋ_ref, f_interact, f_sim_interact, h, τ_{t-1}, e_cum)` ← 对应 priv 组

## 5. Action

- **Action dim: 29**（完整 G1 29-DoF）
- **类型：** PD joint position targets（不是 torque）
- **Formula:** `pos_tgt = default_joint_pos + offset + applied_action × action_scaling`
  - `offset` 是 per-env 随机 joint offset（DR）
  - `action_scaling` 按关节类型分档：
    - elbow / shoulder / wrist → 1.0
    - hip_roll / hip_yaw → 0.25
    - hip_pitch / knee / ankle → 0.5
    - waist → 0.25

### Action 过滤与延迟

- Raw action clamp 到 ±10
- **Communication delay:** per-env 随机 0-4 物理步（0-20ms）
- **EMA smoothing:** α ∈ [0.8, 1.0]，`applied_action = lerp(applied_action, delayed_action, α)`
- **Boot protection:** reset 后若干步输出 initial pose 而非 policy 输出

### PD gains（per-joint，`humanoid.py:105-148`）

- 腿/waist_yaw：~40-99 N·m/rad
- waist_roll/pitch, ankles：~28.5
- 臂/elbow/wrist_roll：~14.25
- wrist_pitch/yaw：~16.77
- Effort limits: 腿 88-139 N·m，臂 25 N·m，腕 ~5 N·m
- Soft joint limit factor: 0.9

## 6. Reward（四组）

Reward 按**组**组织（yaml line 109-135），每组加和后拼成 `[N, num_groups]`，再对组轴求和给 PPO。**Reward 乘以 `dt=0.02`**，所以 nominal weight 是 per-second 的。

Sigma 通过 `_calc_exp_sigma` 实现，对多个 σ 求平均（**scale-mixture of Gaussian kernels**）。

### 6.1 `impedance` 组（compliance，novel 部分）

| Term | Weight | 含义 |
|---|---|---|
| `force_reward` | **2.0** | `exp(-‖f_applied - f_expected‖ / σ)`，σ=[8,4]；`f_applied > τ_safe+10N` 时归零（paper `r_force`） |
| `force_exd_penalty` | **6.0** | `-1` 如果 `f_applied > τ_safe+10N` **且** `> f_expected+5N`（paper `r_pen`） |
| `force_target_tracking` | 2.0 | 6 个上肢 key body 跟踪 **compliant reference** `x_ref_link`（不是 raw motion target），σ=0.3（paper `r_dyn` 位置部分） |
| `force_target_vel_tracking` | 1.0 | 速度版本，σ=[1.0, 0.5]（paper `r_dyn` 速度部分） |
| `keypoint_tracking_imp` | 2.0 | 覆盖 base keypoint tracking：6 个上肢用 admittance 参考，其余（头、膝、踝）用 raw motion |
| `lower_keypoint_tracking` | 2.0 | 下肢刚性跟踪（`.*knee.*`, `.*ankle_roll_link`），保持腿稳定 |

### 6.2 `tracking` 组（motion tracking）

继承自 `active_adaptation`：

| Term | Weight | 说明 |
|---|---|---|
| `root_pos_tracking` | 0.5 | σ=0.3 |
| `root_rot_tracking` | 0.5 | σ=[1.0, 0.5] |
| `root_vel_tracking` | 1.0 | body frame linvel，σ=[1.0, 0.5] |
| `root_ang_vel_tracking` | 1.0 | body frame angvel，σ=3.0 |
| `joint_pos_tracking` | 1.0 | L1 误差，关节：`waist_*`, `hip_*`, `knee`, `wrist*`（不含 shoulder/elbow，那些只走 keypoint） |
| `joint_vel_tracking` | 0.5 | 有限差分速度 |

### 6.3 `loco` 组（locomotion 稳定性）

| Term | Weight | 说明 |
|---|---|---|
| `survival` | **5.0** | +1 每步，absolute +0.1/step |
| `impact_force_l2` | **4.0** | `-Σ(‖contact_force‖/mg)² × first_contact` |
| `feet_slip` | 2.0 | `-Σ v_foot_xy² × in_contact` |
| `feet_air_time_ref` | **10.0** | Reference-gated air time reward with `thres=0.8`，按 motion 里的 feet_standing mask 翻转符号 |
| `action_rate_l2` | 0.1 | `-‖a_t - a_{t-1}‖²` |
| `joint_vel_l2` | 5e-4 | regularizer |
| `joint_pos_limits` | 1.0 | 软阈 90% |

**权重主导：** `feet_air_time_ref=10`, `force_exd_penalty=6`, `survival=5`, `impact_force_l2=4`

## 7. Termination

- **`cum_error`**（唯一显式 termination）
  - `_cum_error` 是 `[pos_err/0.3, rot_err/0.7, keypoint_err/0.25]` 归一化向量
  - 任一分量 > 1.0 **持续 > 50 步** → 终止
  - 回到阈值下计数器归零
- **没有 fall_over / crash termination**（class 存在但未启用）
- **Truncation:**
  - `episode_length ≥ 1000` (= 20s)
  - Motion clip 播完（`finished = t >= lengths - 1`）
  - 前 5 步禁用 termination

## 8. 网络结构（Teacher-Student）

文件：`active_adaptation/learning/ppo/ppo.py` + `common.py`

**所有 MLP：** `[LazyLinear → LayerNorm → Mish] × layers`，**LayerNorm before activation**（而不是 ReLU/ELU）

### Critic

```
[OBS + OBS_PRIV + CRITIC_PRIV] → [512, 512, 256] → Linear(1)
```

### Privileged encoder（teacher 的）

```
OBS_PRIV → [256] → Linear(256) → priv_feature
```

### State estimator（student 的）

```
OBS → [512, 256] → Linear(256) → priv_pred
```

训练中通过 MSE 对齐到 priv_feature（`reg_lambda=0.2`）。

### Actors

**Teacher actor:** `[OBS, priv_feature] → [512, 512, 256] → Actor(29)`
**Student actor:** `[OBS, priv_pred]    → [512, 512, 256] → Actor(29)`（共享结构）

- `Actor`：`LazyLinear(29)` 为 mean，**state-independent 可学习 `std`（`nn.Parameter`）**，`init_noise_scale=1.0`
- `IndependentNormal` (diag Gaussian)
- **Weight init:** orthogonal, gain=0.01
- **No recurrence.** 时间上下文完全靠显式 `joint_pos_history`, `prev_actions` 等 obs

## 9. 算法与超参数

**算法：** PPO（`active_adaptation/learning/ppo/ppo.py`） + 三阶段 teacher-student

| 超参 | 值 |
|---|---|
| `gamma` | 0.99 |
| `lam` (GAE) | 0.95 |
| `train_every` | 32（rollout 长度） |
| `ppo_epochs` | 5 |
| `num_minibatches` | 8 |
| `lr` | 5e-4（train），1e-4（finetune） |
| `clip_param` | 0.2 |
| `desired_kl` | 0.01（adaptive LR：`*1.1` 或 `/1.1`，bounded `[1e-5, 5e-3]`） |
| `entropy_coef` | 0.005 → 0.002（train）；0.002 → 0.0005（finetune），指数衰减 |
| `value_norm` | False（用 `ValueNormFake` 传递） |
| `reg_lambda` | 0.2（priv-feature 对齐 MSE，annealed） |
| `init_noise_scale` | 1.0 |
| Grad-norm clip | 1.0 |

### 独特设计：Symmetry augmentation

`ppo.py:342-384`：每个 minibatch 会 duplicate 一个 left/right mirrored version。Policy loss 只在 original half 计算；额外的 `symmetry_loss_loc=0.2` 和 `symmetry_loss_std=10` 惩罚 `μ_orig` 和 `sym(μ_mirror)` 的差异（以及 σ 的差异）。**这是个很激进的对称性约束。**

### 三阶段细节

1. **Train（PPO teacher）：** PPO 更新 `actor_teacher + encoder_priv`，并**同时**用监督 MSE 训练 `adapt_module`（student encoder）
2. **Adapt（监督蒸馏）：** 只训 estimator，纯 supervised 1B 帧，2 epochs × minibatches inner loop
3. **Finetune（student PPO）：** PPO 更新 `actor_student + adapt_module`，2B 帧，`lr=1e-4`。**开启 `_student_train_`**：用 1s lookahead yaw-compensated root target 避免 student 因为累积 drift 被惩罚

**Multi-GPU：** DDP 包所有模块（`ppo.py:223-231`），NCCL，`cfg.seed = base_seed + rank*10000`

**Checkpoint load：** 从 train 阶段 load 时，teacher weights **硬拷贝**到 student（`ppo.py:557-562`）

## 10. 数据集（AMASS + Inter-X + LAFAN）

### 数据源

- **AMASS** [Mahmood 2019]：大规模 mocap
- **Inter-X** [Xu 2023]：human-human interaction mocap
- **LAFAN** [Harvey 2021]：游戏级 locomotion

### Retargeting

- **GMR**（General Motion Retargeting，[Ze et al. 2025]）的**修改 fork**：`github.com/Axellwppr/GMR`
- 直接输出训练 env 期望的 npz 格式
- **~25 小时过滤后的数据**（paper §III-E.2）
  - AMASS: 7875 files，~31 hrs raw
  - Inter-X: 4207 files，~10 hrs
  - LAFAN: 9 files，~0.26 hrs
  - 总计 ~41 hrs raw，过滤掉"high-dynamic"后约 25 hrs

### Npz 格式

`fps, root_pos, root_rot (xyzw quat), dof_pos, local_body_pos, local_body_rot, body_names, joint_names`

### 数据集预处理（`generate_dataset.sh`）

```bash
scripts/data_process/generate_dataset.py \
  --dataset-root $DATASET_ROOT/{AMASS,InterX,LAFAN} \
  --allowlist scripts/data_process/allowlist_{amass,interx,lafan}.json \
  --mem-path dataset/{amass,interx,lafan}
```

- 每个 allowlist 是 JSON list of `(filename, start, end)` 合法片段
- 存成 **memory-mapped float16 tensors**（16384 envs × 1000-step 需要节省空间）
- **每帧独立平移 z**，使 `min(left_foot_z, right_foot_z)=0`（**按帧 ground-contact normalize**，而不是 constant offset）

### 运行时加载

- `ProgressiveMultiMotionDataset`（`utils/multimotion.py`）
- `path_weights = [0.4, 0.2, 0.4]` for `[interx, lafan, amass]`（**LAFAN 被上采样**）
- Double-buffer（A/B）refresh：每 20000 resets 后台重填然后 swap
- 单 env 最长 1000 步（20s）
- z offset +0.035 全局应用
- `sample_init_robot`：写入 motion 的 root 和 joint state 作为 episode 初始

### 未来窗采样

每步 fetch `[0, 2, 4, 8, 16]` 步的 motion（= `[0, 40, 80, 160, 320]ms` lookahead @ 50Hz），供 reward 和 observation 使用。

### Hand-grid samples

`scripts/data_process/hand_grid_samples.pt`：预计算的手部在躯干系下的候选位置（从 motion 集里采样），用作 guiding-contact 的 anchor 候选。

## 11. Domain randomization 与 curriculum

### 11.1 物理 DR（`G1_gentle.yaml:137-154`）

| 项 | 范围 |
|---|---|
| `perturb_body_com` on pelvis/torso | ±0.02 m per axis（启动时一次） |
| `static_friction` on ankle_roll | [0.3, 1.6]（16 buckets） |
| `dynamic_friction / static` | [0.75, 1.0] |
| `restitution` | [0.0, 0.2] |
| Motor stiffness/damping | ±10% per reset |
| Motor armature | ±25% per reset |
| `random_joint_offset` | ±0.01 rad per reset（同时作为 obs 的减去基线） |

### 11.2 Action-side DR（内置于 `JointPosition`）

- Per-env random control delay `[0, 4]` 物理步
- Per-env EMA α 初始 `[0.9, 0.9]`，每步 jitter ±0.025，clamp `[0.8, 1.0]`
- Boot-protection 随机延迟

### 11.3 State init noise

root_pos ±0.03m, root_ori ±0.1 rad, joint_pos ±0.1 rad 等，clamp ±1σ。

### 11.4 Force / compliance curriculum（核心）

`MotionTrackingCommand_impedance.step_schedule`（`motion_tracking.py:1053-1060`）：

```python
ratio = clamp(progress/0.6, 0.25, 1.0)
force_prob = 0.15 * ratio
force_type_probs = [1 - 4*force_prob, force_prob, force_prob, force_prob, force_prob]
# [none, both_arms, left, right, partial]
```

- **训练前期 ~85% 的环境不受力**
- progress=0.6 时达到 paper 声明的 `[0.4, 0.15, 0.15, 0.15, 0.15]` 分布
- `force_alpha=1.0`（只 ramp 频率，不 ramp 幅度）

### 11.5 Zero-init probability

`zero_init_prob` 从 1.0（永远从 motion 帧 0 开始）在第一次 `step_schedule` 调用后变成 0.0（之后从随机偏移开始）。

### 11.6 Net-wrench limiter

`_limit_net_wrench_about_torso`：6 body 合力超过 30N 或合 torque 超过 20N·m 时，在躯干施加反向残余 wrench 使净量被限制。防止 stacked spring 造成整体非物理加速度。

## 12. Simulator 与机器人

- **Isaac Sim 4.5.0 + Isaac Lab 2.2.0**（PhysX GPU）
- **不是 Isaac Gym**（paper §III-C 提到了但没用）
- **机器人：Unitree G1 29-DoF**
  - USD: `active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd`
  - 腿 12 + 腰 3 + 臂 14 = 29 DoF
- **Keypoint 跟踪 body（~10 点）：** `head_mimic`, `.*_hand_mimic`, `.*wrist_roll_link.*`, `.*shoulder_yaw_link`, `.*knee.*`, `.*ankle_roll_link`
- **Force 施加 body（6 点）：** 双 shoulder_yaw_link, 双 wrist_roll_link, 双 hand_mimic
- **Contact sensor：** `.*ankle_roll_link`（`history_length=3`, `track_air_time=True`）
- **Init pose：** 微蹲（hip_pitch=-0.28, knee=0.5, ankle_pitch=-0.23），臂前伸（elbow=0.87），root z=0.74m
- **分布式：** `torchrun --nproc_per_node=4`，NCCL

## 13. 需要注意的继承关系 / 歧义

1. **大部分是继承自 `active_adaptation` / FACET**，GentleHumanoid 原创只是 compliance 部分（impedance 子类 + force reward + admittance 模型）
2. **Observation 维度是运行时计算的**（取决于数据集 joint 匹配），上面的数字是估计
3. **Reward 乘以 dt**：所以 nominal weight 是 per-second 值，实际 per-step 是 weight × 0.02
4. **Motion dataset 有三份 allowlist**：具体哪些片段被过滤掉不是一次性可见的，需读 JSON
5. `action_rate2_l2`（二阶差分）定义了但**未启用**
6. `setup.py:12` 提到 `mujoco`，但代码里没有 MuJoCo 后端（可能是 eval/deployment 用）
7. **Paper 涉及的 high-level 应用**（vision-based pose estimation、hugging planner、teleop）**不在 training repo 里**

### GentleHumanoid 贡献（paper 的"Gentle" 具体意思）

1. **Admittance 参考动力学**：每个上肢 6 keypoint 的 1D `M ẍ = f_drive + f_interact - D ẋ`（`M=0.1 kg, D=2.0`，4 semi-implicit Euler substeps）。模型积分出的 `x_ref` 取代 raw motion target 作为跟踪目标

2. **Unified spring 力模型**：`f_interact = K_spring (x_anchor - x_cur)`，anchor 要么是初始接触点（resistive），要么是从 motion 数据集采样的姿势（guiding）。`K_spring ~ U(5, 250) N/m`，piecewise-linear gain schedule

3. **One-sided (unilateral) projection**：只保留沿力方向的负分量，模拟"接触不能拉"的物理性质

4. **τ_safe 可调节力阈值**：`U(5, 15)N`，`TemporalLerp` 100-200 步 hold + 25-100 步过渡。同时作为 obs（policy 知道）和 penalty（超过 `τ_safe+10N` 扣分）

5. **核心 novel reward**：`force_target_tracking` + `force_target_vel_tracking` + `keypoint_tracking_imp`（把 admittance 参考替代 raw target） + `force_reward` + `force_exd_penalty`

6. **Net-wrench limiter**（上面讲过）

7. **Force curriculum**：前 60% 训练不加力，再逐步 ramp 概率

---

# 横向对比（便于决策下一步）

| 维度 | UniFP (B2Z1) | GentleHumanoid (G1) |
|---|---|---|
| **任务** | 四足 + 机械臂 loco-manipulation，**同时跟踪位置和主动施力** | 仿人 WBT + **被动承受/兼容外力** |
| **机器人** | Unitree B2 + Z1（19 DoF，policy 控 17） | Unitree G1（29 DoF，policy 全控） |
| **Simulator** | **Isaac Gym Preview 4**（Python ≤ 3.8） | **Isaac Sim 4.5 + Isaac Lab 2.2**（PyTorch） |
| **Control rate** | 50 Hz | 50 Hz |
| **Framework** | 定制 `legged_gym`（PPO + CSE） | `active_adaptation`（PPO + teacher-student） |
| **并行 envs** | 4096 | 16384 × 4 GPUs |
| **总训练量** | ~5.9B 步 | train 4B + adapt 1B + finetune 2B = **7B** 帧 |
| **训练阶段** | 单阶段 PPO + 辅助 estimator loss | **3 阶段**：teacher → student 蒸馏 → student PPO |
| **策略输出** | 17-DoF **关节位置残差** (`a × 0.25 + q_default`) | 29-DoF **关节位置**（per-joint scaling） |
| **PD gains** | 偏软（腿 300-500，臂 64-128） | 标准（腿 40-99，臂 14.25-28.5） |
| **参考轨迹** | **无 mocap**，只有程序化 sinusoidal trot gait clock | **AMASS + Inter-X + LAFAN ~25 小时**（GMR retargeting） |
| **Obs 风格** | Actor 32 帧历史 stack（2336 维）+ 独立 CSE 估计力 | Student/Teacher 分离，显式 history taps + 未来窗，**~450 维** |
| **Critic** | 447 维 privileged（3 帧 × 149） | `[OBS + priv + priv_critic]` 全部 |
| **核心 novel reward** | `tracking_ee_force_world` + `tracking_lin_vel_force_world`（impedance 方程直接入 reward） | `force_reward` + `force_exd_penalty` + `keypoint_tracking_imp`（admittance 参考替换 raw） |
| **Force 处理** | **主动施力**：cmd 流（policy 知道）+ ext 流（policy 通过 CSE 推断） | **被动承受**：unified spring 模型 + τ_safe 安全阈值 |
| **Curriculum** | `force_start_step=8000` 二段式 | `progress/0.6` 渐进 force_prob，前 60% 无力 |
| **Termination** | 姿态阈值 + 20s timeout | `cum_error` sustained > 50 步 + motion clip 结束 |
| **算法特色** | Concurrent State Estimation (CSE) RMA 风格 | Symmetry augmentation + teacher-student + LayerNorm-Mish |
| **Adaptive LR** | `*1.5` / `/1.5`（`bounded [1e-5, 1e-2]`） | `*1.1` / `/1.1`（`bounded [1e-5, 5e-3]`） |
| **开源完整性** | **不完整**（G1 humanoid 变体、imitation pipeline、sim2real 都没开源） | **完整**（3 阶段训练 + 数据集生成 + baselines 都能跑） |
| **Python** | 3.8（硬要求） | 无硬性要求（Isaac Lab 通常 3.10） |

## 两个项目的角色其实互补

- **UniFP 解决的是**：机器人"**施**力给环境"——主动推、拉、按，同时保持 locomotion 和 reaching。关键技术：impedance 方程入 reward，force-sensor-free estimator
- **GentleHumanoid 解决的是**：机器人"**受**力于环境"——被人推、被物体阻挡、拥抱、握手。关键技术：admittance 参考模型，unified spring 力场，安全阈值

两者的 reward 形式看似相似（都用 `exp(-err/σ)`），但目标完全相反：
- UniFP 要 `F_applied ≈ F_cmd`（**力要正确施加**）
- GentleHumanoid 要 `F_applied ≈ F_expected`（**承受力要柔和，不要硬刚**）

---

# 对 Holosoma 下一步的启发

参考 Holosoma 当前状况（CLAUDE.md）：
- 当前的 WBT training 在 `src/holosoma/holosoma/envs/wbt/`，走 `exp:g1-29dof-*` 配置，支持 IsaacGym / IsaacSim / MJWarp
- 有 retargeting pipeline（`src/holosoma_retargeting/`）处理 SMPL-H/OMOMO/LAFAN
- 已经做 multi-motion training（commit `7ce2c04`）

**可以从两个项目各拿什么：**

### 可直接借鉴的（低集成成本）

1. **Force / compliance curriculum 模板**（GentleHumanoid §10d-11.4）：progress-gated 渐进开启外力。Holosoma 已有 curriculum manager，加一个 ForceCommand curriculum term 就能套
2. **两条力流设计**（UniFP §11.2）：cmd 流可见 + ext 流不可见，强制 policy 同时学习"听指令"和"推断未知"。这个 pattern 可以扩展到任何需要 policy 从 proprio 推断隐藏参数的场景
3. **Admittance 参考模型作为 reward target**（GentleHumanoid §12.1）：比直接跟 raw motion 柔和。如果 Holosoma 以后要做 human-robot contact，这是基石
4. **Symmetry augmentation**（GentleHumanoid §9）：+0.2 loss weight，直接 duplicate minibatch。对 humanoid bilateral symmetry 很有效
5. **Teacher-student 三阶段蒸馏**（GentleHumanoid §9）：student PPO finetune 阶段 + student reward 改写（用 lookahead target 补偿 drift）。Holosoma 现在是 single-phase，升级到 multi-phase 会在 sim-to-real 上有明显收益

### 需要设计决定的

1. **Isaac Gym vs Isaac Lab**：UniFP 仍在 Isaac Gym Preview 4 + Python 3.8（已 EOL 方向）；GentleHumanoid 已经全面迁移到 Isaac Lab 2.2。Holosoma 两个都支持——但如果要借鉴哪个项目的具体代码，要先确定 simulator backend
2. **Action scale**：UniFP 用 0.25 残差，GentleHumanoid 用 per-joint scaling `[0.25, 0.5, 1.0]`。Holosoma 当前怎么配？需要对齐
3. **Obs 组织方式**：UniFP 用 32 帧 stack（隐式时间），GentleHumanoid 用显式 history taps + 未来窗（显式时间）。后者更易调，也更易对齐 mocap future frames

### 数据方面

- UniFP 完全不用 mocap —— 如果 Holosoma 的下一步只是 locomotion + force，不需要投入 retargeting 成本
- GentleHumanoid 用 **GMR 修改 fork** 做 retargeting，输出 npz 格式。Holosoma 现在用自己的 retargeting pipeline（SMPL-H / OMOMO / LAFAN），可以对比看是否有兼容可能

### 要避免的坑

1. **UniFP 的 `_push_robot_base` 被注释**——如果借鉴它的 force push 结构，注意这个 bug，要确认 base force 真的施加
2. **GentleHumanoid 的 obs 维度运行时计算**——需要跑起来才知道确切数字，调试时需注意
3. **UniFP 的 G1 humanoid 变体没开源**——别指望直接抄 G1 配置，只能看 paper 自己重新实现

---

## 引用文件索引

### UniFP
| 内容 | 路径 |
|---|---|
| 训练入口 | `third_party/UniFP/legged_gym/scripts/train_b2z1posforce.py` |
| 任务注册 | `third_party/UniFP/legged_gym/envs/__init__.py` |
| 主 env | `third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`（2344 行） |
| Env config | `third_party/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py` |
| Actor/Critic | `third_party/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py` |
| PPO + 估计器 | `third_party/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py` |
| Paper | `third_party/UniFP/UniFP.pdf`（方法见 p4-5，训练细节 p14 §C.1，reward 表 p15 A.1） |

### GentleHumanoid
| 内容 | 路径 |
|---|---|
| 顶层训练 | `third_party/gentle-humanoid-training/train.sh` |
| 数据生成 | `third_party/gentle-humanoid-training/generate_dataset.sh` |
| Python 入口 | `third_party/gentle-humanoid-training/scripts/train.py` |
| **核心 command（compliance）** | `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py`（1177 行） |
| Admittance model | `third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/admittance.py` |
| Env | `third_party/gentle-humanoid-training/active_adaptation/envs/locomotion.py`（128 行，是 WBT env） |
| PPO | `third_party/gentle-humanoid-training/active_adaptation/learning/ppo/ppo.py`（590 行） |
| Main config | `third_party/gentle-humanoid-training/cfg/task/G1/G1_gentle.yaml` |
| Paper | `third_party/gentle-humanoid-training/GentleHumanoid_code.pdf`（11 页） |

### 研究原始稿（更详细，含更多 file:line）

- `/tmp/unifp-research.md`（426 行）
- `/tmp/gentle-humanoid-research.md`（446 行）

> 如果后续要深挖某个具体方面（比如某个 reward 的确切 σ 配置、某个 DR 参数的确切范围），优先查这两份原始稿，里面有精确的行号引用。
