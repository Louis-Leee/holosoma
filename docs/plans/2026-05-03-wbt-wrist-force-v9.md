# WBT Wrist Force Training — 实施计划 (v9.4 — K_virtual as actor obs, GH-style range sampling)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **v9 → v9.1 → v9.2 → v9.3 → v9.4 变更：**
> - v9.1 曾尝试 4-group + `critic_obs` shim 来实现"CHIP selective history"（只给 proprio+actions stack，命令不 stack）。
> - v9.2 回到 Holosoma baseline 2-group 架构，两个 group 都用 `history_length=10`，删除所有 PPO workaround。
> - **v9.3 追加 "Evaluation & Deployment" 完整链路**：训练完成后的 in-training eval（零改动）、ONNX export（自动适配 1620 dim，v9.4 后升级自 v9.3 的 1600）、sim-to-sim (MuJoCo)、真机部署；新增 Phase 7 共 8 个 task（Task 14-21），覆盖 `holosoma_inference` 端的 obs preset、inference config + entry point、F_cmd input provider、WBT force policy 子类、dispatch 扩展、workflow docs、跨-package schema 一致性测试。
> - **v9.3 经 codex review 修正**：(a) 补充 `TaskConfig` + `InputSource` Literal + `inputs/__init__.py` factory 扩 `"wrist_force"` role（缺此步部署时 F_cmd 永远 0）；(b) Task 18 改为加 `BasePolicy._poll_extra_inputs()` hook 而非 override `run()`（保留统一 latency/rate 管控）；(c) `_handle_stop_policy` 必须同时调 `provider.zero()`（防止 provider 内部 buffer 下一 tick repopulate）；(d) Task 14 加 inference 侧 runtime metadata validation（不只写 metadata 还要校验，防止 term 顺序不同但 dim 巧合相等的 ONNX 静默行为漂移）；(e) 键位避开现有 `o`=STOP / `i`=INIT / `]`=START / `m`=MOTION 冲突；(f) `setup.py` 同步注册 `holosoma.policies.wbt` entry point。
> - **v9.4 关键升级：K_virtual 变成 per-env + actor obs**：借鉴 GentleHumanoid 的 `kp_range=[5,250]` 区间采样模式，**但 v1 把 range 写成 `[100.0, 100.0]` 让它退化成 deterministic 100**。Actor obs 新增 `wrist_virtual_stiffness_command` (shape `[N, 2]`：左右 wrist 各 1 维；dim=2 为预留双手不同 K 的能力)；reward 的 `K_virtual` 从 command term 读 per-env buffer 而不是常量。**v2 只需改 range 成 `[50, 300]` 就能立刻跑随机化训练（对齐 GH），不用重写 obs/reward/exporter 链路**。训练端 actor 单步 dim 160 → **162**（baseline 154 + wrist_force_command 6 + wrist_virtual_stiffness_command 2），stacked 1600 → **1620**；critic 单步 dim 298 → **300**，stacked 2980 → **3000**。
> - **v9.4 回答"K 是否进 CHIP/GH 的 critic"**：
>   - **CHIP**: `1/k` **进 actor obs**（即 `g_hind_ptb = g − (1/k)·f`）。paper 没有显式列 critic obs 组成，但 CHIP 用的是 MLP PPO，**critic 通常 ⊇ actor obs**，所以 critic 也看 1/k。
>   - **GentleHumanoid**: `kp` **不进 actor (`command` @`motion_tracking.py:1062-1084`) 也不进 critic privileged (`force_priv` @`motion_tracking.py:1094-1101`)**。GH 的 `force_priv` 只暴露 `force_keypoint_b` / `force_applied_b` / `force_expected_b` / `force_sample_timer`——policy 从 "applied vs expected force" 差值里间接感知 kp，而不是直接看 kp 数值。
>   - **v9.4 设计决定（CHIP 派系）**：K 同时进 actor 和 critic（而不是 GH 的 "actor/critic 都不看 K"），让 policy 直接拿到 F↔Δx 的兑换率。理由：(a) CHIP paper 实证效果更好；(b) v1 退化成常数 100，actor 看到的是常量没成本；(c) 未来做 per-wrist 不同 K 随机化时 policy 能显式区分"这只手该偏多少"。
> - Obs 维度（v9.2 baseline，v9.4 已升级见上方）：actor 1600 dim（160×10）、critic 2980 dim（298×10）—— **v9.4 升级后为 1620 / 3000**（见 v9.4 变更条）。

**目标：** 在 Holosoma 里新增一个 WBT（Whole-Body Tracking）训练变体 `exp:g1-29dof-wbt-force`，让 G1 人形机器人学会**通过主动力指令（F_cmd）控制左右手腕发力**——既做 motion tracking，又能按上层输入施加指定方向和大小的力。

**术语速查（全文通用）：**
- **WBT** = Whole-Body Tracking，让 policy 跟随全身 motion reference（如 mocap 动作）运动
- **F_cmd** = force command（主动力指令，6 维，部署时上层输入）
- **F_ext** = external force（sim 注入的外部扰动力，critic privileged obs）
- **K_virtual** = reward 里的虚拟弹簧刚度（training hyperparameter，单位 N/m）
- **actor** = RL policy 网络（输入 obs → 输出 action；部署时唯一要跑的网络）
- **critic** = 训练时的 value network（可看额外 privileged obs，降 value 估计方差；部署时不用）
- **GH** = GentleHumanoid（`third_party/gentle-humanoid-training`）
- **UniFP** = Unified Force-Position control paper (`third_party/UniFP/`)
- **CHIP** = Adaptive Compliance via Hindsight Perturbation paper (`third_party/CHIP.pdf`)
- **Δx** = wrist target 相对 motion target 的位移量 = `F / K_virtual`
- **proprio / proprioception** = 本体感知，含 `dof_pos` + `dof_vel` 等机器人自身状态（不含 motion reference / command）
- **past actions** = policy 过去若干步输出的 action
- **history_length** = Holosoma `ObsGroupCfg` 的字段，group 里所有 term 都会被 stack 这么多步（group-level 而非 per-term）

---

## 📖 通俗易懂版：我们在做什么

### 一句话

给 G1 训练一个 policy，部署时上层给它 **"左手和右手各应该输出多少牛、往哪个方向"**（一共 6 个数字），policy 就能让手腕产生对应的接触力——同时不耽误 whole-body motion tracking（跟随 mocap 动作）。

### 为什么这么做

现在 Holosoma 的 WBT policy 只会 "跟着 motion reference 走"。要让它在真实世界里擦桌子、开门、推物体，就必须能**主动在 end-effector 输出力**。业界主要有三种思路：

| 思路 | 代表 | 核心想法 | 我们的评价 |
|---|---|---|---|
| Active force via virtual impedance | **UniFP** | reward 里加个"虚拟弹簧"：policy 把 wrist 偏离原 target `F_cmd / K` 米 → 部署时接触表面就自然产生 F_cmd 大小的力 | **采用**（reward 公式直接用） |
| Hindsight perturbation | CHIP | 不改 reward，改 obs 里的 goal；actor 输入 `1/k` 控制 compliance | 不采用（没有 F_cmd 直接接口） |
| Passive compliance | GentleHumanoid | sim 加真实弹簧让机器人感受外力，学"顺应" | 不采用（方向相反，是感受而非主动施力） |

### 核心 idea：虚拟弹簧 reward

训练时 reward 不是让 wrist 跟到原 motion target，而是跟到**偏移后**的 target：

```
target_shifted = motion_target + (F_ext + F_cmd) / K_virtual
                                 ^^^^^^^^^^^^^^^
                                 "想象有根虚拟弹簧；
                                  合力除以弹簧刚度 = wrist 应该偏多远"
```

- `F_cmd`（**actor 的输入**，6 维）：上层告诉 policy "左手 X 方向想要 20N"
- `F_ext`（sim 注入的真实外力，**只给 critic 看**）：训练时模拟真实世界的扰动
- `K_virtual = 100 N/m`（固定 training hyperparameter）：F↔Δx 的兑换率

**训练时没有真接触表面**——policy 学的是"看到 F_cmd 就把 wrist 偏 `F_cmd/K` 米"。

**部署时接触到真实表面**——同样的位置偏移 × 表面实际刚度 ≈ F_cmd，力就自然出来了（这是 impedance control 的经典思想）。

### 5 条硬约束（红线）

1. **力只打在左右 wrist**，其他 12 个 body 一律 0
2. **K_virtual = 100 固定**，不随机化，不进 obs
3. **不碰 Holosoma 现有 reward 项和 experiment object**，只追加
4. **不 import / copy / port 任何 third_party 代码**（GH/UniFP/CHIP 只读参考）
5. **只允许改 6 个现有文件的 DEFAULTS dict 末尾**，其他一字不改

### Observation 策略（v9.2：minimal 2-group history）

**保持 Holosoma baseline 的 2-group 架构**（`actor_obs` + `critic_obs`），只做一件事：两个 group 的 `history_length` 从 1 改成 **10**。

- **历史覆盖**：actor + critic 都看 **过去 10 步的所有 obs**（proprio, motion command, wrist force command 等全部 stack）。
- 对齐 CHIP paper §III 的 "10-step history" 核心 idea，但**不**做 CHIP 的"只 stack proprio+actions"的细粒度选择（Holosoma 的 history 是 group-level，做细粒度选择需要拆 group + PPO shim，v9.1 走过这条路不稳定）。
- **命令被 stack 10 次的代价**：dim 冗余（motion_command 58×10 = 580，wrist_force_command 6×10 = 60 等），但**信息是正确的**（真·历史 `[cmd(t-9), ..., cmd(t)]`），Codex 已确认 `manager.py:240-264` 每 step 都 recompute+append，不是 repeated frame。

### 一些诚实声明

- **K_virtual 不是测得的机器人刚度。** 它是 reward 里的 exchange rate hyperparameter。部署时若 F_real/F_cmd 有偏差，inference 端乘个常数 α 就行，不用重训
- **v9.2 加了 history 后**，actor 原则上能隐式推断 F_ext（通过 q̈ 残差），所以 v1 应该能学出一定程度的 **learned compliance**（不只是 DR）。但真正的 compliance 能力最终由训练曲线决定，不是先验保证

### 工作规模

- 1 个前置 spike（IsaacSim 外力注入 API 验证）
- 14 个实施 task（含常量文件、单测、文档、全局 sanity）
- 核心"新代码"9 个 task（`WristComplianceConfig` dataclass + command term + env 子类 + 2 个 obs term + reward term + 3 个 preset + experiment 注册）
- Actor obs 维度 154 → **1620**（单步 162 × 10 步 history，单步 = baseline 154 + wrist_force_command 6 + wrist_virtual_stiffness_command 2）；Critic 286 → **3000**（单步 300 × 10 步 history）
- 不新增任何依赖，不写新 obs term 给 history（Holosoma 的 `group_cfg.history_length` 原生支持）

---

## 🎯 Scope 硬约束（五条红线）

**红线 1：力只施加在左右 wrist，不施加在 body 其余任何地方。**
- `F_ext` 注入：只往 `left_wrist_yaw_link` 和 `right_wrist_yaw_link` 两个 rigid body 写外力 tensor，其他 12 个 tracked body **一律 0**
- `F_cmd` 指令空间：6-D（左 wrist 3D + 右 wrist 3D），没有 head / torso / root / elbow 等力指令
- reward 只看 wrist 2 body 的误差，其他 12 body 完全走 Holosoma 原 motion tracking reward

**红线 2：`K_virtual = 100.0 N/m`，固定、写死、不随机化、不进 obs。**
- **`K_virtual` 不是测得的机器人 Cartesian 刚度，而是 reward 里的 F↔Δx exchange rate（training-time hyperparameter）。**
- Holosoma 本身是 kinematic motion tracker，**没有一个现成的 "K_arm" 数字** —— 只有 joint-level PD（`config_values/robot.py:498-504`）：shoulder × 3/elbow/wrist_roll `Kp=14.25`, wrist_pitch/yaw `Kp=16.78`，平均 15 Nm/rad
- 真实末端 Cartesian 刚度 `K_cart(q) = J(q)^{-T} · diag(K_q) · J(q)^{-1}` 是构型函数，非常数
- v9 直接选定固定值 **100.0 N/m**，详见下方 "K_virtual 的固定值选取" 节
- Task 0 只负责把这个常量写进 `_k_virtual.py`
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

---

## 🧮 Reward 公式（v9.4：per-env K_virtual）

```
force_w_left  = yaw_quat(base_quat) · force_cmd_b[:, 0]        # body-yaw → world
force_w_right = yaw_quat(base_quat) · force_cmd_b[:, 1]
F_total_w_left  = force_ext_w_left  + force_w_left              # UniFP 双力叠加
F_total_w_right = force_ext_w_right + force_w_right
K_left  = k_virtual[:, 0:1]                                     # ← per-env, from command manager (v1=100.0, v2=range-sampled)
K_right = k_virtual[:, 1:2]
target_shifted_w[left]  = motion_target_w[left]  + F_total_w_left  / K_left
target_shifted_w[right] = motion_target_w[right] + F_total_w_right / K_right
error = Σ_xyz (target_shifted_w − wrist_actual_w)²
reward = exp(−mean_over_2_wrists(error) / σ²)     # Holosoma-style exp-squared kernel
```

**Kernel 选型说明：** target shift 公式沿用 UniFP，但 exp kernel 用 Holosoma 风格（squared error + σ²），不是 UniFP 的 L1 + `exp(-err/σ * 2)`。原因：(a) 与 Holosoma 现有 `motion_relative_body_position_error_exp`（`managers/reward/terms/wbt.py:87`）kernel 一致，便于在同一 reward preset 里与其他 tracking term 加权；(b) squared error 对大偏差惩罚更重，符合"希望 wrist 误差尽量小"的意图。

**注意**：`F_ext` 来自 sim ground truth（`WristComplianceCommand.force_ext_w`），reward 端**不经过 actor obs**，所以不会破坏"actor 不看 F_ext"的信息边界。

### 四个稳态情形（K_virtual=100 N/m）

| 场景 | F_cmd | F_ext | g̃ 相对 g 的偏移 | policy 学到 |
|---|---|---|---|---|
| A（自由空间，无指令）| 0 | 0 | 0 | 纯 motion tracking |
| B（空中"发力"）| 20N +X | 0 | +0.20 m · x̂ | 把 wrist 偏 Δx，接触墙时产生力 |
| C（被外力扰动）| 0 | +20N +X | +0.20 m · x̂ | (v1 是 DR，不是 learned compliance) |
| D（稳态施力）| 20N +X | -20N（墙反推） | 0 | wrist 保持原位，靠 PD 发力 |

### Δx 工作空间 bound

| 合力 `||F_ext + F_cmd||` | Δx | 相对 G1 arm reach (~0.7m) |
|---|---|---|
| 30 N（单路 max） | 0.30 m | 舒适区 |
| 45 N（F_ext=15 + F_cmd=30 同向） | 0.45 m | 舒适区 |
| **60 N（max 叠加，F_ext+F_cmd 都 30 同向）** | **0.60 m** | **arm reach 边缘（kinematic 软上限）** |

> Policy 在 Δx > 0.6m 的采样下无法完全达成 target_shifted —— 这是合理的"上限软截断"，与"wrist 物理可达 ~0.7m"匹配。若训练曲线显示 >20% step 处于 saturation，把 `force_ext_magnitude_range` 上限从 30 降到 15-20 N。

---

## 💥 F_ext 注入规格（只在 wrist）

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

### Physics step vs Control step 频率澄清 + 一致性分析

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
- `_compute_observations` 在 step 之后，actor/critic obs 看到的 `force_cmd_b` / `force_ext_w` 是 `F_{N+1}`，即 **"下一 step 将要施加的力"**——这是 Holosoma 所有 command term 的标准 convention（和 `motion_command` 一样）

**所以**：
- **Command term 梯形状态机在 50 Hz 推进**（RAMP_UP/HOLD/RAMP_DOWN 用 control-step count）
- **F_ext 写入 sim 在每个 physics substep 都执行**（env 子类的 override），每个 control step 内 4 个 substep 写入同一个值（ZOH，与 action ZOH 对齐）
- **Reward 与 physics 同步读取同一个 cached force**（不存在 off-by-one）
- **Observation 看到"下一步将施加的力"**（标准 command convention；与 `motion_command` 时序一致）

**实现要求**：`WristComplianceCommand` 的 `.step()` 必须只更新 state machine + 采样，不能在 `_apply_force_in_physics_step` 或 reward 调用之间修改 `force_ext_w` / `force_cmd_b` buffer。单测 Task 2 加一条：连续调两次 `.force_ext_w`（中间不调 `.step()`）返回同一张量。

**Ramp 实现策略**：在 `WristComplianceCommand` 里用简单 `int8` state + 整数 counter（counter 单位是 control step），自己写 ~50 行，不 port 任何 third-party code。

**F_cmd 剖面**：同样机制，独立实例。F_cmd 也跑梯形剖面是为了让 policy 见过 "指令从 0 升到 X" 的过渡（而不是阶跃），对应真实部署时上层 target 刚移动到表面接触瞬间那段。

**实现**：直接新建 `WristComplianceCommand`，同时维护 `force_cmd_b: [N, 2, 3]` + `force_ext_w: [N, 2, 3]`；obs term 用 `.reshape(N, 6)` 展平。

---

## 🔧 K_virtual 的设计（v9.4：per-env buffer + actor obs + deterministic v1）

**v9.4 决定：K_virtual 变成 per-env、per-wrist 的 buffer，由 `WristComplianceCommand` 管理采样；范围从 `[K_virtual_range[0], K_virtual_range[1]]` 采样；v1 把 range 写成 `[100.0, 100.0]` → 退化 deterministic 100。K 同时作为 actor obs 和 critic obs（CHIP 派系）。**

**重要澄清：`K_virtual` 不是测得或估算得到的 G1 Cartesian 刚度。它是 reward 里 "F↔Δx exchange rate" 这个 training-time hyperparameter**：

- Policy 看到的世界是："F_cmd 越大 → target 偏离原 motion 越远 → wrist 要跟着走"
- "越多 Newton 等于多少米"这个比例关系由训练时的 `K_virtual` 定义，**policy 学到的是"每单位 F_cmd 把 wrist 偏 `1/K_virtual` 米"的位移映射**
- 部署时 F_cmd → 实际接触力的关系由 **policy 学到的位移映射 × 真实接触表面刚度 `K_surface`** 决定，稳态下 F_real ≈ F_cmd · (K_surface / K_virtual)
- 所以：`K_virtual` 确实影响部署行为（它定义了 policy 内化的 scaling），但**误差可以被 inference 端乘常数 α = K_virtual / K_surface 补偿**，不用重训（这是 UniFP 的 calibration convention）

### v9.4 设计：per-env + actor obs + range sampling

**Why per-env + actor obs**：
- CHIP paper §III：`g_hind_ptb = g − (1/k)·f`，`1/k` 直接进 actor 的 goal 观测
- 借 GH 的 `kp_range=[5, 250]` 随机化思路，policy 学到"看到 K=50 就把 wrist 偏 F/50 米，看到 K=200 就偏 F/200 米"
- v1 range 写 `[100, 100]` → 每个 env 采到的 K 都是 100，actor obs 是常量（不起训练作用但占据 obs slot），reward 读 per-env K 的逻辑已跑通
- v2 只需把 range 改成 `[50, 300]` 就能立刻切成随机化训练，**obs / reward / onnx export / inference 代码都不用动**

**Why K 进 critic 而不是 GH 的"critic 也不看"**：
- CHIP: 1/k 进 actor，critic 一般 ⊇ actor，所以 critic 也看 1/k
- GH: kp 不进 command 也不进 force_priv（`motion_tracking.py:1094-1101`），policy 从 `force_applied_b` 间接感知。**GH 这样做能工作是因为它的 `force_applied_b` 本身包含 K × Δx 的结果**，但我们的 reward 是 exp(-||Δ_target - Δ_actual||²)，没有一个"applied_force"桥梁
- v9.4 选 CHIP 派系：actor + critic 都看 K，policy 能直接拿到 F↔Δx 兑换率（代价：2 dim × 10 history = 20 dim 冗余，v1 常量时更是零信息，但简单）

### Per-wrist 独立 K 的意义

`wrist_virtual_stiffness_command` shape `[N, 2]`（左 wrist K_left + 右 wrist K_right）：
- v1：两维都是 100.0（deterministic），但 API 预留双手独立
- v2：若随机化 `K_range=[50, 300]`，左右 wrist 独立采样 → policy 学"哪只手硬哪只手软"
- reward 公式也 per-wrist：`target_shifted[i] = motion_target[i] + F_total[i] / K_virtual[i]`（i ∈ {left, right}）

### 怎么选 `K_virtual = 100`：从 Δx 工作空间反推

**设计目标**：采样分布内 `||F_ext + F_cmd||` 上限除以 `K_virtual` 得到的 Δx 应落在 wrist "可达但不轻松" 的区间——让 policy 既有足够的"力度区分"训练信号，又不会被 kinematic 极限切掉太多 sample。

- G1 arm reach ≈ 0.7 m（physics plausible；参考 G1 URDF）
- 采样范围：`force_ext_magnitude_range = [0, 30] N`（红线 2），`force_cmd_magnitude_range = [5, 30] N`
- 同向最大叠加：`||F_ext + F_cmd|| ≤ 60 N`

| K_virtual | Δx @ 30N（单路 max） | Δx @ 60N（同向叠加 max） | 问题 |
|---|---|---|---|
| 50 N/m | 0.6 m | **1.2 m** | 叠加 Δx 远超 wrist reach |
| **100 N/m ★** | **0.3 m** | **0.6 m** | **叠加 Δx 刚到 wrist reach 边缘** |
| 200 N/m（UniFP 默认） | 0.15 m | 0.3 m | Δx 太小，policy 信号弱 |

**v9 选 `100 N/m`**：保证同向叠加的 Δx 刚好到 arm reach 边缘（合理的 "soft upper bound"），又不过小而让训练信号萎缩。

### 辅助参考（不是主要判据，仅供 sanity）

Holosoma arm joint-level PD（`config_values/robot.py:498-504`）：
- shoulder × 3 / elbow / wrist_roll: Kp = 14.25 Nm/rad
- wrist_pitch/yaw: Kp = 16.78 Nm/rad；**平均 ~15 Nm/rad**

关节空间 Kp（Nm/rad）不能直接当 Cartesian K（N/m）——真实 `K_cart(q) = J(q)^{-T} · diag(K_q) · J(q)^{-1}` 是构型 q 的函数。做极简 K/L² 估算（L=0.4-0.5m, 4-joint arm）得 15-94 N/m 的数量级；100 N/m 跟这个 range 同量级，不会完全脱离物理直觉。

GH `kp_range = [5, 250]` N/m 是真实弹簧刚度的随机化区间，100 N/m 落在 mid-high 段（几何中值 35 偏硬）——我们固定不随机化，选 mid-high 一点让 Δx 更保守。

### 代码落地（v9.4）

`src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py`（手写）：

```python
"""Virtual force-to-displacement exchange rate for G1 wrist force training.

This is a REWARD HYPERPARAMETER, not a measured/calibrated robot stiffness.
It defines how many Newton of force command corresponds to how many meters
of wrist target displacement inside the virtual impedance reward
(target_shifted = motion_target + F_total / K_virtual).

In v9.4, K_virtual is sampled per-env + per-wrist from K_VIRTUAL_RANGE_N_PER_M
by WristComplianceCommand. v1 ships with range = (100.0, 100.0) — i.e. a
deterministic 100 N/m for every env, for every wrist. v2 can flip to a real
range like (50.0, 300.0) to get GH-style stiffness randomization WITHOUT
touching obs/reward/exporter wiring — only this constant changes.

Value chosen so that the maximum combined force draw
(||F_ext + F_cmd|| = 60 N under same-direction sampling) produces a shifted
target at ~0.6 m from the motion target, which is at the edge of G1 arm
reach (~0.7 m) — a natural soft upper bound for training at K=100.

See docs/plans/2026-05-03-wbt-wrist-force-v9.md §K_virtual for full rationale.
If training reveals systematic F_real / F_cmd mismatch < 2x, apply an
inference-time scalar correction rather than retraining.
"""

# v1: deterministic 100 N/m (range collapses to single value).
# v2: widen to e.g. (50.0, 300.0) for GH-style randomization.
K_VIRTUAL_RANGE_N_PER_M: tuple[float, float] = (100.0, 100.0)

# Kept for backwards reference / helper code; equals range[0] when v1.
G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M: float = K_VIRTUAL_RANGE_N_PER_M[0]
```

**关键实现点**：

1. `WristComplianceConfig` dataclass 新增字段：`k_virtual_range: tuple[float, float] = (100.0, 100.0)`（默认从 `_k_virtual.py` 常量读，task preset 构造时填）
2. `WristComplianceCommand.__init__` 分配 `self.k_virtual: torch.Tensor[N, 2]`（左右 wrist 各一维）
3. Reset 时 per-env 从 `k_virtual_range` 均匀采样（左右 wrist **独立采样**，允许双手不同 K；v1 range 是 [100,100] 所以都是 100）
4. Reward term 读 `env.command_manager.wrist_compliance_command.k_virtual` 而不是常量；公式 `target_shifted[i] = motion_target[i] + F_total[i] / k_virtual[:, i:i+1]`
5. Obs term `wrist_virtual_stiffness_command(env) -> [N, 2]`：直接返回 `command.k_virtual.clone()`（v1 常量 100；obs scale 用 `1/100`，让数值在 [0.5, 3.0] 范围，和其他 obs 量级对齐——`1/K` 即 `compliance`，也贴合 CHIP 的 `1/k` 语义）

**v1 → v2 切换成本（零代码改动）**：改 `K_VIRTUAL_RANGE_N_PER_M = (50.0, 300.0)` 一行，**重新训练**即可（obs/reward/exporter 代码不动、inference 端不动）。

**敏感性分析**：训练后若 F_real / F_cmd 偏 0.5-2x → inference 端乘 α 补偿，不重训。若偏 > 5x → 再考虑调此常量重训（且训练曲线若显示 wrist 长期处于 kinematic saturation，应先把 `force_ext_magnitude_range` 上限从 30 降到 15-20 N，而不是动 K）。

---

## 📁 文件结构

### 新建文件

| 路径 | 类型 | 职责 |
|---|---|---|
| `src/holosoma/holosoma/managers/command/terms/wbt_force.py` | **新** | `WristComplianceCommand(CommandTermBase)`：**自己写简单的 per-env 梯形 state machine**（ramp_up → hold → ramp_down → cooldown），方向用 `torch.randn + normalize` 采样，不依赖任何 GH 代码。暴露 `force_cmd_b: [N, 2, 3]`, `force_ext_w: [N, 2, 3]` |
| `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py` | **新** | `WholeBodyTrackingForceInjected(WholeBodyTrackingManager)`：override `_apply_force_in_physics_step`，调 `self.simulator._robot.set_external_force_and_torque(...)`（**world → body frame 转换必需**）；同时暴露 `last_applied_force_w_by_body_id: dict[int, torch.Tensor]` debug accessor 供 e2e 测试断言使用（避免测试依赖 IsaacLab 私有 `_external_force_b` buffer） |
| `src/holosoma/holosoma/managers/observation/terms/wbt_force.py` | **新** | (a) `wrist_force_command(env)` → `force_cmd_b.reshape(N, 6)` 给 actor+critic；(b) `wrist_force_ext_privileged(env)` → `force_ext_w.reshape(N, 6)` 仅 critic；(c) **`wrist_virtual_stiffness_command(env)` → `k_virtual.reshape(N, 2)` 给 actor+critic**（v9.4 新增，v1 常量 100.0，obs scale 用 1/K 的形式见下方 Task 7）。v9.2 **不**包含 `_shim_zero_dim`（4-group shim 已删除） |
| `src/holosoma/holosoma/managers/reward/terms/wbt_force.py` | **新** | `wrist_force_position_tracking_exp(env, sigma, K_virtual, left_wrist_body_name, right_wrist_body_name)` 按 Reward 公式节双力公式 |
| `src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py` | **新** | 手写 `K_VIRTUAL_RANGE_N_PER_M: tuple[float, float] = (100.0, 100.0)` + 向后兼容 alias `G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M = range[0]` + docstring 说明这是 training hyperparameter，v1 range 退化 deterministic 100、v2 可直接改 range 获 GH-style 随机化 |
| `src/holosoma/holosoma/config_values/wbt/g1/command_force.py` | **新** | `g1_29dof_wbt_force_command` preset |
| `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py` | **新** | `g1_29dof_wbt_force_observation` preset：baseline 2-group 架构沿用，两个 group 都设 **`history_length=10`**；actor group 追加 `wrist_force_command` + **`wrist_virtual_stiffness_command`**；critic group 追加 `wrist_force_command` + **`wrist_virtual_stiffness_command`** + `wrist_force_ext_privileged` |
| `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py` | **新** | `g1_29dof_wbt_force_reward` preset：原 reward 全保留，追加 `wrist_force_position_tracking_exp`（weight=2.0，K_virtual 从 `_k_virtual.py` import） |

### 修改现有文件（仅限末尾追加）

| 路径 | 修改方式 |
|---|---|
| `src/holosoma/holosoma/config_types/command.py` | 末尾追加 `WristComplianceConfig` dataclass（F_cmd 采样 + F_ext 采样 + ramp 参数） |
| `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` | 末尾追加 `g1_29dof_wbt_force` experiment，`env_class="holosoma.envs.wbt.wbt_force_injected:WholeBodyTrackingForceInjected"` |
| `src/holosoma/holosoma/config_values/command.py` | DEFAULTS 追加 `"g1_29dof_wbt_force"` |
| `src/holosoma/holosoma/config_values/observation.py` | DEFAULTS 追加 |
| `src/holosoma/holosoma/config_values/reward.py` | DEFAULTS 追加 |
| `src/holosoma/holosoma/config_values/experiment.py` | DEFAULTS 追加 |
| 所有对应的 `tests/test_*.py` | **新增** TDD 测试文件 |

### 必须保持原样（一行不改）

- `src/holosoma/holosoma/envs/wbt/wbt_manager.py`（基类；`wbt_force_injected.py` 继承它不改源）
- `src/holosoma/holosoma/managers/command/terms/wbt.py`（MotionCommand）
- `src/holosoma/holosoma/managers/observation/terms/wbt.py`
- `src/holosoma/holosoma/managers/reward/terms/wbt.py`
- `src/holosoma/holosoma/managers/observation/manager.py`（history 已内置，无需改）
- Holosoma 现有 `push.py` / `virtual_gantry.py`（我们只是参考 API 用法，不改它们）
- 所有 `g1_29dof_wbt*` 现有 preset

### 关于 third-party 代码复用

**v9 决定：不从 GH / UniFP / CHIP port 任何代码。**

- **ramp state machine**：直接在 `WristComplianceCommand` 里用简单的 `int8` 状态 + 整数计数器实现（`cooldown_remaining`, `ramp_up_remaining`, `hold_remaining`, `ramp_down_remaining`），<50 行
- **方向均匀采样**：`d = torch.randn(n, 3); d /= d.norm(dim=-1, keepdim=True)` 一行搞定
- **幅度 clamp**：若需要的话直接 `torch.clamp` 或 `F.normalize(..., dim=-1) * magnitude`

**只从 GH 读源代码（"参考 API 用法"），不 copy 代码**：
- 读 `motion_tracking.py` 看它怎么调 `physx.apply_forces_and_torques_at_position`（已确认 Holosoma 有等价 wrapper `set_external_force_and_torque`）
- 读 `G1_gentle.yaml` 看它的 force perturbation 相关参数配置量级（比如 `max_force`、`force_alpha` 等）
- 这些都是"读一次、做笔记、然后自己重写"，**Holosoma 最终代码里一行 GH 代码都不出现**

**禁令**：
- ❌ `from gentle_humanoid_training ... import ...`
- ❌ `sys.path.append('third_party/gentle-humanoid-training/...')`
- ❌ 在 pyproject.toml 加 GH 依赖
- ❌ 把 `third_party/gentle-humanoid-training/` 加到 PYTHONPATH
- ❌ copy GH 代码片段

**同样原则适用于 UniFP**：只读参考，不 import 不 copy（UniFP 是 IsaacGym quadruped 代码，不适用 IsaacSim humanoid）。

---

## 📊 Observation 结构（v9.2：minimal 2-group history）

### 为什么加 history

三个对标系统的 actor 都用 history（explainer §6.4 有原理推导）：

| 系统 | actor history 配置 | 来源 |
|---|---|---|
| **UniFP** (quadruped+arm) | `frame_stack=32` 全 obs stacked（num_observations = 32 × 73 = 2336） | `third_party/UniFP/.../b2z1_pos_force_config.py:107-111` |
| **CHIP** (humanoid G1) | proprio `s_{t-10:t}` + past actions `a_{t-11:t-1}` | paper §III, line 163-166 |
| **GentleHumanoid** (G1) | actor `joint_pos_history: [0,1,2,3,4,8]`（稀疏 6 步）+ critic `[0..8]`（稠密 9 步） | `cfg/task/G1/G1_gentle.yaml:78-98` |
| **Holosoma baseline WBT** | 无 (`history_length=1`) | baseline，只做 motion tracking 不需要估 F_ext |

**原理**（explainer §6.4）：actor 不直接看 F_ext，但 stacked history `[o_{t-K}, ..., o_t]` 里隐含 `q̈ ≈ (q̇_t − q̇_{t-1})/dt`，policy MLP 理论上能学到隐式 F_ext 估计器。不给 history → actor 无法区分"我自己的加速度"和"外力造成的加速度" → 对 F_ext 梯度方向错，训练不稳。

v9.2 借用 CHIP 的 "10 步" 量级：**两个 group 都设 `history_length=10`**。

### Baseline obs 维度（实测）

| Term | dim | 在 actor 里？ | 在 critic 里？ | 来源 |
|---|---|:---:|:---:|---|
| `motion_command` (`joint_pos + joint_vel`) | **58** | ✅ | ✅ | `wbt.py:827` = `cat([joint_pos_29, joint_vel_29])` |
| `motion_ref_pos_b` | 3 | ✗ | ✅ | privileged（wrist target 世界位置给 critic 降方差） |
| `motion_ref_ori_b` (rot6d) | 6 | ✅ | ✅ | |
| `robot_body_pos_b` | 42 (14×3) | ✗ | ✅ | privileged |
| `robot_body_ori_b` (rot6d) | 84 (14×6) | ✗ | ✅ | privileged |
| `base_lin_vel` | 3 | ✗ | ✅ | 部署时 humanoid base velocity 不可靠 → 只给 critic |
| `base_ang_vel` | 3 | ✅ | ✅ | IMU 可得 |
| `dof_pos` | 29 | ✅ | ✅ | |
| `dof_vel` | 29 | ✅ | ✅ | |
| `actions` | 29 | ✅ | ✅ | 当前 step 的 action（stack 10 步后 = `a_{t-9:t}`）|

**Baseline 加总验证（`history_length=1`）：**
- Actor baseline = 58 + 6 + 3 + 29 + 29 + 29 = **154 dim**
- Critic baseline = 58 + 3 + 6 + 42 + 84 + 3 + 3 + 29 + 29 + 29 = **286 dim**

### v9.4 observation 结构：沿用 baseline 2-group，加 history + 2 个 force term + 1 个 K_virtual term

**核心原则：不破坏 Holosoma 原生 observation 架构。**

完全沿用 baseline 的 2-group 架构 (`actor_obs` + `critic_obs`)，只做三处修改：

1. **两个 group 的 `history_length` 从 1 改成 10**（history-length CLI flag 可以直接 override，不需要 preset 改动）
2. **追加 `wrist_force_command`** 进 actor 和 critic，**`wrist_force_ext_privileged`** 只进 critic
3. **v9.4 新增 `wrist_virtual_stiffness_command`** 进 actor 和 critic（CHIP 派系，让 policy 显式拿到 F↔Δx 兑换率）

| Group | baseline `history_length` | v9.4 `history_length` | baseline 成员 term | v9.4 追加 term |
|---|:---:|:---:|---|---|
| `actor_obs` | 1 | **10** | `motion_command` (58) + `motion_ref_ori_b` (6) + `base_ang_vel` (3) + `dof_pos` (29) + `dof_vel` (29) + `actions` (29) = **154**/step | + `wrist_force_command` (6) + **`wrist_virtual_stiffness_command` (2)** |
| `critic_obs` | 1 | **10** | actor 所有 + `motion_ref_pos_b` (3) + `robot_body_pos_b` (42) + `robot_body_ori_b` (84) + `base_lin_vel` (3) = **286**/step | + `wrist_force_command` (6) + **`wrist_virtual_stiffness_command` (2)** + `wrist_force_ext_privileged` (6) |

**PPO `input_dim` 保持 baseline**（不 override）：
- `actor.input_dim = ["actor_obs"]`
- `critic.input_dim = ["critic_obs"]`

没有 4-group 拆分，没有 `critic_obs` shim，没有 `_shim_zero_dim` dummy term，没有 input_dim override，没有下游 concat 契约。

### 维度加总（v9.4）

| Group | 单步 dim | stacked dim (history=10) |
|---|---|---|
| `actor_obs` | 154 + 6 + 2 = **162** | **1620** |
| `critic_obs` | 286 + 6 + 2 + 6 = **300** | **3000** |

- **Actor obs total: 1620 dim** (baseline 154 × 10.5）
- **Critic obs total: 3000 dim** (baseline 286 × 10.5）

相比 UniFP 的 2336 维 actor 量级接近（1620 vs 2336）；比 CHIP dense humanoid 的 1000-1500 维 actor 略大（CHIP selective history 不 stack 命令；我们 stack 了命令导致额外 ~600 维）。

**v9.3 → v9.4 diff**：actor +20 dim（K_virtual 2 dim × history 10），critic +20 dim。v1 deterministic 100 时这 20 dim 是常量（无训练信号但不伤害），v2 range sampling 时立刻变成有用信号。

### Group-level history 的代价（为什么是"可接受的冗余"）

v9.2 两个 group 都设 `history_length=10` 意味着：
- `motion_command` 58×10 = 580 维（若只 actor，还有 critic 也一份）
- `motion_ref_ori_b` 6×10 = 60 维
- `base_ang_vel` 3×10 = 30 维
- `wrist_force_command` 6×10 = 60 维
- ...等等

**这些命令的 stacked 版本是真·历史信号**，不是 repeated frame：
- `manager.py:240-264` (`_apply_history`) 每 step 都重新 `compute_term(...)` 得到新值，append 进 deque，再 `torch.stack`
- `motion_command(t)` 每 step 由 motion_clip 按时间推进产生新值 → stacked `[cmd(t-9), ..., cmd(t)]` 是"过去 10 帧的轨迹窗口"
- `wrist_force_command` 跑梯形剖面 → stacked `[F_cmd(t-9), ..., F_cmd(t)]` 把 "ramp 上升中 / 在 hold / 开始下降" 的 profile 都放进 obs
- `base_ang_vel` 每 step 变 → stacked 就是真 angular velocity history

**代价**：维度比 CHIP selective history（只 stack proprio+actions）大。Actor 从"CHIP 的 87/step（proprio+actions）" → "v9.4 的 162/step（all baseline + F_cmd + K_virtual）"，多 ~86%。但 MLP 首层 1620→1024 也就 ~1.7M 参数，远非瓶颈；批量训练 4096 envs × 1620 ≈ 6.6M 元素的 batch 在 GPU 上仍轻松。

**相对 v9.1 的 4-group 方案，v9.2 的优势**：
- PPO 代码完全不动（无需 input_dim override、无需 shim、无需 dummy term）
- 下游 `get_inference_policy()` 部署代码**不需要 concat contract**（外部调用者只管 `{"actor_obs": obs_dict["actor_obs"]}` 即可）
- Holosoma 现有的 `history_length` CLI flag（`--observation.groups.actor_obs.history-length 10`）天然可用，用户可以不通过改 preset 就跑变种
- Baseline `g1_29dof_wbt` experiment 完全不受影响（它的 `history_length=1` 不变）

**相对 v9.1 的 4-group 方案，v9.2 的代价**：
- 命令被 stack 10x → actor obs dim 比 CHIP-selective 多 ~600 维
- 从 paradigm 上说，v9.2 不是严格对齐 CHIP 的"proprio+actions only history"，而是更粗粒度的"everything 10-step history"（接近 UniFP 的 `frame_stack=32` 思路，只是 k 更小）

综合 trade-off：**v9.2 选工程稳定性 > paradigm 严格对齐**（用户原话：`motion command 和 force command 重复十次也没什么，尽量就用 holosoma 的原始 code`）。

### 为什么不给 actor F_ext

- F_ext 是 **sim-only ground truth**，部署时 humanoid wrist 无 F/T 传感器 → actor 看不到（CHIP paper §VI 明确讨论）
- critic 可以开挂看 F_ext ground truth（降 value 方差，加速训练收敛）
- F_cmd 两边都给（部署时 actor 真拿得到）

### 隐式 F_ext 估计器的原理（why history works）

actor 通过 `actor_obs` 看到**过去 10 步**所有 term，其中包含 `dof_pos` 和 `dof_vel` 的时序：
- `dof_vel_{t-9:t}` 可反推 q̈ ≈ Δq̇ / dt
- `actions_{t-9:t}` 是过去 10 步自己下过的 τ_cmd
- 由牛顿第二定律 + Lagrange: `q̈ = M⁻¹(τ_cmd - C q̇ - g + J^T F_ext)`
- 所以 `F_ext ≈ (J^T)^+ (M q̈ + C q̇ + g - τ_cmd)`——只要 MLP 容量够，能从 history 里学这个映射
- 没有 history 就没办法算出 q̈，F_ext 完全不可观 → **learned compliance 数学上无法成立**

（命令类 term 被 stack 对这个推断没有帮助但也不伤害——它们只是 "extra context"，MLP 可以学会忽略稳态部分。）

---

## 🔬 前置 Spike（开始 task 前必做）

### Spike 1: IsaacSim wrist-only 外力注入 API 集成验证

目的：确认 `set_external_force_and_torque(..., body_ids=wrist_only)` 的行为符合预期——**只给 wrist 两个 body 传 force tensor，其他 body 的 external force buffer 不被污染**。

```python
# tests/spikes/test_wrist_only_force_injection.py
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
#       行为。生产 e2e 测试 Task 11 禁止读这个 buffer——改用 env 子类的
#       last_applied_force_w_by_body_id debug accessor。）
#    - 传入的 forces tensor 维度跟 body_ids 长度一致（避免 broadcast 错 body）
# 5. 断言（第二层：1 step 后物理合理性）：
#    - 左 wrist world x 速度有变化（力确实被施加）
#    - 非 wrist body（头、胯、右小臂等）没有被施加外力的痕迹
#      （不是"位移 < 1cm" 的硬阈值——articulated body 里关节耦合会 false-fail——
#      而是通过 external force buffer 白名单验证）
# 6. 反向测试：传 body_ids=[wrong_body_id]（比如 pelvis）→ 断言 wrist 的 buffer 保持 0
#    （证明 body_ids 参数真的在做 selection，不是全局广播）
# 7. reset 后断言 external force buffer 全部清零
```

Gate：Spike 1 绿 → 进入 Task 0（写常量文件）→ Task 1 起全面开工。

---

## 📋 任务分解（22 task + 1 spike，分 7 phase；Phase 0-6 训练侧 14 task，Phase 7 部署侧 8 task）

格式：TDD 6 步骤（写失败测试 → 确认失败 → 实现 → 通过 → 回归 → commit）。

### Phase 0（前置验证 & 常量）

- [ ] **Spike 1** — 外力注入 wrist-only 集成验证（见上方 "前置 Spike" 节）

- [ ] **Task 0：创建 `_k_virtual.py` 常量文件**（v9.4：range tuple 而非单值）
  - 手写 `src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py`
  - 内容：
    - `K_VIRTUAL_RANGE_N_PER_M: tuple[float, float] = (100.0, 100.0)`（v1 deterministic，v2 可改 (50.0, 300.0)）
    - `G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M: float = K_VIRTUAL_RANGE_N_PER_M[0]`（向后兼容 alias，便于老代码/脚本引用单值）
  - 文件头 docstring 明确"this is a reward hyperparameter, not a measured stiffness；v1 range=(100,100) collapses to deterministic 100；v2 can widen range without touching obs/reward/exporter"
  - 测试：`src/holosoma/holosoma/config_values/wbt/g1/tests/test_k_virtual.py`
    - import 成功
    - `K_VIRTUAL_RANGE_N_PER_M == (100.0, 100.0)`
    - `isinstance(K_VIRTUAL_RANGE_N_PER_M, tuple)` + 两元素都 `float`
    - `K_VIRTUAL_RANGE_N_PER_M[0] <= K_VIRTUAL_RANGE_N_PER_M[1]`（range 合法性）
    - `G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M == 100.0`（alias）
  - 跑：`pytest src/holosoma/holosoma/config_values/wbt/g1/tests/test_k_virtual.py`
  - Commit

### Phase 1（command 基础设施）

- [ ] **Task 1：`WristComplianceConfig` dataclass**（v9.4：追加 `k_virtual_range`）
  - `src/holosoma/holosoma/config_types/command.py` 末尾追加
  - 字段：
    - F_cmd: `force_cmd_magnitude_range=[5, 30]`, `force_cmd_duration_range_s=[1.0, 3.0]`, `force_cmd_cooldown_range_s=[0.5, 2.0]`, `force_cmd_ramp_frac=0.25`, `force_cmd_activation_prob_per_step=0.01`
    - F_ext: `force_ext_magnitude_range=[0, 30]`, `force_ext_duration_range_s=[1.0, 3.0]`, `force_ext_cooldown_range_s=[0.5, 2.0]`, `force_ext_ramp_frac=0.25`, `force_ext_activation_prob_per_step=0.01`
    - **v9.4 新增**：`k_virtual_range: tuple[float, float] = K_VIRTUAL_RANGE_N_PER_M`（默认从 `_k_virtual.py` 读，preset 构造时覆写可选）
    - 共用：`enable_left=True`, `enable_right=True`, `left_wrist_body_name="left_wrist_yaw_link"`, `right_wrist_body_name="right_wrist_yaw_link"`
  - `@dataclass(frozen=True)`
  - 测试：`src/holosoma/holosoma/config_types/tests/test_wrist_compliance_config.py`
    - 默认值构造成功、所有字段类型正确
    - `frozen=True` 验证：改字段抛 `dataclasses.FrozenInstanceError`
    - `magnitude_range` 下限 ≤ 上限、`cooldown_range_s` 非负、`ramp_frac ∈ [0, 0.5]`
    - **v9.4 新增**：`k_virtual_range[0] > 0`、`k_virtual_range[0] <= k_virtual_range[1]`、默认等于 `(100.0, 100.0)`；构造时传 `(50.0, 300.0)` 成功（验证 v2 路径 API 可用）
    - **v9.4 新增**：`__post_init__` 做**显式 assertion**：`assert k_virtual_range[0] > 0, "K_virtual must be positive"` + `assert k_virtual_range[0] <= k_virtual_range[1], "range lo must ≤ hi"`——不依赖 reward term 的 `clamp(min=1e-3)` 兜底（clamp 是 defense-in-depth，主防线是 dataclass 验证）；传 `(0.0, 100.0)` 或 `(200.0, 50.0)` 断言 raise

- [ ] **Task 2：`WristComplianceCommand` command term**（自己写最简 state machine，不 port GH；v9.4 追加 `k_virtual` buffer）
  - `src/holosoma/holosoma/managers/command/terms/wbt_force.py`
  - per-env + per-wrist 状态机（state ∈ {COOLDOWN, RAMP_UP, HOLD, RAMP_DOWN}），用整数 counter 推进 —— 不引入 `TemporalLerp` 或其他通用 lerp class
  - F_cmd 和 F_ext 两路用**相同的**状态机实现，各自独立参数
  - 方向采样：`d = torch.randn(n_active, 3); d /= d.norm(dim=-1, keepdim=True).clamp(min=1e-6)`
  - 每 step 调 `step()`：
    1. 对每 wrist 判断是否需要触发新 episode（`state == COOLDOWN` + `cooldown_remaining == 0` + Bernoulli(`activation_prob_per_step`)）
    2. 新 episode：采方向 + 幅度 + duration + ramp_up_steps + ramp_down_steps；初始化 `state = RAMP_UP`
    3. 状态推进：RAMP_UP count → HOLD count → RAMP_DOWN count → COOLDOWN（cooldown_steps）→ 可触发下一轮
    4. 输出 force = peak_dir × (当前 ramp 插值系数 ∈ [0, 1])
  - **v9.4 新增 `k_virtual` buffer**：
    - `__init__`: `self.k_virtual = torch.zeros(num_envs, 2, device=device)`（左右 wrist 各 1 维）
    - `reset(env_ids)`: per-env + per-wrist 独立从 `cfg.k_virtual_range` 均匀采样 `torch.rand(len(env_ids), 2) * (hi - lo) + lo`；v1 range=(100,100) 时采样结果恒为 100
    - **不在 `step()` 里重采**：K 在整个 episode 恒定（对标 GH 的 `kp_range` resample per-episode 行为，`motion_tracking.py:781-782`），避免 K 中途跳变破坏训练信号
  - 暴露：`force_cmd_b: [N, 2, 3]`, `force_ext_w: [N, 2, 3]`, **`k_virtual: [N, 2]`**
  - 单元测试（pure CPU）：
    - reset 后 force 全 0，**`k_virtual` per-env 在 `[range[0], range[1]]` 内**
    - F_cmd 和 F_ext 互相独立（一个激活不影响另一个）
    - 梯形剖面形状正确（ramp up → hold → ramp down，峰值等于采样值）
    - `force_ext_w` norm 在 `force_ext_magnitude_range` 内
    - `enable_left=False` 时左 wrist 永远 0
    - **时序一致性测试**：连续调两次 `.force_ext_w`（中间不调 `.step()`）返回同一张量
    - **v9.4 新增 K 测试**：
      - v1 range=(100,100) → `torch.all(k_virtual == 100.0)`
      - mock range=(50,300) → reset 后每个 env 左右独立分布 ∈ [50, 300]、左右值**允许不同**（断言至少 1 个 env 满足 `abs(k_virtual[:, 0] - k_virtual[:, 1]) > 10`）
      - episode 中途（调 N 次 `.step()`）`k_virtual` **不变**；reset 后重新采样

- [ ] **Task 3：`WholeBodyTrackingForceInjected` env 子类**
  - `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py`
  - 继承 `WholeBodyTrackingManager`
  - **Imports（关键）**：
    ```python
    from isaaclab.utils.math import quat_apply_inverse
    # NOTE: IsaacLab 的 body_quat_w 是 wxyz；不要用 holosoma.utils.rotations 里的 helper，
    # 那些 helper 假设 xyzw frame，用错会静默旋转错方向
    ```
  - **Body ID resolution（用 Holosoma 惯用 pattern）**：
    ```python
    def __init__(self, ...):
        super().__init__(...)
        # get_term_cfg returns the CommandTermCfg wrapper; the WristComplianceConfig
        # (Task 1 dataclass) lives at cfg_wrapper.params. Adjust access path to match
        # Holosoma's actual CommandManager API at implementation time — if params is
        # stored under a different attr (e.g. .cfg, .term_cfg), use that.
        cfg_wrapper = self.command_manager.get_term_cfg("wrist_compliance_command")
        cfg = cfg_wrapper.params  # WristComplianceConfig instance
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

### Phase 2（observation）

- [ ] **Task 4：三个 obs term + 单测**（v9.4：加 K_virtual obs）
  - `src/holosoma/holosoma/managers/observation/terms/wbt_force.py`
  - `wrist_force_command(env) -> [N, 6]`：返回 `WristComplianceCommand.force_cmd_b.reshape(N, 6)`
  - `wrist_force_ext_privileged(env) -> [N, 6]`：返回 `WristComplianceCommand.force_ext_w.reshape(N, 6)`
  - **`wrist_virtual_stiffness_command(env) -> [N, 2]`**（v9.4 新增）：返回 `WristComplianceCommand.k_virtual.clone()`（preset 里 obs scale 用 `1/100.0` = `0.01`，让数值 v1 常量时 =1.0，v2 range [50,300] 时 ∈ [0.5, 3.0] —— 这是**数值归一化**，让 K 进入 MLP 时量级和其他 obs 相仿；严格意义上不是 CHIP 的 `1/k`（CHIP 算的是倒数），但训练信号等价——MLP 首层线性层对 `K` 和 `c·K` 同构学习）
  - 单测：用 `SimpleNamespace` + fake `_FakeCommandManager`
    - 验证三个 term 各自 shape / value 正确
    - `wrist_virtual_stiffness_command` v1 情境（mock `k_virtual = full((N,2), 100.0)`）→ 输出 shape `(N, 2)` 且值全 100
    - clone 语义：修改 obs 返回值不影响 `command.k_virtual`

### Phase 3（reward）

- [ ] **Task 5：`wrist_force_position_tracking_exp` reward term**（v9.4：K_virtual 从 command buffer 读）
  - `src/holosoma/holosoma/managers/reward/terms/wbt_force.py`
  - **签名变更**：移除 `K_virtual` 参数（不再从 preset 传常量），改成从 command manager 读 per-env tensor
    - 新签名：`wrist_force_position_tracking_exp(env, sigma, left_wrist_body_name, right_wrist_body_name) -> [N]`
  - 实现按 Reward 公式节双力公式（v9.4 per-env K）：
    ```python
    motion = env.command_manager.get_state("motion_command")
    wrist_cmd = env.command_manager.get_state("wrist_compliance_command")
    force_cmd_b = wrist_cmd.force_cmd_b.view(-1, 2, 3)       # body-yaw frame
    force_ext_w = wrist_cmd.force_ext_w                       # world frame
    k_virtual  = wrist_cmd.k_virtual                          # [N, 2] ★ v9.4: per-env per-wrist
    yq = yaw_quat(env.base_quat, w_last=True)
    force_cmd_w_left  = quat_apply(yq, force_cmd_b[:, 0], w_last=True)
    force_cmd_w_right = quat_apply(yq, force_cmd_b[:, 1], w_last=True)
    force_cmd_w = torch.stack([force_cmd_w_left, force_cmd_w_right], dim=1)
    F_total_w = force_ext_w + force_cmd_w                     # [N, 2, 3] ★ 双力叠加
    left_idx, right_idx = resolve_tracked_index(left_wrist_body_name, right_wrist_body_name)
    wrist_target_w = motion.body_pos_relative_w[:, [left_idx, right_idx], :]
    wrist_actual_w = motion.robot_body_pos_w[:, [left_idx, right_idx], :]
    # per-wrist K: [N, 2] → [N, 2, 1] 以广播到 [N, 2, 3]
    wrist_target_shifted_w = wrist_target_w + F_total_w / k_virtual.unsqueeze(-1).clamp(min=1e-3)
    error = torch.sum(torch.square(wrist_target_shifted_w - wrist_actual_w), dim=-1)
    return torch.exp(-error.mean(-1) / (sigma ** 2))
    ```
  - 单测（pure CPU）：
    - 覆盖 4 个稳态情形（Reward 公式节表格，mock `k_virtual = full((N,2), 100.0)`）
    - yaw rotation 验证
    - **v9.4 新增 per-env K 测试**：mock `k_virtual[:, 0] = 50.0, k_virtual[:, 1] = 200.0` → 同样的 F_cmd=20N 下左 wrist Δx=0.4m、右 wrist Δx=0.1m，reward kernel 对应不同误差
    - 边界：`k_virtual = 0` 或极小值 → clamp(min=1e-3) 保证不除零（assert 不 NaN）

### Phase 4（config presets & experiment）

- [ ] **Task 6：command preset `g1_29dof_wbt_force_command`**
  - `src/holosoma/holosoma/config_values/wbt/g1/command_force.py`
  - motion_command（原样继承）+ `wrist_compliance_command`
  - `src/holosoma/holosoma/config_values/command.py` DEFAULTS 追加
  - 回归测试：断言 `g1_29dof_wbt_command` object identity 未变

- [ ] **Task 7：observation preset `g1_29dof_wbt_force_observation`**（v9.4：沿用 baseline 2-group，加 history=10 + 2 force term + 1 K_virtual term）
  - `src/holosoma/holosoma/config_values/wbt/g1/observation_force.py`
  - **沿用 baseline 2-group 架构**（见 "Observation 结构 v9.4" 节）：
    - `actor_obs` (`history_length=10`): baseline 所有 term + **`wrist_force_command` (6)** + **`wrist_virtual_stiffness_command` (2, scale=0.01)**
    - `critic_obs` (`history_length=10`): baseline 所有 term + **`wrist_force_command` (6)** + **`wrist_virtual_stiffness_command` (2, scale=0.01)** + **`wrist_force_ext_privileged` (6)**
  - **Obs scale 选择**：`wrist_virtual_stiffness_command` scale = `1 / 100.0` = `0.01`，让 v1 常量 100 → obs 值 1.0；v2 range [50, 300] → obs 值 ∈ [0.5, 3.0]（**数值归一化**让 K 和其他 obs 量级相仿，避免首层 BN 崩溃；不是 CHIP 的 `1/k`——CHIP 算的是倒数、我们做的是线性缩放，但 MLP 对两者同构学习所以等价）
  - **不改 baseline** `g1_29dof_wbt_observation` 对象：新 preset 通过**新建 `ObservationManagerCfg` + 新建两个 `ObsGroupCfg`** 构造；baseline 的 `terms` dict 可以 `copy.copy` + mutate 到新实例（或手写完整 term 列表以避免任何 alias）
  - DEFAULTS 注册 `"g1_29dof_wbt_force"`
  - 回归测试：
    - `assert g1_29dof_wbt_observation is module.g1_29dof_wbt_observation`（identity check）
    - `actor_obs.history_length == 10`（baseline 是 1）
    - `critic_obs.history_length == 10`
    - `actor_obs` 的 terms 包含 baseline 所有 term + `wrist_force_command` + `wrist_virtual_stiffness_command`（但 **不含** `wrist_force_ext_privileged`）
    - `critic_obs` 的 terms 包含 `wrist_force_command` + `wrist_virtual_stiffness_command` + `wrist_force_ext_privileged`
    - `wrist_virtual_stiffness_command` 的 `scale == 0.01`（两个 group 都是）
    - 维度测试（flat dim via `ObservationManager.get_obs_dims()` 或等价 API）：
      - `actor_obs` single-step dim = 154 + 6 + 2 = **162**，stacked = 162 × 10 = **1620**
      - `critic_obs` single-step dim = 286 + 6 + 2 + 6 = **300**，stacked = 300 × 10 = **3000**

- [ ] **Task 8：reward preset `g1_29dof_wbt_force_reward`**（v9.4：不再传 K_virtual 参数）
  - `src/holosoma/holosoma/config_values/wbt/g1/reward_force.py`
  - 原 9 个 reward term 全部继承（"以 holosoma 为主"）
  - 追加 `wrist_force_position_tracking_exp`（weight=2.0，σ=0.3；**不传 K_virtual**——reward term 自己从 `command_manager.wrist_compliance_command.k_virtual` 读 per-env tensor）
  - DEFAULTS 注册
  - 回归测试：断言 `g1_29dof_wbt_reward` 未变；新 preset 的前 9 项 hash 等于原 preset；新 reward term 的 params dict **不含** `K_virtual` key（v9.4 的新架构保证）

- [ ] **Task 9：experiment `g1_29dof_wbt_force`**（v9.2：沿用 baseline algo，不 override input_dim）
  - `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` 末尾追加（不改 `g1_29dof_wbt`）
  - `env_class="holosoma.envs.wbt.wbt_force_injected:WholeBodyTrackingForceInjected"`
  - command/observation/reward 用新 preset
  - **algo 完全沿用 baseline 的 `input_dim`**：`actor.input_dim=["actor_obs"]`、`critic.input_dim=["critic_obs"]` 保持不变（baseline `config_values/algo.py:44,99`）
  - **MLP 层宽**：baseline `[512, 256, 128]`，actor 从 154 → **1620** dim（扩 10.5x）。**v9.4 建议首层上 `[1024, 512, 256]`**（input 翻 10x，首层容量翻 2x）。若训练后发现 overfit / 参数太多，v2 可回到 `[512, 256, 128]`
  - 层宽的具体决定留给 task 执行时做一次 smoke training（5min 看 loss 曲线）：
    - 若 baseline `[512, 256, 128]` 能收敛 → 保持
    - 若不能 → 上 `[1024, 512, 256]`
  - algo 的其他字段（lr, gamma, entropy coef, clip_range, epochs, minibatches）完全继承 baseline
  - 其他（robot、simulator、curriculum、randomization、terrain、termination）完全继承
  - `src/holosoma/holosoma/config_values/experiment.py` DEFAULTS 追加
  - 回归测试：
    - baseline `g1_29dof_wbt` object 的 `algo` field identity 未动 (`assert g1_29dof_wbt.algo is ORIGINAL_ALGO`)
    - 新 preset 的 `actor.input_dim == ["actor_obs"]`、`critic.input_dim == ["critic_obs"]`（跟 baseline 一样）

### Phase 5（集成测试 & 文档）

- [ ] **Task 10：tyro CLI smoke test**
  - `tests/test_wbt_force_cli.py`
  - subprocess 跑 `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-force --help` 几秒内 exit 0
  - 断言 baseline `exp:g1-29dof-wbt --help` 依旧 ok
  - 断言 `--help` 输出里出现 `wrist_compliance_command` 可配参数

- [ ] **Task 11：IsaacSim e2e（`@pytest.mark.isaacsim`）**
  - `tests/e2e/test_wbt_wrist_force_e2e.py`
  - **测试夹具覆盖 activation_prob = 1.0**：config 的默认 `activation_prob_per_step=0.01` 意味着 20 step × 2 wrist 有 ~67% 概率全程零力，e2e 断言会 flaky。测试时用 `replace(cfg.wrist_compliance_command.params, force_ext_activation_prob_per_step=1.0, force_cmd_activation_prob_per_step=1.0, force_ext_cooldown_range_s=(0.0, 0.0), force_cmd_cooldown_range_s=(0.0, 0.0))` override，保证每个 step 都有力激活
  - 初始化新 experiment、跑 20 control step
  - 断言：
    - **红线 1 强约束**：通过 env 子类暴露的 `last_applied_force_w_by_body_id` debug accessor 验证——它的 keys 只能是 `{left_wrist_isaac_id, right_wrist_isaac_id}`；**不**直接读 IsaacLab 私有 `_external_force_b` buffer
    - 辅助断言：command term 内部 `force_ext_w` 形状是 `[N, 2, 3]`（只有 2 个 wrist channel），没有其他 body channel
    - `force_ext_w` 连续 20 step 每步都有非零 sample（activation=1.0 下应该确定性非零，flake-free）
    - `force_cmd_b` 的 ramp 剖面可观察到（连续 step 值平滑变化）
    - reward 张量非 NaN、在 `[0, 1]`
    - `obs_dict["actor_obs"].shape[1] == 1620`（154 baseline + 6 wrist_force_command + 2 wrist_virtual_stiffness_command，× 10 history）
    - `obs_dict["critic_obs"].shape[1] == 3000`（286 baseline + 6 wrist_force_command + 2 wrist_virtual_stiffness_command + 6 wrist_force_ext_privileged，× 10 history）
    - **v9.4 新增**：`command_manager.wrist_compliance_command.k_virtual` shape `(N, 2)` 且 v1 全 100.0
    - `obs_dict` 只有 2 个 group key（`actor_obs` 和 `critic_obs`），没有 `actor_current` / `critic_extra` 之类的额外 group
  - **独立的 random-activation smoke 断言**：跑 200 step 默认 config，仅断言 `force_ext_w.abs().sum() > 0` 至少命中一次（200 步下 P(全零) = `0.99^400 ≈ 1.8%`）

- [ ] **Task 12：用户文档 `docs/wbt-wrist-force-training.md`**
  - 启动命令 + 关键参数表
  - "F_ext 和 F_cmd 只施加 wrist" 显式声明（红线 1）
  - K_virtual 来源（`_k_virtual.py` 常量 + §K_virtual 节说明"这是 training hyperparameter，不是测得刚度"）
  - 部署端：上层 API 只需给 F_cmd 6-D body-yaw frame
  - 调参建议（σ、F_cmd 量级）
  - **v9.2 定位声明**：learned compliance 的数学基础已具备（10 步 history 让 actor 能隐式估 F_ext）；能否真学到需要看训练曲线
  - **部署端调用**：v9.2 因为保持 baseline 2-group + `input_dim=["actor_obs"]`，**不需要任何 concat 契约**。调用者按 baseline Holosoma 的标准部署路径即可：
    ```python
    policy_fn = ppo_agent.get_inference_policy()
    # obs_dict 来自 env / sim，actor_obs 已经由 obs manager 自动 stack 10 步
    action = policy_fn({"actor_obs": obs_dict["actor_obs"]})  # shape [N, 1620]
    ```
    行为与 baseline `g1_29dof_wbt` 完全一致，只是 dim 从 154 → 1620（v9.4 加了 wrist_virtual_stiffness_command 所以比 v9.2 的 1600 多 20）
  - v2 扩展说明（若 history 学不到 F_ext 推断，考虑加 RMA-style 显式 F_ext estimator / adaptive compliance 1/k）

### Phase 6（sanity）

- [ ] **Task 13：全局 sanity**
  - pre-commit 全绿
  - mypy 全绿
  - `pytest -s --ignore=thirdparty --ignore=src/holosoma_inference -m "not isaacsim and not requires_inference"` 全绿
  - `pytest -m isaacsim` 新增 Spike 1 + Task 3 + Task 11 全绿
  - `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt --help` 依旧 ok（baseline 不受污染）
  - `demo_scripts/demo_omomo_wb_tracking.sh` 第 4 步的 baseline 命令仍能启动（不需要真跑完，启动 5s 后 SIGINT 就行）

---

## 🚀 Evaluation & Deployment（inference-side requirements，v9.3 新增）

**目的**：保证训练完成后的 eval (`eval_agent.py`) + sim-to-sim (MuJoCo) + 真机部署都能直接跑起来，**无需回头重训**。

### 关键链路梳理

| 阶段 | 入口 | 读什么 config | 备注 |
|---|---|---|---|
| **In-training eval** | `src/holosoma/holosoma/eval_agent.py` | checkpoint 内置 config（训练时冻结）| 同 simulator，obs + action 自动匹配。**零改动** |
| **ONNX export** | `ppo.py:707` / `fast_sac:962` `export_motion_and_policy_as_onnx` | 训练时 `actor.input_dim` = 1620 | `_OnnxMotionPolicyExporter.input_dim` 自动从 `actor_module[0].in_features` 读取 → 自适应扩到 1620 dim，**无需改 exporter 代码** |
| **Sim-to-sim eval (MuJoCo)** | `run_policy.py inference:g1-29dof-wbt` + `holosoma.run_sim` | `holosoma_inference` 独立 config | **需要新 inference config** `g1-29dof-wbt-force` |
| **真机部署** | `run_policy.py inference:g1-29dof-wbt` + Unitree SDK | 同上 | **需要新 policy 子类** + F_cmd input channel |

### 潜在 Silent-Failure 点（ONNX 不匹配 → shape error 或行为偏离）

1. **history_length 不一致** — 训练 `history_length=10`、inference `wbt` config 默认 `history_length_dict={"actor_obs": 1}`。不覆盖会让 `obs_buf_dict["actor_obs"]` = `(1, 162)` 而 ONNX 期望 `(1, 1620)` → `InvalidArgument` 崩溃
2. **Obs term 缺失** — inference 的 `WholeBodyTrackingPolicy.get_current_obs_buffer_dict`（`wbt.py:225-256`）只填 baseline 6 个 term；不追加 `wrist_force_command` → `KeyError` 或者 term 被 0-padding 顶替（parse_current_obs_dict `wbt.py:570` 会 raise KeyError）
3. **Obs term 排序错位** — 训练端 `manager.py:141` 用 `sorted(obs_tensors.keys())`；inference 端 `base.py:130` 用 `sorted(term_names)`。**两端都 alphabetical → term 顺序一致**（`wrist_force_command` 排最后）。已有保证；回归测试显式断言即可
4. **F_cmd 永远 0（CRITICAL silent failure）** — 现有 input API 只有 `VelCmdProvider` + `StateCommandProvider`；`InputSource` Literal (`config_types/task.py:9`) 也没有 "wrist_force" 通道；`BasePolicy._create_input_providers` (`base.py:342`) 只按 velocity/state 角色 `create_input`。**即便新加 `WristForceCmdProvider` Protocol，若不同时扩 `TaskConfig` 字段 + `InputSource` Literal + `inputs/__init__.py` factory + `_create_input_providers` 调用，部署时 `_wrist_force_cmd` 永远是全 0 tensor**，policy 看似正常跑但 F_cmd 等于 baseline WBT，不报错
5. **Obs scale 不一致** — 训练 `wrist_force_command: 1.0`；inference obs config 必须同值（否则 policy 看到的 F_cmd 量级偏移）
6. **Body-yaw frame 约定漂移** — training reward 用 `yaw_quat(base_quat) · force_cmd_b`，即 F_cmd 已在 body-yaw frame。operator 给的数必须同 frame（左 wrist +X = 机器人前方），不能误当 world frame
7. **Stop policy 后 provider 不清零（CRITICAL silent failure）** — 子类 `_handle_stop_policy` 清 `_wrist_force_cmd` 的**局部缓存**还不够——`base.py:816` 主循环每 cycle 先 `poll_wrist_force()` 再 `policy_action()`，provider 内部状态（如 ROS2 subscriber 的 last-received buffer）可能 hold 前一个 F_cmd 值，stop 后下 tick 又被 repopulate。**必须同时调 `self._wrist_force_provider.zero()`**
8. **ONNX metadata 写入但不校验（HIGH silent failure）** — Task 14 把 `history_length`/`obs_term_names_sorted`/`obs_group_dims` 写进 metadata，但 `WholeBodyTrackingPolicy.setup_policy` (`wbt.py:131-142`) 只读 `robot_urdf`/`kp`/`kd`。若换成一个 term 顺序或 history 长度不同的 ONNX，总 dim 巧合相等时 policy 会跑但行为错。**Task 14 必须加 runtime check**：policy 加载时对齐自身 `history_length_dict["actor_obs"]` 和 `obs_terms_sorted["actor_obs"]` 与 ONNX metadata 的记录，不匹配直接 raise

### 新增文件清单（inference 侧）

| 路径 | 类型 | 职责 |
|---|---|---|
| `src/holosoma_inference/holosoma_inference/config/config_values/observation.py` | 追加 | 新 `wbt_force` ObservationConfig：`obs_dict["actor_obs"]` 追加 `wrist_force_command`；`obs_dims["wrist_force_command"]=6`；`obs_scales["wrist_force_command"]=1.0`；`history_length_dict={"actor_obs": 10}` |
| `src/holosoma_inference/holosoma_inference/config/config_values/inference.py` | 追加 | 新 `g1_29dof_wbt_force` InferenceConfig：`robot=_g1_29dof_wbt_robot`（复用 stiff startup 设置）、`observation=wbt_force`、`task=task.wbt`、`secondary=_g1_safety_secondary`；DEFAULTS 注册 |
| `src/holosoma_inference/setup.py` | 追加 | `holosoma.config.inference` entry point 追加 `g1-29dof-wbt-force = ...config_values.inference:g1_29dof_wbt_force` |
| `src/holosoma_inference/holosoma_inference/policies/wbt_force.py` | **新** | `WholeBodyTrackingForcePolicy(WholeBodyTrackingPolicy)`：持 `_wrist_force_cmd: np.ndarray[1,6]`；override `get_current_obs_buffer_dict` 追加 `wrist_force_command`；override `_handle_stop_policy` 清零；初始化新 input provider；poll loop 更新 `_wrist_force_cmd` |
| `src/holosoma_inference/holosoma_inference/inputs/api/commands.py` | 追加 | `WristForceCmd` dataclass：`force_left: tuple[float,float,float]`, `force_right: tuple[float,float,float]`（body-yaw frame，N） |
| `src/holosoma_inference/holosoma_inference/inputs/api/base.py` | 追加 | `WristForceCmdProvider` Protocol：`poll_wrist_force() -> WristForceCmd \| None`, `start()`, `zero()` |
| `src/holosoma_inference/holosoma_inference/config/config_types/task.py` | 修改 | 扩 `InputSource` Literal（或单独新 Literal）+ `TaskConfig` 追加 `wrist_force_input: InputSource = "keyboard"`, `wrist_force_magnitude_cap: float = 30.0`, `ros_wrist_force_topic: str = "holosoma/wrist_force_cmd"` |
| `src/holosoma_inference/holosoma_inference/inputs/__init__.py` | 修改 | `create_input` factory 扩"wrist_force" role，按 `InputSource` dispatch 到 keyboard/ros2/joystick 实现 |
| `src/holosoma_inference/holosoma_inference/inputs/impl/keyboard.py` | 追加 | F_cmd 键位 mapping（避免和现有 `o`=STOP / `i`=INIT / `]`=START / `m`=MOTION / `=`=STAND_TOGGLE 冲突：左 wrist 用 `u/j/h/k/y/n`（X+/X−/Y+/Y−/Z+/Z−），右 wrist 用数字区 `8/2/4/6/9/3`；`/` 归零；magnitude 步长用 `,` 减 / `.` 加）—— 键位具体布局待 dogfood |
| `src/holosoma_inference/holosoma_inference/inputs/impl/ros2.py` | 追加 | Subscribe `holosoma/wrist_force_cmd` (std_msgs/Float32MultiArray[6])；topic 名 CLI 可配 |
| `src/holosoma_inference/holosoma_inference/policies/base.py` | 修改 | `BasePolicy.run()` (`base.py:816`) 内 `policy_action()` 之前加一个 no-op hook `self._poll_extra_inputs()`；基类默认 pass。理由：避免子类 duplicate 整个 run loop → 破坏 dual-mode 下统一 latency/rate 管控 |
| `src/holosoma_inference/holosoma_inference/policies/dual_mode.py` | 修改 | `_select_policy_class`：若 `"wrist_force_command" in actor_obs` → 新 WBT force policy（entry point 优先，兜底 import） |
| `src/holosoma_inference/holosoma_inference/README.md` | 修改 | "Whole-Body Tracking" controls 表 + `## Observation History Length` 之前追加 "Wrist Force Command" 小节（键位表 + ROS2 topic 表 + F_cmd magnitude cap 说明） |
| `src/holosoma/holosoma/utils/inference_helpers.py` | 修改 | ONNX 导出调用点（PPO `ppo.py:731-742` + FastSAC `fast_sac_agent.py:986-997` 的 metadata 构造 dict）追加：`history_length`（int）、`obs_term_names_sorted`（list[str]）、`obs_group_dims`（dict）；**同时** `attach_onnx_metadata` 用法无需改，只需调用点的 metadata dict 多塞 3 个 key |
| `src/holosoma_inference/setup.py` | 修改 | 除 `holosoma.config.inference` 入口外，新增 `holosoma.policies.wbt` entry point group，注册 `g1-29dof-force = holosoma_inference.policies.wbt_force:WholeBodyTrackingForcePolicy`，让 dual_mode dispatch 能通过 entry point 找到（和 `dual_mode.py:28-32` pattern 保持一致） |
| `src/holosoma_inference/docs/workflows/sim-to-sim-wbt-force.md` | **新** | MuJoCo 下 F_cmd 只影响 wrist Δx（无接触面 → 无真实力）；验证方法是观察 wrist 相对 motion target 偏移 |
| `src/holosoma_inference/docs/workflows/real-robot-wbt-force.md` | **新** | 首次部署 F_cmd ≤ 5N、ramp 手动渐增、紧急停止（`o` 键）与 F_cmd=0 的关系 |
| 对应 `tests/test_*.py` | **新** | unit + integration 测试 |

### Phase 7（inference & deployment）

- [ ] **Task 14：ONNX metadata schema extension + inference 侧 runtime 校验**
  - **训练侧**：PPO export (`ppo.py:731-742`) + FastSAC export (`fast_sac_agent.py:986-997`) 构造 metadata dict 时追加：
    - `"history_length"`: `cfg.observation.groups.actor_obs.history_length`（int）
    - `"obs_term_names_sorted"`: `sorted(cfg.observation.groups.actor_obs.terms.keys())`
    - `"obs_group_dims"`: `{"actor_obs": 1620, "critic_obs": 3000}`（critic 对部署无用但便于 debug）
  - **inference 侧 runtime 校验（CRITICAL）**：`WholeBodyTrackingPolicy.setup_policy` (`wbt.py:131-142`) 读出 metadata 后 vs `self.obs_terms_sorted["actor_obs"]` 和 `self.history_length_dict.get("actor_obs", 1)` 做 strict compare，不匹配 **直接 raise**（shape 巧合相等时避免静默行为漂移）
  - 向后兼容：metadata key 缺失（老 checkpoint）→ fallback `logger.warning` + 允许继续（不 raise），方便 baseline ONNX 仍能部署
  - 测试：
    - 训练侧：export → `onnx.load` → 断言 metadata 字段存在 + 值正确
    - inference 侧 unit：构造 fake metadata（不匹配 term order / 不匹配 history）→ 断言 policy 初始化 raise
    - 老 checkpoint 缺 key → warning（不 raise）；baseline `g1-29dof-wbt` smoke 不因 Task 14 变更而坏

- [ ] **Task 15：Inference obs preset `wbt_force`**
  - 在 `src/holosoma_inference/holosoma_inference/config/config_values/observation.py` 追加
  - `obs_dict["actor_obs"]` = baseline 6 个 term + `wrist_force_command`
  - `obs_dims["wrist_force_command"] = 6`；`obs_scales["wrist_force_command"] = 1.0`
  - `history_length_dict = {"actor_obs": 10}`
  - 测试：alphabetical sort 后 term list 和训练端 (`g1_29dof_wbt_force_observation.actor_obs.terms.keys()`) 排序一致（跨 package 断言需把训练 preset 的 term 名 hardcode 到测试常量里，避免引入 isaacsim 依赖）

- [ ] **Task 16：Inference preset `g1_29dof_wbt_force` + entry point**
  - `config_values/inference.py` 追加 `g1_29dof_wbt_force`：复用现有 `_g1_29dof_wbt_robot`（stiff startup 不变）、`observation=wbt_force`、`task=task.wbt`、`secondary=_g1_safety_secondary`
  - DEFAULTS 注册 `"g1-29dof-wbt-force"`
  - `setup.py` entry point 追加 `g1-29dof-wbt-force = holosoma_inference.config.config_values.inference:g1_29dof_wbt_force`
  - 测试：`python src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-force --help` exit 0；subcommand 在 help 输出中出现
  - 测试：重装后 `entry_points(group="holosoma.config.inference")` 能发现新 key

- [ ] **Task 17：`WristForceCmd` + Provider + impl + task config + factory wiring**（顺序依赖：必须在 Task 18 之前完成整套 end-to-end wiring，否则 Task 18 子类初始化时 provider 无法 inject）
  - `inputs/api/commands.py` 追加 `WristForceCmd` frozen dataclass（字段见上表）
  - `inputs/api/base.py` 追加 `WristForceCmdProvider` Protocol（含 `poll_wrist_force()`、`start()`、`zero()`）
  - **`config_types/task.py` 扩 `InputSource` Literal + 新增 3 个 TaskConfig 字段**（`wrist_force_input`, `wrist_force_magnitude_cap`, `ros_wrist_force_topic`）—— 缺了这步 CLI 不给你 `--task.wrist-force-input` 开关，provider 永远不初始化
  - **`inputs/__init__.py`：`create_input` factory 扩 "wrist_force" role**，按 `InputSource` dispatch 到实现（类似 velocity/command role 已有的 pattern）
  - `inputs/impl/keyboard.py` 追加 F_cmd 键位（避开 `o`=STOP/`i`=INIT/`]`=START/`m`=MOTION/`=`=STAND_TOGGLE：左 wrist `u/j/h/k/y/n`，右 wrist numpad `8/2/4/6/9/3`，`/`=归零，`,` 减 / `.` 加 magnitude 步长；magnitude 受 `wrist_force_magnitude_cap` 硬 clamp）
  - `inputs/impl/ros2.py` 追加 subscriber，topic 名从 `task.ros_wrist_force_topic` 读取
  - 测试：
    - pure-protocol 单测（fake keystrokes → `poll_wrist_force` 返回正确 WristForceCmd）
    - magnitude clamp 单测：输入 `[100, 0, 0]` + cap=30 → 断言 output norm ≤ 30
    - `zero()` 后 `poll_wrist_force()` 返回 `WristForceCmd(force_left=(0,0,0), force_right=(0,0,0))`
    - ROS2 subscriber 只做 mock 导入校验（不依赖 rclpy 运行时）
    - factory dispatch：`create_input(policy, "keyboard", "wrist_force")` 返回实现了 `WristForceCmdProvider` 的对象

- [ ] **Task 18：`BasePolicy._poll_extra_inputs()` hook + `WholeBodyTrackingForcePolicy`**
  - **`BasePolicy.run()` (`base.py:816`) 加 hook**：在 `self.policy_action()` 之前插入一行 `self._poll_extra_inputs()`；基类默认 pass。理由（codex review 推荐）：避免子类 override 整个 `run()` → 破坏 dual-mode 下 latency / rate / log 的统一管控；最小基类改动 (~3 行) 把 F_cmd 轮询变成可扩展 extension point
  - `policies/wbt_force.py` 继承 `WholeBodyTrackingPolicy`
  - `__init__`：
    - `self._wrist_force_cmd = np.zeros((1, 6), dtype=np.float32)`
    - 通过 `create_input(self, config.task.wrist_force_input, "wrist_force")` 初始化 provider，保存到 `self._wrist_force_provider`；`.start()`
    - 读 `self._magnitude_cap = config.task.wrist_force_magnitude_cap`
  - override `_poll_extra_inputs()`：调 `self._wrist_force_provider.poll_wrist_force()`，若非 None 则 clip norm ≤ `self._magnitude_cap` 后写入 `self._wrist_force_cmd`
  - override `get_current_obs_buffer_dict(robot_state_data)`：调 super 得 baseline 6 个 term dict，追加 `d["wrist_force_command"] = self._wrist_force_cmd.copy()`
  - override `_handle_stop_policy`：调 super + `self._wrist_force_cmd.fill(0.0)` + **`self._wrist_force_provider.zero()`**（关键：防止 provider 内部 last-received buffer 在下一 tick 又 repopulate F_cmd；见 "Silent-Failure 点 #7"）
  - 测试：
    - fake provider → 验证 `get_current_obs_buffer_dict` 输出字典包含 `wrist_force_command` 且 shape=`(1, 6)`
    - stop policy → 断言 `_wrist_force_cmd` 为 0 **且 `provider.zero()` 被调一次**
    - magnitude cap：provider 输出 norm=50 + cap=30 → `_wrist_force_cmd` norm ≤ 30
    - hook 回归：baseline `LocomotionPolicy` / baseline `WholeBodyTrackingPolicy` 下 `_poll_extra_inputs()` = no-op，行为不变

- [ ] **Task 19：policy dispatch extension**
  - `dual_mode.py:_select_policy_class` (`dual_mode.py:12-37`)：若 `"wrist_force_command" in actor_obs` → entry point 优先（`holosoma.policies.wbt` group, 按 `robot_type` 键查），兜底 `WholeBodyTrackingForcePolicy`
  - `setup.py` `entry_points` 新增 `holosoma.policies.wbt` group：`g1-29dof-force = holosoma_inference.policies.wbt_force:WholeBodyTrackingForcePolicy`（注意：现有 `dual_mode.py:29` 已按 `ep.name == robot_type` 查，需要 **pip reinstall** 后 entry point 才生效——Task 20 smoke 前提醒用户重装）
  - 测试：
    - fake config with `wrist_force_command` in actor_obs → dispatch 到 `WholeBodyTrackingForcePolicy`（identity 断言）
    - baseline WBT config → 仍 `WholeBodyTrackingPolicy`（identity 断言，保证现有行为不变）
    - entry point 发现测试：mock `entry_points` 返回 WristForceSubclass → dispatch 优先用 entry point 版本

- [ ] **Task 20：sim-to-sim smoke + workflow docs**（前置依赖：Task 15-19 全部完成 + `pip install -e src/holosoma_inference` 重装以生效新 entry points）
  - 新建 `src/holosoma_inference/docs/workflows/sim-to-sim-wbt-force.md`：
    - 启动命令（两个 terminal：`run_sim.py` + `run_policy.py inference:g1-29dof-wbt-force --task.wrist-force-input keyboard`）
    - 明确 body-yaw frame 约定（+X=前方，+Y=左方，+Z=上方；单位 N）
    - **MuJoCo 无接触面 → F_cmd 效果观察的是 wrist Δx，不是真实力**
    - 验证：F_cmd=0 → wrist 跟 motion；F_cmd=[5,0,0] 左 wrist → 左 wrist 相对 motion target 前移 ~0.05m（K_virtual=100, 稳态）
    - 键位表（对齐 Task 17 的最终键位）+ ROS2 topic 格式（`Float32MultiArray[6]`）
  - 新建 `real-robot-wbt-force.md`：
    - 首次部署流程：F_cmd=0 先跑完整 motion clip；再 F_cmd=1-5N 小幅增量；确认 wrist 接触物体时力感合理
    - 紧急停止：`o` 键 → stop policy → 回 stiff hold（**Task 18 保证 `_wrist_force_cmd.fill(0)` + `provider.zero()` 同时执行**）
    - 安全红线：F_cmd magnitude 默认硬 clamp 上限 30N（与训练一致）；operator 可通过 CLI `--task.wrist-force-magnitude-cap` 降低；上调到 >30 会 raise
  - `src/holosoma_inference/README.md` 追加 "Whole-Body Tracking" controls 表下方小节：F_cmd 键位表 + ROS2 topic 用法
  - e2e smoke test（marker `requires_inference`）：
    - `run_sim.py` 起 MuJoCo + `run_policy.py` 起 policy，5s 后 SIGINT
    - 断言：`force_cmd` provider 成功初始化、`obs_buf_dict["actor_obs"].shape[1] == 1620`、无 crash
    - 额外断言：启动后 `_wrist_force_cmd` 初值全 0；dispatch 路由到 `WholeBodyTrackingForcePolicy` 而非 baseline `WholeBodyTrackingPolicy`（instance 类型断言）

- [ ] **Task 21：Inference regression + cross-package schema 一致性 + runtime metadata validation**
  - `tests/e2e/test_wbt_force_inference_schema.py`：**跨 package 静态断言**（不跑 isaacsim，只 import config_values 比对）：
    - term 名集合相等：训练端 `g1_29dof_wbt_force_observation` actor group 的 term names vs inference 端 `wbt_force.obs_dict["actor_obs"]`
    - alphabetical sort 后顺序一致
    - dim 一致（单步 162，stacked 1620）
    - `history_length` 一致（=10）
    - scale 一致：每个 term 的 `obs_scale` 两端相等（特别是 `wrist_force_command: 1.0`）
  - **runtime metadata validation 测试**（配合 Task 14）：
    - 构造 metadata 不匹配的 ONNX（history=5 但 policy 初始化期望 10）→ 断言 `setup_policy` raise
    - 构造 term 顺序不匹配的 ONNX → 断言 raise
    - 老 baseline ONNX（无新 metadata key）→ warning 不 raise
  - 回归：baseline `wbt` config 不受影响（identity assertion + `g1_29dof_wbt_observation` unchanged）；baseline `g1-29dof-wbt` ONNX 仍能加载跑 MuJoCo smoke



## ✅ Self-Review

| 用户要求 | 覆盖 |
|---|---|
| "F_ext 和 F_cmd 只加 wrist" | 红线 1；Task 3 注入到 `left/right_wrist_yaw_link`；Task 11 断言通过 env debug accessor（`last_applied_force_w_by_body_id` 的 keys ⊆ wrist body ids） |
| "K 保持跟 holosoma 原始一致而固定" | 红线 2 + K_virtual 节说明这是 **training-time hyperparameter 不是测得刚度**；Task 0 手写 `_k_virtual.py = 100.0 N/m`；inference 端可用常数 α 校正 |
| "F_cmd 作为 actor 输入" | Task 4 obs term；Task 7 actor group；Observation 节显式论证不给 F_ext 给 actor |
| "v1 加 history（对齐 CHIP）" | v9.4：沿用 baseline 2-group，两个 group 都设 `history_length=10`；actor **1620** dim / critic **3000** dim（v9.2 的 1600/2980 + v9.4 新增 20 dim 的 K_virtual history）。借用 CHIP "10 步" 量级但不做 CHIP 的 proprio-only selective stacking（trade-off 换工程稳定性） |
| "尽量用 Holosoma 原始 code" | v9.2：observation 架构不拆 group，PPO `input_dim` 不 override，没有 shim term，没有 concat 契约。`history_length=10` 这一项可以通过 tyro CLI flag `--observation.groups.actor_obs.history-length 10` 直接 override（对任意 obs preset 生效）；但新 obs term `wrist_force_command` / `wrist_force_ext_privileged` 必须通过新 preset 注册，CLI 无法追加新 term——**因此实施 Task 7 的新 preset 仍是必需的**，不能仅靠 CLI flag 取代 |
| "不加 1/k（compliance 可调）" | "已知未决项" 列为 future extension，v1 不实现 |
| "CHIP 仅作对照参考，没有可 port 的代码" | CHIP 没开源 → 我们没用 CHIP 的任何实现细节，只借鉴"F_ext 注入 + critic privileged"这种 architecture-level 思想 |
| "UniFP 是 IsaacGym + quadruped → 不 port 代码" | 只借用 reward 公式 `g̃ = g + (F_ext + F_cmd) / K`（Task 5 实现），**不**抄 UniFP 的 Python 代码 |
| "GH 的 code 只作参考，不 import 不 copy" | §文件结构 明确：只读 GH 源看 API 用法，最终 Holosoma 代码里**一行 GH 代码都不出现**；Task 2 的 ramp state machine 自己写简单版；Task 3 直接用 Holosoma 已有的 `set_external_force_and_torque` |
| "reward 以 holosoma 为主" | Task 8 原 9 reward 全部保留，只追加 1 项；回归测试断言 identity |
| "不影响 Holosoma 已有 training/testing code" | 红线 5 枚举所有允许修改的文件（6 个）+ 禁止所有其他文件；每 preset 都有 `assert original is module.original` identity check；Task 13 Sanity 跑 `exp:g1-29dof-wbt --help` 验证 baseline 启动没坏；Task 13 跑 `demo_scripts/demo_omomo_wb_tracking.sh` 第 4 步验证 demo script 没坏 |
| "训练完 eval + deploy 不回头" | Phase 7（Task 14-21）覆盖 inference 侧全链路：ONNX metadata 扩展（Task 14）、inference obs preset 严格对齐训练端 alphabetical sort（Task 15 + 21）、新 inference config + entry point（Task 16）、F_cmd input provider（Task 17）、新 policy 子类（Task 18）、dispatch 扩展（Task 19）、sim-to-sim + 真机 workflow docs（Task 20）。In-training eval + ONNX export 因 exporter 自适应读 `actor_module[0].in_features`（`inference_helpers.py:40`），**1620** dim 不需改 exporter 代码 |
| "K_virtual 对齐 GH 的 range 采样 + 进 actor obs" | v9.4：`WristComplianceConfig.k_virtual_range`（Task 1）+ `WristComplianceCommand.k_virtual` per-env buffer（Task 2）+ `wrist_virtual_stiffness_command` obs term（Task 4 + 7）+ reward 从 command buffer 读 per-env K（Task 5 + 8）。v1 range=(100,100) 退化 deterministic，v2 改 range 即可 GH-style 随机化不改 obs/reward/exporter。K 同时进 actor 和 critic（CHIP 派系，GH 是 actor/critic 都不看） |

**v9.2 诚实声明：**
- **Learned compliance 的数学基础 v9.2 已具备，但能否真学到要看训练曲线。** v9.2 两个 obs group 都设 `history_length=10`，actor 原则上能从 `dof_vel_{t-9:t}` 反推 q̈，再结合 `actions_{t-9:t}` 从 dynamics 残差估 F_ext（explainer §6.4）。但这只是**上界**——policy 会不会真学到隐式 estimator，取决于：(a) MLP 容量够不够（Task 9 smoke training 判断是否需要扩层）；(b) F_ext 激活频率 0.01/step 是否够；(c) PPO exploration 够不够。若训练后 F_real/F_cmd 相关性低，先调 (b) 和层宽，再考虑 actor 看 F_ext estimate 作为额外 obs
- **v9.2 的命令 stacking 代价是已知的 trade-off。** `motion_command` 58×10 = 580 维、`wrist_force_command` 6×10 = 60 维等命令类 term 被 stack 是冗余（MLP 需自己学会忽略稳态部分）。我们接受这个 dim 膨胀换取工程稳定性（不需动 PPO 代码、不需部署 concat 契约、不需 shim term）。若训练曲线显示 learning speed 差，可以在 v2 退回 v9.1 的 4-group selective history 方案
- **K_virtual=100 N/m 不是测得的 G1 Cartesian 刚度。** 它是 reward exchange rate hyperparameter，选这个值是为了让同向叠加合力（max 60N）除以它得到的 Δx 刚好到 arm reach 边缘，产生合理的 kinematic "soft upper bound"。若推演发现 F_real/F_cmd 系统性偏离，inference 端用标量 α 校正即可

---

## 🔮 已知未决项（留给执行期/训练后决定）

v1 的"不做、先看"清单：

1. **MLP 层宽决定**（v9.4）：actor obs 扩到 **1620**（baseline 154×10.5），Task 9 需要做 smoke training 判断 `[512,256,128]` 是否够；不够就用 `[1024,512,256]`
2. **history_length = 10 是否合适**：CHIP 用 10，UniFP 用 32，GH actor 最远回看 8 步。若 MLP 学不到 F_ext estimator 可试 15-20；若过 fit 可降到 5。v9.2 额外考量：命令类 term 也被 stack，所以加大 `history_length` 的维度代价比 v9.1 4-group 更重，优化时倾向降低而非提升
3. **是否回到 v9.1 的 4-group selective history**：v9.2 选了工程稳定性。若训练曲线显示 MLP 容量不够（即便扩到 `[1024,512,256]` 仍 overfit），或者 learning speed 明显慢于 CHIP 基准，可考虑在 v2 切回 v9.1 的 4-group（只 stack proprio+actions），代价是重新处理 PPO shim / concat 契约
4. **F_ext 激活概率 0.01/step**：对应平均每 100 step（2s @ 50Hz）一次 episode。若 compliance 行为学不出可升到 0.02-0.05（给 actor 更多"见过外力"的训练机会）
5. **双 wrist 是否同时采样力**：v1 两路 Bernoulli 独立，允许同时发力。若同时双力 degrade motion tracking 严重 → v2 限制"最多一路激活"
6. **force range saturation 监控**：若训练曲线显示 `||F_ext + F_cmd|| > 45 N` 的 step 占比 > 20% 且 wrist tracking 在这些 step 明显退化 → v2 把 `force_ext_magnitude_range` 上限从 30 降到 15-20 N
7. **是否加 `force_exd_penalty`**：v1 不加，先看 baseline 曲线；若 policy 学到 overshoot 发力（F_real >> F_cmd）再补
8. **显式 F_ext estimator 作为 actor obs**（RMA-style teacher-student distill）：若 history 隐式估计学不到，可以考虑先训一个 critic-side F_ext regressor，然后把它的预测作为 actor 的额外 obs（类似 Kumar et al. 2021 RMA）。3-5 task
9. **`1/k` adaptive compliance input**（CHIP 的核心）：若未来需要 stiff↔compliant 可调，在现有基础上加 `compliance_range=[0, 0.05]` 的 episode-level 采样 + actor obs；reward 改成 `target_shifted = g + (F_ext + F_cmd)·(1/k)`。3-5 task，不推翻现有结构
10. **F_cmd 键位布局最终确定**（Task 17）：v9.3 初版已选 `u/j/h/k/y/n`（左）+ numpad `8/2/4/6/9/3`（右），避开所有现有 StateCommand 键位冲突（`o`/`i`/`]`/`m`/`=` 等）。但真机 dogfood 时 operator 双手操作可能发现"左手同时控 F_cmd 和按 stop 键不顺"——实装后若反馈不好，可考虑让 `--task.wrist-force-input=joystick` 走右摇杆+扳机（joystick 本身已是 velocity 通道占用者，需要 multiplex 或独立 joystick device）
11. **MuJoCo 接触面建模**（Task 20）：MJWarp/MuJoCo 没有"虚拟墙"让 policy 真正接触测量 F_real。sim-to-sim 只能验证 wrist Δx 偏移 + 不崩溃；F_real / F_cmd 比值只能在真机上测。若真机测得 α = K_virtual / K_surface 显著偏 1（比如 >3x），inference 端加常数补偿即可，不重训
12. **`--task.wrist-force-magnitude-cap` 安全上限**（Task 20）：真机部署时 CLI 可提供硬 clamp，防止操作员误输入 100N；默认 30N（与训练一致）。v1 放在 `task.py` config_value，v2 可做 runtime 可调
13. **K_virtual range 何时拓宽（v2 → GH-style randomization）**（Task 0 + 2 + 7）：v1 `K_VIRTUAL_RANGE_N_PER_M = (100.0, 100.0)` 只是"把 API 搭起来但不加训练信号"。何时切到真 range：
    - a) baseline v1 训练收敛（F_cmd 行为可验证）
    - b) 希望 policy 学到"手硬时小幅偏移、手软时大幅偏移"的区分
    - c) 部署现场发现不同接触面 K_surface 差异大 → 需要 policy 有 scaling 适应能力
    
    建议 v2 值：`(50.0, 300.0)`——覆盖 GH `kp_range=[5, 250]` 的主要工作区间，但下限从 5 抬到 50（G1 arm 太软时 Δx 会超 reach，v9.4 Δx 工作空间表依然适用，取 lo=50 → 60N 合力下 Δx=1.2m 超限，故更保守取 lo=100 或加 reach 限制；先用 50 试，观察 saturation 比例）。**切换不改代码，只改一行常量**
14. **K_virtual 是否让 critic 看更多（GH 的 `force_priv` 思路）**：GH 暴露 `force_applied_b` 给 critic 而不是 kp（`motion_tracking.py:1094-1101`）。我们 v9.4 选 CHIP 派系（actor + critic 都看 K）。若 v2 range randomization 下 critic value estimator 方差大，可考虑补充一个 critic-only privileged obs `wrist_force_applied_priv`（从 `K × Δx_actual` 反算），让 critic 同时看到"指令 K"和"实际展现出的 K·位移"——这是 GH 的 hybrid 做法。留到 v3 再说

---

## 📚 参考

- 原始 plan（含 Path A / Path B 历史参考）：`docs/plans/2026-05-01-wbt-wrist-force-controller.md`
- Force training paradigms 对比（理论深度）：`docs/force-training-paradigms-explainer.md`
- UniFP reward 参考：`third_party/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891-1900`
- GH force API 参考（只读）：`third_party/gentle-humanoid-training/active_adaptation/envs/mdp/commands/motion_tracking.py`
- CHIP paper：`third_party/CHIP.pdf`
- Holosoma 外力注入现用法参考：`src/holosoma/holosoma/agents/callbacks/push.py:222-252`、`src/holosoma/holosoma/simulator/shared/virtual_gantry.py:425`
- Holosoma 实际 callback 顺序（WBT）：`src/holosoma/holosoma/envs/base_task/base_task.py:405-460`
- G1 joint Kp（辅助参考）：`src/holosoma/holosoma/config_values/robot.py:498-524`
