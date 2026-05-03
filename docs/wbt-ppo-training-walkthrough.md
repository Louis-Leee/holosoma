# Whole-Body Tracking PPO 训练流程详解

端到端解析 `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt ...`
（即 `demo_scripts/demo_omomo_wb_tracking.sh` / `demo_lafan_wb_tracking.sh` 里的训练阶段）：
输入输出 shape、网络结构、reward、loss、backprop 发生在哪里，以及关键 hyperparameters。

---

## 0. 数字先行（G1 29-DOF, WBT）

| 维度 | 值 | 来源 |
|---|---|---|
| 并行环境 `num_envs` | **4096** | `config_values/wbt/g1/experiment.py:22` |
| 动作维度 `num_act` | **29**（G1 有 29 DOF） | `config_values/robot.py:14` + `base_task.py:315` 断言 |
| 每 rollout 步数 `num_steps_per_env` | **24** | `config_values/algo.py:33`（PPO default） |
| 控制频率 | **200 Hz / 4 = 50 Hz**（sim 200 Hz + `control_decimation=4`） | `config_values/simulator.py:40-41` |
| 一集最长 `max_episode_length_s` | 10 s → 500 env steps | `config_values/wbt/g1/experiment.py:49` |
| 总迭代 `num_learning_iterations` | **30 000** | `config_values/wbt/g1/experiment.py:29` |

**一次 PPO iteration 做了什么**：`4096 × 24 = 98 304` 条 transition 进 storage → `num_learning_epochs=5 × num_mini_batches=4 = 20` 次梯度更新。

---

## 1. 入口与 `train()` 构造（`train_agent.py`）

```
main() ─ tyro.cli(AnnotatedExperimentConfig)      # L323
  → train(tyro_cfg)                              # L325
      → init_sim_imports()  # 必须在 `import torch` 之前
      → configure_multi_gpu()                    # L179
      → get_class(env_target)(...)   # L261 构造 WholeBodyTrackingManager
      → algo_class(env, config, ...) # L282-289 构造 PPO
      → algo.setup()                             # L290
      → algo.learn()                             # L300
```

多 GPU 下 `num_envs` 会按 GPU 数均分（L247-256）；`seed` 加上 rank 保证各 GPU 采样不同（L199-201）。

---

## 2. Observation：actor/critic 非对称

配置在 `config_values/wbt/g1/observation.py`。两个 group 都 `concatenate=True, history_length=1`。

### Actor obs（`actor_obs_shared`）—— 带噪声，对齐真实机器人可观测量

| Term | 函数 | per-env shape |
|---|---|---|
| `motion_command` | `joint_pos ⊕ joint_vel`（来自参考动作片段） | `2 × 29 = 58` |
| `motion_ref_ori_b` | 目标躯干姿态在基座系（6D：旋转矩阵前两列） | `6` |
| `base_ang_vel` | 基座角速度（基座系） | `3` |
| `dof_pos` | `dof_pos − default_dof_pos` | `29` |
| `dof_vel` | 关节速度 | `29` |
| `actions` | 上一步动作 | `29` |

**→ actor_obs = 154 维 / env。** 每个 term 带一个 `noise` 值（`dof_vel=0.5`、`base_ang_vel=0.2` …），训练时加上作为 sensor noise 的 domain randomization。

### Critic obs —— 无噪声 + privileged info

额外包括：`motion_ref_pos_b`(3)、`robot_body_pos_b`（14 个被追踪 body × 3 = 42）、`robot_body_ori_b`(14 × 6 = 84)、`base_lin_vel`(3)。
**≈ 283 维 / env**（精确值由 env 构造后的 `observation_manager.get_obs_dims()` 吐出）。

**关键**：actor 看不到 `base_lin_vel` 和全身 body 位姿 —— 这是标准的 asymmetric actor-critic：降低 value 的方差，但不把不可观测信号泄漏进可部署的 policy。

**进 PPO 的形状**：
- `obs_dict["actor_obs"]`: `(4096, 154)`, `torch.float32`
- `obs_dict["critic_obs"]`: `(4096, ~283)`, `torch.float32`

### 2.1 数据来源：OMOMO reference 还是 simulator 实测？

WBT 的每个 obs term 本质上都是「**reference**（从 OMOMO 重定向后的 .npz 查表得到）」或「**current state**（从 IsaacSim 物理引擎当前帧读出来的）」两类之一，或者两者的**相对量**（误差 / 相对变换）。

在 `managers/command/terms/wbt.py` 里，MotionCommand 把两类量严格分开：
- `## Robot from motion data` 段（L826-890）—— 全部从 `.npz` 按 `self.time_steps` 查表
- `## Robot from simulator` 段（L895-909）—— 全部从 `self._env.simulator` 实时读

| Obs term | 数据源 | 物理含义 |
|---|---|---|
| `motion_command` = `joint_pos ⊕ joint_vel` | **OMOMO** → `motion.joint_pos[t]`, `motion.joint_vel[t]`（`wbt.py:834-839`） | 当前 reference 帧的 29 维目标关节角 + 目标关节速度 |
| `motion_ref_ori_b` | **OMOMO - simulator 相对量** | `ref_quat_w`（reference 躯干姿态，世界系，来自 OMOMO）相对 `robot_ref_quat_w`（机器人当前躯干姿态，世界系，来自 simulator）的差，表达在机器人躯干系下 |
| `motion_ref_pos_b`（critic 独有） | **OMOMO - simulator 相对量** | 同上，reference 躯干位置相对当前躯干位置的 delta，在机器人躯干系下 |
| `base_ang_vel` | **simulator**（`simulator.robot_root_states[:, 10:13]` 旋到 base frame） | 基座当前角速度 |
| `base_lin_vel`（critic 独有） | **simulator**（`simulator.robot_root_states[:, 7:10]` 旋到 base frame） | 基座当前线速度 |
| `dof_pos` | **simulator**（`simulator.dof_pos − default_dof_pos`） | 当前关节角相对默认姿态的偏移 |
| `dof_vel` | **simulator**（`simulator.dof_vel`） | 当前关节速度 |
| `actions` | **policy 自身**（`action_manager.action`） | 上一步网络输出 |
| `robot_body_pos_b`（critic 独有） | **simulator**（14 个 body 的当前世界位置，转到躯干系） | 14 个被追踪 body 的当前相对位置 |
| `robot_body_ori_b`（critic 独有） | **simulator**（14 个 body 的当前世界姿态，转到躯干系） | 14 个被追踪 body 的当前相对姿态 |

一点容易混淆的地方：
- **`motion_command` 里的 `joint_pos/joint_vel` 是 reference，不是当前关节状态**。当前状态分别由 `dof_pos` 和 `dof_vel` term 提供。
- **所有 `*_ref_*` 名字的量都是 OMOMO**；所有 `robot_*` 前缀的量都是 **simulator 实测**。
- Reward 里也遵循同样的命名 —— 比如 `ref_pos_w` vs `robot_ref_pos_w` 的 `(ref − robot)` 误差就是 tracking loss。

### 2.2 Actor vs Critic 输入差别一览

| 维度 | Actor（部署可用） | Critic（训练特权） |
|---|---|---|
| 总维度 | **154** | **≈ 283** |
| `noise` | 全部 term 带噪声（`dof_vel=0.5`, `base_ang_vel=0.2`, …） | 全 0（干净值） |
| `motion_command`（ref joint pos/vel） | ✅ | ✅ |
| `motion_ref_ori_b`（ref 躯干姿态相对量） | ✅ | ✅ |
| `motion_ref_pos_b`（ref 躯干位置相对量） | ❌ | ✅ |
| `base_ang_vel`（当前基座角速度） | ✅ | ✅ |
| `base_lin_vel`（当前基座线速度） | ❌ | ✅ |
| `dof_pos`, `dof_vel` | ✅ | ✅ |
| `actions`（上一步动作） | ✅ | ❌ |
| `robot_body_pos_b`（14 body 当前相对位置，42 维） | ❌ | ✅ |
| `robot_body_ori_b`（14 body 当前相对姿态，84 维） | ❌ | ✅ |

设计意图：
- **Actor 只用真机上能拿到的量**（IMU → 基座角速度；编码器 → `dof_pos/vel`；ref 通过上位机给）。所以 `base_lin_vel`（真机上要估计，很难）、全身 body 世界位姿（需要 MoCap / SLAM）都**不给 actor**。
- **Critic 可以开天眼**：看 reference 位置、基座线速度、14 个 body 的完整当前位姿。这些只在 sim 里才拿得到，但 critic 只用来训练时估计 value function，部署时不需要，所以没关系。

### 2.3 Actor 和 Critic 的 output

| | Actor 输出 | Critic 输出 |
|---|---|---|
| Shape | `(B, 29)` | `(B, 1)` |
| 物理含义 | **29 个关节位置偏移量** —— 相对 `default_dof_pos` 的 delta；经 `action_scales` 放大后，加回 `default_dof_pos` 作为 PD 目标位置 q* | **标量 value function V(s)** —— 该状态下 critic 估计的折扣累计 reward |
| 分布 | 训练时采样自 `Normal(μ, σ)`（学到的对角 Gaussian），推理时直接用 `μ` | 确定性标量 |
| 下游 | 丢进 `JointPositionActionTerm.apply_actions` → PD 算力矩 → sim | 进 GAE 算 `Rₜ, Âₜ`；再进 PPO value loss |
| 是否导出 | **✅ 导出为 ONNX** (`ppo.py:379-385`)，给 `holosoma_inference` 部署用 | ❌ 训练完就扔，不导出 |

一个要点：**actor 的输出永远不是力矩**。网络输出 29 个浮点数 → `a_scaled = a × action_scales` → `q* = default_dof_pos + a_scaled` → PD law `τ = kp·(q* − q) − kd·q̇`。所以 policy 学的是"给 PD 一个什么样的目标位置"，真正把目标位置变成力矩的是 PD —— 这也是为什么 policy + PD（+ `default_dof_pos` + gains）必须一起部署。

---

## 3. 网络结构（`agents/modules/ppo_modules.py` + `config_values/algo.py`）

Actor 和 Critic 都是 **3 层 MLP，`hidden_dims=[512, 256, 128]`，activation 为 `ELU`**（`algo.py:46, 52`）。

### Actor（`PPOActor`，`ppo_modules.py:14-101`）

```
input  : (B, 154)     # actor_obs（经 EmpiricalNormalization 归一化）
MLP    : 154 → 512 → 256 → 128 → 29
output : mean μ ∈ (B, 29)
std    : nn.Parameter(init_noise_std * ones(29))   # WBT 里 init_noise_std = 1.0
dist   : π_θ = Normal(μ, σ)                         # 对角 Gaussian
sample : a ~ π_θ（训练） / a = μ（推理）
```

`init_noise_std = 1.0` 是 WBT override（`experiment.py:33`），比 locomotion 默认值偏大。

### Critic（`PPOCritic`，`ppo_modules.py:104-125`）

```
input  : (B, ~283)    # critic_obs
MLP    : ~283 → 512 → 256 → 128 → 1
output : V(s) ∈ (B, 1)
```

### `EmpiricalNormalization`（`ppo.py:40-101`）

对 obs 的 running mean / variance（Welford 算法；多 GPU 下 all-reduce），应用 `(x − μ) / (σ + ε)`。它是一个 `nn.Module`，但内部是 `@torch.no_grad()` —— **不参与 backprop**。只有 `.training` 时才更新；推理和 final-value 计算时关闭（见 `ppo.py:438` 的 `update=False`）。

---

## 4. Action → PD 力矩（"low-level controller" 的真身）

`apply_actions` 在 `managers/action/terms/joint_control.py:156-199`：

```
a_scaled = a * action_scales                           # (B, 29)
τ = kp_scale · P · (a_scaled + default_dof_pos − q)     # 位置式 PD（control_type='P'）
    − kd_scale · D · q̇
simulator.apply_torques_at_dof(τ)
```

- `action_scale = 0.25`，且 `action_scales_by_effort_limit_over_p_gain=True`：每关节标度变为 `0.25 × effort_limit / p_gain`，让弱关节也能饱和（`joint_control.py:326-337`）。
- 一次 env.step 内，PD 会在 4 个 physics substep 里反复打力矩（`base_task.py:417-425`，`control_decimation=4`）。
- **Policy 输出的是关节位置偏移量**（相对于 `default_dof_pos`），PD 把它变成力矩。所以 policy + 固定 PD 一起才构成一个 low-level controller。

---

## 5. Reward（per-step scalar，**无梯度**）

`base_task.py:500-501` 调用 `self.reward_manager.compute(self.dt)`，写入 `rew_buf: (num_envs,)`。

**Reward 聚合**（`managers/reward/manager.py:148-184`）：

```
rew[b] = Σ_term  term(env, **params)  × weight × dt
```

**WBT 的 term**（`config_values/wbt/g1/reward.py`）—— 全部是指数型 tracking reward + 正则项：

| Term | 公式 | weight | σ |
|---|---|---|---|
| `motion_global_ref_position_error_exp` | `exp(−‖ref_pos − robot_ref_pos‖² / σ²)` | **+0.5** | 0.3 |
| `motion_global_ref_orientation_error_exp` | `exp(−quat_err² / σ²)` | +0.5 | 0.4 |
| `motion_relative_body_position_error_exp` | 14 个 body 在躯干系下的位置误差 | **+1.0** | 0.3 |
| `motion_relative_body_orientation_error_exp` | 14 个 body 的姿态误差 | +1.0 | 0.4 |
| `motion_global_body_lin_vel` | body 线速度误差 | +1.0 | 1.0 |
| `motion_global_body_ang_vel` | body 角速度误差 | +1.0 | 3.14 |
| `penalty_action_rate` | `Σ (aₜ − aₜ₋₁)²` | **−0.1** | — |
| `limits_dof_pos` | 软关节限位越界量 | **−10** | — |
| `UndesiredContacts` | 非足 / 腕之外的 body 发生接触次数 | **−0.1** | — |

**关键**：reward 是纯数值（完全由 sim state 算出），**没有梯度流过它**。PPO 的梯度只来自 log-prob 和 value —— 见 §7。

---

## 6. Rollout 收集（`ppo.py:387-450`）

`num_steps_per_env = 24` 步，全程在 `torch.inference_mode()` 里 —— **不建计算图**：

```python
actor_obs  = cat(obs_dict[k] for k in actor_obs_keys)        # (4096, 154)
critic_obs = cat(obs_dict[k] for k in critic_obs_keys)       # (4096, ~283)
actor_obs  = normalize_actor_obs(actor_obs)                   # EmpiricalNorm（无 grad）
critic_obs = normalize_critic_obs(critic_obs)

actions = self.actor.act({"actor_obs": actor_obs})            # (4096, 29) sample
values  = self.critic.evaluate({"critic_obs": critic_obs}).detach()   # (4096, 1)

obs_dict, rewards, dones, infos = env.step({"actions": actions})
# rewards: (4096,), dones: (4096,)

# Timeout bootstrap：对被时间截断的 episode，把 γ · V(s_last) 加到最后一步的 reward 上
if infos["time_outs"].any():
    final_values  = critic(final_obs).detach()
    final_rewards += gamma * final_values * time_outs

storage.add(actor_obs, critic_obs, actions, values,
            actions_log_prob = actor.get_actions_log_prob(a).detach(),
            action_mean=μ.detach(), action_sigma=σ.detach(),
            rewards=(rewards + final_rewards).view(-1,1),
            dones=dones.view(-1,1))
```

24 步存完后，算 **GAE**（`_compute_returns_and_advantages`，L452-472）：

```
δₜ = rₜ + γ · V(sₜ₊₁) · (1 − doneₜ) − V(sₜ)
Âₜ = δₜ + γ · λ · (1 − doneₜ) · Âₜ₊₁
Rₜ = Âₜ + V(sₜ)
```

`γ = 0.99`、`λ = 0.95`（`algo.py:19-20`）。Advantages 按整个 batch standardize（多 GPU 下全局 standardize，L468）。

**Storage 布局**（`ppo.py:309-332`，`RolloutStorage`）：`[T=24, N=4096]`，做 mini-batch 时 flatten 为 `[T·N = 98 304, D]`：

| Key | Shape | Dtype |
|---|---|---|
| `actor_obs` | `(T, N, 154)` | float |
| `critic_obs` | `(T, N, ~283)` | float |
| `actions` | `(T, N, 29)` | float |
| `rewards / dones / values / returns / advantages / actions_log_prob` | `(T, N, 1)` | — |
| `action_mean / action_sigma` | `(T, N, 29)` | float |

---

## 7. Training step —— loss 与 backprop 在哪（`ppo.py:474-621`）

`_training_step` 跑 `num_learning_epochs × num_mini_batches` 次更新：

```
for _ in range(5):                      # num_learning_epochs = 5（WBT override）
    for minibatch in split(storage, 4): # num_mini_batches = 4 → mbs ≈ 24 576
        _update_algo_step(minibatch)
```

**`_compute_ppo_loss` —— loss 本体：**

```python
self.actor.act({"actor_obs": actor_obs})                     # 重新建分布（这次带 grad）
value_batch          = self.critic.evaluate({"critic_obs": critic_obs})
log_prob_new         = self.actor.get_actions_log_prob(actions_batch)
μ_new, σ_new, H_new  = self.actor.action_mean, action_std, entropy

# ---- PPO-clip actor（surrogate）loss ----
ratio        = exp(log_prob_new − old_log_prob)
surr         = −A · ratio
surr_clip    = −A · clamp(ratio, 1−ε, 1+ε)            # ε = clip_param = 0.2
surrogate_loss = max(surr, surr_clip).mean()

# ---- clipped value loss ----
v_clipped   = old_v + clamp(v_new − old_v, −ε, ε)
value_loss  = max((v_new − R)², (v_clipped − R)²).mean()

# ---- entropy bonus ----
entropy_loss = H_new.mean()

actor_loss  = surrogate_loss − entropy_coef · entropy_loss
                             + symmetry_actor_coef  · symmetry_actor_loss
critic_loss = value_loss_coef · value_loss + symmetry_critic_coef · symmetry_critic_loss
```

**`_update_algo_step` —— backprop 就在这里（L488-505）：**

```python
actor_optimizer.zero_grad(); critic_optimizer.zero_grad()
ppo_loss = actor_loss + critic_loss
ppo_loss.backward()                                   # ← 整个训练里唯一一次 .backward()
if multi_gpu: _reduce_parameters()                    # all-reduce gradients
nn.utils.clip_grad_norm_(actor.parameters(),  max_grad_norm)   # = 1.0
nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
actor_optimizer.step()
critic_optimizer.step()
```

**Optimizer**：`AdamW`，actor 和 critic **分开**，LR 各 `1e-3`（`experiment.py:34-35`），`weight_decay=0.0`（WBT override，`experiment.py:38-40`）。

**KL-adaptive LR**（`ppo.py:637-648`，`desired_kl=0.01`）：
- `kl > 2·desired_kl` → LR /= 1.5
- `kl < desired_kl / 2` → LR *= 1.5
- 夹在 `[min_lr, max_lr]` 内

---

## 8. 梯度流向示意

```
reward（sim）           ──┐                ← 纯数值，无梯度
actions, log_prob, V    │ storage（.detach()） ← 在 inference_mode 下产生
                        │
                        ▼
minibatch ───────────▶ actor(actor_obs) ──▶ μ, σ ──▶ Normal ──▶ log_prob_new
                        │                                  └──▶ entropy
                        │  （梯度经 surrogate_loss & entropy_loss 回到 actor MLP 权重）
                        │
                        └──▶ critic(critic_obs) ──▶ V_new
                                                 （梯度经 value_loss 回到 critic MLP 权重）

actor_loss + critic_loss = ppo_loss  ──► .backward()  ──► grad_clip ──► AdamW.step()
```

**除此之外的一切都不带梯度。** `EmpiricalNormalization`、`env.step`、reward terms、PD 力矩转换都在 autograd graph 之外。梯度只活在 actor 和 critic 两张 MLP 内部。

---

## 9. 关键 hyperparameters（`g1_29dof_wbt` 生效值）

| 参数 | 值 | 来源 |
|---|---|---|
| `num_envs` | 4096 | `experiment.py:22` |
| `num_steps_per_env` | 24 | `algo.py:33` |
| `num_learning_iterations` | 30 000 | `experiment.py:29` |
| `num_learning_epochs` | 5 *(override)* | `experiment.py:30` |
| `num_mini_batches` | 4 | `algo.py:17` |
| `gamma` | 0.99 | `algo.py:19` |
| `lam`（GAE λ） | 0.95 | `algo.py:20` |
| `clip_param` (ε) | 0.2 | `algo.py:18` |
| `entropy_coef` | 0.005 *(override)* | `experiment.py:32` |
| `value_loss_coef` | 1.0 | `algo.py:21` |
| `max_grad_norm` | 1.0 | `algo.py:27` |
| `actor_lr` / `critic_lr` | 1e-3 / 1e-3 *(override)* | `experiment.py:34-35` |
| `schedule` | `adaptive`（KL-based） | `algo.py:28` |
| `desired_kl` | 0.01 | `algo.py:29` |
| `init_noise_std` | 1.0 *(override)* | `experiment.py:33` |
| `empirical_normalization` | True *(override)* | `experiment.py:37` |
| `use_symmetry` | False（WBT 关闭） | `experiment.py:38` |
| `save_interval` | 4000 iters | `experiment.py:31` |
| Actor/Critic MLP | `[512, 256, 128]`, ELU | `algo.py:46, 52` |
| `weight_decay` | 0.0 *(override)* | `experiment.py:39-40` |
| `control_decimation × sim fps` | 4 × 200 Hz → 50 Hz 控制 | `simulator.py:40-41` |
| Episode 长度 | 10 s ≈ 500 env steps | `experiment.py:49` |

---

## 10. 一次 iteration 端到端（用数字走一遍）

1. **Rollout（不建图）** —— `24 × 4096 = 98 304` 条 transition；每次 env.step 内部跑 4 个 physics substep + PD。
2. **GAE** —— 对整个 batch 算 `Rₜ, Âₜ`，再对 `Âₜ` standardize。
3. **Training** —— `98 304 / 4 = 24 576` 条 / mini-batch；`5 × 4 = 20` 次 SGD 更新；每次 = 一轮 `actor.forward + critic.forward + backward + grad_clip(1.0) + AdamW.step`。
4. **KL-adaptive LR** —— 本轮 batch-mean KL > 0.02 则两个 LR 都 /1.5；< 0.005 则 ×1.5；夹到 `[min, max]`。
5. **每 4 000 iters** —— rank 0 保存一个 `.pt`，**并导出 `.onnx`**（`ppo.py:379-385`）。`holosoma_inference/run_policy.py` 就是加载这份 ONNX 做 sim-to-sim 和真机部署。

---

## 11. 容易踩坑的点

- **两个独立 optimizer**。Actor 和 critic 各自一个 AdamW。仍然只调用一次 `ppo_loss.backward()` —— 因为 `actor_loss` 的图里没有 critic 参数，反之亦然，两者分开 step 不冲突。
- **Reward 乘以 `dt`**（`manager.py:171`）—— `dt ≈ 0.02 s`（50 Hz）。所以 `weight=+1.0` 的 term 每步实际贡献 ≈ 0.02，一局 10 s 累计到 ≈ 10。这决定了 value head 的量级，也是 PPO 稳定训练的前提。
- **Timeout bootstrap**（`ppo.py:407-413`）。Truncation ≠ termination —— 不 bootstrap `γ · V(s_last)` 到最后一步 reward 里，长 horizon 的 value 估计会被系统性低估。
- **`EmpiricalNormalization` 是 buffer 不是 parameter** —— 会写进 checkpoint，加载时还原（`ppo.py:656-659`）。`.eval()` 时停止更新（L337）。
- **WBT 里 symmetry augmentation 是关着的**（`use_symmetry=False`，`experiment.py:38`），所以 `ppo.py:527-547` 那段是 no-op。打开后 batch 会翻倍，额外加一项 MSE symmetry loss。
- **IsaacGym 的 import 顺序很脆弱**。`init_sim_imports` 必须在 `import torch` 之前运行 —— 训练的 context manager 保证了这一点。

---

## 12. 想再往下挖？

- **`MotionCommand.step()`** —— adaptive timesteps sampler 怎么从参考动作里挑片段：`managers/command/terms/wbt.py:743+`
- **FastSAC 变体**（`exp:g1-29dof-wbt-fast-sac`）—— distributional critic，off-policy，完全不同的 loss：`agents/fast_sac/`
- **ONNX 导出** —— 最终部署用的是什么：`ppo.py:820+` 和 `utils/inference_helpers.py`

---

## 附录：文件地图

| 文件 | 作用 |
|---|---|
| `src/holosoma/holosoma/train_agent.py` | 入口；tyro 解析；sim-app 生命周期；distributed 初始化 |
| `src/holosoma/holosoma/config_values/experiment.py` | 注册 `exp:g1-29dof-wbt` 等 subcommand |
| `src/holosoma/holosoma/config_values/wbt/g1/experiment.py` | 具体 WBT recipe（"这次实验的配置"） |
| `src/holosoma/holosoma/config_values/algo.py` | PPO 默认值（MLP 形状、clip、gamma、lam …） |
| `src/holosoma/holosoma/envs/base_task/base_task.py` | `step / _pre_physics_step / _physics_step / _post_physics_step`；reward/termination 分发 |
| `src/holosoma/holosoma/envs/wbt/wbt_manager.py` | WBT 专用 env（断言 IsaacSim、motion-command 钩子） |
| `src/holosoma/holosoma/managers/command/terms/wbt.py` | `MotionCommand` —— 参考轨迹回放 |
| `src/holosoma/holosoma/managers/action/terms/joint_control.py` | PD "low-level controller"：`process_actions`、`apply_actions` |
| `src/holosoma/holosoma/managers/observation/terms/wbt.py` | 所有 WBT observation terms |
| `src/holosoma/holosoma/managers/reward/terms/wbt.py` | 所有 WBT reward terms（指数型 + 正则） |
| `src/holosoma/holosoma/managers/reward/manager.py` | 加权求和 `Σ term × weight × dt` |
| `src/holosoma/holosoma/agents/ppo/ppo.py` | `PPO.learn`、`_rollout_step`、GAE、`_compute_ppo_loss`、`.backward()` |
| `src/holosoma/holosoma/agents/modules/ppo_modules.py` | `PPOActor` / `PPOCritic`（MLP 的外包装） |
| `src/holosoma/holosoma/agents/modules/modules.py` | `BaseModule` —— 真正的 MLP builder |
