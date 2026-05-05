# WBT Wrist-Force v10 Walkthrough Report

<!-- 元数据 -->
| 字段 | 值 |
|------|-----|
| branch | `wbt-wrist-force-v10-training` |
| base | `origin/main` |
| commit range | `c16e5c9..70d89ad`（19 commits） |
| 报告日期 | 2026-05-04 |
| 总 LOC（branch 净增） | +3944 行（39 个文件） |
| CPU 测试函数数 | **85 个**（13 个测试文件，全部纯 CPU，无需 GPU / IsaacSim） |

---

## 1. TL;DR（一屏摘要）

1. **新任务**：`g1-29dof-wbt-force` — Whole-Body Tracking + 双腕外力注入，用 virtual-impedance compliance reward 替代纯运动学跟踪。
2. **零回归**：基线 WBT env 类、command / observation / reward presets 全部未修改；新能力通过子类 + 新 preset 组合实现（composition over modification）。
3. **新入口**：CLI selector `exp:g1-29dof-wbt-force` + 训练 launcher `demo_scripts/demo_wbt_wrist_force_training.sh`。
4. **85/85 CPU 单元测试全绿**；IsaacSim e2e test（`tests/e2e/test_wbt_wrist_force_e2e.py`，`@pytest.mark.isaacsim`）及 Spike 1（`tests/spikes/test_wrist_only_force_injection.py`）已延期至 training-host CI 执行。

---

## 2. 背景与问题定义

**基线 WBT 的局限**：`g1_29dof_wbt`（`WholeBodyTracking` env）在每个控制步只追踪运动捕捉参考轨迹的运动学目标（关节角、末端执行器位置），不施加任何外力，也没有让策略感知或产生接触力的信号。策略无法学习"被推时如何顺应"这一行为。

**本 branch 加了什么**：在每个 env 的每个控制步，向两个腕部刚体（`left_wrist_yaw_link` / `right_wrist_yaw_link`）注入两路 6-D wrench：
- **F_cmd**（body-yaw frame）：RL policy 被要求产生的腕力指令，actor + critic 均可见。
- **F_ext**（world frame）：由仿真器直接施加的随机外部干扰力，仅 critic 可见（privileged info）。

两路信号都由梯形状态机（`COOLDOWN → RAMP_UP → HOLD → RAMP_DOWN → COOLDOWN`）驱动，以概率 `force_*_activation_prob_per_step` 触发。

**Virtual-impedance reward**：tracking reward 目标点根据合力做偏移：
```
target_shifted = motion_target + (F_ext + F_cmd_world) / K_virtual
reward = exp(-||wrist_actual - target_shifted||².mean() / σ²)
```
`K_virtual`（100 N/m，v1 为常数）是"虚弹簧"刚度，它的含义是：如果你施 30 N 的力，运动目标就自动平移 30/100 = 0.3 m，reward 仍然满分。这样策略无需真实接触仿真，却能学会"力到哪里，手就顺应哪里"。

---

## 3. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│ Config / 数据契约 (Tasks 0-1)                                     │
│  _k_virtual.py → K_VIRTUAL_RANGE_N_PER_M = (100.0, 100.0)       │
│  config_types/command.py → WristComplianceConfig (dataclass)     │
└────────────────────────┬─────────────────────────────────────────┘
                         │ 实例化
┌────────────────────────▼─────────────────────────────────────────┐
│ Command 层 (Task 2)                                               │
│  WristComplianceCommand                                          │
│   ├── F_cmd: TemporalLerp 梯形状态机 (body-yaw frame)            │
│   ├── F_ext: TemporalLerp 梯形状态机 (world frame)               │
│   ├── k_virtual: per-env per-wrist 标量                          │
│   └── update_metrics() → 14 个 force/* wandb keys               │
└──────────┬──────────────────────────┬────────────────────────────┘
           │ get_state()              │ get_state()
┌──────────▼──────────┐  ┌───────────▼──────────────────────────┐
│ Observation 层       │  │ Env 层 (Task 3)                       │
│ (Task 4, 3 个 terms) │  │ WholeBodyTrackingForceInjected        │
│  wrist_force_command │  │  ├── _apply_force_in_physics_step()  │
│  wrist_force_ext_*   │  │  │    apply F_ext via sim API        │
│  wrist_virtual_k     │  │  └── _update_log_dict()              │
└──────────────────────┘  │       调 update_metrics() +          │
                          │       6 个 env-owned force/* keys    │
┌─────────────────────────▼──────────────────────────────────────┐
│ Reward 层 (Task 5)                                              │
│  wrist_force_position_tracking_exp                             │
│   target_shifted = motion_target + F_total / K_virtual        │
│   return exp(-sq_err.mean() / σ²)                             │
└─────────────────────────┬──────────────────────────────────────┘
                          │ extras["to_log"] / log_dict
┌─────────────────────────▼──────────────────────────────────────┐
│ Wandb metrics pipeline (Task 9.5)                              │
│  PPO._post_epoch_logging  /  FastSAC.logging_helper            │
│   → 14 cmd-owned + 6 env-owned = 20 force/* keys visible      │
│     in wandb dashboard                                         │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. 逐阶段 Walkthrough

### 4.1 Phase 0–1（Tasks 0–1）：数据契约

#### Task 0 — K_VIRTUAL 常量（`_k_virtual.py`）

**WHAT**：新建常量文件，定义 v1 virtual spring stiffness 为 100 N/m。  
**WHY**：把 reward 超参从 `reward_force.py` 或 `command.py` 中解耦出来，当 v2 需要扩展成范围采样时只改一处。  
**WHERE**：`src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py:17-20`

```python
K_VIRTUAL_RANGE_N_PER_M: tuple[float, float] = (100.0, 100.0)
G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M: float = K_VIRTUAL_RANGE_N_PER_M[0]
```

v1 两端点相等 → per-env/per-wrist 采样退化为常数 100.0，无随机性。  
**TESTS**：`src/holosoma/tests/config_values/wbt/g1/test_k_virtual.py`（4 个 test function）— 验证 lo == hi，lo > 0，alias 与 lo 相等。

---

#### Task 1 — `WristComplianceConfig` 数据类（`config_types/command.py:140`）

**WHAT**：在 `command.py` 追加 `WristComplianceConfig` dataclass，声明 F_cmd / F_ext 两路信号的全部超参。  
**WHY**：用 tyro-serializable dataclass 统一持久化与 CLI override 路径（`--command.setup_terms.wrist_compliance_command.params.wrist_compliance_config.k_virtual_range "[50,300]"`）。  
**WHERE**：`config_types/command.py:140-197`  
关键字段（注：`config_types/command.py` 140 行起）：
```
force_cmd_magnitude_range: (5.0, 30.0)   # N
force_ext_magnitude_range: (0.0, 30.0)   # N
k_virtual_range: K_VIRTUAL_RANGE_N_PER_M  # 从 _k_virtual.py 导入
enable_left / enable_right: True          # 可单独关闭某侧
left_wrist_body_name / right_wrist_body_name: "left/right_wrist_yaw_link"
```
`__post_init__` 做严格断言（k_virtual_range lo > 0，lo ≤ hi 等）。  
**TESTS**：`config_types/tests/test_wrist_compliance_config.py`（11 个 test function）— 验证默认值、断言路径、各范围不变式。

---

### 4.2 Phase 2（Tasks 2–3）：运行时管道

#### Task 2 — `_gh_utils.py` + `WristComplianceCommand`

**WHAT**：  
- `_gh_utils.py`：从 `third_party/gentle-humanoid-training` 选取性复制三个原语（`TemporalLerp`, `clamp_norm`, `random_uniform`），保留零运行时依赖（文件内只 import `torch` + `typing`）。  
- `wbt_force.py`：实现 `WristComplianceCommand`（CommandTermBase 子类，~467 行），内部维护两套梯形状态机，驱动 `F_cmd`（body-yaw frame）和 `F_ext`（world frame）各自独立的 magnitude + duration + cooldown 采样。  

**WHY**：将"力信号生成"隔离到 Command 层，与 Env 层的"力注入"解耦；`update_metrics()` 在每个 epoch 输出 14 个标量到 wandb，而不是散布在 env 里。  

**WHERE**：  
- `managers/command/terms/_gh_utils.py`（全文件）  
- `managers/command/terms/wbt_force.py:303`（`WristComplianceCommand` 类定义）  
- `wbt_force.py:407`（`update_metrics` 方法）

`update_metrics()` 输出的 14 个 command-owned keys：
```python
metrics["force/cmd_magnitude_l"]    # [N]
metrics["force/cmd_magnitude_r"]
metrics["force/cmd_magnitude_max"]
metrics["force/ext_magnitude_l"]
metrics["force/ext_magnitude_r"]
metrics["force/ext_magnitude_max"]
metrics["force/cmd_ext_alignment"]  # cosine(F_cmd, F_ext)，两腕平均
metrics["force/k_virtual_l"]
metrics["force/k_virtual_r"]
metrics["force/active_frac_cmd"]
metrics["force/active_frac_ext"]
metrics["force/phase_ramp_up"]
metrics["force/phase_hold"]
metrics["force/phase_ramp_down"]
```

**TESTS**：`managers/command/terms/tests/test_wbt_force_command.py`（13 个 test function）— 覆盖状态机触发、magnitude 边界、梯形 ramp 单调性、k_virtual 采样等。

---

#### Task 3 — `WholeBodyTrackingForceInjected` env 子类

**WHAT**：继承基线 `WholeBodyTracking`，在 physics step 中通过 `set_external_force_and_torque` API 注入外力，并扩展 `_update_log_dict` 计算 env-owned 指标。  
**WHY**：遵循 "ship by composition" 原则——基线 env 类零修改，新能力全部在子类里。子类只覆盖两个方法。  
**WHERE**：`envs/wbt/wbt_force_injected.py:74`（`_apply_force_in_physics_step`）、`wbt_force_injected.py:219`（`_update_log_dict`）

`_update_log_dict` 在 command-owned metrics 之外再追加 6 个 env-owned keys：
```python
"force/wrist_target_shift_l/r"   # K_virtual 导致的目标点偏移量 [m]
"force/wrist_pos_error_l/r"      # 实际位置 vs. shifted target 的 L2 误差
"force/applied_f_body_l/r"       # 实际施加力的 norm（world→body 旋转 sanity）
```

**TESTS**：`envs/wbt/tests/test_wbt_force_injected.py`（9 个 test function）— mock simulator，验证 `_update_log_dict` keys 存在、shape 正确、力只施加在腕部（红线 1）。

---

### 4.3 Phase 3（Tasks 4–5）：信号层

#### Task 4 — 三个观测 terms

**WHAT**：新建 `managers/observation/terms/wbt_force.py`，定义三个函数供 obs preset 注册。  
**WHY**：把 F_cmd / F_ext / K_virtual 暴露给 actor + critic，policy 才能"感知到"自己被要求产生什么力。  
**WHERE**：`managers/observation/terms/wbt_force.py:37-52`

```python
def wrist_force_command(env) -> Tensor:      # [N, 6]  actor+critic
def wrist_force_ext_privileged(env) -> Tensor: # [N, 6]  critic-only
def wrist_virtual_stiffness_command(env) -> Tensor: # [N, 2]  actor+critic
```

三者均从 `command_manager.get_state("wrist_compliance_command")` 读取，调用 `.clone()` 防止 obs history buffer 污染 command 状态。  
**TESTS**：`managers/observation/terms/tests/test_wbt_force_obs.py`（5 个 test function）— mock command term，验证 shape、clone 行为、缺少 command term 时的 RuntimeError。

---

#### Task 5 — `wrist_force_position_tracking_exp` reward

**WHAT**：实现 v10 reward 公式，返回 shape `[N]` 的 per-env 奖励值，值域 `[0, 1]`。  
**WHY**：使 reward 对腕部外力"免疫"——有外力时目标自动平移，策略不需要抗拒力就能得满分。  
**WHERE**：`managers/reward/terms/wbt_force.py:94`（函数定义）、`wbt_force.py:125-133`（核心公式）

```python
k_virtual = wrist_cmd.k_virtual.clamp(min=_K_FLOOR).unsqueeze(-1)  # [N,2,1]
shift = f_total_w / k_virtual                    # virtual spring displacement
target_shifted_w = motion_target_w + shift       # shifted tracking target
sq_err = torch.sum((target_shifted_w - wrist_actual_w)**2, dim=-1)  # [N,2]
return torch.exp(-sq_err.mean(dim=-1) / (sigma**2))
```

sigma 默认 0.3 m（preset 里可 CLI 覆盖），weight = 2.0。`_K_FLOOR = 1e-3` 防止除零。  
**TESTS**：`managers/reward/terms/tests/test_wbt_force_reward.py`（9 个 test function）— 纯 CPU 张量测试，验证零力时公式退化为标准 WBT reward、正力时目标平移量正确、NaN-free。

---

### 4.4 Phase 4（Tasks 6–8）：Preset 组合

#### Task 6 — `command_force` preset（`config_values/wbt/g1/command_force.py`）

**WHAT**：用 `replace(g1_29dof_wbt_command, ...)` 在基线 command preset 的 setup/reset/step_terms 字典里追加 `"wrist_compliance_command"` 一项，不改任何已有条目。  
**WHERE**：`config_values/wbt/g1/command_force.py:31-43`  
**TESTS**：`src/holosoma/tests/config_values/wbt/g1/test_command_force_preset.py`（5 个 test function）。

#### Task 7 — `observation_force` preset（`config_values/wbt/g1/observation_force.py`）

**WHAT**：`actor_obs` group 在基线 6 个 term 之上追加 `wrist_force_command`（scale=1.0）和 `wrist_virtual_stiffness_command`（scale=0.01）；`critic_obs` 再额外追加 `wrist_force_ext_privileged`（scale=1.0，privileged）。两组均 `history_length=10`。  
**WHERE**：`config_values/wbt/g1/observation_force.py:42-82`  
期望维度：`actor_obs` = 162 per step × 10 = **1620**；`critic_obs` = 300 × 10 = **3000**。  
**TESTS**：`test_observation_force_preset.py`（7 个 test function）。

#### Task 8 — `reward_force` preset（`config_values/wbt/g1/reward_force.py`）

**WHAT**：`replace(g1_29dof_wbt_reward, terms={**baseline_terms, "wrist_force_position_tracking_exp": ...})` 添加一个 reward term，weight=2.0，sigma=0.3。  
**WHERE**：`config_values/wbt/g1/reward_force.py:14-27`  
**TESTS**：`test_reward_force_preset.py`（4 个 test function）。

---

### 4.5 Phase 5（Task 9）：实验注册

#### Task 9 — `g1_29dof_wbt_force` 实验 preset + tyro 注册

**WHAT**：在 `config_values/wbt/g1/experiment.py` 用 `replace(g1_29dof_wbt, ...)` 构造 `g1_29dof_wbt_force`，换掉 `env_class`、`command`、`observation`、`reward` 四个字段，保留 `simulator`（仍绑 `isaacsim`）、`terrain`、`algo` 不变。在 `config_values/experiment.py:24` 加入 `DEFAULTS["g1_29dof_wbt_force"]`，同样在 `config_values/command.py:19`、`observation.py:16`、`reward.py:23` 各注册一行。

注册后 tyro CLI 可解析：
```bash
python train_agent.py exp:g1-29dof-wbt-force simulator:isaacsim logger:wandb
```
（注：`g1_29dof_wbt_force` → `exp:g1-29dof-wbt-force`，下划线换连字符由 `{k.replace('_', '-')` 完成，见 `config_values/experiment.py:31`）

**WHERE**：`config_values/wbt/g1/experiment.py`（`g1_29dof_wbt_force = replace(...)` 块）；`config_values/experiment.py:11,24`  
**TESTS**：`test_experiment_force_preset.py`（7 个 test function）— 验证 env_class 字符串正确、command/obs/reward preset 是新 force 版本、simulator 未被替换。

---

### 4.6 Phase 5.5（Tasks 9.5 + 13.5）：Wandb 指标 + ONNX schema

#### Task 9.5 — PPO / FastSAC wandb 指标接入

**WHAT**：让 `_update_log_dict` 生成的 `log_dict`（20 个 `force/*` key）在 PPO 的 `extra_log_dicts` 路径 / FastSAC 的 `extras["to_log"]` 路径进入 wandb dashboard，无需改 logging infra。  
**WHY**：两个 agent 已有不同的 logging helper 接口，这里确认两者都能传递 env 级别的 `log_dict`。  
**WHERE**：  
- `agents/ppo/ppo.py:763-771`（`extra_log_dicts` 构造处，+8 行）  
- `agents/fast_sac/fast_sac_agent.py:89-102`（`extras["to_log"]` 路径，+9 行）

**TESTS**：`src/holosoma/tests/test_wbt_force_wandb_metrics.py`（218 行，6 个 test function）— 用 mock env 验证 `COMMAND_OWNED_KEYS`（14 个）和 `ENV_OWNED_KEYS`（6 个）全部出现在 `log_dict` 里，且 tensor shape/dtype 符合约定。

---

#### Task 13.5 — ONNX metadata schema 扩字段

**WHAT**：在 `ppo.py:730-746` 和 `fast_sac_agent.py` 对应位置的 `attach_onnx_metadata` 调用里追加三个新字段：`history_length`、`obs_term_names_sorted`、`obs_group_dims`。这些字段在 Phase 8 inference 侧用于训练/部署 obs 漂移检测。  
**WHY**：ONNX ckpt 是训练产物，必须在训练时就写入 schema，否则 Phase 8 部署测试需要重训（见计划 §6 Phase 5.5 说明）。  
**WHERE**：`agents/ppo/ppo.py:730-746`  
**TESTS**：`src/holosoma/tests/utils/test_onnx_metadata_schema.py`（2 个 test function）— `test_schema_fields_round_trip` 验证三字段 JSON round-trip；`test_attach_on_baseline_style_metadata_does_not_break_onnx_load` 验证旧 ckpt（不含新字段）加载不崩。

---

### 4.7 Phase 6（Tasks 10–13）：质量门

#### Task 10 — tyro CLI smoke test（`tests/test_wbt_force_cli.py`，3 个 test function）

验证 tyro 能解析 `exp:g1-29dof-wbt-force`、`command:g1-29dof-wbt-force`、`observation:g1-29dof-wbt-force`、`reward:g1-29dof-wbt-force` 选择器，不抛异常，且返回正确的 Python 对象类型。

#### Task 11 — IsaacSim e2e test（`tests/e2e/test_wbt_wrist_force_e2e.py`，`@pytest.mark.isaacsim`）

标记为 `isaacsim`，**已延期至 training-host CI**。测试内容：用强制激活（`force_*_activation_prob_per_step=1.0`）的 `g1_29dof_wbt_force` 启动 2-env，验证外力只施加在腕部（红线 1）、obs shape 1620/3000、k_virtual=100.0、reward 在 [0,1] 无 NaN。延期原因：沙盒环境缺少 `carb` / `AppLauncher`（IsaacSim 启动链），执行 `@pytest.mark.isaacsim` 标记的测试需在有完整 IsaacSim 安装的 training-host 上以 `pytest -m isaacsim` 运行。

#### Task 12 — 训练用户文档（`docs/wbt-wrist-force-training.md`）

详见文档本身。内容包含 canonical 训练命令、关键超参 CLI 覆盖、Phase 7 TRAIN GATE 指标解读、调参 debug 触发点。本报告 §9 给出完整复现步骤，不重复文档。

#### Task 13 — `cfg.command` None guard

`tests/e2e/test_wbt_wrist_force_e2e.py:54` 在 `replace` 链之前加 `assert cfg.command is not None`，修复 mypy `Optional[CommandManagerCfg]` union-attr 报错（commit `2bc8f73`）。严格来说 preset 总是非 None，但显式断言让类型检查通过。

---

### 4.8 Phase 7（Task 13.8）：TRAIN GATE

Phase 7 是"训练做完才做部署"的**硬门**。下面 8 条指标在 wandb 全绿后才能进 Phase 8。

> 详见 `demo_scripts/demo_wbt_wrist_force_training.sh` 的 `TRAIN GATE` banner + `docs/plans/2026-05-03-wbt-wrist-force-v10.md §6 Phase 7 Task 13.8`。

---

## 5. Baseline vs. 新路径对比

| 维度 | `demo_omomo_wb_tracking.sh` | `demo_wbt_wrist_force_training.sh` |
|------|-----------------------------|------------------------------------|
| **Retargeting 步骤** | 需要：SMPL-H → robot (`robot_retarget.py`) | **跳过** — motion file 随仓库发布，preset 默认指向它 |
| **Data conversion** | 需要：`.npz` → MuJoCo format (`convert_data_format_mj.py`) | **跳过** — 同上，converted file 已在 repo |
| **`exp` 选择器** | `exp:g1-29dof-wbt` | `exp:g1-29dof-wbt-force` |
| **Env 类** | `WholeBodyTracking` | `WholeBodyTrackingForceInjected`（子类） |
| **Simulator default** | `isaacsim`（WBT preset 绑定） | `isaacsim`（同，继承自 `g1_29dof_wbt`） |
| **可覆盖 env vars** | （无，hardcoded `CONVERTED_FILE`） | `SIMULATOR`, `LOGGER`, `SEED`, `MOTION_FILE`, `SKIP_REINSTALL` |
| **额外 reward term** | 无 | `wrist_force_position_tracking_exp`（weight=2.0） |
| **额外 obs（actor）** | 无 | `wrist_force_command`（+6 dim）+ `wrist_virtual_stiffness_command`（+2 dim） |
| **额外 obs（critic）** | 无 | 同 actor + `wrist_force_ext_privileged`（+6 dim） |

**为什么跳过 retargeting / conversion**：`command_force.py` 的 `wrist_compliance_command` 沿用基线 motion command，其默认 `motion_file` 路径指向仓库内已有的 `g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz`（已经过 retargeting + conversion）。不需要重跑。  
**为什么 simulator 默认 isaacsim**：`config_values/wbt/g1/experiment.py` 里 `g1_29dof_wbt_force = replace(g1_29dof_wbt, ...)` 保留了基线的 `simulator=replace(simulator.isaacsim, ...)` 绑定，所以 CLI 省略 `simulator:` 选择器时自动选 isaacsim。

---

## 6. Wandb TRAIN GATE 速查表

以下 8 条指标**同时满足**才算 Phase 7 通过（来源：`docs/plans/2026-05-03-wbt-wrist-force-v10.md §6 Phase 7 Task 13.8`）：

1. `Train/mean_reward` 前 200 iter 单调上升 → 稳定在正区间（不崩）
2. `Episode/rew_wrist_force_position_tracking_exp` 随训练上升到 **≥ 0.4**（`exp(-err/σ²)` ≥ 0.4 对应稳态 wrist 跟踪误差约 σ=0.3m 的一半）
3. `Env/force/active_frac_cmd` + `Env/force/active_frac_ext` 稳态 ∈ [0.2, 0.5]（激活采样率对）
4. `Env/force/k_virtual_l` / `Env/force/k_virtual_r` ≡ **100.0**（v1 配置正确；非平线说明 reset 采样 broken）
5. `Env/force/cmd_magnitude_max` ≤ 30 N（不爆）
6. `Env/force/applied_f_body_l` / `applied_f_body_r` **等于** `Env/force/ext_magnitude_l/r`（sanity：world→body quat 转换正确；不等说明 Task 3 的 `quat_apply_inverse` 有 bug）
7. 基线 `Episode/rew_motion_tracking_*` **不塌陷**（虚拟弹簧 reward 不能把 motion tracking 挤掉）
8. actor/critic loss 不 diverge；`Episode/rew_survival` 不塌（G1 没频繁摔倒）

**Gate 不过时调参入口**（详见 `docs/wbt-wrist-force-training.md`）：
- reward 停滞 → 增大 MLP 或调大 `history_length`
- `active_frac` 过低 → `force_ext_activation_prob_per_step` 0.01 → 0.02
- `applied_f_body ≠ ext_magnitude` → 检查 Task 3 body quat wxyz 顺序
- motion tracking 塌 → 降 `wrist_force_position_tracking_exp.weight` 2.0 → 1.0

---

## 7. 哪些文件没有改动（回归保护）

以下文件在本 branch **全部保持原样**（通过 `git diff origin/main..HEAD -- <file>` 验证为空）：

- `envs/wbt/wbt_manager.py`（基线 `WholeBodyTracking` env 类）
- `config_values/wbt/g1/command.py`（基线 `g1_29dof_wbt_command` preset）
- `config_values/wbt/g1/observation.py`（基线 `g1_29dof_wbt_observation` preset）
- `config_values/wbt/g1/reward.py`（基线 `g1_29dof_wbt_reward` preset）
- `config_values/wbt/g1/experiment.py` 中的 `g1_29dof_wbt` 实例本身（只追加 `g1_29dof_wbt_force`）
- `agents/callbacks/` 目录（IsaacGym callback 顺序敏感，见 commit `470fd78`，本 branch 未触碰）
- `demo_scripts/demo_omomo_wb_tracking.sh`（基线 demo launcher，仅作为对比参考）

**原则**：通过组合（`replace(baseline, ...)`）而非修改实现扩展。所有新能力（env 子类、preset 扩展、command term 追加）均以"插件"方式叠加在基线之上，基线 test suite 的回归保障完全不受影响。

---

## 8. 已知延期项

### Spike 1 — `tests/spikes/test_wrist_only_force_injection.py`

验证仿真器 `set_external_force_and_torque` API 对"只写腕部"的幂等性（加力后其他刚体不受影响）。标记为 spike（探针测试），不计入 CPU 测试集，需在有完整 IsaacSim 环境的 training-host 上手动验证。

### Task 11 — `tests/e2e/test_wbt_wrist_force_e2e.py`

标记 `@pytest.mark.isaacsim`，沙盒执行时报 `No module named carb`（IsaacSim 的 C++ 核心模块，只有完整 IsaacSim 安装才有）。运行方式：

```bash
source scripts/source_isaacsim_setup.sh
pytest tests/e2e/test_wbt_wrist_force_e2e.py -m isaacsim -s
```

在 training-host 上作为 nightly CI 的一部分执行（见 `tests/ci/` 目录）。

### Phase 8（Tasks 14–21）— 部署侧完全延期

inference metadata 校验（Task 14）、policy 子类（Task 15-16）、F_cmd 输入链（Task 17-18）、sim-to-sim / 真机 workflow（Task 19-20）、schema 一致性（Task 21）——**全部等待 Phase 7 TRAIN GATE 通过后**才开始。详见 `docs/plans/2026-05-03-wbt-wrist-force-v10.md §6 Phase 8`。

---

## 9. 附录 — 如何复现

### 最简单的起点

```bash
cd /data/project/holosoma-wbt-force-v10
git pull origin wbt-wrist-force-v10-training
bash demo_scripts/demo_wbt_wrist_force_training.sh
```

launcher 内置 Phase 7 TRAIN GATE banner，训练启动前会打印 8 条指标提示。

### env var 旋钮

```bash
# 使用 mjwarp 后端（跳过 IsaacSim 安装）
SIMULATOR=mjwarp bash demo_scripts/demo_wbt_wrist_force_training.sh

# 用 stdout logger 快速本地调试（不需要 wandb account）
LOGGER=stdout bash demo_scripts/demo_wbt_wrist_force_training.sh

# 换一个 random seed
SEED=42 bash demo_scripts/demo_wbt_wrist_force_training.sh

# 用自己的 motion file
MOTION_FILE=/abs/path/to/your_motion_mj.npz bash demo_scripts/demo_wbt_wrist_force_training.sh

# 已安装过依赖，跳过 pip install / isaaclab check
SKIP_REINSTALL=1 bash demo_scripts/demo_wbt_wrist_force_training.sh
```

### tyro 直接覆盖超参（forwarded verbatim）

```bash
# 扩宽 K_virtual 范围（v2 模式，训练 K-robust policy）
bash demo_scripts/demo_wbt_wrist_force_training.sh \
    "--command.setup_terms.wrist_compliance_command.params.wrist_compliance_config.k_virtual_range" "[50.0, 300.0]"

# 提高外力激活率（active_frac 过低时）
bash demo_scripts/demo_wbt_wrist_force_training.sh \
    "--command.setup_terms.wrist_compliance_command.params.wrist_compliance_config.force_ext_activation_prob_per_step" "0.02"
```

### 只跑 CPU 单元测试（无需 GPU / IsaacSim）

```bash
source scripts/source_isaacsim_setup.sh  # 激活 conda env
pytest src/holosoma/holosoma/config_types/tests/test_wrist_compliance_config.py \
       src/holosoma/holosoma/managers/command/terms/tests/test_wbt_force_command.py \
       src/holosoma/holosoma/managers/observation/terms/tests/test_wbt_force_obs.py \
       src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force_reward.py \
       src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected.py \
       src/holosoma/tests/test_wbt_force_wandb_metrics.py \
       src/holosoma/tests/utils/test_onnx_metadata_schema.py \
       src/holosoma/tests/config_values/wbt/g1/ \
       tests/test_wbt_force_cli.py \
       -v
# 预期输出：85 passed
```
