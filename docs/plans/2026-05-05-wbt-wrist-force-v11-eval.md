# WBT Wrist Force — Evaluation 计划（v11，S0 开工档）

> **v11 与 v10 的分工**：
> - **v11（本文档）**：**开工档**。只保留现在要动手做的 S0（in-training force-aware sanity eval 入口，`eval_agent_force.py`）所需的所有信息——技术方案、文件清单、acceptance、运行命令、测试覆盖。**直接可执行**，不含 backlog / 其它 scenario 的任何冗余文字。
> - **v10**（[2026-05-05-wbt-wrist-force-v10-eval.md](./2026-05-05-wbt-wrist-force-v10-eval.md)）：**backlog 档**。保留 S1（paired cancellation）/ S2（F_ext sweep）/ S3（sim-to-sim MuJoCo）/ S4（baseline A/B）+ UniFP / codex review 的所有技术细节、acceptance 阈值、motion clip 设计等。不删，留作后续 approve 后直接可接续的参考。
> - **约束**：S0 实现**不引入任何** backlog task 才用得到的基础设施（eval callback、scenario schedule、motion clip synthesizer、MuJoCo 注入器 等）。

---

## 1. S0 是什么（TL;DR）

**S0 = `eval_agent.py` 的 force-aware 版本**。

- Platform：**IsaacSim**，和训练同 env 同 config。
- 输入：parent plan Phase 7 产出的 force-aware ckpt（`exp:g1-29dof-wbt-force` 训出来的 `.pt`）。
- 机制：**复用** `eval_agent.py` 的 rollout 引擎（load ckpt → 重建 env → `algo.evaluate_policy()`），**加两个独立 CLI flag** `--enable-force-cmd` 和 `--enable-force-ext`。
- F_cmd / F_ext 来源（两个通道**独立**控制）：
  - `--enable-force-cmd True` → F_cmd 走训练同款随机 schedule；`False` → F_cmd 通道清零
  - `--enable-force-ext True` → F_ext 走训练同款随机 schedule；`False` → F_ext 通道清零
  - 两个默认都是 `True`（等同训练时的完整 schedule）
- 判定：**全部 qualitative**（肉眼看 IsaacSim viewer + wandb 指标），不产 NPZ、不自动打分、不写 markdown 报告。
- 目的：policy 训好了没有 —— 这是最薄的"至少是能看的"门槛。**拆成两个 flag 的意义**：可以**渐进式**对比 baseline-like / 只 F_cmd / 只 F_ext / 完整 schedule 四种行为，定位 policy 对哪一路的响应在工作。

---

## 2. 为什么这一步够用

和 README 里 `### In-Training Evaluation` 小节做的事**在同一层**：
- 自动从 ckpt 读 saved config，原封不动重建训练时 env（同 simulator / manager / reward / randomization）
- policy 不再学习（`actor.eval()`），只跑 rollout
- 可选导出 ONNX

我们**只加**两个独立 flag：`--enable-force-cmd` 和 `--enable-force-ext`。其它都 reuse `eval_agent.py` 的机制：
- `attach_checkpoint_metadata` + `load_checkpoint`：reuse
- `setup_simulation_environment`：reuse
- `evaluate_policy()` 主 loop：reuse
- ONNX export / log dir / wandb sync：reuse
- **`_update_log_dict` 里 parent plan Task 9.5 打的 `Env/force/*` 指标**：reuse（wandb 自动可见）

---

## 3. 两个观察维度（都 qualitative，**不**自动判定）

### 3.1 肉眼看 IsaacSim viewer（渐进式 4 档对比）

两个 flag 组合出 4 档，**推荐按下表顺序**依次跑，一档看明白再进下一档：

| 档位 | `--enable-force-cmd` | `--enable-force-ext` | 预期行为 | 验证的是什么 |
|---|---|---|---|---|
| **档 0 — baseline-like** | `False` | `False` | G1 跟 motion 行为等同 baseline WBT；两路 force buffer 恒零 | policy 在无 force 时不退化 |
| **档 1 — 只 F_cmd** | `True` | `False` | G1 跟 motion + wrist 在 F_cmd 触发瞬间有偏移；sim 物理层**不**被外力推 | policy 读 `wrist_force_command` obs 并响应 |
| **档 2 — 只 F_ext** | `False` | `True` | G1 跟 motion + sim 物理层被 F_ext 推动 wrist（operator 视角 "policy 接到外扰能稳住不摔倒 + 尽量回 motion"）| policy 对没进 obs 的隐藏外扰有隐式鲁棒性 |
| **档 3 — 完整**（训练同款）| `True` | `True` | 两路都开，两种 schedule 叠加；应该和训练末期的行为最接近 | 两路组合下 policy 还能跟住 motion + reward 公式落到行为上 |

### 3.2 Wandb 指标（parent plan Task 9.5 已有的 `Env/force/*` 和 `Episode/rew_*`）

- `Episode/rew_motion_tracking_*` 不塌陷（所有档位都要满足）
- `Episode/rew_wrist_force_position_tracking_exp` 稳定在训练末期水平（档 3 完整 schedule 下预期 ≥ 0.4；档 0/1/2 由于 F_ext 或 F_cmd 清零，reward 值域会偏移，只看**不崩**就行）
- `Env/force/active_frac_cmd`、`active_frac_ext` 和所选档位一致：
  - 档 0：两路都 ≈ 0
  - 档 1：`active_frac_cmd` 稳态 ∈ `[0.2, 0.5]`，`active_frac_ext` ≈ 0
  - 档 2：`active_frac_cmd` ≈ 0，`active_frac_ext` 稳态 ∈ `[0.2, 0.5]`
  - 档 3：两路都稳态 ∈ `[0.2, 0.5]`（和训练一致）
- `Env/force/applied_f_body_{l,r}` == `Env/force/ext_magnitude_{l,r}`（world→body quat 转换对了；只在 `enable_force_ext=True` 的档 2 / 档 3 下有意义）

---

## 4. 硬约束

1. **append-only**：不动 parent plan 任何已有文件；不动 `eval_agent.py`
2. **不引入**：eval callback、scenario schedule、motion clip synthesizer、MuJoCo 注入器、acceptance 自动判定 —— 都在 v10 backlog
3. **ckpt 兼容性**：v10 force ckpt 可跑；baseline `g1_29dof_wbt` ckpt 不应跑进这个入口（提前 raise，引导用 `eval_agent.py`）
4. **训练路径零污染**：eval-only CLI flag；training 完全不受影响

---

## 5. 技术方案（直接开工用）

### 5.1 新文件 1：`src/holosoma/holosoma/config_types/eval_force.py`

```python
"""CLI config for eval_agent_force.py."""
from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ForceEvalConfig:
    """Force-aware eval CLI options for ``eval_agent_force.py``.

    Two independent flags control whether each force channel keeps its
    training-time random schedule or gets zeroed out. Combining them
    yields 4 diagnostic modes (see §3.1):
        (False, False) -> baseline-like (no force injection at all)
        (True,  False) -> F_cmd only (obs channel active; sim untouched)
        (False, True ) -> F_ext only (sim-injected ext force; obs zero)
        (True,  True ) -> full training schedule (default)
    """

    enable_force_cmd: bool = True
    """If True, keep the training-time random schedule for F_cmd
    (actor obs ``wrist_force_command``). If False, zero
    ``force_cmd_magnitude_range`` and ``force_cmd_activation_prob_per_step``
    so F_cmd stays at 0 throughout the rollout."""

    enable_force_ext: bool = True
    """If True, keep the training-time random schedule for F_ext
    (sim-injected external force on wrist bodies). If False, zero
    ``force_ext_magnitude_range`` and ``force_ext_activation_prob_per_step``
    so no external force is applied to the wrists during rollout."""
```

### 5.2 新文件 2：`src/holosoma/holosoma/eval_agent_force.py`

**结构 mirror `eval_agent.py`**（入口函数签名、tyro 多 `return_unknown_args=True` 链、`run_eval_with_tyro` 调用）。差别只三处：

1. **CLI schema**：在 `main()` 里多解析一层 `ForceEvalConfig`（和 `CheckpointConfig` / `EvalCallbacksConfig` 同级，`return_unknown_args=True` 串联）
2. **ckpt 兼容性检查**（在 `setup_simulation_environment` 之前）：
   ```python
   saved_cmd_setup = saved_cfg.command.setup_terms  # dict[str, CommandTermCfg]
   if "wrist_compliance_command" not in saved_cmd_setup:
       raise RuntimeError(
           f"Checkpoint is not force-aware (command.setup_terms has no "
           f"'wrist_compliance_command' key). Use eval_agent.py for baseline "
           f"WBT ckpts instead."
       )
   ```
3. **按 flag 选择性清零 force config**（在 `setup_simulation_environment` 之前对 `tyro_config` 做 `replace`）：
   ```python
   from dataclasses import replace
   from holosoma.config_types.command import WristComplianceConfig

   def _apply_force_eval_flags(
       cfg: WristComplianceConfig,
       *,
       enable_force_cmd: bool,
       enable_force_ext: bool,
   ) -> WristComplianceConfig:
       """Zero out the F_cmd and/or F_ext channel independently."""
       updates: dict = {}
       if not enable_force_cmd:
           updates.update(
               force_cmd_magnitude_range=(0.0, 0.0),
               force_cmd_activation_prob_per_step=0.0,
           )
       if not enable_force_ext:
           updates.update(
               force_ext_magnitude_range=(0.0, 0.0),
               force_ext_activation_prob_per_step=0.0,
           )
       return replace(cfg, **updates) if updates else cfg

   # Rewrite only if at least one flag is False
   if not (force_eval_cfg.enable_force_cmd and force_eval_cfg.enable_force_ext):
       # Must override across all three buckets (setup / reset / step) —
       # v10 plan §Task 6 registered `wrist_compliance_command` in all
       # three, so missing any of them leaks the default schedule back in.
       tyro_config = _rewrite_wrist_compliance_cfg(
           tyro_config,
           lambda c: _apply_force_eval_flags(
               c,
               enable_force_cmd=force_eval_cfg.enable_force_cmd,
               enable_force_ext=force_eval_cfg.enable_force_ext,
           ),
       )
   ```

   辅助函数 `_rewrite_wrist_compliance_cfg(exp_cfg, transform)` 是一个纯 data transform —— 拷贝 `exp_cfg.command` 的 3 个 bucket dict，对每个 bucket 里叫 `wrist_compliance_command` 的 `CommandTermCfg`，把它 `params["wrist_compliance_config"]` 过一下 `transform`，再用 `dataclasses.replace` 组装回新的 `CommandManagerCfg` → `ExperimentConfig`。

**其它完全复用 `eval_agent.py`**：`run_eval_with_tyro(...)` 本身可以直接复用原函数（把 signature 暴露出来 import 即可，不改它）；或者如果需要最小侵入性，`eval_agent_force.py` 自己写一个 `main()` 但调用 `eval_agent.run_eval_with_tyro`。

### 5.3 新文件 3：`tests/eval/test_eval_agent_force_cli.py`

Pure CPU，subprocess 起 `--help`，不真起 sim。覆盖 **6 个 case**（flag 拆分后多了 3 个组合 case）：

1. **`--help` smoke**：subprocess 跑 `python src/holosoma/holosoma/eval_agent_force.py --help`，exit 0，stdout 同时含 `enable-force-cmd` **和** `enable-force-ext`（两个 flag 都注册成功）
2. **不兼容 ckpt 提前 raise**：mock `load_saved_experiment_config` 返回一个 `command.setup_terms` 不含 `wrist_compliance_command` 的 saved config → 调 `main()` 断言 `RuntimeError` 带相应 message
3. **档 0（`cmd=False, ext=False`）全部清零**：直接调 `_rewrite_wrist_compliance_cfg` on 一个 synthetic `ExperimentConfig`；断言**三个 bucket**（setup / reset / step）里的 `wrist_compliance_config` 满足：`force_cmd_magnitude_range == (0.0, 0.0)` AND `force_cmd_activation_prob_per_step == 0.0` AND `force_ext_magnitude_range == (0.0, 0.0)` AND `force_ext_activation_prob_per_step == 0.0`
4. **档 1（`cmd=True, ext=False`）只清零 F_ext**：三个 bucket 里 `force_cmd_*` 保持训练默认值（`(5.0, 30.0)` / `0.01`），`force_ext_*` 全零
5. **档 2（`cmd=False, ext=True`）只清零 F_cmd**：三个 bucket 里 `force_ext_*` 保持训练默认值，`force_cmd_*` 全零
6. **档 3（`cmd=True, ext=True`）不触发 rewrite**：断言 `_rewrite_wrist_compliance_cfg` 根本没被调（配置对象 identity 不变，用 `is` 比较）—— 这条同时 cover "默认值 = 档 3 = 不动 config"的合理性

### 5.4 修改 1：`src/holosoma/README.md`（append-only，不动现有段落）

在 `## Evaluation` 段的 `### In-Training Evaluation` 小节**后面**追加一个 `### In-Training Force-Aware Evaluation` 小节。**完全 mirror** 原节格式（命令块 + bullet list），示例：

````markdown
### In-Training Force-Aware Evaluation

For WBT wrist-force checkpoints (`exp:g1-29dof-wbt-force`), use the
force-aware evaluation entry point, which reuses the training-time
random force schedule:

```bash
# Mode 0 — baseline-like (both force channels disabled; regression check)
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint=<CHECKPOINT_PATH> \
    --enable-force-cmd=False --enable-force-ext=False

# Mode 1 — F_cmd only (actor obs channel active; sim untouched)
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint=<CHECKPOINT_PATH> \
    --enable-force-cmd=True --enable-force-ext=False

# Mode 2 — F_ext only (sim-injected ext force; actor gets no F_cmd obs)
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint=<CHECKPOINT_PATH> \
    --enable-force-cmd=False --enable-force-ext=True

# Mode 3 — Full training schedule (defaults; both channels active)
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint=wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME>
```

Flags (two independent channels; both default to `True`):
- `--enable-force-cmd`: if `True`, F_cmd is sampled with the training-time
  random schedule (`force_cmd_magnitude_range=(5,30) N`, ramp/hold/cooldown
  state machine, activation prob 0.01 per step). If `False`, F_cmd stays
  at zero and the actor `wrist_force_command` obs is effectively muted.
- `--enable-force-ext`: if `True`, F_ext is sampled with the same
  training-time schedule and injected into the sim on the wrist bodies.
  If `False`, no external force is applied — sim wrists see no external
  perturbation.

Run modes in order for a full sanity pass (hardest to detect issues last):
1. Mode 0 — verify no regression vs pure motion tracking.
2. Mode 1 — verify policy responds to `wrist_force_command` obs.
3. Mode 2 — verify policy stays up under hidden external force.
4. Mode 3 — verify full training-time behavior reproduces.

Sanity checks to eyeball during rollout:
- G1 tracks the motion clip (baseline WBT behavior in all modes)
- Mode 1: wrist positions show visible shifts when F_cmd triggers
- Mode 2: sim physics pushes the wrist; policy compensates to stay on motion
- Mode 3: Wandb `Episode/rew_wrist_force_position_tracking_exp >= 0.4`
- Mode 2 / Mode 3: Wandb `Env/force/applied_f_body_{l,r} == Env/force/ext_magnitude_{l,r}`
````

---

## 6. 文件清单

### 6.1 新建

| 路径 | 作用 | 预估 LoC |
|---|---|---|
| `src/holosoma/holosoma/config_types/eval_force.py` | `ForceEvalConfig` frozen dataclass（1 个字段） | ~25 |
| `src/holosoma/holosoma/eval_agent_force.py` | CLI 入口：mirror `eval_agent.py` + 解析 `ForceEvalConfig` + 兼容性 assert + `replace`-rewrite | ~100 |
| `tests/eval/test_eval_agent_force_cli.py` | 3 个 pure-CPU test | ~120 |

### 6.2 修改（append-only）

| 路径 | 改什么 |
|---|---|
| `src/holosoma/README.md` | 追加 `### In-Training Force-Aware Evaluation` 小节 |

### 6.3 必须保持原样

- `src/holosoma/holosoma/eval_agent.py`
- `src/holosoma/holosoma/config_types/eval_callback.py`
- 任何 parent plan 的 training config value
- `envs/` / `managers/` / `agents/` 全部不动

---

## 7. 运行命令（dogfood）

主仓 `feat/wbt-wrist-force-v10` 分支上已经有 hardlink 过来的 force-aware ckpt（`logs/WholeBodyTracking/20260504_*_g1_29dof_wbt_force_manager-locomotion/model_*.pt`），**推荐按 4 档递进**跑（每次关一个 viewer 再开下一个；或用 4 个 Terminal 并排看，但要注意 IsaacSim 多实例 GPU 占用）：

```bash
source scripts/source_isaacsim_setup.sh
CKPT=logs/WholeBodyTracking/20260504_172258-g1_29dof_wbt_force_manager-locomotion/model_08000.pt

# 档 0 — baseline-like（两路都关，验证无 force 下不退化）
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --enable-force-cmd False --enable-force-ext False

# 档 1 — 只 F_cmd（policy 读 obs 响应，sim 物理层不被推）
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --enable-force-cmd True --enable-force-ext False

# 档 2 — 只 F_ext（sim 推 wrist，policy 看不到 F_cmd obs）
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --enable-force-cmd False --enable-force-ext True

# 档 3 — 完整（训练同款；两个 flag 都是默认 True，可省略）
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT"
```

### 7.1 IsaacSim viewer 里的 debug arrow（必须打开 viewer 才看得到）

Parent plan v10 Task 3 已经在 `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py` 里实现了 **wrist 力箭头** viz，S0 直接复用 —— 不需要写新代码。**但训练默认 headless，必须在 eval 时把 viewer 和 debug-viz 显式打开**，否则一片黑什么都看不见。

- **颜色规约**（v11 版：`wbt_force_injected.py` 的 `COLOR_F_CMD / COLOR_F_EXT` 常量）：
  - **F_cmd = 橙色**（`(1.0, 0.5, 0.0)`）—— actor obs `wrist_force_command`
  - **F_ext = 绿色**（`(0.0, 0.8, 0.0)`）—— sim 注入的外力
  - **不画 F_total**（v11 决定：F_cmd 和 F_ext 分开看更直观，几何和由肉眼脑补）
- **画的位置**：左右 wrist 位置发出的直线箭头（端点一个小球），只画 `env_id=0`
- **长度换算**：`arrow_length_m = ‖F‖ / debug_arrow_scale_n_per_m`（默认 50 N/m → 30N 画 0.6m）
- **Torque 不画**（没训练 torque 通道，画出来全是零；如果未来加 torque 再扩颜色 + 绕轴箭头）

**跑 S0 时必须加的 viewer flag**：

```bash
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --enable-force-cmd True --enable-force-ext True \
    --training.headless False \
    --simulator.config.debug-viz True
```

- `--training.headless False` 弹出 IsaacSim 窗口（默认 True 不弹）
- `--simulator.config.debug-viz True` 打开 per-render `draw_debug_viz()` 调用（默认 False 不画；这个 flag 名字确认一下 parent plan `simulator.config` schema 里的精确字段名，在实现 Task 13.9 时对齐）

**肉眼 check list**（4 档对应的箭头表现）：
| 档位 | 橙 F_cmd | 绿 F_ext |
|---|---|---|
| 档 0（F, F）| 无（恒零）| 无（恒零）|
| 档 1（T, F）| 有（随机触发）| 无 |
| 档 2（F, T）| 无 | 有（随机触发）|
| 档 3（T, T）| 有 | 有 |

**看不到箭头的 debug 顺序**：
1. IsaacSim 窗口根本没开 → 加 `--training.headless=False`
2. 窗口开了但无箭头 → 加 `--simulator.config.debug-viz=True`
3. debug-viz 开了但 `sim.draw` 是 None（IsaacSim 侧渲染 adapter 没初始化）→ 查 `src/holosoma/holosoma/simulator/isaacsim/isaacsim.py:488` 附近逻辑
4. 都开了但还是零 → `env_id=0` 那一 env 刚好没在发力，等几秒到 `activation_prob` 触发下一次

---

## 8. GATE S0（qualitative，不自动判定）

4 档 rollout 全部满足：

- **通用**（所有档）：跑满 `max_eval_steps`，不 crash、不 NaN；G1 跟 motion
- **档 0**（`cmd=F, ext=F`）：motion reward 不塌；行为等同 baseline WBT；`Env/force/active_frac_{cmd,ext}` ≈ 0；IsaacSim viewer 里**橙 / 绿 两种箭头都不画**（§7.1）
- **档 1**（`cmd=T, ext=F`）：肉眼看到 wrist 对 F_cmd 有响应；`active_frac_cmd` 稳态 ∈ [0.2, 0.5]，`active_frac_ext` ≈ 0；IsaacSim viewer 里**橙色 F_cmd 箭头随机出现**，**绿色 F_ext 箭头不出现**
- **档 2**（`cmd=F, ext=T`）：sim 物理层看到 wrist 被推，policy 仍能站住 + 尽量回 motion；`active_frac_cmd` ≈ 0，`active_frac_ext` 稳态 ∈ [0.2, 0.5]；`applied_f_body_{l,r} == ext_magnitude_{l,r}`；IsaacSim viewer 里**绿色 F_ext 箭头随机出现**，**橙色 F_cmd 箭头不出现**
- **档 3**（`cmd=T, ext=T`）：和训练末期行为最接近；`Episode/rew_wrist_force_position_tracking_exp ≥ 0.4`；两路 active_frac 都在 [0.2, 0.5]；`applied_f_body == ext_magnitude`；IsaacSim viewer 里**橙 / 绿 两种箭头都出现**
- **异常兼容**：baseline ckpt（`g1_29dof_wbt`）丢进来 → 立即 raise 明确错误，不进 sim

过了 GATE S0 → 可以往 parent plan Phase 8 部署推；或者 approve **v10 backlog** 的 Task 13.9a / 13.9b / 22 / 23 其中之一继续。

---

## 9. Backlog 入口（v10 文档）

所有以下内容**留在 v10**，v11 不重复：
- **S1**：paired cancellation callback + raised-arms motion + C1-C8 acceptance
- **S2**：F_ext sweep benchmark + bootstrap CI
- **S3**：sim-to-sim MuJoCo 部署 eval（`WristExtForceInjector` + `ScriptedWristForceCmdProvider`）
- **S4**：baseline vs force-aware A/B（5 seeds × 3 motions）
- codex review 意见汇总（C3 cosine fix、C4 adaptive threshold、C7 body-id validation、C8 transient response）
- 扩展 scenarios（asymmetric F、vertical F、cap saturation、K 随机化、non-static motion + F_cmd）

要激活 backlog 任意一项 → 用户明确 approve → 把那一项的技术档从 v10 复制进**新的** v12/v13 开工档，一次推一个。

---

## 10. References

- **Parent plan**：[`2026-05-03-wbt-wrist-force-v10.md`](./2026-05-03-wbt-wrist-force-v10.md)
- **Backlog eval plan**：[`2026-05-05-wbt-wrist-force-v10-eval.md`](./2026-05-05-wbt-wrist-force-v10-eval.md)
- **参考入口 `eval_agent.py`**：`src/holosoma/holosoma/eval_agent.py`
- **参考 config `WristComplianceConfig`**：`src/holosoma/holosoma/config_types/command.py:161-175`
- **参考 `EvalCallbacksConfig`**（后续 backlog 用到，v11 不需要）：`src/holosoma/holosoma/config_types/eval_callback.py`
- **在主仓可用的 force-aware ckpt**：`logs/WholeBodyTracking/20260504_*_g1_29dof_wbt_force_manager-locomotion/model_*.pt`（hardlink 自 `/data/project/holosoma-wbt-force-v10/logs/WholeBodyTracking/`）
