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
- `Env/force/applied_f_body_{l,r}` ≈ `Env/force/ext_magnitude_{l,r}`（**watch-only smoke，不是 gate**）：
  - 注意这条**当前实际上是同义反复** —— `wbt_force_injected.py:95-96` 把 `force_ext_w[:, 0/1]` 直接 clone 进 `last_applied_force_w_by_body_id`，**没**经过 world→body 旋转，所以 `applied_f_body_*` 的 norm 就是 `ext_magnitude_*` 的 norm，两者**数值上必然相等**。
  - 这条仅能证明 "world F_ext 被挪进了 buffer"；**不能**证明 `_rotate_force_world_to_body` / `set_external_force_and_torque` 的 frame convention 对（norm 对旋转不变，quaternion 错 / 左右手互换 / body_quat index 错都能过）。
  - 路径 A 决定：S0 **不**动 code 修这条，只在 plan 留 caveat；operator dogfood 档 2/档 3 时靠肉眼看 "wrist 被推方向 vs 绿 F_ext 箭头方向" 是否一致（§8 frame sanity 缺口）。如果肉眼看到方向不符，升级到 backlog S1（paired cancellation，间接证 frame 正确）或给这个 buffer 改成真的 body-frame + vector-level 断言。

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
2. **ckpt 兼容性检查**（在 `setup_simulation_environment` 之前，**结构性合约**，不只是看 key 存在）：
   ```python
   from holosoma.config_types.command import WristComplianceConfig
   from holosoma.managers.command.terms.wbt_force import WristComplianceCommand

   # env_class values in the codebase use dot-separators
   # (e.g. "holosoma.envs.wbt.wbt_force_injected.WholeBodyTrackingForceInjected"),
   # but the .endswith("WholeBodyTrackingForceInjected") guard in (d) works either way.
   FORCE_AWARE_ENV_CLASS = (
       "holosoma.envs.wbt.wbt_force_injected.WholeBodyTrackingForceInjected"
   )

   def _resolve_command_func(func_str: str):
       """Mirror ``managers/command/manager.py:_resolve_function`` — CommandTermCfg.func
       uses ``"module.path:ClassName"`` (colon), which ``utils.helpers.get_class`` does
       not accept. Keep this local so we don't reach into the command manager."""
       module_path, _, attr = func_str.partition(":")
       import importlib
       return getattr(importlib.import_module(module_path), attr)

   def _assert_force_aware_ckpt(saved_cfg) -> None:
       """Raise RuntimeError with a single actionable message if saved_cfg
       is not a force-aware WBT checkpoint. Checks the whole contract
       (not just that one key exists), to catch:
         - partial registration (term in setup_terms but not reset/step)
         - stale ``func`` pointing to a different class
         - missing / wrong params type
         - wrong env class
         - actor obs missing ``wrist_force_command``
       """
       cmd = saved_cfg.command
       if cmd is None:
           raise RuntimeError(
               "Checkpoint command config is None. Expected a force-aware WBT "
               "config with a `wrist_compliance_command` term. Use eval_agent.py "
               "for baseline WBT ckpts."
           )

       # (a) The term must be registered in all three lifecycle buckets.
       for bucket_name in ("setup_terms", "reset_terms", "step_terms"):
           bucket = getattr(cmd, bucket_name)
           if "wrist_compliance_command" not in bucket:
               raise RuntimeError(
                   f"Checkpoint is not force-aware: command.{bucket_name} "
                   f"has no 'wrist_compliance_command' key. Use eval_agent.py "
                   f"for baseline WBT ckpts."
               )

       # (b) func must resolve to WristComplianceCommand exactly.
       setup_term = cmd.setup_terms["wrist_compliance_command"]
       try:
           resolved = _resolve_command_func(setup_term.func)
       except Exception as exc:  # noqa: BLE001
           raise RuntimeError(
               f"wrist_compliance_command.func ({setup_term.func!r}) does not "
               f"resolve: {exc}"
           ) from exc
       if resolved is not WristComplianceCommand:
           raise RuntimeError(
               f"wrist_compliance_command.func resolves to {resolved.__name__}, "
               f"expected WristComplianceCommand."
           )

       # (c) params must carry a WristComplianceConfig.
       params = setup_term.params or {}
       wcfg = params.get("wrist_compliance_config")
       if not isinstance(wcfg, WristComplianceConfig):
           raise RuntimeError(
               "wrist_compliance_command.params['wrist_compliance_config'] is "
               f"{type(wcfg).__name__}, expected WristComplianceConfig."
           )

       # (d) env class must be force-injected. `env_class` is a top-level field
       # on ExperimentConfig — there is no `.experiment` wrapper.
       env_class = saved_cfg.env_class
       if not env_class.endswith("WholeBodyTrackingForceInjected"):
           raise RuntimeError(
               f"Checkpoint env_class is {env_class!r}, expected "
               f"{FORCE_AWARE_ENV_CLASS!r} (or a subclass). Is this really a "
               f"g1-29dof-wbt-force ckpt?"
           )

       # (e) actor obs must include wrist_force_command. `observation.groups`
       # is a ``dict[str, ObsGroupCfg]`` (see config_types/observation.py:48-56),
       # not a dataclass with a named ``actor_obs`` field.
       if saved_cfg.observation is None:
           raise RuntimeError("Checkpoint observation config is None.")
       actor_group = saved_cfg.observation.groups.get("actor_obs")
       if actor_group is None or "wrist_force_command" not in actor_group.terms:
           raise RuntimeError(
               "Checkpoint actor_obs has no 'wrist_force_command' term. Policy "
               "cannot be force-aware without this obs. Check training config."
           )

   _assert_force_aware_ckpt(saved_cfg)
   ```
   一次调用覆盖 5 条合约，错一条就 raise 一条可操作的 error；不要只看 key 名。

   **Spec→code 差异（2026-05-05 实现时修正）**：早版 plan 用 `saved_cfg.experiment.env_class` 和 `saved_cfg.observation.groups.actor_obs.terms`，都不对 —— `ExperimentConfig` 本身就是 top-level（没有 `.experiment` wrapper），`groups` 是 `dict[str, ObsGroupCfg]`（不是 dataclass）。另外 `CommandTermCfg.func` 用冒号分隔，必须走本地 `_resolve_command_func`，`utils.helpers.get_class` 只吃点号会挂。
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

Pure CPU，subprocess 起 `--help`，不真起 sim。覆盖 **11 个 case**（flag 拆分 + 结构性 preflight 合约）：

**CLI + config rewrite（6 case）**：
1. **`--help` smoke**：subprocess 跑 `python src/holosoma/holosoma/eval_agent_force.py --help`，exit 0，stdout 同时含 `enable-force-cmd` **和** `enable-force-ext`（两个 flag 都注册成功）
2. **档 0（`cmd=False, ext=False`）全部清零**：直接调 `_rewrite_wrist_compliance_cfg` on 一个 synthetic `ExperimentConfig`；断言**三个 bucket**（setup / reset / step）里的 `wrist_compliance_config` 满足：`force_cmd_magnitude_range == (0.0, 0.0)` AND `force_cmd_activation_prob_per_step == 0.0` AND `force_ext_magnitude_range == (0.0, 0.0)` AND `force_ext_activation_prob_per_step == 0.0`
3. **档 1（`cmd=True, ext=False`）只清零 F_ext**：三个 bucket 里 `force_cmd_*` 保持训练默认值（`(5.0, 30.0)` / `0.01`），`force_ext_*` 全零
4. **档 2（`cmd=False, ext=True`）只清零 F_cmd**：三个 bucket 里 `force_ext_*` 保持训练默认值，`force_cmd_*` 全零
5. **档 3（`cmd=True, ext=True`）不触发 rewrite**：断言 `_rewrite_wrist_compliance_cfg` 根本没被调（配置对象 identity 不变，用 `is` 比较）—— 这条同时 cover "默认值 = 档 3 = 不动 config"的合理性
6. **`_rewrite_wrist_compliance_cfg` immutability**：断言原 `saved_cfg` 不被 mutate —— `replace` 应该返回新对象，原对象 identity 不变

**Preflight 结构性合约（5 case，对应 §5.2 `_assert_force_aware_ckpt` 的 5 条断言）**：

7. **(a) command=None**：`saved_cfg.command = None` → `RuntimeError` message 含 "command config is None"
8. **(a') bucket 缺失**：`setup_terms` 里有 `wrist_compliance_command` 但 `reset_terms` 或 `step_terms` 缺 → raise 带"has no 'wrist_compliance_command' key"（参数化 3 种缺失组合）
9. **(b) func 解析错**：`setup_term.func = "holosoma.envs.wbt.wbt_manager:WholeBodyTrackingManager"`（不是 `WristComplianceCommand`）→ raise 带"resolves to WholeBodyTrackingManager"
10. **(c) params 类型错**：`params["wrist_compliance_config"] = {"force_cmd_magnitude_range": (5,30)}`（dict 不是 `WristComplianceConfig`）→ raise 带"is dict, expected WristComplianceConfig"
11. **(d) env_class 错 + (e) actor obs 缺失**：一个 fixture `env_class = "holosoma.envs.wbt.wbt_manager:WholeBodyTrackingManager"`（baseline）→ raise；另一个 fixture env_class 对但 `actor_obs.terms` 不含 `wrist_force_command` → raise 带"Policy cannot be force-aware"

所有 preflight test 用小 fixture `make_force_aware_exp_config(overrides)` 构造，只改要测的字段，其他默认值从 `ExperimentConfig` 的 dataclass default 来。避免每个测试手写 full config。

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
| `src/holosoma/holosoma/config_types/eval_force.py` | `ForceEvalConfig` frozen dataclass（2 个字段：`enable_force_cmd` + `enable_force_ext`） | ~35 |
| `src/holosoma/holosoma/eval_agent_force.py` | CLI 入口：mirror `eval_agent.py` + 解析 `ForceEvalConfig` + 结构性 preflight（5 条合约）+ `replace`-rewrite（两 flag 独立清零）| ~170 |
| `tests/eval/test_eval_agent_force_cli.py` | 11 个 pure-CPU test（6 CLI/rewrite + 5 preflight）| ~280 |

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

主仓 `feat/wbt-wrist-force-v10` 分支上已经有 hardlink 过来的 force-aware ckpt（`logs/WholeBodyTracking/20260504_*_g1_29dof_wbt_force_manager-locomotion/model_*.pt`），**推荐按 4 档递进**跑（每次关一个 viewer 再开下一个；或用 4 个 Terminal 并排看，但要注意 IsaacSim 多实例 GPU 占用）。

**步数约定**：对齐原 `eval_agent.py` 的行为 —— 默认 `--training.max-eval-steps` 不设置（`None`），`evaluate_policy` 内部是 `itertools.islice(count(), None)` = **无限循环**，由 operator Ctrl+C 停。这是 holosoma eval 的系统约定，S0 不额外约束。operator 自己判断看够了就停。

**bool flag 语法**：两个 flag 默认都是 `True`，用 tyro 的 `--no-<name>` 前缀来禁用（空格分隔 `--enable-force-cmd False` 会被 tyro 认成子命令 positional，**不可用**）。

```bash
source scripts/source_isaacsim_setup.sh
CKPT=logs/WholeBodyTracking/20260504_172258-g1_29dof_wbt_force_manager-locomotion/model_08000.pt

# 档 0 — baseline-like（两路都关，验证无 force 下不退化）
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --no-enable-force-cmd --no-enable-force-ext

# 档 1 — 只 F_cmd（policy 读 obs 响应，sim 物理层不被推）
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --no-enable-force-ext

# 档 2 — 只 F_ext（sim 推 wrist，policy 看不到 F_cmd obs）
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --no-enable-force-cmd

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

- **通用**（所有档）：跑到 operator 主动 Ctrl+C 停为止（默认 unlimited loop），不 crash、不 NaN；G1 跟 motion
- **档 0**（`cmd=F, ext=F`）：motion reward 不塌；行为等同 baseline WBT；`Env/force/active_frac_{cmd,ext}` ≈ 0；IsaacSim viewer 里**橙 / 绿 两种箭头都不画**（§7.1）
- **档 1**（`cmd=T, ext=F`）：肉眼看到 wrist 对 F_cmd 有响应；`active_frac_cmd` 稳态 ∈ [0.2, 0.5]，`active_frac_ext` ≈ 0；IsaacSim viewer 里**橙色 F_cmd 箭头随机出现**，**绿色 F_ext 箭头不出现**
- **档 2**（`cmd=F, ext=T`）：sim 物理层看到 wrist 被推，policy 仍能站住 + 尽量回 motion；`active_frac_cmd` ≈ 0，`active_frac_ext` 稳态 ∈ [0.2, 0.5]；`applied_f_body_{l,r}` ≈ `ext_magnitude_{l,r}`（幅值 smoke，见 §3.2 caveat —— **不是** frame correctness gate）；IsaacSim viewer 里**绿色 F_ext 箭头随机出现**，**橙色 F_cmd 箭头不出现**
- **档 3**（`cmd=T, ext=T`）：和训练末期行为最接近；`Episode/rew_wrist_force_position_tracking_exp ≥ 0.4`；两路 active_frac 都在 [0.2, 0.5]；`applied_f_body ≈ ext_magnitude`（同上 caveat）；IsaacSim viewer 里**橙 / 绿 两种箭头都出现**
- **步数约定**：不强制 `max_eval_steps`，对齐 holosoma 系统约定默认 `None`（无限 loop；operator Ctrl+C 停）。operator 自己看够几次 F 的 activation 周期就停。
- **frame sanity 缺口**（S0 明知的盲区，watch-item，操作时留意）：
  - **已知的同义反复**：`applied_f_body_{l,r}` **当前**存的就是 `force_ext_w` 的 clone（`wbt_force_injected.py:95-96` 把 `force_w[:, 0/1]` 直接塞进 `last_applied_force_w_by_body_id`，**没做 world→body quaternion 旋转**），所以它的 norm 等于 `ext_magnitude` 是**数值上注定**的 —— S0 这条 "sanity" 仅是"把 world F_ext 转到 buffer 里没掉精度"的极弱 smoke，**不能**证明 `_rotate_force_world_to_body` / `set_external_force_and_torque` 的 frame convention 对。
  - **norm 还是盲区**：就算未来把 `last_applied_force_w_by_body_id` 改存 body-frame（真的 quaternion rotate 过），norm 也对旋转不变，幅值比较依然测不出 quaternion 错、左右手换了、body_quat index 错这些 frame bug。
  - **S0 的位置**：按路径 A 处理 —— **不动 code**，仅在 plan 里 flag 这条 caveat，operator dogfood 时注意"档 2 /档 3 viewer 里 wrist 真被推的方向/大小"对得上 F_ext 箭头方向就行。如果肉眼感觉 wrist 被推的方向和绿箭头不一致（比如箭头朝 +x 但 wrist 往 -x 偏），那就是 frame bug，operator 抓这个现象反馈，我们再启动路径 B（改 code 存 body-frame + 加 vector-level 断言）或 S1 backlog（paired cancellation 间接证 frame）。
  - **要严格验这条**：做 vector-level 比较（log 实际传给 `set_external_force_and_torque` 的 body-frame 3-vector，vs 期望 `quat_apply_inverse(body_quat_w, force_ext_w)` 的 3-vector，按分量比）。S0 不做，留给 backlog S1。
- **异常兼容**：baseline ckpt（`g1_29dof_wbt`）丢进来 → 立即 raise 明确错误，不进 sim（详见 §5.2 preflight contract）

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
