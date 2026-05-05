# WBT Wrist Force — Evaluation 计划（v10-eval v4）

> **v4 修订说明（2026-05-05，用户反馈后）**：**收窄执行范围**
> - 用户要求：**先只做一个 sanity-check 级的 in-training eval 入口**（`eval_agent_force.py`），带一个 `--enable-force-cmd` 开关；F_cmd 用训练同款随机 schedule 采样；其它 task（S1 / S2 / S3 / S4）全部降级为"待 approve 的 backlog"，要用户一个个点头才动
> - §2 scenario 表：新增 **S0（in-training sanity eval）** 作为首发，原 S1/S2/S3/S4 改名为 BACKLOG
> - §4 Task 清单：现役只有 **Task 13.9** = 实现 `eval_agent_force.py` + `--enable-force-cmd` flag；原来的 13.9a / 13.9b / 22 / 23 全部打 `[BACKLOG — 待 approve]` 标记
> - 其它章节（§3、§5-9）保持 as-is，作为 backlog 任务的技术档备查
>
> **v3 修订说明（2026-05-05，用户反馈后）**：
> - §3.1 motion 来源：放弃"取现有 motion 第一帧 freeze"的方案，改用 G1 WBT 部署态的 **default "raised arms" pose**（即 `src/holosoma_inference/holosoma_inference/config/config_values/robot.py:70` 的 `default_dof_angles`，对应 README "Standing with raised arms"）。和 S3（MuJoCo 部署）的 stiff control 初始 pose 语义一致
> - 新增 §4.5「baseline ckpt 依赖」说明：S4（Task 23）依赖用户另跑一个 `g1_29dof_wbt` baseline ckpt；plan 本身不负责训 baseline
>
> **v2 修订说明（2026-05-05，codex review 之后）**：根据 `tasks/a5a0c2ef54be4f878.output` 的审查意见收紧。
> - §3.3 acceptance 阈值：C2 放宽到 `[0.025, 0.075] m`，C3 修成 cosine similarity（原来的 raw dot 维度错了），C4 改成 adaptive（`max(0.02, 0.4 * ||B_offset||)`）
> - Task 13.9a：加显式 cleanup 语义 + body-id 断言（C7）+ transient response 通道（C8）
> - Task 22：把 pexpect 驱动键盘换成新增 `ScriptedWristForceCmdProvider`
> - Task 23：扩大到 **5 seeds × 3 motions**，引入 bootstrap 95% CI
> - 新增 §9「扩展 scenarios (v2/v3)」，覆盖 asymmetric F、vertical F、cap saturation、transient response、K 随机化、non-static motion + F_cmd

---

## 1. 动机（TL;DR）

UniFP 论文 Fig. 5(a) 通过挂 2.5 kg 负载（= 25 N 向下 F_ext）+ 发 25 N 向上 F_cmd 让末端原地不动，演示 force control。这就是**力控的 canonical sanity test**：如果 policy 真的把 `target_shifted = motion_target + (F_ext + F_cmd) / K` 落到了行为上，那么成对的 (F_cmd, -F_cmd) 注入应该在末端互相抵消。

parent plan 没有这个测试。parent plan 同时也缺：

1. 一种**能从 CLI 确定性地下发 F_cmd** 的方式（parent 的 `WristComplianceCommand` 用随机 state machine 采样 F_cmd，不可测）
2. 一种**在 MuJoCo 部署推理时注入 F_ext** 的方式（parent 的 `WholeBodyTrackingForceInjected` 只在 IsaacSim 训练里跑）
3. 一个量化 **Δx** 的观测 loop，把它和 reward 公式对上
4. 一套 **baseline vs force A/B** 对比（force-aware policy 在外力下是不是真的比 baseline 好？）

这份 plan 通过 4 个新 task（Task 13.9a / 13.9b / 22 / 23）补齐上述 4 个能力。

---

## 2. Eval 设计总览

### 2.1 Scenario 结构（v4：先只做 S0，其它 backlog 等 approve）

| Scenario | 要验证什么 | 跑在哪 | F_cmd / F_ext 怎么驱动 | 状态 |
|---|---|---|---|---|
| **S0 — In-training sanity eval**（Task 13.9）| **policy 训好没有** — 把训练时的 env + motion + random force schedule 原封搬出来跑 rollout，肉眼/wandb 看机器人是不是能跟 motion + 能对 F_cmd 有响应 | IsaacSim `eval_agent_force.py` | `--enable-force-cmd True` → 训练同款随机 schedule；`False` → F_cmd 归零（对照） | **当前唯一 active** |
| **S1 — Paired cancellation**（Task 13.9a）| reward 公式 `F_ext + F_cmd = 0 → wrist 回到 motion_target` 是否**行为上成立** | IsaacSim `eval_agent.py` + 新 eval callback | eval callback 按 scripted schedule override `WristComplianceCommand` buffer | **BACKLOG — 待 approve** |
| **S2 — F_ext sweep benchmark**（Task 13.9b）| 在隐藏 F_ext 下 tracking error 是否受控 + 不摔倒 | IsaacSim `eval_agent.py` + 同一个 callback | callback 在 4 个幅值 bin × 3 seeds 注 F_ext；F_cmd = 0 | **BACKLOG — 待 approve** |
| **S3 — Sim-to-sim paired cancellation**（Task 22）| 端到端部署 stack：ONNX + keyboard/ROS2 F_cmd provider + MuJoCo F_ext 注入 全链路 | MuJoCo `run_sim.py` + `run_policy.py` | 操作员通过 keyboard + MuJoCo CLI `--wrist-ext-force-schedule` | **BACKLOG — 待 approve** |
| **S4 — Baseline A/B**（Task 23）| force-aware v10 在外力下是否**真的**比 baseline WBT 强 | IsaacSim（复用 S2 callback）| 同 motion + 同 F_ext bin 对比 v10 vs baseline WBT ckpt | **BACKLOG — 待 approve** |

### 2.1.a S0 在做什么（v4 首发）

**S0 就是 README 那段 "In-Training Evaluation" 的 force 版本**。README 里 `eval_agent.py` 对 locomotion / baseline WBT ckpt 做的事：

- 自动从 ckpt 读 saved config，**原封不动**重建训练时的 env（同 simulator / 同 manager / 同 reward / 同 randomization）
- policy 不再学习（`actor.eval()`），只跑 rollout
- 可选导出 ONNX
- locomotion ckpt 额外支持键盘 `w/a/s/d` 交互改 velocity

我们做的 `eval_agent_force.py` 是在上面基础上加**一个** CLI flag：

```bash
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint=<ckpt> \
    --enable-force-cmd=True   # 默认 True，F_cmd 随机采样；设 False 则 F_cmd 恒为 0
```

### 2.1.b S0 是怎么判定 "policy 训好了"

两个观察维度，**全部 qualitative**（这版 sanity check 不做自动 pass/fail）：

1. **肉眼看 IsaacSim viewer**：
   - G1 能稳定跟 motion（搬箱子）— baseline WBT 能力
   - `--enable-force-cmd True` 时，wrist 在 F_cmd 触发瞬间有明显偏移（不只是被 motion 拖着走）— force channel 有响应
   - `--enable-force-cmd False` 时，行为和 baseline WBT 几乎一致 — F_cmd=0 没有 side effect
2. **wandb 看训练期已经打好的指标**（parent plan Task 9.5 产物）：
   - `Episode/rew_motion_tracking_*` 不塌陷
   - `Episode/rew_wrist_force_position_tracking_exp` ≥ 0.4
   - `Env/force/active_frac_cmd`、`active_frac_ext` 稳态在 [0.2, 0.5]
   - `Env/force/applied_f_body_{l,r}` == `Env/force/ext_magnitude_{l,r}`（world→body quat 转换对了）

**S0 不产 NPZ、不自动打分、不写 markdown 报告**。它是最轻量的"这 ckpt 至少是能看的"门槛。

### 2.2 为什么 S1 是最重要的一道 gate（backlog 文档备查）

S1 是 **unit test**，不是 benchmark。它回答两件事：
- policy 是不是真的读 `wrist_force_command` obs？（A→B 的偏移验证）
- `F_ext + F_cmd` 的加性模型是不是真的体现在行为里？（B→C 的抵消验证）

如果 S1 过不了，`force/*` 的 wandb 指标**可能还是好看**、S2/S3 的吞吐数字**可能也还凑合**——但 policy 实质上**没学会把 F_cmd 当成可控输入使用**。S1 捕捉"力通道 silent failure"的作用，和 regression test 捕捉"parser silent failure"的作用一样。**S1 必须先过，S2-S4 才跑**。

### 2.3 硬约束（5 条红线，和 parent plan 一致）

1. 力只施加在 `left_wrist_yaw_link` / `right_wrist_yaw_link`
2. S1 里 `K_virtual` 固定 100 N/m（对应 v1 config）；S2 sweep 可在 v2 变 K
3. **append-only**：除 `EvalCallbacksConfig` 注册 + `run_sim.py` 新 CLI flag 外，不动 parent plan 里任何已有文件
4. **运行时不 import `third_party/`**（同 parent）
5. 所有新代码只在 `src/holosoma/holosoma/agents/callbacks/`、`tests/eval/`、`src/holosoma_inference/` 下；parent 的 `envs/` / `managers/` 保持不动

---

## 3. Canonical scenario：raised-arms paired cancellation

这是 S1。把它定具体，让所有 task 说同一种"语言"。

### 3.1 Motion clip 要求

S1 用的 motion clip **必须保持静态 pose**，至少 25s（3 phase × ≥5s + gap + warmup）。parent plan 训练用的 motion（LAFAN 跳舞、OMOMO 抓箱子）都在动，直接用不合适。

**v3 的做法（用户反馈后简化）**：**直接用 G1 WBT 部署态的 default "raised arms" pose** 作为静态 motion。这个 pose 是：
- `src/holosoma_inference/holosoma_inference/config/config_values/robot.py:70-76` 的 `default_dof_angles`
- 对应 `src/holosoma_inference/README.md:85` 描述的 "**Default pose**: Standing with raised arms"
- 部署时 MuJoCo 的 stiff control mode + real-robot `Y` 按钮都是把机器人调到这个 pose
- 肩 pitch=0.2、肩 roll=±0.2、肘=0.6、腿部半蹲 —— **手已经抬起来有一定力臂**，F_cmd 5N 能产生的偏移方向（Y/Z 世界轴）都不会撞躯干

**为什么 v3 比 v2 简单**：
- v2 方案要从 motion NPZ 切帧、tile 时间轴、写 helper script、`replay.py` 肉眼验证 → **~80 LoC 工程量 + 手工挑帧** + schema 兼容风险
- v3 方案 motion 直接来自部署态 config，**0 LoC 工程量**，而且天然和 S3（MuJoCo 部署 eval）共用同一个起手 pose，两边行为可直接比对

**实现路径**：
- **新建**：`src/holosoma/holosoma/data/motions/g1_29dof/eval/raised_arms_hold.npz` — 把 `default_dof_angles` 常量"rasterize"成一段 30s × 50fps 的静态 NPZ
- **生成脚本**：`src/holosoma/holosoma/data/motions/g1_29dof/eval/tools/make_raised_arms_motion.py`（~50 LoC），逻辑：
  1. 读 `default_dof_angles`（29 维）作为每一帧的 `joint_pos`
  2. 走 G1 URDF 的 FK（`holosoma.utils.kinematics` 或 isaaclab 的 `compute_body_poses`）算出对应的 `body_pos_w` / `body_quat_w`（14+ 个 body，匹配 WBT tracked body 列表）
  3. velocities 全部置零，tile 到 1500 帧（30s × 50fps）
  4. 保留 `MotionLoader` 的完整 schema（`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、body 线/角速度、`body_names`、`joint_names`、metadata）
- **FK 实现选择**（2 选 1，按实现难度）：
  - **Option A（推荐）**：调 IsaacLab 的 forward kinematics（`isaaclab.utils.math.compute_body_poses` 或等价 API）一次，缓存单帧 → tile → 写 NPZ。~30 LoC 核心逻辑
  - **Option B**：若 IsaacLab FK 在环境外难调，用 mujoco python FK（`mujoco.mj_kinematics`）替代 —— G1 URDF 已经被 `src/holosoma_inference/holosoma_inference/models/g1/g1.xml` 加载过，可以 reuse
- **Schema 校验**：`tests/eval/test_static_motion_schema.py` — 用 `MotionLoader` load 新 NPZ，断言所有 required key 都在且 shape / dtype 对。**关键 assert**：`body_pos_w[:, wrist_idx, 2] > 0.8 m`（手是抬起来的不是垂下的）

**Sanity 验证（跑 S1 前一次性）**：用 `src/holosoma/holosoma/replay.py` 确认 G1 能稳定保持 pose 30s 不漂。

### 3.2 三段式 schedule（per wrist）

```
时间：    0s ─ 2s ─ 7s ─ 9s ─ 14s ─ 16s ─ 21s     （总 21s + 5s warmup）
          warmup  A   gap1  B   gap2   C
Phase A:  F_cmd = 0,         F_ext = 0         →  x_A  （baseline）
Phase B:  F_cmd = ±5N 外,    F_ext = 0         →  x_B  （预期 Δ ≈ +0.05m 往外）
Phase C:  F_cmd = ±5N 外,    F_ext = ∓5N 外    →  x_C  （预期 Δ ≈ 0，回到 x_A）
```

- `±` 指**左右 wrist 对称**（左：-Y 方向，右：+Y 方向，G1 world frame）
- `gap1` / `gap2`（各 2s）让瞬态稳下来；只取每段**最后 3s** 算 steady-state 平均
- `warmup`（5s）让 policy 从 spawn 稳定

### 3.3 Acceptance 准则（S1）— v2 阈值（codex review 后）

对每个 wrist `w ∈ {left, right}`：

| 检查 | 准则 | 理由 |
|---|---|---|
| **C1 — Phase A 稳定性** | 最后 3s `std(x_A[w]) < 0.01 m` | policy 连静态 motion 都 hold 不住，后面都是白测 |
| **C2 — Phase B 幅值** | `‖mean(x_B[w]) - mean(x_A[w])‖ ∈ [0.025, 0.075] m` | 预测 0.05m（F/K=5/100）±50%；比算术模型宽，因为 learned policy 不是理想弹簧。v3 拿到数据再收 |
| **C3 — Phase B 方向** | `cosine_similarity(x_B[w] - x_A[w], F_cmd_direction) > 0.8` | 偏移是沿着 F_cmd 方向（不是瞎飘）。**必须先归一化 delta** — 原来的 raw dot 维度错 |
| **C4 — Phase C 抵消** | `‖mean(x_C[w]) - mean(x_A[w])‖ < max(0.02, 0.4 * ‖B_offset‖)`，其中 `B_offset = mean(x_B[w]) - mean(x_A[w])` | 自适应：B 偏移小 → 抵消容差也小；B 饱和了 → 抵消容差宽一点。下限 2cm（sim 数值噪声 + policy 抖动） |
| **C5 — 不摔倒** | 26s 全程 `root_pos_w.z > 0.5 m` | 安全门槛 |
| **C6 — Baseline WBT reward** | Phase A 期间 `mean(reward_motion_tracking_exp) ≥ 0.4` | 力路径没把 motion reward 搞塌 |
| **C7 — Body-id / frame 校验** | eval 开头 log `left_wrist_body_id` 和 `right_wrist_body_id`；注入的 F_ext world-frame 值只在这两个 body 出现——diff `sim._robot._external_force_b[eid]` 验证别处都为零 | 防 Task 13.9a config 里 body name 打错（如 `left_wrist_pitch_link`）静默写到错的 body。correctness gate，不是 coverage gate |
| **C8 — Transient response** | A→B 切换后 wrist 位置在 500ms 内到达稳态偏移的 90% | 反应慢说明 policy 被 motion inertia 盖过去、没真 track F_cmd——silent failure 模式 |

**Gate**：C1-C8 在 ≥ 3 seeds 上全绿 → S1 过 → 进 S2。任何一项不过 → 找 root cause 修；不准跳过 S2-S4。

### 3.4 阈值背后的理由（v2 rationale）

- **C2 `[0.025, 0.075] m`**：codex 说原来 `[0.04, 0.06]` "aggressive — learned policy plus robot dynamics is not an algebraic spring"。真实有瞬态尾巴、policy deadband、workspace 小饱和。首跑允许 ±50% 带宽；看到分布再收到 30-40%
- **C3 cosine similarity**（不是 raw dot）：raw dot 的值随 delta 幅值（m 为单位）scale——微小移动就天然过不了。必须先归一化
- **C4 自适应**（`max(0.02, 0.4 * |B_offset|)`）：修两个问题 —— 原来固定 0.015m 和 C1 的 0.01m 噪声地板不自洽；而且如果 B 产生了很大偏移（如 policy overshoot 到 0.15m），3cm 的抵消残差其实已经很好，但固定 0.015m 会误判
- **C7 新加**：没它的话，config typo（`left_wrist_roll_link` vs `left_wrist_yaw_link`）会默默注力到错 body，其他检查**可能**还过（如果 roll link 和 yaw link 位置接近）。成本低，拦一整类 bug
- **C8 新加**：只看 steady-state 会漏掉"policy 最终能补偿但用了 3 秒"——部署体感差，操作员控制不了。500ms 先放粗估，跑完再调

所有原始值（per-timestep x、F_cmd、F_ext、motion_target、reward 各项）都落 NPZ，阈值离线重调不用重跑 sim。

---

## 4. Task 清单

task 前缀的 Phase 号是嵌进 parent plan 编号用的。

### Phase 7.5 — EVAL SANITY（**v4：当前唯一 active**）

- [ ] **Task 13.9 — S0 in-training force-aware sanity eval 入口**

  目标：做 `src/holosoma/holosoma/eval_agent.py` 的 force 版本，给 v10 ckpt 一个最薄的 rollout 入口 —— F_cmd 走训练同款随机 schedule，用于肉眼 + wandb sanity check 判断 policy 训好了没有。**不涉及 callback、scenario schedule、acceptance 判定、motion clip 合成、MuJoCo 部署等**（都在 backlog）。

  - **新文件**：
    - `src/holosoma/holosoma/eval_agent_force.py` —— 结构基本 copy `eval_agent.py`，差别只在 CLI schema 多一个 `--enable-force-cmd` flag + 在 `run_eval_with_tyro` 里落地这个 flag 的含义
      - **改 CLI schema**：在 `main()` 里 `tyro.cli` 层级间额外解析一个 `ForceEvalConfig(enable_force_cmd: bool = True)` frozen dataclass（放 `config_types/eval_force.py`），和 `CheckpointConfig` / `EvalCallbacksConfig` 同级，走 `return_unknown_args=True` 串联
      - **落地 flag 的含义**：`enable_force_cmd=False` 时，在 `setup_simulation_environment(tyro_config)` 之前对 `tyro_config.command` 做一次 `dataclasses.replace`，把三个 bucket（`setup_terms` / `reset_terms` / `step_terms`）里 `wrist_compliance_command` term 的 `wrist_compliance_config` 字段（`WristComplianceConfig`）替换成：
        ```python
        replace(cfg,
          force_cmd_magnitude_range=(0.0, 0.0),
          force_cmd_activation_prob_per_step=0.0,
          force_ext_magnitude_range=(0.0, 0.0),
          force_ext_activation_prob_per_step=0.0,
        )
        ```
        —— 这样 `WristComplianceCommand` 的 state machine 永远不会触发、buffer 恒为 0，actor obs 里 `wrist_force_command` 也恒为 0，相当于"干净 WBT rollout"的对照组
      - **`enable_force_cmd=True` 时完全不动 config** —— F_cmd / F_ext 走训练时的随机采样（`force_cmd_magnitude_range=(5,30)`，`force_cmd_activation_prob_per_step=0.01` 等 `WristComplianceConfig` 默认值）
      - **ckpt 兼容性检查**：跑前 assert `saved_cfg` 里 `command.setup_terms` 包含 `wrist_compliance_command`；不含（即 baseline WBT ckpt）→ `raise RuntimeError` 提示"该 ckpt 不是 force-aware，请用 `eval_agent.py`"
      - **ONNX 导出 / log dir / checkpoint 迁移** 等路径和 `eval_agent.py` **完全一致** —— 直接 reuse 原函数
    - `src/holosoma/holosoma/config_types/eval_force.py` —— `ForceEvalConfig` frozen dataclass（只有 `enable_force_cmd: bool = True` 一个字段 + docstring）
    - `tests/eval/test_eval_agent_force_cli.py`（pure CPU，subprocess 起 `--help`，秒级退出 0）—— 3 个 test：
      - `--help` 不 crash，输出含 `enable-force-cmd`
      - 用 mock ckpt（没 `wrist_compliance_command`）跑 → 提前 raise
      - `--enable-force-cmd False` 下，经过 replace 的 config 里所有 `force_*_magnitude_range` 和 `force_*_activation_prob_per_step` 都是 0（通过 dry-run 拦截，不真起 sim）
  - **append-only 修改**：无（不动 parent plan 任何文件，不动 `eval_agent.py`）
  - **文档修改**：
    - `src/holosoma/README.md` —— 在 `## Evaluation / ### In-Training Evaluation` 小节后面**追加**一个 `### In-Training Force-Aware Evaluation` 小节，**文风、格式、示例完全 mirror 原节**（只换命令 + 加一段 flag 说明），内容：
      ```markdown
      ### In-Training Force-Aware Evaluation

      For WBT wrist-force checkpoints (`exp:g1-29dof-wbt-force`), use the
      force-aware evaluation entry point, which reuses the training-time
      random force schedule:

      ```bash
      # Evaluate force-aware checkpoint from Wandb (F_cmd random, same as training)
      python src/holosoma/holosoma/eval_agent_force.py \
          --checkpoint=wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME> \
          --enable-force-cmd=True

      # Evaluate with F_cmd disabled (baseline-like behavior, regression check)
      python src/holosoma/holosoma/eval_agent_force.py \
          --checkpoint=<CHECKPOINT_PATH> \
          --enable-force-cmd=False
      ```

      `--enable-force-cmd` flag:
      - `True` (default): wrist F_cmd / F_ext sampled with the same random
        schedule used during training (`force_cmd_magnitude_range=(5,30) N`,
        ramp/hold/cooldown state machine, etc.). Use this to sanity-check
        that the policy responds to force commands.
      - `False`: all force channels zeroed; policy runs like baseline WBT.
        Use this to verify no regression vs a pure motion-tracking policy.

      Sanity checks to eyeball during rollout:
      - G1 tracks the motion clip (baseline WBT behavior)
      - With `--enable-force-cmd=True`, wrist positions show visible shifts
        when F_cmd triggers (not just motion inertia)
      - Wandb `Episode/rew_wrist_force_position_tracking_exp >= 0.4`
      - Wandb `Env/force/applied_f_body_{l,r} == Env/force/ext_magnitude_{l,r}`
      ```
  - **运行**：
    ```bash
    source scripts/source_isaacsim_setup.sh
    python src/holosoma/holosoma/eval_agent_force.py \
        --checkpoint <train_gate_ckpt> \
        --enable-force-cmd True
    ```
    对照组：改成 `--enable-force-cmd False` 再跑一次
  - **GATE S0**（qualitative，不自动判定）：
    - 两次 rollout 都跑满（不 crash、不 NaN）
    - `--enable-force-cmd True` 下肉眼看到 wrist 有 F_cmd 响应 + wandb `Episode/rew_wrist_force_position_tracking_exp` 稳定在训练末期水平
    - `--enable-force-cmd False` 下行为等同 baseline WBT（motion reward 不退化）
  - **依赖**：parent plan Phase 7 TRAIN GATE 产出的 ckpt；无其它

### Phase 7.6 — EVAL BACKLOG（**v4：暂缓，等用户 approve**）

下面这些 task 技术方案写在 §3、§4.5、§5-9 里做参考；**未经过用户明确 approve 不要开写**。按 approve 顺序执行。

- [ ] **Task 13.9a — [BACKLOG 待 approve] S1 paired-cancellation eval callback + motion clip + acceptance 测试**

  - **新文件**：
    - `src/holosoma/holosoma/data/motions/g1_29dof/eval/raised_arms_hold.npz`（由下面的 helper 从 G1 部署态 `default_dof_angles` rasterize；详见 §3.1）
    - `src/holosoma/holosoma/data/motions/g1_29dof/eval/tools/make_raised_arms_motion.py`（§3.1；读 `default_dof_angles` + FK 算出 body 位姿，tile 到 N 帧，velocities 清零，保留完整 `MotionLoader` schema）
    - `src/holosoma/holosoma/agents/callbacks/wrist_force_eval.py` — `WristForceEvalCallback(RLEvalCallback)`，结构 mirror `EvalPayloadCallback`：
      - 内部持一张 **declarative schedule**（`list[Phase(name, t_start, t_end, f_cmd_left_world, f_cmd_right_world, f_ext_left_world, f_ext_right_world)]`）
      - `on_pre_evaluate_policy`：
        - 通过 `simulator.find_rigid_body_indice("left_wrist_yaw_link")` 解左右 wrist `isaac_body_id`，断言 **resolved id != -1**；log 两个 id（C7 审计用）
        - 拿 `env.command_manager.get_state("wrist_compliance_command")` 的 handle
        - monkey-patch `env._apply_force_in_physics_step` 注 scripted F_ext（同 `EvalPayloadCallback` 的 pattern）
        - 置 `term._eval_override = True`，**把原值存到 `self._prev_override`** 备恢复
        - 以上全部包一层 `try:... except: self._cleanup(); raise`，防止初始化半截泄漏
      - `on_pre_eval_env_step`：读当前 sim-time `t`；对当前激活的 Phase，**override** command term 的内部 buffer：`term.force_cmd_b[:, wrist_idx] = quat_apply_inverse(base_yaw_quat, f_cmd_world)`、`term.force_ext_w[:, wrist_idx] = f_ext_world`；通过 flag 跳过 state machine（command `step()` 早 return）。同时验证当前时刻落在某个 Phase 内（无 gap 未处理）
      - `on_post_eval_env_step`：写 NPZ buffer——**v2 扩展集**：
        - `f_cmd_w_commanded[T,2,3]`、`f_ext_w_commanded[T,2,3]`
        - `wrist_pos_w[T,2,3]`、`motion_target_w[T,2,3]`
        - `reward_motion_tracking[T]`、`reward_wrist_force_tracking[T]`
        - `phase_name[T]`（变长 string 数组或 categorical int）
        - `sim_time[T]`、`root_pos_w[T,3]`（给 C5 用）
        - **`applied_force_by_body[T, n_bodies, 3]`** — sim 端完整外力 tensor（C7 验证 wrist 之外全为零）
        - **`transient_90pct_time[per phase transition]`** — 在线计算，给 C8 用
      - `on_post_evaluate_policy`：**一定**（finally clause）调 `self._cleanup()`：恢复 `_eval_override` 到 `self._prev_override`，恢复 `env._apply_force_in_physics_step` 到保存的原 handler，关 NPZ。然后跑 C1-C8，写 `acceptance_json`；任一 fail 则 `raise RuntimeError`（CI fail-loud）
      - `_cleanup()`：idempotent，二次调用安全。显式写出来是为了让异常路径不把 `WristComplianceCommand` 留在 override 状态污染同 GPU 上后续的训练
    - `src/holosoma/holosoma/config_types/eval_callback.py` — append `WristForceEvalConfig` + `WristForceEvalCallbackConfig`；把 `wrist_force_eval` 字段注册进 `EvalCallbacksConfig`
    - `src/holosoma/holosoma/agents/callbacks/scenarios/paired_cancellation.py` — schedule factory：`build_paired_cancellation_schedule(f_cmd_magnitude: float = 5.0, k_virtual: float = 100.0) -> list[Phase]`
    - `tests/eval/test_wrist_force_eval_callback_unit.py`（pure CPU，mock env / command term / sim）— **v2 扩展**：10 个测试
      - schedule 解析、Phase time 查询、buffer override 语义
      - acceptance pass/fail（C1-C8 每项 pass + fail fixture）
      - phase_name string 数组、`_eval_override` flag 生效
      - **正常退出路径 cleanup 恢复 state**
      - **异常路径 cleanup 恢复 state**（在 `on_pre_eval_env_step` 里注入 fault → 断言 `_eval_override=False`、`_apply_force_in_physics_step` 是原 handler）
      - body-name 错 → 断言 raise
      - transient 90pct 时间在合成 step response 上算对
    - `tests/eval/test_static_motion_schema.py` — load `raised_arms_hold.npz` 进 `MotionLoader`，断言所有 required key + shape/dtype（对应 §3.1 验证）
  - **Append-only 修改**（很轻）：
    - `src/holosoma/holosoma/managers/command/terms/wbt_force.py` — `WristComplianceCommand.step()` 开头 `if getattr(self, "_eval_override", False): return`（一行；训练路径完全不受影响）
    - `src/holosoma/holosoma/config_types/eval_callback.py` — append 新 dataclass + 注册（≤ 30 LoC）
  - **运行 & gate**：
    ```bash
    python src/holosoma/holosoma/eval_agent.py \
        --checkpoint <train_gate_ckpt> \
        --command.setup_terms.motion_command.params.motion_config.motion_file \
            "holosoma/data/motions/g1_29dof/eval/raised_arms_hold.npz" \
        --eval-callbacks.wrist_force_eval.config.enabled True \
        --eval-callbacks.wrist_force_eval.config.scenario paired_cancellation \
        --eval-callbacks.wrist_force_eval.config.output-path eval_wbt_force_s1.npz \
        --training.num-envs 1 \
        --training.seed 1
    ```
    改 seed 重复 2、3 跑。跑完 aggregator `tools/summarize_wrist_force_eval.py` → 打印 pass/fail 表
  - **GATE S1**：3/3 seeds 全过 C1-C8 → 进 Task 13.9b

- [ ] **Task 13.9b — [BACKLOG 待 approve] S2 F_ext sweep benchmark**

  - 复用 Task 13.9a 的 `WristForceEvalCallback`，只换 scenario：`build_ext_force_sweep_schedule`
  - Schedule：4 个 bin（F_ext 幅值 0 / 10 / 20 / 30 N）× 3 个方向（body frame +X / +Y / +Z，每步 world 转换）× 10s 每段 = 120s；F_cmd 全程为 0
  - 3 seeds × 1 motion（LAFAN 跳舞或 OMOMO pickup — 挑一个非静态 motion 压测真实 tracking）
  - 聚合指标（`tools/summarize_wrist_force_eval.py` 里）：
    - 每 F_ext bin 的 `wrist_pos_error_p50 / p95`
    - `fall_rate`（二元 per run）
    - 每 bin 的 `reward_motion_tracking_mean`
  - **GATE S2**：
    - 3 × 1 = 3 个 run 里跌倒次数 = 0
    - 所有 bin `wrist_pos_error_p95 < 0.30 m`（软阈值，跑完再调）
    - F_ext=0 bin 的 motion-tracking reward 均值在 TRAIN GATE baseline 5% 以内
  - **产物**：`docs/plans/eval-artifacts/wbt_force_s2_results.md` — 自动生成的每 bin 统计表
  - GATE S2 过 → 进 Task 22

- [ ] **Task 22 — [BACKLOG 待 approve] S3 sim-to-sim paired cancellation（MuJoCo 部署 eval）**

  - **新文件**：
    - `src/holosoma_inference/docs/workflows/sim-to-sim-wbt-force.md`（模板见 §5）
    - `src/holosoma/holosoma/utils/wrist_ext_force_injector.py` — 读 JSON schedule + wrist body 名；每步拿 `mujoco_backend.get_applied_forces_view()`，按 schedule 写 `xfrc_applied[body_id, :3]`。在 `run_sim.py` 主 loop 里调。~80 LoC。**backend 专用**：classic shape `[num_bodies, 6]`；warp shape `[num_envs, num_bodies, 6]` —— injector 通过 `isinstance(view, np.ndarray) vs torch.Tensor` 区分调度。**不和 IsaacSim 训练侧注力器（parent plan 的 `wbt_force_injected.py`）共享抽象** —— 两边 API 差很多，强行共享会掩盖 backend bug。**只共享 schedule JSON schema**（§6.1）
    - `src/holosoma/holosoma/config_values/wrist_ext_force_schedules/paired_cancellation.json` — canonical 3 段 schedule（和 S1 同）
    - **`src/holosoma_inference/holosoma_inference/inputs/impl/scripted.py`** — `ScriptedWristForceCmdProvider(WristForceCmdProvider)` 测试专用 provider。读 JSON 时间轴（和 `paired_cancellation.json` 用同一个 Phase schema，**单一来源**），按当前时间确定性返回 `WristForceCmd`。`start()` 记 `t_start = time.monotonic()`；`poll_wrist_force()` 查 `(time.monotonic() - t_start)` 所在的 Phase；`zero()` 强制输出零
    - `src/holosoma_inference/holosoma_inference/inputs/__init__.py` — factory 扩 `"scripted"` → 实例化 `ScriptedWristForceCmdProvider`
    - `tests/e2e/test_sim_to_sim_wbt_force_s3.py`（marker：`requires_inference`）— test driver 用 `subprocess.Popen` 起 `run_sim.py`（带 `--wrist-ext-force-schedule`）和 `run_policy.py`（带 `--task.wrist-force-input scripted --task.scripted-schedule-path paired_cancellation.json`），**不用 pexpect**。两个进程读**同一份 schedule JSON**，F_cmd / F_ext 天然时间对齐。驱动等 SIGTERM 后读 `run_sim.py` NPZ recording callback 的文件，跑 C1-C8。~150 LoC（比 pexpect 版本简单）
  - **Append-only 修改**：
    - `src/holosoma/holosoma/config_types/run_sim.py` — 追加 `wrist_ext_force_schedule: str | None = None`（CLI path）
    - `src/holosoma/holosoma/utils/sim_utils.py` `DirectSimulation.run()` 主 loop — 配置了 schedule 就起 `WristExtForceInjector`、每物理 step 调 `.step(sim_time)`（加 backend 兼容 guard：只支持 MuJoCo classic+warp；IsaacSim 路径 warn + skip）
    - `src/holosoma_inference/holosoma_inference/config_types/task.py` — `InputSource` Literal 扩 `"scripted"`，加 `scripted_schedule_path: str | None = None`
  - **运行**：
    ```bash
    # 终端 A
    source scripts/source_mujoco_setup.sh
    python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
        --wrist-ext-force-schedule src/holosoma/holosoma/config_values/wrist_ext_force_schedules/paired_cancellation.json

    # 终端 B
    source scripts/source_inference_setup.sh
    python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-force \
        --task.model-path wandb://<entity>/<project>/<run>/model_<step>.onnx \
        --task.wrist-force-input keyboard \
        --task.interface lo \
        --task.use-sim-time \
        --task.rl-rate 50
    ```
    操作员按 §5 的 keystroke 序列操作
  - **GATE S3**：C1-C8 在 MuJoCo 轨迹上全过（容差放宽：C4 < 0.025m，MuJoCo ↔ IsaacSim 物理 gap 吃掉一些精度）。无 crash。操作员照文档一次能复现
  - GATE S3 过 → 进 Task 23

- [ ] **Task 23 — [BACKLOG 待 approve] S4 baseline-vs-force A/B 对比** _(v2：按 codex 意见扩大)_

  - 跑两遍 S2 sweep：一遍用 v10 force-aware ckpt，一遍用 baseline `g1_29dof_wbt` ckpt（**同 compute budget** —— 同 step 数、同 motion set、无 force term）
  - baseline ckpt 没有 `wrist_force_command` obs —— 由 `WholeBodyTrackingPolicy`（dual-mode dispatch 自动识别）加载
  - **样本量（v2）**：**5 seeds × 3 motions × 4 F_ext bin = 每 ckpt 60 个 rollout，总 120**。motion：1 个静态（`raised_arms_hold`）+ 2 个非静态（一跳舞、一 pickup）；非静态 motion 用来抓"policy 在 motion 中段受力就掉链子"这种情况
  - 指标：`delta_error = error_baseline - error_force_aware`，按 `(motion, seed, F_ext bin)` 三元组**配对**。F_ext > 0 时预期正（force-aware 更好），F_ext = 0 时预期近零
  - **统计上报**：每 bin 对 paired delta 做 bootstrap 95% CI（1000 次重采样）。报 median + CI，不只报点估计。按 motion 拆开报，单个 motion 的离群不能劫持整个 gate
  - **GATE S4**（v2，工程 gate 用，非发论文声明）：
    - F_ext = 20 N bin 在全部 3 个 motion 上 `median(delta_error) > 0.03 m` **且** `95% CI 下界 > 0`（force-aware 是*可靠*更好，不只是"平均更好"）
    - F_ext = 0 bin 在全部 3 个 motion 上 `|median(delta_error)| < 0.02 m` **且** `95% CI 包含 0`（不退化）
    - 任何 motion、任何 F_ext > 0 bin 上 force-aware 都不差于 baseline
  - **要发论文级 claim**：扩到 10 seeds × 5 motions（codex 建议）；本 plan 只覆盖工程 gate
  - **产物**：`docs/plans/eval-artifacts/wbt_force_s4_ab.md`：
    - 每 bin bootstrap CI 表
    - 每 motion 分组柱状图（png）
    - 原始 CSV（每 rollout 的 per-timestep error）
    - "fall-rate delta" 列：baseline vs force-aware —— 高 F_ext 下 force-aware 不应该更容易摔

---

### 4.5 S4 baseline ckpt 依赖

Task 23（S4 A/B）要用**两个** ckpt 跑同一批 rollout：

| ckpt | 来源 | 谁跑 |
|---|---|---|
| `v10 force-aware` | parent plan Phase 7 TRAIN GATE 产出 | 已在 parent plan 里跑 |
| `baseline g1_29dof_wbt` | 用 parent plan baseline config 单独训 | **需要用户单独跑**，本 plan 不负责 |

**baseline 训练命令**（用户跑，不在 S1-S3 依赖路径上）：

```bash
# baseline ckpt：g1_29dof_wbt（无 force term）
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

**关键一致性约束**（不遵守 S4 对比无意义）：
- 用**同一套 motion**（和 v10 训练的 motion 集合一致 —— 通常是 `sub3_largebox_003_mj.npz` 等 OMOMO/LAFAN 条目）
- 用**同样的 seed 集**（S4 会 paired 比，seed 1/2/3/4/5 两边都要有）
- **同样 step 数 / compute budget**（避免 baseline 训练不足造成假 A/B 优势）

**如果 baseline 不可得**（时间预算紧）：
- 降级 plan A：只跑 S1-S3，Task 23 留到后续 iteration
- 降级 plan B："v10 带 F_cmd 输入（provider 正常发）vs v10 `provider.zero()`"—— 这是 ablation 不是 A/B，结论会弱很多，但能做

**触发时机**：S1（Task 13.9a）绿之后就可以 kick off baseline 训练，让它和 S2/S3 并行跑，等 S3 完成 baseline 也应该训完 → 再跑 S4。

---

## 5. Sim-to-sim workflow 文档 — `sim-to-sim-wbt-force.md` 结构

完整内容在 Task 22 里写。大纲：

1. **Overview** — 复用 `sim-to-sim-wbt.md` 的双终端架构（MuJoCo 物理 + policy 大脑）；在 policy 终端加 F_cmd keyboard channel，在 MuJoCo 终端加 F_ext scheduler
2. **Prerequisites** — 和 sim-to-sim-wbt 同 + Phase 7 TRAIN GATE 产出的 v10 ONNX ckpt
3. **Quick start — position-only replay（sanity）** — `--task.wrist-force-input constant --task.wrist-force-magnitude 0` 先确认 ckpt 能 load + 能跟 motion；行为应该和 sim-to-sim-wbt 一致
4. **Scenario A：Force control（只 F_cmd）** — 操作员用 keyboard `u/j/h/k/y/n`（左）和 numpad `8/2/4/6/9/3`（右）发 F_cmd；MuJoCo 不注 F_ext；验证 wrist 相对 motion 的偏移
5. **Scenario B：Paired cancellation（S1 的翻版）** — 分步 keystroke + MuJoCo CLI flag，对应 Task 22 acceptance sequence：
   - 按 `]` 启动 policy，`m` 启动 motion
   - 等 5s warmup / stiff control 稳
   - Phase A：不按任何键（F_cmd=0，MuJoCo schedule 也还在 F_ext=0 段——用 `paired_cancellation.json`，MuJoCo 侧 A/B/C 段自动推进）
   - Phase B：*policy* 终端按 `u` + numpad `8` 发 ±5N 向外 F_cmd；MuJoCo schedule 还在 F_ext=0
   - Phase C：MuJoCo schedule 自动推进到注 -5N 配对 F_ext；操作员保持 F_cmd 不变
   - 预期：B 段 wrist 外移，C 段 wrist 回位
6. **Controls 参考** — keyboard 表（F_cmd 键位继承自 parent Task 17）+ ROS2 topic
7. **安全** — `--task.wrist-force-magnitude-cap 30.0` 强制；stop-policy（`o` 键）同时清零 F_cmd 和 provider buffer
8. **Troubleshooting** — wrist 偏移幅值不对 → 查 ckpt metadata 的 K_virtual；抵消不回位 → sim↔train 有 gap 或 policy 没学会加性模型（回 IsaacSim 重跑 S1）
9. **链回 eval plan** — 原样引用 GATE S3 准则，备注 tolerance

---

## 6. 文件清单（create / modify 汇总）

### 6.1 新建

| 路径 | 作用 |
|---|---|
| `src/holosoma/holosoma/agents/callbacks/wrist_force_eval.py` | `WristForceEvalCallback` — schedule 驱动的 F_cmd/F_ext override + NPZ recorder + acceptance checker |
| `src/holosoma/holosoma/agents/callbacks/scenarios/paired_cancellation.py` | S1 三段式 Phase list |
| `src/holosoma/holosoma/agents/callbacks/scenarios/ext_force_sweep.py` | S2 sweep schedule |
| `src/holosoma/holosoma/data/motions/g1_29dof/eval/raised_arms_hold.npz` | 30s 静态 "raised arms" default pose（和 G1 WBT 部署态 stiff-control 起手一致）|
| `src/holosoma/holosoma/data/motions/g1_29dof/eval/tools/make_raised_arms_motion.py` | 从 `default_dof_angles` + FK rasterize 成 NPZ |
| `src/holosoma/holosoma/utils/wrist_ext_force_injector.py` | MuJoCo 侧 F_ext scheduler（读 JSON 写 `xfrc_applied`）|
| `src/holosoma/holosoma/config_values/wrist_ext_force_schedules/paired_cancellation.json` | S3 用的 MuJoCo 侧 schedule JSON |
| `src/holosoma_inference/holosoma_inference/inputs/impl/scripted.py` | `ScriptedWristForceCmdProvider` — 读 JSON 时间轴的测试专用 provider |
| `src/holosoma_inference/docs/workflows/sim-to-sim-wbt-force.md` | S3 操作员 runbook |
| `tools/summarize_wrist_force_eval.py` | NPZ 聚合器；打印 S1 / S2 表 |
| `tests/eval/test_wrist_force_eval_callback_unit.py` | pure CPU callback 测试 |
| `tests/eval/test_static_motion_schema.py` | `raised_arms_hold.npz` 的 schema 校验 |
| `tests/e2e/test_sim_to_sim_wbt_force_s3.py` | `requires_inference` marker；双进程 scripted 测试 |
| `docs/plans/eval-artifacts/wbt_force_s2_results.md` | 自动生成的 S2 总结 |
| `docs/plans/eval-artifacts/wbt_force_s4_ab.md` | 自动生成的 S4 A/B 总结 |

### 6.2 修改（append-only）

| 路径 | 改什么 |
|---|---|
| `src/holosoma/holosoma/config_types/eval_callback.py` | append `WristForceEvalConfig` + `WristForceEvalCallbackConfig`；注册进 `EvalCallbacksConfig` |
| `src/holosoma/holosoma/managers/command/terms/wbt_force.py` | `WristComplianceCommand.step()` 加一行 `_eval_override` 早 return |
| `src/holosoma/holosoma/config_types/run_sim.py` | 追加 `wrist_ext_force_schedule: str | None = None` |
| `src/holosoma/holosoma/utils/sim_utils.py` | `DirectSimulation.run()` 可选调 `WristExtForceInjector.step(sim_time)` |
| `src/holosoma_inference/holosoma_inference/config_types/task.py` | `InputSource` Literal 扩 `"scripted"`；加 `scripted_schedule_path` |
| `src/holosoma_inference/holosoma_inference/inputs/__init__.py` | `"scripted"` 路由到 `ScriptedWristForceCmdProvider` |

### 6.3 必须保持原样

- `envs/` 下所有东西（包括 parent plan 的 `wbt_force_injected.py`）
- `managers/reward/terms/wbt_force.py`
- 任何 parent plan 的 training config value
- `train_agent.py`

---

## 7. 执行顺序（v4：只画出 active path）

```
Phase 7 TRAIN GATE（parent plan Task 13.8）           ─── 绿 ───┐
                                                               ▼
Phase 7.5 EVAL SANITY（本 plan，v4 首发）
  Task 13.9（S0 in-training force-aware sanity eval）          ←── 当前位置

                                                                    │
                                                                    ▼
Phase 7.6 EVAL BACKLOG（v4：以下全部 GATE by 用户 approve）
  Task 13.9a（S1 paired cancellation）          [待 approve]
  Task 13.9b（S2 F_ext sweep）                  [待 approve]
  Task 22  （S3 sim-to-sim paired cancellation）[待 approve]
  Task 23  （S4 baseline A/B）                  [待 approve]
```

**v4 原则**：先只做 Task 13.9 的最薄 sanity 入口，跑出来肉眼看 ckpt 行不行。行 → 用户 approve Task 13.9a，写 scripted cancellation；不行 → 回 parent plan 找 training bug。
**不要**越级去写 backlog 任务的代码。

---

## 8. 待决项（首次跑完后再决定）

1. **S1 motion pose**：v3 用 "raised arms" default pose（已确定）。如果这个 pose 下 F_cmd 5N 产生的偏移太小（被 policy deadband 淹没），升到 10N
2. **在测的 K_virtual 值**：v10 v1 config 固定 100 N/m；parent plan 升到随机 range（v9.4 note）时，S1 要从 ckpt metadata 读 per-env K 然后按比例缩放 Δx 预期
3. **S3 时间预算**：每个 S3 run 大约 30s 真实时间（MuJoCo + stiff-control init + 21s schedule）。如果 MuJoCo 里抵消不干净，可能要把每段 hold 从 5s 加到 10s 积累 steady-state 统计
4. **S4 baseline ckpt**：见 §4.5 —— 由用户另外跑 `g1_29dof_wbt` baseline；plan 不负责训

---

## 9. 扩展 scenarios（v2/v3 — 初版不覆盖）

codex 标出这些重要但不 must-have。列为 future work，不卡 S1-S4。

| ID | Scenario | 重要性 | 何时加 |
|---|---|---|---|
| **E1** | F_cmd / F_ext 幅值不对称（如 F_cmd=5N, F_ext=-10N）| 测 policy 在不对称下的部分补偿 —— 真实接触几乎不会是干净的 ±对 | S1 绿以后 |
| **E2** | F_cmd 方向错（操作员手抖）| 确认 policy 跟 F_cmd 即便对抗 motion —— 安全属性 | S1 绿以后 |
| **E3** | 垂直 F_cmd（向下 / 向上 推 wrist）| 重力 + 接触耦合让 Z 轴和 X/Y 不一样；payload 类测试 | S1 绿以后 |
| **E4** | cap saturation（F_cmd = 30N+，预期被 clamp）| 验 `wrist_force_magnitude_cap` 端到端 | 和 Task 22（S3）一起 |
| **E5** | 只左 / 只右 不对称 F | 测力响应是否真的 per-wrist，还是被 MLP 共享特征耦合了 | S1 绿以后 |
| **E6** | K_virtual 随机化（parent plan v2）—— F_cmd=5N 加 K∈[50, 300] sweep | parent 把 `K_VIRTUAL_RANGE_N_PER_M` 从 (100,100) 打开到 range 后，S1 Δx 预测就变成 K 的函数；需要新 acceptance 曲线 | parent v2 落地后 |
| **E7** | 非静态 motion + F_cmd（跳舞时发力）| 静态 pose 是 easy case；真实部署是"跟复杂 motion 的同时加力" | 和 S4（Task 23）一起 |
| **E8** | 任意 motion 阶段的 F_cmd step response | 确认 policy 不止训过"motion 中段的 F"，也能 handle 新 F 在 held pose 上 | 未来 |

这些 scenario 复用 `WristForceEvalCallback` + schedule 基础设施；每个只是一张新 scenario JSON。

---

## 10. References

- Parent plan：[`docs/plans/2026-05-03-wbt-wrist-force-v10.md`](./2026-05-03-wbt-wrist-force-v10.md)
- UniFP 论文：`third_party/UniFP/UniFP.pdf` — Fig. 5(a) 是 paired cancellation 的类比；§4.1 是 F_ext sweep 的模板
- Holosoma eval callback pattern：`src/holosoma/holosoma/agents/callbacks/payload.py`
- Holosoma eval 入口：`src/holosoma/holosoma/eval_agent.py`
- Sim-to-sim WBT baseline workflow：`src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md`
- MuJoCo applied-force API：`src/holosoma/holosoma/simulator/mujoco/backends/base.py:210`（`get_applied_forces_view`）
