# WBT Wrist-Force v12 — Quaternion Convention Fix + Retrain

> **为什么有这份 plan**：v10 Task 3 实现 body-yaw→world 旋转时把 `env.base_quat` 当 wxyz（`w_last=False`）用了，实际全 repo baseline 都按 xyzw（`w_last=True`）处理 —— 这个 bug 同时污染了 (a) 训练时 reward 的 F_cmd→world 旋转（`reward/terms/wbt_force.py`），和 (b) eval 时 F_cmd 箭头可视化（`envs/wbt/wbt_force_injected.py`）。2026-05-05 dogfood v11 S0 时肉眼发现 F_cmd 橙色箭头方向不对，溯源回 quaternion convention。
>
> **影响**：现有 8000-step ckpt (`fx2yfu3r/model_08000.pt`) 在**错误的 reward 信号**下训成，policy 学到的是"错方向下的 F_cmd 响应"。训练收敛是因为 F_cmd 在 body-yaw 里各向随机采样 + reward 只看 `norm²` 的 error，错的 yaw 旋转不会让 reward 信号坍塌，但**训练/obs/reward 三者之间的 frame 语义不一致**，部署到真机或跨 sim 必然出问题。
>
> **范围**：这份 plan 只做 quat convention fix + 重训，不动 v11 S0 eval 入口（已经解耦）。

---

## 1. Bug 根因（TL;DR）

| 约定 | w 的位置 | `quat_apply(..., w_last=?)` |
|---|---|---|
| **xyzw**（全 repo baseline） | 最后一位 | `w_last=True` |
| **wxyz**（isaaclab `body_quat_w`） | 第一位 | `w_last=False` |

`env.base_quat` 来源：`simulator.base_quat = robot_root_states[:, 3:7]`（`simulator/isaacsim/isaacsim.py:777`，注释明确 `xyzw`）。全 repo 其它地方都用 `w_last=True` 对它做旋转（见 `observation/terms/wbt.py` / `locomotion.py` / `terrain/terms/locomotion.py`）。

**我们（v10 Task 3）错用 `w_last=False` 的两个位置**：
1. `src/holosoma/holosoma/managers/reward/terms/wbt_force.py:42-50` 里的 `_yaw_rotate_body_to_world` —— 训练时 F_cmd body-yaw → world 给 virtual-spring reward 用的
2. `src/holosoma/holosoma/envs/wbt/wbt_force_injected.py:309-327` 里的 `_rotate_body_yaw_to_world` / `_rotate_body_yaw_to_world_batch` —— eval 时 F_cmd 可视化 + env-owned wandb 指标用的

**正确无需动**：
- `_rotate_force_world_to_body` (`wbt_force_injected.py:296-307`) 用的是 `sim._robot.data.body_quat_w`（**IsaacLab articulation data，wxyz**）+ `isaaclab.utils.math.quat_apply_inverse`（也吃 wxyz）。这条路径 convention 对 —— user dogfood 时绿色 F_ext 箭头方向对，从侧面印证了 F_ext 的 world→body 旋转走的是正确分支。

---

## 2. 影响面（grep 结果）

仅 wrist-force v10 Task 3 代码受影响：

| 文件 | 行 | 现状 | 修正 |
|---|---|---|---|
| `managers/reward/terms/wbt_force.py` | 42 | `def _yaw_rotate_body_to_world(base_quat_wxyz: ...)` | rename → `base_quat_xyzw` |
| `managers/reward/terms/wbt_force.py` | 46 | `yaw_quat(base_quat_wxyz, w_last=False)` + `# (N, 4) wxyz` | → `w_last=True` + `# (N, 4) xyzw` |
| `managers/reward/terms/wbt_force.py` | 49 | `quat_apply(yaw_q_nw, flat, w_last=False)` | → `w_last=True` |
| `envs/wbt/wbt_force_injected.py` | 148-149 | `# base_quat on the env subclass is already wxyz` (comment) | → `xyzw`（删掉误导） |
| `envs/wbt/wbt_force_injected.py` | 313 | `base_q = self.base_quat[...]  # (1, 4) wxyz` | → `xyzw` |
| `envs/wbt/wbt_force_injected.py` | 314 | `yaw_quat(base_q, w_last=False)` | → `w_last=True` |
| `envs/wbt/wbt_force_injected.py` | 317 | `quat_apply(yaw_q2, force_b, w_last=False)` | → `w_last=True` |
| `envs/wbt/wbt_force_injected.py` | 323 | `yaw_quat(self.base_quat, w_last=False)` + `# (N, 4) wxyz` | → `w_last=True` + `xyzw` |
| `envs/wbt/wbt_force_injected.py` | 326 | `quat_apply(yaw_q_nw, flat, w_last=False)` | → `w_last=True` |

**不动**：
- `envs/wbt/wbt_force_injected.py:296-307` `_rotate_force_world_to_body`（用 `body_quat_w` + `isaaclab.utils.math.quat_apply_inverse`，wxyz 正确）
- `managers/command/terms/wbt_force.py` —— F_cmd 在 body-yaw 里采样，根本没做过 quat 旋转

### 2.1 测试 fixture 同步更新

| 文件 | 行 | 现状 | 修正 |
|---|---|---|---|
| `managers/reward/terms/tests/test_wbt_force_reward.py` | 45 | param name `base_quat_wxyz` | → `base_quat_xyzw` |
| `managers/reward/terms/tests/test_wbt_force_reward.py` | 65-66 | identity `[1,0,0,0]` + `# wxyz identity` | → `[0,0,0,1]` + `# xyzw identity` |
| `managers/reward/terms/tests/test_wbt_force_reward.py` | 139-150 | 90deg yaw wxyz fixture | → 90deg yaw xyzw fixture |
| `managers/reward/terms/tests/test_wbt_force_reward.py` | 200 | `torch.zeros(2, 4)` 无 w → 这是纯 zero test，不改 | 保持 |
| `envs/wbt/tests/test_wbt_force_injected.py` | 94-95 | `env.base_quat[:, 0] = 1.0  # wxyz identity` | → `env.base_quat[:, 3] = 1.0  # xyzw identity` |
| `src/holosoma/tests/test_wbt_force_wandb_metrics.py` | 139-140, 202-203 | 同上 wxyz identity 错在 index 0 | → index 3 xyzw identity（**codex review 发现漏改，否则新 code 会把 `[1,0,0,0]` 当 180° roll 解读，`yaw_quat` 剥 yaw 后归零，测试仍然 pass 但没真正 exercise xyzw 语义）|

### 2.2 具体 quaternion 数值

90° yaw 绕 z 轴（body +x → world +y）：
- wxyz：`[cos(45°), 0, 0, sin(45°)] = [0.7071, 0, 0, 0.7071]`
- xyzw：`[0, 0, sin(45°), cos(45°)] = [0, 0, 0.7071, 0.7071]`

identity：
- wxyz：`[1, 0, 0, 0]`
- xyzw：`[0, 0, 0, 1]`

---

## 3. 重训必要性（**codex review 后修正**：先做 A/B 决定，再投 GPU-hour）

### 3.1 先验 corrected-reward on old ckpt（便宜的 gate，先做这个）

在花 4-6 GPU-hour 重训之前，先用**新 code（xyzw）+ 旧 ckpt `model_08000.pt`** 跑 eval，观察：
```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint logs/WholeBodyTracking/20260504_172258-g1_29dof_wbt_force_manager-locomotion/model_08000.pt \
    --training.headless True --training.max-eval-steps 2000
```
关注 wandb：
- `Episode/rew_wrist_force_position_tracking_exp` 在新 reward 下 steady 在哪个量级？
  - 若 ≥ 0.35：policy 已经学到的"wrist 对 F_cmd 响应"在新 frame 下还能工作（yaw 对称性救了我们），可能只需**短 fine-tune**
  - 若 0.2-0.35：部分保留，fine-tune 1-2k step 应该能救回
  - 若 < 0.2：方向完全错位，必须 from-scratch 重训

### 3.2 A/B 短 smoke（decision gate，~30 min）

```bash
# Arm A: fine-tune resume from old ckpt, 500 step
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-force simulator:isaacsim logger:wandb \
    --training.seed 1 --training.max-steps 500 \
    --checkpoint logs/WholeBodyTracking/20260504_172258-g1_29dof_wbt_force_manager-locomotion/model_08000.pt

# Arm B: fresh seed, 500 step
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-force simulator:isaacsim logger:wandb \
    --training.seed 2 --training.max-steps 500
```
比较 500 step 时两 arm 的 `rew_wrist_force_position_tracking_exp`：
- Arm A 明显领先（差 ≥ 0.1）：resume 路线省，继续 fine-tune 到 2000-3000 total step 就收工
- 两 arm 相近：from-scratch 8000 step 更稳，放弃 resume
- Arm B 领先（罕见）：旧 ckpt 被旧 reward 陷入局部最优，必须 fresh

### 3.3 全量重训（若 A/B 判定需要）

parent plan v10 Phase 7 training，`demo_scripts/demo_wbt_wrist_force_training.sh`，预计 4-6 GPU-hour 到 8000 steps。

### 3.4 已训 ckpt 的处置

- `logs/WholeBodyTracking/20260504_*` (fx2yfu3r) —— **归档，不删**。留作 "pre-quat-fix baseline"，做 sanity 对比
- 新 ckpt 命名按日期：`20260505_*-g1_29dof_wbt_force_manager-locomotion`

---

## 4. 执行步骤

### 4.1 Code fix（这台机器）

1. 按 §2 表改 6 处代码（`reward/terms/wbt_force.py` 3 处 + `envs/wbt/wbt_force_injected.py` 6 处）
2. 按 §2.1 改 4 处测试 fixture
3. 跑全套 CPU test：
   ```bash
   pytest src/holosoma/holosoma/managers/reward/terms/tests/test_wbt_force_reward.py \
          src/holosoma/holosoma/envs/wbt/tests/test_wbt_force_injected.py \
          src/holosoma/holosoma/config_types/tests/test_wrist_compliance_config.py \
          tests/eval/test_eval_agent_force_cli.py \
          src/holosoma/tests/test_wbt_force_wandb_metrics.py
   ```
   期望全绿。新增 2 个 **vector-level**（非 norm）direction-sensitive tests（codex [high] 建议）：
   - `test_yaw_rotation_vector_level_xyzw_identity`：identity → force 原样透过，按分量断言
   - `test_yaw_rotation_vector_level_90deg_xyzw`：90° yaw 下 body +x → world +y, +y → -x，按分量断言。"同模长错方向"不能通过 —— 这条是 convention 对错的硬 gate（codex finding [high]）

4. ruff / pre-commit clean
5. `codex:review` 给 diff 过一遍
6. 提交 + push

### 4.2 重训（按 §3 gate 决定路径）

先按 §3.1 + §3.2 跑 A/B 决定 fine-tune vs from-scratch。落定后：

**Fine-tune 路径（resume 旧 ckpt）**：
```bash
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-force simulator:isaacsim logger:wandb \
    --training.seed 1 --training.max-steps 3000 \
    --checkpoint logs/WholeBodyTracking/20260504_172258-g1_29dof_wbt_force_manager-locomotion/model_08000.pt
```

**From-scratch 路径**：
```bash
bash demo_scripts/demo_wbt_wrist_force_training.sh
# 或：
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-force simulator:isaacsim logger:wandb \
    --training.seed 1
```

新 ckpt 落地到 `logs/WholeBodyTracking/<timestamp>-g1_29dof_wbt_force_manager-locomotion/`。
训练中检查 wandb：
- `Episode/rew_wrist_force_position_tracking_exp` 应该在 ~3000-5000 step 爬到 ≥ 0.4（fine-tune 应该 ≤ 1000 step 就到）
- `Env/force/active_frac_cmd` / `active_frac_ext` 稳态 ∈ [0.2, 0.5]（schedule 没变）
- 若曲线比旧 run 显著更低（< 0.3 在 8000 step / from-scratch，< 0.35 在 3000 step / fine-tune），停下来 debug

### 4.3 Re-dogfood v11 S0

新 ckpt 训完 → 按 `docs/plans/2026-05-05-wbt-wrist-force-v11-eval.md` §7 跑 4 档。
重点肉眼确认：
- **档 1 (F_cmd only)**：橙色箭头从 wrist 发出、方向和 wrist 被推的方向**一致**
- **档 2 (F_ext only)**：绿色箭头同上（旧 ckpt 已确认对，新 ckpt 应该保持对）
- **档 3**：两色箭头共存不冲突

---

## 5. 风险 / 盲点

- **reward 方向翻转后训练不收敛的风险**：低但非零。F_cmd 采样完全 body-yaw 对称，reward 是 norm²，方向变了只改变哪个具体 yaw 下的 obs-action 映射。policy 要重学映射，但不需要新技能。**监控阈值**：如果 5000 step `rew_wrist_force_position_tracking_exp < 0.3`，说明还有其它 bug，立即停。
- **测试 fixture 翻转后现有 test 依然 pass 但语义悄悄变了**：低风险 —— 如果 fixture 的 "90deg yaw + body +x F" 和 "expect world +y" 这组断言在 identity 下能过、在 90°下也能过，说明旋转逻辑端到端对。
- **quat_conjugate / quat_apply_inverse 的 convention**：不动，因为我们不碰这些调用。但 codex review 要点一下：有没有**别的**地方对 `env.base_quat` 做过 `w_last=False` 假设被我 grep 漏了？
- **motion clip / command term 自身有没有 quat**：`force_cmd_b` 在采样时是 `sampled_magnitude * unit_vector`，unit_vector 本身在 3-D 里均匀采样，不涉及 quat —— 没有 frame 假设。安全。
- **部署路径**：`src/holosoma_inference/` 目前不消费 `wrist_force_command` obs（S3 backlog 才做）。这个 fix 不影响部署代码。但将来实现 S3 时，inference 侧如果也写 body-yaw→world，要从一开始用 xyzw。
- **F_ext 物理注入链路确认正确**（codex 问点 (f) 回答）：
  - `_rotate_force_world_to_body` (`wbt_force_injected.py:296-307`) 用的是 `sim._robot.data.body_quat_w`（**IsaacLab articulation data，wxyz**）+ `isaaclab.utils.math.quat_apply_inverse`（wxyz）。convention 对。
  - `set_external_force_and_torque(..., body_ids=..., is_global=<默认 False>)`（IsaacLab articulation.py:968 docstring 明确 "in their local frame"，且我们调用时没传 `is_global=True`）吃 **body-local frame**，和我们传入的 body-frame 3-vector 匹配。
  - `last_applied_force_w_by_body_id` 存的是 **F_ext 世界系原始值的 clone**（未经旋转），不参与任何物理注入 —— 纯作为 wandb 诊断用。这条自觉是 v11 plan §3.2 已经 flag 过的"同义反复"watch-item，和 quat fix 互相独立。

---

## 6. Codex review 要点（给审稿人）

请重点看：
1. §2 grep 结果是否**完整覆盖**了所有 `env.base_quat` / `wrist_cmd.force_cmd_b` 相关的 quat 旋转路径？有没有遗漏？
2. §2.2 的 xyzw / wxyz 数值换算对不对？90° yaw 绕 z 是否真的是 `[0,0,sin(π/4),cos(π/4)]` (xyzw)？
3. §3 "不重训直接跑新 reward policy 会退化" 这个判断对不对？有没有可能 fine-tune 一下就够（v.s. from scratch 8000 step）？
4. §5 的"测试 fixture 翻转后现有 test 依然 pass" —— 能否构造一个**独立验证**（比如构造已知 body-frame F 和已知 yaw，手算 world F，看代码输出是否一致）？这一步比纯 fixture 翻转更严格。
5. 还有没有**第三个** convention 相关的 bug —— 比如 reward 里用 F_cmd 的 magnitude metric 有没有被方向影响？（按理说 norm 是旋转不变，应该无事）

---

## 7. References

- Parent plan: [`2026-05-03-wbt-wrist-force-v10.md`](./2026-05-03-wbt-wrist-force-v10.md)
- v11 eval plan (不变): [`2026-05-05-wbt-wrist-force-v11-eval.md`](./2026-05-05-wbt-wrist-force-v11-eval.md)
- `env.base_quat` convention 来源: `src/holosoma/holosoma/simulator/isaacsim/isaacsim.py:777`
- `isaaclab.utils.math.quat_apply_inverse` 约定 wxyz —— 我们在 `_rotate_force_world_to_body` 用对了
- `holosoma.utils.rotations` 的 `yaw_quat` / `quat_apply` 签名: `src/holosoma/holosoma/utils/rotations.py:20-58`
