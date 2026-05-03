# WBT Training — Step-by-Step Code Tour（合并版）

这份文档是 Claude 和 Codex **各自独立**把 `demo_scripts/demo_omomo_wb_tracking.sh` 的训练阶段走一遍之后，交叉核对、对齐共识、合并冲突之后的最终版本。原始独立稿保留在：

- `docs/wbt-training-code-tour.claude.md`（Claude 的结构化 Phase A–G 版本）
- `docs/wbt-training-code-tour.codex.md`（Codex 的 325 步细粒度版本）

本合并版采用 Claude 的 **Phase 分段 / 时间线叙事**结构，同时吸收 Codex 补充的更多**细粒度步骤**（主要是 simulator 和 video recorder 内部调用）。两版存在分歧的地方在文末"**分歧与共识**"一节显式列出。

---

## 命令与上下文

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE
```

- 这是 `demo_scripts/demo_omomo_wb_tracking.sh` 的第 4 步，前置 3 步是：
  1. `source scripts/source_retargeting_setup.sh`（切到 retargeting 环境）
  2. `examples/robot_retarget.py ...`（SMPL-H → G1 中间结果）
  3. `convert_data_format_mj.py ...`（中间结果 → Holosoma 格式 `.npz`，50 FPS）—— 产出 `$CONVERTED_FILE`，指向 `src/holosoma_retargeting/.../converted_res/robot_only/sub3_largebox_003_mj_fps50.npz`
  4. `source scripts/source_isaacsim_setup.sh`（切到 IsaacSim 环境）
- 然后才是本 tour 覆盖的训练命令。`source_isaacsim_setup.sh` 决定了本次跑的是 **IsaacSim**；WBT manager 会在构造时 assert 拒绝 IsaacGym（`envs/wbt/wbt_manager.py:16`）。
- **[WBT]** = WBT 专属；**[IsaacSim]** = IsaacSim 专属（即与 IsaacGym/MJWarp 不同的执行路径）。

覆盖范围：**Python 启动 → 第一次 `.backward()` → 之后每个 PPO iteration → checkpoint/ONNX → 关停**。

---

## Phase A — 进程启动、tyro 解析

### A1. 顶层 import（`train_agent.py:1-25`）
加载 `tyro / loguru / holosoma.config_* / holosoma.utils.*`。**注意此时还没 `import torch`、也没 `import isaacsim/isaacgym`**。torch 的 import 被推迟到 sim app 启动之后（`train_agent.py:167-168`），以满足 IsaacGym 的 "torch must come after isaacgym" 约束；IsaacSim 则要求 `AppLauncher` 最先完成。公用模块里如果要用到 torch，会走 `holosoma.utils.safe_torch_import`（`utils/safe_torch_import.py:1`）——它先尝试 `import isaacgym` 再 `import torch`。

### A2. `main()` → tyro CLI（`train_agent.py:322-325`）
```python
tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
print(tyro_cfg.curriculum)   # L324，调试用打印
train(tyro_cfg)
```

- `AnnotatedExperimentConfig`（`config_values/experiment.py:25-32`）是 tyro 的 subcommand union，基于 `DEFAULTS`（L14-23）把 `exp:g1-29dof-wbt` 映射到 `g1_29dof_wbt`（`config_values/wbt/g1/experiment.py:18-82`）。
- `logger:wandb` 是 `ExperimentConfig.logger` 字段上的另一个 union（`config_types/experiment.py:160` + `config_values/logger.py:5`），选中 `WandbLoggerConfig(mode="online")`。
- `--command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE` 是 tyro 的 dotted-path override；落点是 `config_values/wbt/g1/command.py:17-38` 里 `MotionConfig.motion_file`。
- `TYRO_CONIFG`（`utils/tyro_utils.py:4`）启用 cascading subcommands、禁用 flag 转换、允许 Python literal 收集语法。
- 返回的 `tyro_cfg` 是 **完全展开的 `ExperimentConfig` dataclass**。

### A3. `exp:g1-29dof-wbt` 默认值一览（`config_values/wbt/g1/experiment.py:18-82`）
```
env_class                = "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager"   # L24 [WBT]
algo                     = PPO, override:
    num_learning_iterations=30000  (L29)     num_learning_epochs=5 (L30)
    save_interval=4000   entropy_coef=0.005  init_noise_std=1.0
    actor_lr=1e-3  critic_lr=1e-3  init_at_random_ep_len=True
    empirical_normalization=True  use_symmetry=False  weight_decay=0.0
simulator                = simulator.isaacsim  (L43-52)      [IsaacSim]
    max_episode_length_s = 10.0 s  (L49)
robot                    = g1_29dof, override: action_scale=0.25,
                           action_scales_by_effort_limit_over_p_gain=True,
                           enable_self_collisions=True, init pos=[0,0,0.76]
training.num_envs        = 4096   (L22)
```
PPO 基准值（`config_values/algo.py:12-56`）：`num_steps_per_env=24, clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0, max_grad_norm=1.0, schedule="adaptive", desired_kl=0.01, MLP hidden=[512,256,128] ELU`。

---

## Phase B — Sim app / torch / distributed 初始化

### B1. `train()` 入口（`train_agent.py:147-164`）
`main()` 直接调 `train(tyro_cfg)`，没传 `TrainingContext`，因此走 else 分支：
```python
simulation_app = init_sim_imports(tyro_config)   # L163
auto_close = True
```

### B2. `init_sim_imports` —— **最关键的 import 顺序**（`utils/eval_utils.py:272-298`）
```python
setup_simulator_imports(tyro_config)         # utils/sim_utils.py:33
# → set_simulator_type(...) 记录 SimulatorType.ISAACSIM
# → 若 IsaacGym/MuJoCo 就现在 import；IsaacSim 留给下一步
if get_simulator_type() == SimulatorType.ISAACSIM:
    return setup_isaaclab_launcher(tyro_config)     # utils/sim_utils.py:56-115
```

`setup_isaaclab_launcher` 做了：
1. **[IsaacSim]** `from isaaclab.app import AppLauncher`（延迟 import 到这里）。
2. 构造 `argparse.Namespace`：`num_envs`（多 GPU 下 `// world_size`，L86-88，避免单 rank 过度分配）、`seed`、`env_spacing`、`output_dir`、`headless`、`device`、`enable_cameras`（L103-106，只在 video 启用时）。
3. `AppLauncher(args_cli)` 启动 IsaacSim 的 `SimulationApp` —— **这一步间接触发了 `import torch` 和 omni.* 一大堆东西**。
4. 返回 `simulation_app`。

### B3. 延后的 torch / dist / wandb import（`train_agent.py:167-170`）
```python
import torch
import torch.distributed as dist
import wandb
```
torch 已经被 AppLauncher 预加载，这里再 import 只是拿引用。

### B4. 多 GPU / device / seed
- `configure_multi_gpu()`（`train_agent.py:62-100`）：读 `WORLD_SIZE`，>1 才 `init_process_group(backend=nccl, timeout=7200s)` 并 `torch.cuda.set_device(LOCAL_RANK)`。demo 单卡 → 返回 `None`。
- `get_device(...)`（`train_agent.py:103-121`）→ `"cuda:0"`。
- `seeding(seed+global_rank)`（`train_agent.py:198-201`）统一 seed torch/np/random/cuda。

### B5. Logger / 实验目录（`train_agent.py:185-195`）
- `wandb_enabled = True`（因为 `logger:wandb`）。
- `get_timestamp()` + `get_experiment_dir(..., task_name="locomotion")`。
- **注意（两稿共同指出）**：这里硬编码 `task_name="locomotion"` 给 WBT 用也没改，产出的实验目录会带 `locomotion`。Codex 把这一条明确标为"naming quirk"；`BaseTask._get_task_name()` 之后会再生成一次，两次并不同步 —— 属于代码历史遗留的小坑，不影响训练正确性，但路径上会让人困惑。
- `configure_logging(...)` 重配 loguru，rank0 INFO，其余 rank ERROR，并把 stdlib logging 桥过来。

### B6. rank0 `wandb.init`（`train_agent.py:206-244`）
从 `WandbLoggerConfig + TrainingConfig` 组 `project/name/entity/group/id/tags/mode/resume`；`dataclasses.asdict(tyro_config)` 整段 dump 进 wandb config；`wandb_run_path = "entity/project/run_id"` 存着，供 checkpoint 元数据用。

### B7. 多 GPU num_envs 拆分（`train_agent.py:247-256`）
`num_envs //= world_size`；用 `dataclasses.replace` 克隆配置。单卡 no-op。

---

## Phase C — 构造 env（WBT Manager + IsaacSim）

### C1. 取 class 构造 env（`train_agent.py:258-261`）
```python
env_target      = tyro_config.env_class      # WholeBodyTrackingManager
tyro_env_config = get_tyro_env_config(tyro_config)
env             = get_class(env_target)(tyro_env_config, device=device)
```

### C2. `WholeBodyTrackingManager.__init__`（`envs/wbt/wbt_manager.py:13-17`）
```python
super().__init__(tyro_config, device=device)
assert not hasattr(self.simulator, "gym"), "WBT requires IsaacSim — IsaacGym is not supported."
```
**[WBT]** 一行 assert 明确硬约束。

### C3. `BaseTask.__init__`（`envs/base_task/base_task.py:24-182`）
按行走：

1. **拆 config**（L42-73）—— 校验 11 段 manager 配置非 None，缺一 raise。
2. **关 JIT profiling**（L76-77）`torch._C._jit_set_profiling_*(False)`，减少首次 forward 抖动。
3. **二次 `get_experiment_dir`**（L80-85），`task_name = BaseTask._get_task_name()` 取类名小写（`wholebodytrackingmanager`）—— 与 A/B5 的 `"locomotion"` 不一致。
4. **`FullSimConfig`**（L88-94）把 simulator+robot+training+logger+experiment_dir 打包，给 simulator 构造函数。
5. **维度**（L96-100）：`num_envs=4096, dim_obs/dim_critic_obs/dim_actions` 来自 `robot_config`。
6. **TerrainManager + Simulator**（L102-105）：
   - `TerrainManager(terrain_config, self, device)` 先于 simulator，因为 simulator 要 terrain 提供的 env 原点。
   - `SimulatorClass = get_class("holosoma.simulator.isaacsim.isaacsim.IsaacSim")` → 实例化成 `self.simulator`。
7. **`IsaacSim.__init__` 内部**（Codex 深入的部分）：
   - **[IsaacSim]** `BaseSimulator` 拿 `logger.video` / `headless_recording`（`simulator/base_simulator/base_simulator.py:149,170,177`）。
   - 构造 IsaacLab `SimulationCfg`（dt/render_interval/device/PhysX/material）（`simulator/isaacsim/isaacsim.py:70`）。
   - 单例化 `SimulationContext`（L92）—— 重复构造会 raise。
   - `InteractiveSceneCfg(num_envs, env_spacing, replicate_physics)` + `InteractiveScene(...)` + `_setup_scene()`（L116-123）。
   - Terminal 启动时 `self.sim.reset()`（L145）；若启用了 video，构造 `IsaacSimVideoRecorder`（L186 / `simulator/isaacsim/video_recorder.py:30`）。
8. **Simulator 生命周期 call chain**（回到 `base_task.py:108-134`）：
   - `simulator.set_headless(...)` → `simulator.setup()`（`isaacsim.py:562`，设 `sim_dt = 1/fps = 1/200 = 0.005 s`）。
   - `self.dt = control_decimation × sim_dt = 4 × 0.005 = 0.02 s`（**50 Hz 控制频率，恰与转换后的 `.npz` 一致**）。
   - `max_episode_length = ceil(10.0 / 0.02) = 500`。
   - `simulator.setup_terrain()`（**[IsaacSim]** 目前是 pass，terrain 状态由 TerrainManager 独立管）。
   - `_load_assets()` → `simulator.load_assets()`（`isaacsim.py:568`），读 IsaacLab articulation 的 DOF/body 名；`isaacsim.py:612` 保留 config order；`isaacsim.py:655` assert DOF/body 名称与 robot config 对齐 —— WBT 的 motion index 化依赖这个对齐。
   - **[IsaacSim]** `is_isaacgym_manager = False` → `_update_tasks_before_termination = False`（L125）；randomization manager 推迟到后面构造（L149）。
   - `_create_envs()` → `simulator.create_envs(num_envs, env_origins, base_init_state)`（`isaacsim.py:665`）。
   - `get_dof_limits_properties()` 拉 `dof_pos_limits / dof_vel_limits / torque_limits`（`isaacsim.py:672`）。
   - `simulator.prepare_sim()`（`isaacsim.py:715`）建 root state proxy + state adapter；若有 video recorder，`setup_recording()`（L751）建 USD camera + replicator render product。
9. **7 个 manager 顺序构造**（L144-152）：
   ```
   ObservationManager (L144)
   ActionManager      (L145) → 实例化 JointPositionActionTerm (managers/action/terms/joint_control.py:15)
                                      → _configure_pd_gains + _configure_action_scales (L66-67)
   RewardManager      (L146) → 解析 WBT reward terms
   TerminationManager (L147)
   RandomizationManager (L149)   [IsaacSim 在这里才构造]
   CommandManager     (L151) → 注册 MotionCommand 到 setup/reset/step 三阶段
                                      (managers/command/manager.py:86)
   CurriculumManager  (L152)
   ```
10. **`_init_buffers()`**（L154 → `base_task.py:184-200` + **[WBT]** override `wbt_manager.py:18-26`）：
    - Base：`rew_buf / reset_buf / episode_length_buf / time_out_buf / extras / log_dict / _pending_*`。
    - **[WBT]** 追加：`base_quat`, `need_to_refresh_envs`, `_configure_default_dof_pos()`（把 `default_dof_pos` 广播成 `(num_envs, 29)`），domain rand push buffer。
11. **`simulator.prepare_manager_fields(...)`**（L158-162）扫装饰器元数据，把 per-env randomization tensor 扩张。
12. **各 manager `setup()`**（L165-174）：
    - `randomization_manager.setup()` **[IsaacSim]** 在此调，setup 完写 scene data + refresh sim tensors（`managers/randomization/manager.py:181`）。
    - `action_manager.setup()` → `JointPositionActionTerm.setup()`（`joint_control.py:78-103`）分配 `torques_substep / dof_pos_substep / dof_vel_substep`（shape `(num_envs, decimation=4, num_dof=29)`），attach actuator randomizer scale。
    - **[WBT]** `command_manager.setup()` → `MotionCommand.setup()`（`managers/command/terms/wbt.py:524-592`）—— **motion 文件在这里第一次被打开**（详见 C4）。
    - `curriculum_manager.setup()`、`terrain_manager.setup()`。
13. **`ResetEventManager`**（L177-179）wrap reset 时触发的事件。

### C4. MotionCommand.setup 深入（`managers/command/terms/wbt.py:524-592`）
- L525-531：记 `num_envs/device`，拿 `simulator._body_list / dof_names`；对 body name 应用 `FAKE_BODY_NAME_ALIASES`。
- L534：assert `motion_file OR motion_dir`。
- L538-551：demo 路径有 `motion_file`（=`$CONVERTED_FILE`），走 `MotionLoader(motion_file, robot_body_names_alias, robot_joint_names, device)`。
- **`MotionLoader.__init__`**（L33-51）→ `resolve_data_file_path(motion_file)`（L42）解析路径 → `_load_data_from_motion_npz()`（L73+）。
- **`_load_data_from_motion_npz`**（L73-150+）：
  - `cached_open(motion_file, "rb")` + `np.load(f)` 打开 `.npz`。
  - 校验 key：`fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, body_names, joint_names` 一个不能缺（缺则 raise）。
  - 读 `fps` / body/joint name / 原始 array。
  - **Holosoma 格式约定**（L123-125）：`joint_pos` 前 7 列是根 DOF（`xyz + wxyz`），`joint_vel` 前 6 列是根速度（`vel_xyz + vel_wxyz`）—— 都 strip 掉只留关节。
  - L137-138：`body_quat_w` 从 `.npz` 里是 **wxyz**，这里转成 **xyzw**。
  - L143-149：若含 object，`object_pos_w / object_quat_w / ...` 同样 load，`object_quat_w` 也 wxyz→xyzw。
- 回到 `setup()`（L554-592）：
  - 存 body/joint index 映射。
  - `_maybe_add_default_pose_transition(prepend=True/False)` 前后各 prepend/append 一段从默认姿态插值过来的过渡帧，避免起止处跳变。
  - 查 `ref_body_index`（L564，demo 是 `torso_link`）、`tracked_body_indexes`（14 个 body，见 `command.py:19-34`）。
  - `has_object` → `assert` **[IsaacSim]** `object` actor 只在 IsaacSim 支持（`wbt.py:575`）。
  - `use_adaptive_timesteps_sampler=True`（WBT default）→ 构造 `AdaptiveTimestepsSampler(motion.time_step_total, device, 1/env.dt)`（L580-583）。
  - `self.metrics = {}`；`init_buffers()`（L588）分配 `time_steps / body_pos_relative_w / body_quat_relative_w / ...`。
  - **[IsaacSim]** 非 headless 挂 visualization markers。

### C5. 回到 `train()`：存 config + 构造 PPO（`train_agent.py:264-289`）
- `env.observation_manager` 存在性断言。
- rank0：`tyro_config.save_config(experiment_save_dir / holosoma_config.yaml)` 并 `wandb.save()` 上云。
- `algo_class = get_class(tyro_config.algo._target_)` → `PPO`。

---

## Phase D — 构造 PPO + setup + 可选 load

### D1. `PPO.__init__`（`agents/ppo/ppo.py:178-200`）
- `super().__init__(env, config, device, multi_gpu_cfg)` —— `BaseAlgo` 存 env/config/device/gpu info。
- 建 `TensorboardSummaryWriter(log_dir)` + `LoggingHelper(num_envs, num_steps_per_env, num_learning_iterations, ...)`.
- `_init_config()`（L202-224）：
  - `algo_obs_dim_dict = env.observation_manager.get_obs_dims()` —— 决定 actor/critic 输入维度。
  - `actor_obs_keys = ["actor_obs"]`、`critic_obs_keys = ["critic_obs"]`（来自 `module_dict`，`algo.py:43-53`）。
  - `num_act = 29`（来自 `robot_config.actions_dim`，C3-5 assert 过 `num_dof == num_act`）。
  - 各 LR min/max 上下界。
- **首次 `env.reset_all()`（`ppo.py:200`）** —— 这是一次**实实在在的 reset + 一次 step**，把 obs_dict 填满。Codex 也把这一条单列出来（step 169）。
- 这次 reset 会触发 `wbt_manager.reset_all`（`wbt_manager.py:82-86`）→ 先 `motion_command.init_buffers()` 清 state，然后 `super().reset_all()`（`base_task.py:218-230`）。

### D2. `algo.setup()`（`train_agent.py:290` → `ppo.py:230-239`）
1. `_setup_models_and_optimizer()`（L241-278）：
   - `setup_ppo_actor_module(obs_dim_dict, module_cfg=actor, num_actions=29, init_noise_std=1.0, device, history_length)` → `PPOActor`（内部 MLP `154→512→256→128→29` + `nn.Parameter(log σ)`；具体数字在 §"维度"节）。
   - `setup_ppo_critic_module(...)` → `PPOCritic`（`~283→512→256→128→1`）。
   - **[WBT]** `empirical_normalization=True` → 两个 `EmpiricalNormalization`（`ppo.py:40-101`）被注册；Welford 算法，支持多 GPU all-reduce。
   - **[WBT]** `use_symmetry=False` → `symmetry_utils` 不建；后面 loss 里对应项为 0。
   - 多 GPU：`_synchronize_model_weights()` broadcast src=0（`ppo.py:787-797`）。
   - `actor_optimizer = AdamW(actor.parameters(), lr=1e-3, weight_decay=0.0)`；critic 同样一份。**两个 optimizer 完全独立**。
2. `_setup_storage()`（L309-332）：
   - `RolloutStorage(num_envs=4096, num_steps_per_env=24, device)` —— 底下 buffer shape `[T, N, *D]`（`agents/modules/data_utils.py:44`）。
   - register 11 个 key：`actor_obs / critic_obs / actions / rewards / dones / values / returns / advantages / actions_log_prob / action_mean / action_sigma`。

### D3. `attach_checkpoint_metadata` + 可选 `load`（`train_agent.py:291-297`）
- 把 `tyro_config + wandb_run_path` 挂到后续要保存的 checkpoint dict 里。
- 本 demo 没传 `--training.checkpoint`，跳过 `load_checkpoint(...)` / `algo.load(...)`（`ppo.py:650-669`）。

---

## Phase E — `algo.learn()`：主循环

`ppo.py:346-385`：

```python
self._train_mode()                              # L347  actor/critic/normalizer .train()
obs_dict = self.env.reset_all()                 # L349  ← 第二次全体 reset（Codex 亦注明）
if init_at_random_ep_len:                       # L353
    env.episode_length_buf = randint(0, 500)    # 打散各 env 的 episode phase
for obs_key in obs_dict: obs_dict[k] = obs_dict[k].to(device)

for it in range(current_learning_iteration, current_learning_iteration + 30000):
    if is_multi_gpu: self._synchronize_curriculum_metrics()
    with record_collection_time(): obs_dict = self._rollout_step(obs_dict)   # F1
    with record_learn_time():      loss_dict = self._training_step()         # F3
    if is_main_process: self._post_epoch_logging(it, loss_dict)              # F6
    if it % 4000 == 0 and is_main_process:
        self.save(f"model_{it:05d}.pt")          # F7
        self.export(f"model_{it:05d}.onnx")
```

**[WBT]** `init_at_random_ep_len=True`（`experiment.py:36`）—— 避免 4096 个 env 同步 done 切断 GAE。

---

## Phase F — 一次 iteration 内部

### F1. `_rollout_step`（`ppo.py:387-450`） — **24 步采样，不建图**
整体包在 `torch.inference_mode()` 内（L388）：forward 不建 autograd graph，返回 tensor 全 `requires_grad=False`，`actions_log_prob` 必须 `.detach()` 存；训练时重新 forward 一次才能拿到带梯度的 log-prob。

**每步循环**（24 次）：

1. **拼 obs**（L391-394）
   ```python
   actor_obs_raw  = cat([obs_dict[k] for k in actor_obs_keys],  dim=1)   # (4096, 154)
   critic_obs_raw = cat([obs_dict[k] for k in critic_obs_keys], dim=1)   # (4096, ~283)
   actor_obs  = actor_obs_normalizer(actor_obs_raw,  update=True)        # Welford 更新
   critic_obs = critic_obs_normalizer(critic_obs_raw, update=True)
   ```
   `EmpiricalNormalization.forward`（L53-62）在 `.training and update` 时顺手更新 running mean/std；多 GPU all-reduce（L69-83）。

2. **Actor sample + Critic value**（L396-397）
   ```python
   actions = actor.act({"actor_obs": actor_obs})            # Normal(μ, σ).sample(), (4096, 29)
   values  = critic.evaluate({"critic_obs": critic_obs}).detach()   # (4096, 1)
   ```

3. **`env.step({"actions": actions})`**（L399） → 详见 F2。

4. **Timeout bootstrap**（L405-413）—— truncation vs termination 的经典处理：
   ```python
   if infos["time_outs"].any():
       final_critic_obs = critic_obs_normalizer(cat(infos["final_observations"][k] for k...), update=False)
       final_values = critic.evaluate(...).detach()
       final_rewards += gamma * final_values * time_outs   # 把 γV(s_last) 加到这步 reward
   ```
   被 `max_episode_length` 切掉的 episode 不当作 terminal，critic 对 s_last 的估计要补偿回去，否则长 horizon value 被系统性低估。

5. **写 storage**（L416-426）—— `actor_obs / critic_obs / actions / values / actions_log_prob / action_mean / action_sigma / rewards(+final_rewards) / dones`，全部 `.detach()`。`RolloutStorage.add()`（`data_utils.py:48`）按当前 time index 拷贝。

6. **Actor/Critic 状态重置**（L429-430）`.reset(dones)` 对 MLP 是 no-op（无 RNN），接口留作 future hook。

7. **Logging helper** 累积 episode reward/length + 收 `infos["to_log"]`（`ppo.py:432-434` → `logging_utils.py:120`）。

**循环外**（L436-448）：
```python
last_values = critic.evaluate(last_critic_obs).detach()   # V(s_T)
returns, advantages = self._compute_returns_and_advantages(last_values, values, dones, rewards)
storage["returns"]    = returns
storage["advantages"] = advantages
```
`_compute_returns_and_advantages`（L452-472）反向递推标准 GAE：δ=r+γV'−V，Â=δ+γλÂ'，最后 advantages 做 standardize（多 GPU 全局 standardize，L467-471）。

### F2. `BaseTask.step` 内部（`envs/base_task/base_task.py:405-473`）

```python
self._pre_physics_step(actions)    # L408
self._physics_step()               # L409  decimation × (PD + simulate)
self._post_physics_step()          # L410  refresh + term + reward + reset + obs
return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras
```

**`_pre_physics_step`**（L413-415）→ `action_manager.process_actions(actions)`（`managers/action/manager.py:154`）校验 shape + 按 term 分发 → `JointPositionActionTerm.process_actions`（`joint_control.py:110-140`）：
- 存 `_raw_actions`；若 `clip_actions`，逐元素 clip + log `action_clip_frac`；若启用 delay 随机化走延迟队列，否则 `_actions_after_delay = _processed_actions`。

**`_physics_step`**（L417-421）4 个 substep（`control_decimation=4`）：
```python
self.render()
for _ in range(4):
    self._apply_force_in_physics_step()      # → action_manager.apply_actions()
    self.simulator.simulate_at_each_physics_step()
```
- `JointPositionActionTerm.apply_actions`（`joint_control.py:156-168`）：
  ```
  self.torques = self._compute_torques(_actions_after_delay)
  torques_substep[:, substep_idx] = self.torques
  simulator.apply_torques_at_dof(self.torques)
  _prev_dof_vel.copy_(simulator.dof_vel)
  ```
- `_compute_torques`（L170-214） —— **"low-level controller" 本尊**，`control_type='P'`：
  ```
  τ = kp_scale * kp * (action * action_scale + default_dof_pos - dof_pos)
    - kd_scale * kd * dof_vel
  ```
  - **[WBT]** `action_scale=0.25`，且 `action_scales_by_effort_limit_over_p_gain=True`（`experiment.py:57-58`）→ `_configure_action_scales`（`joint_control.py:326-337`）把每关节 `action_scale = 0.25 * effort_limit / kp`，让弱关节也能饱和。
  - 可选 torque RFI 随机化；可选 `torch.clip(τ, ±torque_limits)`。
- **[IsaacSim]** `simulator.apply_torques_at_dof` → `_robot.set_joint_effort_target(...)`（`simulator/isaacsim/isaacsim.py:805`）。
- **[IsaacSim]** `simulator.simulate_at_each_physics_step`（`isaacsim.py:812`）：step counter + bridge + write_data_to_sim + physics step + render + scene update；若 video 启用，`capture_video_frame()`（L854 → `simulator/shared/video_recorder.py:174`）。

**`_post_physics_step`**（L427-473） —— 顺序**对 reward/observation 正确性有实质影响**：

1. `_refresh_sim_tensors()`（L428 → `isaacsim.py:773`）同步 root state / dof state / contact force / rigid body state。
2. `episode_length_buf += 1`（L429）。
3. **[WBT]** `_pre_compute_observations_callback`（`wbt_manager.py:42-43`）刷 `base_quat`。
4. **Callback 顺序分歧点**（L434-444）：
   - **IsaacGym** (`_update_tasks_before_termination=True`)：先 `_update_tasks_callback()`（command/curriculum/randomization step）再 termination+reward。
   - **[IsaacSim]** / **[WBT]**：**先 termination → reward → log，再 `_update_tasks_callback()`**（L457-458）。**这是因为 WBT 要在参考帧 advance 之前检查 tracking error 作为 termination 条件**，否则 error 永远看起来在变小。历史背景见 commit `470fd78`。
5. `_check_termination()`（L522-531）清 `reset_buf/time_out_buf` → `termination_manager.check()`（`managers/termination/manager.py:82`）遍历 WBT termination terms → `reset_buf |= reset_flags | time_out_buf`。
6. `_compute_reward()`（L500-503）：`rew_buf[:] = reward_manager.compute(dt=0.02)`。`RewardManager.compute`（`managers/reward/manager.py:126-184`）逐 term 调 `func(env, **params) * weight * dt` 累加。**Reward 是纯数值，无梯度。**
7. **[WBT]** `_update_log_dict`（`wbt_manager.py:71-80`）写入 `average_episode_length` + `motion_command.update_metrics()` 产生的 tracking 指标。
8. **Final-obs**（L446-449）：对 `reset_buf` 非零的 env 先算一份 `modify_history=False` 的 terminal obs，存进 `extras["final_observations"]`，给 timeout bootstrap 用。
9. `reset_envs_idx(env_ids)`（L232-271）—— 子步骤串行：
   - `simulator.on_episode_end(env_id)`（用于 video 录制 stop hook，`simulator/shared/video_recorder.py:346`）。
   - `observation_manager.reset(env_ids)` 清 history buffer。
   - `_reset_envs_idx_impl`（L273-290）：`_reset_buffers_callback` → `_reset_tasks_callback` → `_reset_robot_states_callback` → `_fill_extras`。
     - **[WBT]** `_reset_robot_states_callback`（`wbt_manager.py:88-91`）**刻意** pass —— 因为 robot/object 状态的 reset 由 `MotionCommand.reset` 接管。
   - 各 manager `.reset(env_ids)`：
     - **[WBT]** `command_manager.reset` → `MotionCommand.reset`（`wbt.py:594-...`）：
       - adaptive sampler 采 phase（L600-608），或 `torch.rand` 均匀采样。
       - 从 motion data 按 `time_steps` 查 root pose/vel/ori + dof pose/vel。
       - 加 initial-pose noise（`NoiseToInitialPoseConfig`），dof 加 noise 后 clip 到 soft joint limit。
       - 写回 simulator tensor（L714+）；有 object 还写 object state（L723 `simulator.set_actor_states(["object"], ...)`）。
   - `reset_manager.reset_scene(env_ids)`（`managers/reset_events/manager.py:50`）串行 reset events。
   - `simulator.on_episode_start(env_id)`（用于 video start hook，`video_recorder.py:323`）。
10. **[WBT]** `_refresh_envs_after_reset`（`wbt_manager.py:55-61`）—— 写回 root/dof state tensor、清 contact force history、`refresh_sim_tensors`、再跑一次 `_pre_compute_observations_callback`。
11. **[IsaacSim]** `_update_tasks_callback()`（L457-458 → `base_task.py:511`）：
    - `command_manager.step()` → `MotionCommand.step()`（`wbt.py:743-824`）：
      - time_steps += 1（可选 `freeze_at_timestep_zero_prob` 保留第一帧）；
      - 若片段走完，`self.reset(ended_env_ids)` 在线 resample（不 terminate 整个 episode）；
      - 更新 `body_pos_relative_w` / `body_quat_relative_w` —— **reward 和 observation 真正比较的 "target pose"**。
    - `curriculum_manager.step()`、`randomization_manager.step()`。
12. `_compute_observations()`（L505-506）→ `observation_manager.compute()`（`managers/observation/manager.py:74+`）：按 group 逐 term 求 → apply noise → apply scale → push history → cat 成 per-group tensor。**[WBT]** `actor_obs` group 带 noise（`dof_vel=0.5, base_ang_vel=0.2 ...`）；`critic_obs` group 全 0 noise。
13. `_store_final_observations` / `_post_compute_observations_callback` / `_clip_observations`（按 `clip_observations` 整体 clip）。
14. `extras["to_log"] = log_dict`；viewer-only hook。

### F3. `_training_step`（`ppo.py:474-486`）
```python
generator = storage.mini_batch_generator(num_mini_batches=4, num_learning_epochs=5)
# flatten [T=24, N=4096, D] → [T*N=98304, D]；shuffle；切 4 片 → mbs=24576；外层 5 epoch 复用同 storage
for minibatch in generator:
    loss_dict = self._update_algo_step(minibatch, loss_dict)
storage.clear()
```
`mini_batch_generator`（`data_utils.py:105-126`）：`batch_size = num_envs * num_transitions_per_env = 98304`；`mini_batch_size = 98304/4 = 24576`；总更新次数 `5*4 = 20`。

### F4. `_update_algo_step` —— **整个训练里唯一一次 `.backward()`**（`ppo.py:488-516`）
```python
ppo_loss_dict = self._compute_ppo_loss(minibatch)
actor_optimizer.zero_grad(); critic_optimizer.zero_grad()
ppo_loss = ppo_loss_dict["actor_loss"] + ppo_loss_dict["critic_loss"]
ppo_loss.backward()                                            # ← 唯一 .backward()
if is_multi_gpu: self._reduce_parameters()                     # all-reduce grads
nn.utils.clip_grad_norm_(actor.parameters(),  max_grad_norm=1.0)
nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm=1.0)
actor_optimizer.step(); critic_optimizer.step()
```
**为什么两个独立 optimizer 还能共享一次 backward：** actor_loss 图只触及 actor MLP，critic_loss 图只触及 critic MLP（它们的 forward 输入都 detach 自 storage），所以一次 `.backward()` 同时建两张梯度，两个 optimizer 各 step 各的互不冲突。

### F5. `_compute_ppo_loss`（`ppo.py:518-621`）
**[WBT]** `use_symmetry=False` 的路径：
```python
actor.act({"actor_obs": minibatch["actor_obs"]})             # 重建 Normal(μ,σ)，这次带 grad
value_batch           = critic.evaluate({"critic_obs": minibatch["critic_obs"]})
log_prob_new          = actor.get_actions_log_prob(minibatch["actions"])
μ_new, σ_new, H_new   = actor.action_mean, actor.action_std, actor.entropy

# Adaptive KL LR（schedule=="adaptive", desired_kl=0.01） —— 见 F5.1
if schedule == "adaptive":
    kl_mean = KL(Normal(old_μ,old_σ) || Normal(new_μ,new_σ))   # 多 GPU all-reduce
    self._update_learning_rate(kl_mean)

# PPO-clip actor surrogate（ε=0.2）
ratio         = exp(log_prob_new - old_log_prob)
surrogate_loss= max(-A*ratio, -A*clamp(ratio, 1-ε, 1+ε)).mean()

# Clipped value loss
v_clipped     = old_v + clamp(v_new - old_v, -ε, +ε)
value_loss    = max((v_new - R)², (v_clipped - R)²).mean()

# Symmetry → 全 0 （WBT 关着）
entropy_loss  = H_new.mean()
actor_loss    = surrogate_loss - entropy_coef * entropy_loss + 0
critic_loss   = value_loss_coef * value_loss + 0
```
- **[WBT]** `entropy_coef=0.005`（`experiment.py:32`；比 locomotion 的 0.01 小），`value_loss_coef=1.0`。

#### F5.1 Adaptive KL LR（`ppo.py:637-648`）
- `kl > 2 * desired_kl` → LR /= 1.5
- `kl < desired_kl / 2` → LR *= 1.5
- 夹在 `[min_lr, max_lr]`（默认 `[actor_lr/100, actor_lr*10]` 之类，`ppo.py:214-219`）
- 分别写回 `actor_optimizer.param_groups[0]['lr']` / `critic_optimizer...`。

### F6. `_post_epoch_logging`（`ppo.py:754-763`）
把 `loss_dict + actor_lr + critic_lr + mean_noise_std` 交给 `LoggingHelper.post_epoch_logging`（`logging_utils.py:151`）→ `_logging_to_writer`（L280）写 TB；rank0 若有 wandb run，同步 `wandb.log(metrics, step=global_step)`（`logging_utils.py:341`）。

### F7. 每 `save_interval=4000` iter checkpoint + ONNX（`ppo.py:379-385`）
- `save`（`ppo.py:671-690`）：`{actor_state_dict, critic_state_dict, actor_optim, critic_optim, actor_obs_normalizer, critic_obs_normalizer, iter, infos, env_state, metadata}` → `logging_helper.save_checkpoint_artifact` 写 `.pt` + 上传 wandb。
- `export`（`ppo.py:692-752`）：
  - `actor_onnx_wrapper`（L819-833）把 `actor_obs_normalizer + actor.act_inference` 打包成单一 `nn.Module` —— 导出的 ONNX **自带** obs 归一化，部署端直接喂原始 obs。
  - **[WBT]** `motion_command is not None` → `export_motion_and_policy_as_onnx`（`utils/inference_helpers.py`）把当前 motion 段一起塞进 ONNX，便于 `holosoma_inference/run_policy.py` 做离线回放。非 WBT 走 `export_policy_as_onnx` + zero actor obs。
  - `attach_onnx_metadata(...)` 挂 `dof_names / kp / kd / action_scale / command_ranges / robot_urdf(_path)` 到 ONNX metadata —— 部署端从一个文件就能读到全部 PD 参数和 URDF。
  - `logging_helper.save_to_wandb(onnx_path)` 上传。

`learn()` 末尾（L383-385）再额外 save + export 一次，保证即使 `num_learning_iterations` 不是 `save_interval` 倍数也能落盘最终模型。

---

## Phase G — 关停

`train()` 结束（`train_agent.py:303-319`）：

1. rank0 `wandb.teardown()`（**必须在 SimApp close 之前**，否则 IsaacLab 关机卡住 wandb socket）。
2. `dist.destroy_process_group()`（多 GPU 才有）。
3. `finally: close_simulation_app(simulation_app)`（`utils/sim_utils.py:286-334`）—— **[IsaacSim]** 两个关键 workaround：
   - monkey-patch `omni.usd.get_context().close_stage` 为 no-op（否则 headless 挂起）。
   - 设 `SimulationContext._disable_app_control_on_stop_handle = True`（否则 `app_control_on_stop_handle_fn` 进死循环 render）。
   最后 `simulation_app.close(wait_for_replicator=False)`。
4. `logger.info("Training shutdown complete.")`（L319）。

---

## 维度与 hyperparameter 总表

| 维度 | 值 | 来源 |
|---|---|---|
| `num_envs` | 4096 | `config_values/wbt/g1/experiment.py:22` |
| `num_steps_per_env` | 24 | `config_values/algo.py:33` |
| `num_learning_iterations` | 30 000 | `experiment.py:29` |
| `num_learning_epochs` | 5 | `experiment.py:30` |
| `num_mini_batches` | 4 | `algo.py:17` |
| mini-batch size | 24 576 | `98304 / 4` |
| `sim fps` × `control_decimation` | 200 × 4 → 50 Hz 控制 | `config_values/simulator.py:40-41` |
| `max_episode_length_s` | 10.0 → 500 env steps | `experiment.py:49` |
| `gamma`, `lam`, `clip_param` | 0.99, 0.95, 0.2 | `algo.py:18-20` |
| `entropy_coef`, `value_loss_coef` | 0.005, 1.0 | `experiment.py:32` + `algo.py:21` |
| `max_grad_norm` | 1.0 | `algo.py:27` |
| `actor_lr` / `critic_lr` | 1e-3 / 1e-3 (WBT override) | `experiment.py:34-35` |
| `schedule`, `desired_kl` | adaptive, 0.01 | `algo.py:28-29` |
| `init_noise_std` | 1.0 (WBT override) | `experiment.py:33` |
| `empirical_normalization` | True (WBT override) | `experiment.py:37` |
| `use_symmetry` | False (WBT override) | `experiment.py:38` |
| `weight_decay` | 0.0 (WBT override) | `experiment.py:39-40` |
| Actor/Critic MLP | `[512, 256, 128]`, ELU | `algo.py:46,52` |
| Actor obs dim | 154*（14 body tracking + proprio） | `get_obs_dims()` 运行时 |
| Critic obs dim | ≈283*（actor + privileged） | `get_obs_dims()` 运行时 |
| `num_act` | 29 (G1 DOF) | `robot_config.actions_dim` |
| `action_scale` | 0.25, scaled by `effort_limit/kp` | `experiment.py:57-58` |
| `save_interval` | 4000 iters | `experiment.py:31` |

（* 精确值依赖运行时 `observation_manager.get_obs_dims()`；154 / 283 参见 `wbt-ppo-training-walkthrough.md §2`，未在本 tour 独立计算。）

---

## 调用栈速查（时间线）

```
main()                                                      # train_agent.py:322
  tyro.cli → tyro_cfg                                       # train_agent.py:323
  print(tyro_cfg.curriculum)                                # train_agent.py:324
  train(tyro_cfg)                                           # train_agent.py:325

train()                                                     # train_agent.py:147
  init_sim_imports                                          # utils/eval_utils.py:272
    setup_simulator_imports                                 # utils/sim_utils.py:33
    [IsaacSim] setup_isaaclab_launcher → AppLauncher        # utils/sim_utils.py:56-115
  import torch / dist / wandb                               # train_agent.py:168-170
  configure_multi_gpu                                       # train_agent.py:179
  get_device / seeding                                      # train_agent.py:180, 201
  wandb.init                                                # train_agent.py:242
  env = WholeBodyTrackingManager(tyro_env_config, device)   # train_agent.py:261
    BaseTask.__init__                                       # envs/base_task/base_task.py:24
      TerrainManager + IsaacSim(...)                        # L102-134
        [IsaacSim] SimulationCfg, SimulationContext,        # simulator/isaacsim/isaacsim.py:70-186
                   InteractiveScene, video recorder
      _load_assets, prepare_sim                             # base_task.py:118,134
      7 managers                                            # L144-152
      _init_buffers (base + WBT override)                   # L154 + wbt_manager.py:18
      prepare_manager_fields + manager.setup() ×            # L158-174
        MotionCommand.setup → MotionLoader.load_npz($FILE)  # managers/command/terms/wbt.py:524
    assert not simulator.gym  (WBT guard)                   # wbt_manager.py:16
  tyro_config.save_config + wandb.save                      # train_agent.py:276-280
  PPO(env, cfg, log_dir, device)                            # agents/ppo/ppo.py:178
    env.reset_all()                                         # ppo.py:200  (#1)
  algo.setup()                                              # ppo.py:230
    _setup_models_and_optimizer (actor/critic/norm/AdamW×2) # ppo.py:241
    _setup_storage (RolloutStorage 24×4096×…)               # ppo.py:309
  algo.learn()                                              # ppo.py:346
    env.reset_all()                                         # ppo.py:349  (#2)
    if init_at_random_ep_len: randomize ep lens             # ppo.py:353
    for it in 0..30000:
      _rollout_step  (inference_mode, 24 step)              # ppo.py:387
        for step in 24:
          actor.act + critic.evaluate                       # ppo.py:396-397
          env.step → pre/physics/post                       # base_task.py:405/413/417/427
            action.process → clip                           # joint_control.py:110
            for d in decimation:
              action.apply → PD τ → simulator               # joint_control.py:156 + isaacsim.py:805
              simulator.simulate_at_each_physics_step       # isaacsim.py:812
            refresh_sim_tensors                             # isaacsim.py:773
            [IsaacSim] termination → reward → log →         # base_task.py:442-444
                       reset_envs_idx → MotionCommand.reset # base_task.py:451 / wbt.py:594
                       [IsaacSim] _update_tasks_callback    # base_task.py:457
                           MotionCommand.step               # wbt.py:743
            observation_manager.compute                     # base_task.py:505
          timeout bootstrap                                 # ppo.py:407-413
          storage.add                                       # ppo.py:416
        _compute_returns_and_advantages (GAE)               # ppo.py:452
      _training_step                                        # ppo.py:474
        for minibatch in (5 epoch × 4 mb):
          _compute_ppo_loss (adaptive KL LR inside)         # ppo.py:518, 559, 637
          zero_grad → ppo_loss.backward()                   # ppo.py:491-495  ← 唯一 backward
          grad_clip(1.0) + AdamW.step ×2                    # ppo.py:501-505
      _post_epoch_logging (TB + wandb)                      # ppo.py:754
      if it % 4000 == 0:
        save (.pt) + export (.onnx + motion + metadata)     # ppo.py:379-385
  wandb.teardown + dist.destroy_process_group                # train_agent.py:305/310
  close_simulation_app (IsaacLab hang workaround)           # utils/sim_utils.py:286-334
```

---

## 分歧与共识（Claude vs Codex 交叉核对）

两稿**主要事实完全一致**。几个值得记录的差别：

| 议题 | Claude 稿 | Codex 稿 | 共识（以代码为准） |
|---|---|---|---|
| `env.reset_all()` 被调了几次 | 在 F1 前置里只明确提到 1 次（在 `learn()` 入口） | 单列 2 次：`ppo.py:200`（`__init__`）+ `ppo.py:349`（`learn`） | **2 次**。合并版已采纳 Codex 的更准确表述 |
| `print(tyro_cfg.curriculum)` | 未提 | 明确提到（step 29） | **存在**（`train_agent.py:324`），合并版保留 |
| `task_name="locomotion"` 硬编码 | 在 B5 提到"已知粗糙点" | 明确标 "naming quirk"（step 55） | 两稿一致 |
| `simulator.setup_terrain()` 对 IsaacSim 行为 | 未展开 | 明确说 **[IsaacSim]** 是 pass | 合并版采纳 Codex |
| IsaacSim 内部 `SimulationContext` 单例 | 未提 | 明确 `isaacsim.py:92` 重复会 raise | 合并版保留 |
| Video recorder 的 start/stop/capture call sites | 未覆盖 | 全覆盖 | 合并版引用 Codex 的行号 |
| `BaseSimulator.video` / `headless_recording` 逻辑 | 未覆盖 | 覆盖到 `L149,170,177` | 合并版引用 |
| ONNX metadata 项 | 给了列表 | 给了列表 | 一致 |
| IsaacSim 特有的 `capture_video_frame` 在 physics substep 里的调用点 | 未提 | 指到 `isaacsim.py:854` | 合并版引用 |
| Callback 顺序（IsaacSim 要先 term+reward 再 task advance） | 明确，附历史 commit `470fd78` | 明确 | 一致 |
| "唯一一次 `.backward()`" | 明确 | 明确 | 一致（`ppo.py:495`） |
| Two-optimizer-one-backward 原理 | 给了解释 | 没展开 | 合并版保留 Claude 的解释 |
| KL-adaptive LR 阈值公式 | 明确 `2×desired_kl` / `desired_kl/2` + `/×1.5` | 明确同公式 | 一致 |

**两稿都未在静态分析中完全验证的点**（合并版继承该"不保证"声明）：
- Actor/Critic obs 精确维度（154 / 283）—— 依赖运行时 `observation_manager.get_obs_dims()`。
- `MotionLoader._maybe_add_default_pose_transition` 具体拼接帧数。
- `MultiMotionLoader` 路径（demo 走 `motion_file`，不触发）。
- IsaacGym 专属分支（`_update_tasks_before_termination=True`）在本命令下不触发。
- `MotionCommand.init_buffers()` / `update_metrics()` 的完整实现（仅核实了调用点）。

---

## 与 `wbt-ppo-training-walkthrough.md` 的分工

- **`wbt-ppo-training-walkthrough.md`** — 按架构/维度组织：shape 先行、actor vs critic obs 对照表、reward term 权重、hyperparameter 速查。适合 **"查值"**。
- **本文（code tour）** — 按调用栈/时间顺序组织，每步有 `file:line`。适合 **"Debug / 第一次读通 pipeline / 修 bug 时定位"**。
- 两文数字已相互核对（`num_envs=4096 / num_steps_per_env=24 / iterations=30000 / 5 epochs × 4 mini-batches / control 50 Hz / episode 500 steps / 唯一一次 backward / timeout bootstrap` 等），不存在矛盾。
