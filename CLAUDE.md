# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Holosoma is a humanoid robotics framework for training and deploying RL policies (PPO, FastSAC) and for retargeting human motion capture data to robots (Unitree G1, Booster T1). It spans four simulators — IsaacGym and IsaacSim and MJWarp for training, MuJoCo for evaluation/deployment — with a shared inference pipeline that runs in sim or on the real robot.

## Repository Layout

The repo is a mono-repo with three independently installable Python packages under `src/`:

- `src/holosoma/` — core training framework. Tasks: locomotion (velocity tracking) and whole-body tracking (WBT). The RL-facing entry points live here: `train_agent.py`, `eval_agent.py`, `run_sim.py`, `replay.py`.
- `src/holosoma_inference/` — inference/deployment pipeline. Runs ONNX policies in MuJoCo (sim-to-sim) or on real G1/T1 hardware via SDK bridges. Entry point: `run_policy.py`.
- `src/holosoma_retargeting/` — converts human motion capture (SMPL-H, mocap, LAFAN) into robot-executable motions for WBT training.

Each subpackage has its own `pyproject.toml`/`setup.py`, its own README, and its own tests. Install them with `pip install -e src/<pkg>`. The CI scripts in `tests/ci/` show the canonical install order (e.g. `pip install -e 'src/holosoma[unitree,booster]'` then `pip install -e src/holosoma_inference`).

## Environment Setup

Because each simulator has incompatible dependency stacks, the repo ships **one setup script per stack** in `scripts/`. Always pair `setup_*.sh` with the matching `source_*_setup.sh` — the `source_*_setup.sh` activates the right conda env / uv venv and prepends needed paths. Examples:

```bash
bash   scripts/setup_isaacgym.sh       && source scripts/source_isaacgym_setup.sh
bash   scripts/setup_isaacsim.sh       && source scripts/source_isaacsim_setup.sh   # Ubuntu 22.04+
bash   scripts/setup_mujoco.sh         && source scripts/source_mujoco_setup.sh     # MJWarp train + MuJoCo eval
bash   scripts/setup_mujoco_via_uv.sh  && source scripts/source_mujoco_uv_setup.sh  # uv alternative
bash   scripts/setup_inference.sh      && source scripts/source_inference_setup.sh
bash   scripts/setup_retargeting.sh    && source scripts/source_retargeting_setup.sh
```

`scripts/source_common.sh` defines the shared `WORKSPACE_DIR=$HOME/holosoma_deps` and `CONDA_ROOT=$HOME/miniconda3` used by all setup scripts — external SDKs/simulators are installed under `WORKSPACE_DIR`, not inside the repo.

## Common Commands

### Training

```bash
# Locomotion — G1 + FastSAC on IsaacGym
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:isaacgym logger:wandb --training.seed 1

# Locomotion — G1 + FastSAC on MJWarp (beta)
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:mjwarp logger:wandb

# Whole-body tracking demo pipeline (retargeting → data conversion → training)
bash demo_scripts/demo_omomo_wb_tracking.sh
bash demo_scripts/demo_lafan_wb_tracking.sh
```

All training/eval scripts accept `--help`. CLI is driven by **tyro** — colon-prefixed tokens (`exp:...`, `simulator:...`, `logger:...`) select **registered config variants** (see "Config System" below); `--dotted.path` flags override individual fields.

### Inference / Deployment

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py --help
```

Workflow guides live in `src/holosoma_inference/docs/workflows/` (real-robot-locomotion, real-robot-wbt, sim-to-sim-locomotion, sim-to-sim-wbt).

### Retargeting

```bash
python src/holosoma_retargeting/.../examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh
python src/holosoma_retargeting/.../examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new ...
```

### Tests

Tests use **pytest markers** registered in `conftest.py`:

- `isaacsim` — requires Isaac Sim
- `multi_gpu` — requires multiple GPUs
- `requires_inference` — requires the inference env

Invoke pytest with a marker filter, mirroring CI (`tests/ci/isaacgym_ci_tests.sh`):

```bash
# IsaacGym CI selection
pytest -s --ignore=thirdparty --ignore=src/holosoma_inference -m "not isaacsim and not requires_inference"

# Single file / single test
pytest -s path/to/test_file.py::TestClass::test_name
```

End-to-end tests live in `tests/e2e/` (`test_training.py`, `test_run_policy.py`). Nightly training sweeps are in `tests/nightly/`. Per-simulator CI scripts are in `tests/ci/`.

**IsaacGym import order is fragile.** `conftest.py` imports `from holosoma.utils.safe_torch_import import torch` at collection time to guarantee torch loads before isaacgym. Do not add earlier imports that could reverse this order.

### Linting / Type-Checking

```bash
pre-commit install        # one-time
pre-commit run --all-files
mypy .
```

Pre-commit enforces: **ruff** (lint + format), **mypy** (with pydantic plugin), **clang-format** for C/C++/CUDA, plus EOF/whitespace/LF fixers. Config lives in `pyproject.toml` (ruff), `mypy.ini`, and `.pre-commit-config.yaml`.

Ruff runs `select = ["ALL"]` with a large targeted ignore list — check `pyproject.toml` before arguing with a lint error; many rules are intentionally disabled. Also note the per-file ignores, e.g. `src/holosoma/holosoma/simulator/isaacsim/**` is fully disabled until Jenkins gets IsaacLab access.

mypy **excludes** `src/holosoma_inference/` and IsaacSim sim code; don't expect type coverage there. IsaacGym/MuJoCo/IsaacSim/ROS2/etc. are listed as `ignore_missing_imports`.

## Architecture

### Multi-simulator abstraction

`src/holosoma/holosoma/simulator/` defines a `base_simulator` interface implemented by `isaacgym/`, `isaacsim/`, and `mujoco/` (plus `shared/` helpers and `types.py`). The training loop is simulator-agnostic; picking `simulator:isaacgym` vs `simulator:mjwarp` vs `simulator:isaacsim` at the CLI swaps the backend without touching task/env/agent code. MuJoCo on the training side is MJWarp-only; plain MuJoCo is inference-only.

### Env + managers

RL tasks live under `src/holosoma/holosoma/envs/`:

- `base_task/` — shared base
- `locomotion/` — velocity tracking
- `wbt/` — whole-body tracking

Each task composes **managers** from `src/holosoma/holosoma/managers/`: `action`, `command`, `curriculum`, `observation`, `randomization`, `reset_events`, `reward`, `termination`, `terrain`. This is the primary extension seam — new terms (rewards, terminations, observations, etc.) are added as new managers or new entries inside a manager module, not by editing the env class.

### Agents

`src/holosoma/holosoma/agents/` holds the RL algos:

- `ppo/`, `fast_sac/` — the two supported trainers
- `base_algo/`, `modules/` — shared infra (networks, etc.)
- `callbacks/` — training hooks (video logging, ONNX export, push/payload evaluators, etc.). **IsaacGym is sensitive to callback ordering** — see commit `470fd78` for context before reordering callbacks.

### Config system (tyro + entry points)

Configs are **dataclasses** under `config_types/` with concrete instances in `config_values/`. `train_agent.py` loads an `ExperimentConfig` via `get_tyro_env_config` and tyro. Variant selectors (`exp:...`, `simulator:...`, `logger:...`) are bound through tyro's union-of-subcommands pattern — add new experiments/robots/simulators by registering new config values and annotating them.

`holosoma_inference` uses **Python entry points** (declared in `src/holosoma_inference/setup.py`) for plug-in discovery:

- `holosoma.sdk` — SDK bridges (`unitree`, `booster`)
- `holosoma.config.robot` — robot configs (`g1-29dof`, `t1-29dof`)
- `holosoma.config.inference` — inference configs (`g1-29dof-loco`, `t1-29dof-loco`, `g1-29dof-wbt`)

`src/holosoma/pyproject.toml` does the same for `holosoma.bridge` (`unitree`, `booster`). When adding a new robot or SDK bridge, register the entry point — don't hard-code it.

### Inference pipeline

`src/holosoma_inference/holosoma_inference/` splits responsibilities into:

- `policies/` — locomotion, wbt, dual_mode policy wrappers (loaded from ONNX or Wandb)
- `inputs/` — input providers (`api/`, `impl/`) for StateCommand, VelCmd; ROS2 imports are **deferred** (see per-file ignores)
- `sdk/` — hardware SDK adapters (unitree, booster)
- `models/`, `config/`, `utils/`

Many files use deferred imports (lazy imports inside functions) to keep optional heavy deps (ROS2, rclpy, joystick) from being imported when unused — this is why `pyproject.toml` exempts those files from ruff's `PLC0415`. Preserve the pattern when editing.

### Bridges

`src/holosoma/holosoma/bridge/` plugs trained policies into live hardware/middleware: `unitree/`, `booster/`, `ros2/`. Selected via the `holosoma.bridge` entry-point group.

## Wandb

`logger:wandb` turns on video logging, automatic ONNX checkpoint upload, and supports loading checkpoints directly from Wandb. Version is pinned (`wandb==0.22.0`) — do not bump casually; see comment in `src/holosoma/pyproject.toml`.

## Python Version

`requires-python = ">=3.8"` but IsaacGym wheels are cp310 and inference SDK wheels are cp310 — the practical target is **Python 3.10**. Mypy runs with `python_version = 3.8` so avoid 3.10+-only syntax in typed code (the ruff config also has `target-version = "py38"`).

## Sanity-Check Before Committing

- Did you `source` the correct `source_*_setup.sh` for the env you're working in? (Wrong env = wrong simulator = cryptic failures.)
- Pre-commit clean? (`pre-commit run --files <changed>` or just let the hook run.)
- If touching types: `mypy .` green? (Note: `src/holosoma_inference/` is excluded.)
- If touching managers/envs/agents: run the matching CI script locally (`tests/ci/isaacgym_ci_tests.sh` etc.) or at minimum `pytest -m "not isaacsim and not requires_inference"`.
- If reordering IsaacGym callbacks: verify against commit `470fd78` — ordering can cause sim instabilities.
