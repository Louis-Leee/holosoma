# Setup: Force-Aware WBT Evaluation on a Fresh Workstation

End-to-end steps to run `eval_agent_force.py` on a workstation that has **only miniconda + uv** preinstalled, starting from `git clone`.

Target audience: operator setting up a new box to dogfood the S0 eval entry point from `docs/plans/2026-05-05-wbt-wrist-force-v11-eval.md`.

---

## 0. Prerequisites

The workstation must already have:

- **miniconda** (default path `~/miniconda3`; override with `export CONDA_ROOT=/path/to/miniconda3` before running any setup script).
- **uv** (only used by the MuJoCo uv setup path; IsaacSim path does not need it).
- **NVIDIA GPU + 550+ driver** with CUDA 12.x runtime. `nvidia-smi` must work.
- **Ubuntu 22.04+**.
- **>= 80 GB free disk** (`~/holosoma_deps/` holds IsaacSim + torch; the repo itself is small).

`scripts/source_common.sh` pins `WORKSPACE_DIR=$HOME/holosoma_deps` and `CONDA_ROOT=$HOME/miniconda3`. Override via environment variables if you need different paths; otherwise accept the defaults.

---

## 1. Clone and switch branch

```bash
cd ~   # or your workspace directory
git clone git@github.com:Louis-Leee/holosoma.git
cd holosoma
git checkout feat/wbt-wrist-force-v10   # S0 code lives here
```

---

## 2. One-time IsaacSim stack install (~30-50 min)

```bash
bash scripts/setup_isaacsim.sh
```

What the script does (abridged):

- Installs miniconda at `$CONDA_ROOT` if missing (skipped if you already have it).
- `mamba create -n hssim python=3.11`
- `pip install torch==2.7.0 torchvision==0.22.0` (CU128 wheel)
- `pip install isaacsim[all,extscache]==5.0.0`
- Clones IsaacLab into `$WORKSPACE_DIR`, runs `./isaaclab.sh -i`
- `pip install -e 'src/holosoma[unitree,booster]'`
- `pip install -e src/holosoma_inference`
- Writes a sentinel file `$WORKSPACE_DIR/.env_setup_finished_hssim` so reruns are idempotent.

If the script fails mid-way (usually a network hiccup), re-run the same command — it picks up where it left off. **Do not** delete the sentinel unless you actually want a full reinstall.

---

## 3. Activate the environment

```bash
source scripts/source_isaacsim_setup.sh
```

Run this in every new shell. It does `conda activate hssim`, sets `OMNI_KIT_ACCEPT_EULA=1`, and prepends common paths.

---

## 4. Get a force-aware checkpoint

`logs/` is gitignored, so `git clone` does **not** bring checkpoints. Pick one:

### (a) Load from Wandb (cleanest)

`eval_agent_force.py --checkpoint` accepts a `wandb://<entity>/<project>/<run_id>[/<model_name>]` URI directly. Log in once:

```bash
wandb login   # paste API key when prompted
```

Then use the URI at eval time, e.g.

```bash
--checkpoint=wandb://dfnet/WholeBodyTracking/fx2yfu3r/model_08000.pt
```

(Run id `fx2yfu3r` is the 2026-05-04 force training run — find it in the startup log line `Checkpoint originated from W&B run`.)

### (b) rsync from another machine

```bash
rsync -av --progress \
    other-host:/data/project/holosoma/logs/WholeBodyTracking/20260504_172258-g1_29dof_wbt_force_manager-locomotion/ \
    logs/WholeBodyTracking/20260504_172258-g1_29dof_wbt_force_manager-locomotion/
```

### (c) Train a new checkpoint (hours of GPU)

Follows parent plan v10:

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-force simulator:isaacsim logger:wandb
```

---

## 5. Smoke test the pipeline (headless, 100 steps)

```bash
CKPT='wandb://dfnet/WholeBodyTracking/fx2yfu3r/model_08000.pt'

# Mode 0 — both force channels off
python src/holosoma/holosoma/eval_agent_force.py \
    --checkpoint "$CKPT" \
    --no-enable-force-cmd --no-enable-force-ext \
    --training.headless True \
    --training.max-eval-steps 100
```

Success looks like:

```
INFO | Shutting down simulation app...
INFO | Successfully patched close_stage method
```

If you see those two lines, the pipeline is wired up end-to-end (preflight contract passed, rehydrate worked, rewrite applied, rollout ran, ONNX exported, sim closed).

---

## 6. Full 4-mode dogfood run (per plan §7)

Operator Ctrl+C's when they have seen enough; there is no fixed step count.

```bash
# Mode 0 — baseline-like (both channels off)
python src/holosoma/holosoma/eval_agent_force.py --checkpoint "$CKPT" \
    --no-enable-force-cmd --no-enable-force-ext

# Mode 1 — F_cmd only
python src/holosoma/holosoma/eval_agent_force.py --checkpoint "$CKPT" \
    --no-enable-force-ext

# Mode 2 — F_ext only
python src/holosoma/holosoma/eval_agent_force.py --checkpoint "$CKPT" \
    --no-enable-force-cmd

# Mode 3 — full training schedule (defaults)
python src/holosoma/holosoma/eval_agent_force.py --checkpoint "$CKPT"
```

---

## 7. Seeing the IsaacSim viewer + debug arrows (plan §7.1)

When you want to eyeball the wrist force arrows, add:

```bash
--training.headless False --simulator.config.debug-viz True
```

To actually see the window from a remote workstation, pick one:

- **Option A (recommended): Isaac Sim Livestream.** Workstation starts a streaming server; your Mac runs the Omniverse Streaming Client. See the IsaacSim livestream docs.
- **Option B: X11 forwarding.** `ssh -X`. Slow, and often breaks on OpenGL driver mismatches.

Arrow colors (from `wbt_force_injected.py`):

- F_cmd = orange `(1.0, 0.5, 0.0)` — actor obs channel
- F_ext = green  `(0.0, 0.8, 0.0)` — sim-injected external force

---

## Common pitfalls

| Symptom | Cause / fix |
|---|---|
| `setup_isaacsim.sh` hangs during IsaacSim init | GPU / driver check. `nvidia-smi` must work; driver >= 550. |
| `Failed to startup plugin carb.windowing-glfw.plugin` in headless logs | Warning only. Expected when there is no X display. |
| `ModuleNotFoundError: No module named 'holosoma.config_types.eval_force'` after a fresh branch switch | The editable install cached `site-packages` paths. Re-run `pip install -e src/holosoma` or re-run `setup_isaacsim.sh`. |
| `Unrecognized options: False, False` from tyro | Wrong bool flag syntax. Tyro does not accept `--enable-force-cmd False` or `--enable-force-cmd=False` for a `default=True` bool. Use `--no-enable-force-cmd` to disable. |
| `wrist_compliance_config is dict, expected WristComplianceConfig` | Should not happen — `_rehydrate_wrist_compliance_configs` handles this. If it does, the rehydrate pass was skipped or the ckpt has a genuinely broken `CommandTermCfg`; inspect `load_saved_experiment_config` output. |
| `Dropping N unknown field(s) from ckpt WristComplianceConfig` warning | Expected for older ckpts that still carry removed fields like `debug_draw_total_arrow`. Harmless. |

---

## References

- Plan: [`docs/plans/2026-05-05-wbt-wrist-force-v11-eval.md`](plans/2026-05-05-wbt-wrist-force-v11-eval.md)
- Entry point: `src/holosoma/holosoma/eval_agent_force.py`
- Config: `src/holosoma/holosoma/config_types/eval_force.py`
- Tests: `tests/eval/test_eval_agent_force_cli.py`
- Setup scripts: `scripts/setup_isaacsim.sh`, `scripts/source_isaacsim_setup.sh`, `scripts/source_common.sh`
