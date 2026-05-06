#!/usr/bin/env bash

# v14 launcher for WBT wrist-force force-tracking training.
#
# Starts `train_agent.py exp:g1-29dof-wbt-force-v2`. This is the v14
# formulation (docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md):
# reward is the baseline WBT motion-tracking reward (no wrist_force reward
# term); F_cmd is sampled on the wrist; F_ext = -F_cmd (world) is injected
# into the sim; the actor observes F_cmd and the critic observes F_cmd + F_ext.
# The robot has to produce F_cmd to cancel F_ext and stay on the motion
# reference — learning force tracking, not virtual-spring compliance.
#
# Requires Ubuntu/Linux (IsaacSim is not supported on Mac).
#
# Usage:
#   bash demo_scripts/demo_wbt_wrist_force_training_v2.sh
#   bash demo_scripts/demo_wbt_wrist_force_training_v2.sh --training.seed 2
#   SIMULATOR=mjwarp bash demo_scripts/demo_wbt_wrist_force_training_v2.sh
#
# Overridable env vars:
#   SIMULATOR           (default: isaacsim)   one of {isaacgym, isaacsim, mjwarp}
#                       WBT presets bind simulator.isaacsim at construction,
#                       so `isaacsim` matches the reference demo_omomo script.
#   LOGGER              (default: wandb)      disable with LOGGER=stdout
#   SEED                (default: 1)
#   MOTION_FILE         (default: preset's baseline)
#                       absolute path; overrides motion_command.motion_file
#   SKIP_REINSTALL      (default: 0)          set 1 to skip pip install/isaaclab check
#   NUM_GPUS            (default: 1)          set >1 to launch via torchrun
#
# Any additional CLI flags are forwarded verbatim to train_agent.py after the
# built-in ones, so the last-write-wins order (tyro) lets callers override
# defaults, e.g.:
#     bash demo_scripts/demo_wbt_wrist_force_training_v2.sh \
#         --command.setup_terms.wrist_force_tracking_command.params.wrist_force_tracking_config.force_cmd_magnitude_range "[10.0, 40.0]"

set -e  # Exit on error

# Resolve SCRIPT_DIR / PROJECT_ROOT even if invoked via symlink or from another cwd
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect operating system and check if it's supported
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        echo "Detected Linux OS - proceeding..."
        ;;
    Darwin*)
        echo "Error: Mac OS is not supported. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
    CYGWIN*|MINGW*)
        echo "Error: Windows is not supported. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
    *)
        echo "Error: Unsupported operating system: ${OS}. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
esac

SIMULATOR="${SIMULATOR:-isaacsim}"
LOGGER="${LOGGER:-wandb}"
SEED="${SEED:-1}"
SKIP_REINSTALL="${SKIP_REINSTALL:-0}"
MOTION_FILE="${MOTION_FILE:-}"
NUM_GPUS="${NUM_GPUS:-1}"

# Sanity-check NUM_GPUS against CUDA_VISIBLE_DEVICES / nvidia-smi. NOTE:
# plain `python src/.../train_agent.py ...` does NOT use any GPU past cuda:0 —
# IsaacSim spreads some context across visible GPUs so nvidia-smi may show
# small allocations on every card, but the actual training process is
# single-rank unless launched via torchrun. Set NUM_GPUS>1 to opt in.
if [ "$NUM_GPUS" -gt 1 ]; then
    if ! command -v torchrun >/dev/null 2>&1; then
        echo "Error: NUM_GPUS=${NUM_GPUS} requested but torchrun is not on PATH."
        echo "Install pytorch or source scripts/source_isaacsim_setup.sh first."
        exit 1
    fi
fi

# Step 1: Source IsaacSim setup (matches demo_omomo_wb_tracking.sh step 3)
echo "Sourcing IsaacSim setup..."
cd "$PROJECT_ROOT"
unset CONDA_ENV_NAME
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

# Step 2: Ensure holosoma and isaaclab are installed in the IsaacSim env
if [ "$SKIP_REINSTALL" != "1" ]; then
    HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}"
    pip install -e "$PROJECT_ROOT/src/holosoma[unitree,booster]" --quiet
    if ! python -c "import isaaclab" 2>/dev/null; then
        echo "isaaclab not found, reinstalling..."
        pip install 'setuptools<81' --quiet
        echo 'setuptools<81' > /tmp/hs-build-constraints.txt
        PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \
            pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet
        rm /tmp/hs-build-constraints.txt
    fi
else
    echo "SKIP_REINSTALL=1 — skipping holosoma/isaaclab reinstall."
fi

# Step 3: Assemble train_agent command
#
# The v14 experiment is `exp:g1-29dof-wbt-force-v2` (this plan).
# The command preset defaults to the baseline wbt motion file shipped in the
# repo, so we do NOT need to pass --command.setup_terms.motion_command.params.motion_config.motion_file
# unless the user opts in via $MOTION_FILE.
TRAIN_ARGS=(
    "exp:g1-29dof-wbt-force-v2"
    "simulator:${SIMULATOR}"
    "logger:${LOGGER}"
    "--training.seed" "${SEED}"
)

if [ -n "$MOTION_FILE" ]; then
    TRAIN_ARGS+=(
        "--command.setup_terms.motion_command.params.motion_config.motion_file=${MOTION_FILE}"
    )
fi

# Forward any extra flags caller passed (tyro: last write wins)
if [ "$#" -gt 0 ]; then
    TRAIN_ARGS+=("$@")
fi

echo "Running WBT wrist-force-v2 (v14 force-tracking) training..."
echo "  exp        : g1-29dof-wbt-force-v2"
echo "  simulator  : ${SIMULATOR}"
echo "  logger     : ${LOGGER}"
echo "  seed       : ${SEED}"
echo "  NUM_GPUS   : ${NUM_GPUS}"
if [ -n "$MOTION_FILE" ]; then
    echo "  motion_file: ${MOTION_FILE}"
fi
if [ "$#" -gt 0 ]; then
    echo "  extra args : $*"
fi

# Print v14 TRAIN GATE checklist for convenience
# (docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md)
cat <<'EOF'

-----------------------------------------------------------------------
v14 TRAIN GATE — wandb indicators to eyeball during training
  (docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md):

  1. Train/mean_reward monotonically rises in first ~200 iter
  2. Baseline Episode/rew_motion_tracking_* holds / rises
     (this is the ONLY reward — no wrist_force reward term in v14)
  3. Env/force/active_frac_cmd steady-state in [0.2, 0.5]
  4. Env/force/cmd_magnitude_max <= 30 N
  5. Env/force/cmd_ext_alignment ~ -1.0  (anti-parallel sanity)
  6. Env/force/ext_magnitude_l ~ Env/force/cmd_magnitude_l   (|F_ext|=|F_cmd|)
     Env/force/ext_magnitude_r ~ Env/force/cmd_magnitude_r
  7. Env/force/applied_f_body_{l,r} ~ Env/force/ext_magnitude_{l,r}
     (sanity on world->body quat rotation at injection time)
  8. Actor/critic loss not diverging; Episode/rew_survival stable
-----------------------------------------------------------------------

EOF

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching multi-GPU via torchrun --nproc_per_node=${NUM_GPUS}"
    echo "  Each rank instantiates its own sim + policy; PPO all-reduces gradients."
    echo "  Per-rank num_envs defaults to preset (4096); total batch = 4096 * NUM_GPUS"
    echo "  (Strategy A — scale total batch). Override via --training.num_envs."
    torchrun --nproc_per_node="${NUM_GPUS}" --standalone \
        src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"
else
    python src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"
fi

echo "Done!"
