#!/usr/bin/env bash

# Phase 7 launcher for WBT wrist-force training (v10 plan §6 Task 13.8).
#
# Starts `train_agent.py exp:g1-29dof-wbt-force` directly — retargeting +
# data conversion are already in place (the baseline WBT motion file at
# src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz
# ships with the repo and the wbt-force command preset defaults to it, so
# unlike demo_omomo_wb_tracking.sh we do not rerun retargeting/conversion).
#
# Requires Ubuntu/Linux (IsaacSim is not supported on Mac).
#
# Usage:
#   bash demo_scripts/demo_wbt_wrist_force_training.sh
#   bash demo_scripts/demo_wbt_wrist_force_training.sh --training.seed 2
#   SIMULATOR=mjwarp bash demo_scripts/demo_wbt_wrist_force_training.sh
#
# Overridable env vars:
#   SIMULATOR           (default: isaacgym)   one of {isaacgym, isaacsim, mjwarp}
#   LOGGER              (default: wandb)      disable with LOGGER=stdout
#   SEED                (default: 1)
#   MOTION_FILE         (default: preset's baseline)
#                       absolute path; overrides motion_command.motion_file
#   SKIP_REINSTALL      (default: 0)          set 1 to skip pip install/isaaclab check
#
# Any additional CLI flags are forwarded verbatim to train_agent.py after the
# built-in ones, so the last-write-wins order (tyro) lets callers override
# defaults, e.g.:
#     bash demo_scripts/demo_wbt_wrist_force_training.sh \
#         --command.setup_terms.wrist_compliance_command.params.wrist_compliance_config.k_virtual_range "[50.0, 300.0]"

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

SIMULATOR="${SIMULATOR:-isaacgym}"
LOGGER="${LOGGER:-wandb}"
SEED="${SEED:-1}"
SKIP_REINSTALL="${SKIP_REINSTALL:-0}"
MOTION_FILE="${MOTION_FILE:-}"

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
# The new experiment is `exp:g1-29dof-wbt-force` (v10 plan §6 Task 9).
# The command preset defaults to the baseline wbt motion file shipped in the
# repo, so we do NOT need to pass --command.setup_terms.motion_command.params.motion_config.motion_file
# unless the user opts in via $MOTION_FILE.
TRAIN_ARGS=(
    "exp:g1-29dof-wbt-force"
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

echo "Running WBT wrist-force training..."
echo "  exp        : g1-29dof-wbt-force"
echo "  simulator  : ${SIMULATOR}"
echo "  logger     : ${LOGGER}"
echo "  seed       : ${SEED}"
if [ -n "$MOTION_FILE" ]; then
    echo "  motion_file: ${MOTION_FILE}"
fi
if [ "$#" -gt 0 ]; then
    echo "  extra args : $*"
fi

# Print Phase 7 TRAIN GATE checklist for convenience (v10 plan §6 Task 13.8)
cat <<'EOF'

-----------------------------------------------------------------------
Phase 7 TRAIN GATE — wandb indicators to eyeball during training
  (v10 plan §6 Task 13.8; gate must be fully green before starting
   Phase 8 inference/deploy):

  1. Train/mean_reward monotonically rises in first ~200 iter
  2. Episode/rew_wrist_force_position_tracking_exp rises to >= 0.4
  3. Env/force/active_frac_cmd & active_frac_ext steady-state in [0.2, 0.5]
  4. Env/force/k_virtual_l & k_virtual_r == 100.0 (flat line in v1)
  5. Env/force/cmd_magnitude_max <= 30 N
  6. Env/force/applied_f_body_{l,r} == Env/force/ext_magnitude_{l,r}
     (sanity on world->body quat conversion)
  7. Baseline Episode/rew_motion_tracking_* does not collapse
  8. Actor/critic loss not diverging; Episode/rew_survival stable
-----------------------------------------------------------------------

EOF

python src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"

echo "Done!"
