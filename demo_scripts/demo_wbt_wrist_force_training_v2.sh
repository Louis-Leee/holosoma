#!/usr/bin/env bash

# v14 launcher for WBT wrist-force force-tracking training.
#
# This is an INDEPENDENT launcher. The structure (OS check,
# SKIP_REINSTALL, NUM_GPUS, torchrun) is *borrowed* from V10 as a
# reference pattern, but V14 and V10 launchers are maintained
# separately — changes to one will not affect the other.
#
# V14 has two obs-frame variants for F_cmd (ablation experiment):
#   FRAME=yaw       (default) -> exp:g1-29dof-wbt-force-v2
#                               F_cmd_obs = -R_yaw inverse (base_quat) . F_ext_w
#                               learning-compliance style; roll/pitch do NOT
#                               enter the force observation.
#   FRAME=fullbase           -> exp:g1-29dof-wbt-force-v2-fullbase
#                               F_cmd_obs = -R inverse (base_quat) . F_ext_w
#                               gentle-humanoid style; full base rotation
#                               enters the force observation.
# Both variants share env / reward / obs / algo; only the command preset
# (and therefore the obs frame) differs. Train one of each and compare
# motion-tracking reward curves under active F_ext windows.
#
# Common to both variants:
#   * F_ext (world frame) is sampled per-wrist with a trapezoidal schedule
#     (V14 owns its own _ForceChannel; no V10 dependency) and injected into
#     the sim via PhysX onto the two wrist bodies.
#   * Reward is the baseline WBT motion-tracking reward (tracking the
#     unmodified dataset reference motion, not a shifted target).
#
# Requires Ubuntu/Linux (IsaacSim is not supported on Mac).
#
# Usage:
#   bash demo_scripts/demo_wbt_wrist_force_training_v2.sh
#   FRAME=fullbase bash demo_scripts/demo_wbt_wrist_force_training_v2.sh
#   CURRICULUM=on bash demo_scripts/demo_wbt_wrist_force_training_v2.sh
#   bash demo_scripts/demo_wbt_wrist_force_training_v2.sh --training.seed 2
#   SIMULATOR=mjwarp bash demo_scripts/demo_wbt_wrist_force_training_v2.sh
#   NUM_GPUS=4 bash demo_scripts/demo_wbt_wrist_force_training_v2.sh
#
# Overridable env vars:
#   FRAME               (default: yaw)        one of {yaw, fullbase}
#   CURRICULUM          (default: off)        one of {off, on}; on -> enable force
#                                             magnitude curriculum (ramp 0-5 N to
#                                             5-30 N over CURRICULUM_RAMP_STEPS)
#   CURRICULUM_RAMP_STEPS (default: 10000)    ramp horizon in command steps
#   SIMULATOR           (default: isaacsim)   one of {isaacgym, isaacsim, mjwarp}
#   LOGGER              (default: wandb)      disable with LOGGER=stdout
#   SEED                (default: 1)
#   MOTION_FILE         (default: preset's baseline)
#   SKIP_REINSTALL      (default: 0)          set 1 to skip pip install/isaaclab check
#   NUM_GPUS            (default: 1)          set >1 to launch via torchrun
#
# Extra flags are forwarded verbatim to train_agent.py (tyro last-write-wins),
# e.g.:
#     bash demo_scripts/demo_wbt_wrist_force_training_v2.sh \
#         --command.setup_terms.wrist_force_tracking_command.params.wrist_force_tracking_config.force_ext_magnitude_range "[10.0, 40.0]"

set -e

# Resolve SCRIPT_DIR / PROJECT_ROOT even if invoked via symlink or from another cwd
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OS="$(uname -s)"
case "${OS}" in
    Linux*)
        echo "[v14] Detected Linux OS - proceeding..."
        ;;
    Darwin*|CYGWIN*|MINGW*)
        echo "[v14] Error: this script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
    *)
        echo "[v14] Error: unsupported OS: ${OS}."
        exit 1
        ;;
esac

FRAME="${FRAME:-yaw}"
case "${FRAME}" in
    yaw)      EXP_KEY="g1-29dof-wbt-force-v2" ;;
    fullbase) EXP_KEY="g1-29dof-wbt-force-v2-fullbase" ;;
    *)
        echo "[v14] Error: FRAME must be 'yaw' or 'fullbase', got '${FRAME}'."
        exit 1
        ;;
esac

SIMULATOR="${SIMULATOR:-isaacsim}"
LOGGER="${LOGGER:-wandb}"
SEED="${SEED:-1}"
SKIP_REINSTALL="${SKIP_REINSTALL:-0}"
MOTION_FILE="${MOTION_FILE:-}"
NUM_GPUS="${NUM_GPUS:-1}"

CURRICULUM="${CURRICULUM:-off}"
CURRICULUM_RAMP_STEPS="${CURRICULUM_RAMP_STEPS:-10000}"
case "${CURRICULUM}" in
    off|on) ;;
    *)
        echo "[v14] Error: CURRICULUM must be 'off' or 'on', got '${CURRICULUM}'."
        exit 1
        ;;
esac

if [ "$NUM_GPUS" -gt 1 ]; then
    if ! command -v torchrun >/dev/null 2>&1; then
        echo "[v14] Error: NUM_GPUS=${NUM_GPUS} requested but torchrun is not on PATH."
        echo "[v14] Install pytorch or source scripts/source_isaacsim_setup.sh first."
        exit 1
    fi
fi

echo "[v14] Sourcing IsaacSim setup..."
cd "$PROJECT_ROOT"
unset CONDA_ENV_NAME
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

if [ "$SKIP_REINSTALL" != "1" ]; then
    HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-$HOME/holosoma_deps}"
    pip install -e "$PROJECT_ROOT/src/holosoma[unitree,booster]" --quiet
    if ! python -c "import isaaclab" 2>/dev/null; then
        echo "[v14] isaaclab not found, reinstalling..."
        pip install 'setuptools<81' --quiet
        echo 'setuptools<81' > /tmp/hs-build-constraints-v14.txt
        PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints-v14.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \
            pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet
        rm /tmp/hs-build-constraints-v14.txt
    fi
else
    echo "[v14] SKIP_REINSTALL=1 — skipping holosoma/isaaclab reinstall."
fi

TRAIN_ARGS=(
    "exp:${EXP_KEY}"
    "simulator:${SIMULATOR}"
    "logger:${LOGGER}"
    "--training.seed" "${SEED}"
)

if [ -n "$MOTION_FILE" ]; then
    TRAIN_ARGS+=(
        "--command.setup_terms.motion_command.params.motion_config.motion_file=${MOTION_FILE}"
    )
fi

# Force magnitude curriculum — ablation toggle. Default off.
if [ "${CURRICULUM}" = "on" ]; then
    for _group in setup_terms reset_terms step_terms; do
        TRAIN_ARGS+=(
            "--command.${_group}.wrist_force_tracking_command.params.wrist_force_tracking_config.enable-force-curriculum=True"
            "--command.${_group}.wrist_force_tracking_command.params.wrist_force_tracking_config.curriculum-ramp-steps=${CURRICULUM_RAMP_STEPS}"
        )
    done
fi

if [ "$#" -gt 0 ]; then
    TRAIN_ARGS+=("$@")
fi

echo "[v14] Launching wrist-force-v2 training..."
echo "  FRAME      : ${FRAME}   (obs frame: $([ "${FRAME}" = "yaw" ] && echo yaw_only || echo full_base))"
echo "  CURRICULUM : ${CURRICULUM}$([ "${CURRICULUM}" = "on" ] && echo "   (ramp steps: ${CURRICULUM_RAMP_STEPS})")"
echo "  exp        : ${EXP_KEY}"
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

cat <<'EOF'

-----------------------------------------------------------------------
v14 TRAIN GATE — wandb indicators to eyeball during training
  (docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md):

  1. Train/mean_reward monotonically rises in first ~200 iter
  2. Baseline Episode/rew_motion_tracking_* holds / rises
     (this is the ONLY reward — no wrist_force reward term in v14)
  3. Env/force/active_frac_ext steady-state in [0.2, 0.5]
  4. Env/force/ext_magnitude_max <= 30 N (CURRICULUM=off) or curriculum ramp
     approaching 30 N (CURRICULUM=on; track force/curriculum_current_hi)
  5. Env/force/cmd_ext_alignment ~ -1.0  (anti-parallel sanity; holds for
     both yaw_only and full_base variants — the metric rotates F_cmd_obs
     back to world using the variant's own forward rotation before comparing)
  6. Env/force/cmd_magnitude_l ~ Env/force/ext_magnitude_l
     Env/force/cmd_magnitude_r ~ Env/force/ext_magnitude_r   (|F_cmd|=|F_ext|;
     holds for both variants — rotation preserves magnitude)
  7. Env/force/applied_f_body_{l,r} ~ Env/force/ext_magnitude_{l,r}
     (sanity on world->body quat rotation at injection time)
  8. Actor/critic loss not diverging; Episode/rew_survival stable
  9. Env/force/curriculum_progress:
       CURRICULUM=off -> constant 1.0
       CURRICULUM=on  -> linear ramp 0 -> 1.0 over CURRICULUM_RAMP_STEPS
     Env/force/curriculum_current_hi ramps in lock-step up to 30 N
 10. Q6 termination hazard:
     Env/force/term_rate_active_ext  vs  Env/force/term_rate_inactive_ext
       if active >> inactive (5x+), the 30 N pull is destabilising — flip
       CURRICULUM=on. If close, off is safe.
 11. Q1 closure-mechanism diagnostic (log-only, not in reward):
     Env/force/wrist_reaction_vs_cmd_cos_{l,r}   ideal approx 1.0
     Env/force/wrist_reaction_vs_cmd_mag_ratio_{l,r} ideal approx 1.0
     These metrics do NOT feed reward; they inform follow-up ablations.

  Ablation comparisons (same seed, same motion clip, up to 2x2 = 4 runs):
    - FRAME: yaw_only vs fullbase
    - CURRICULUM: off vs on
    - Record the FRAME and CURRICULUM values in each run_name.
-----------------------------------------------------------------------

EOF

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[v14] Launching multi-GPU via torchrun --nproc_per_node=${NUM_GPUS}"
    torchrun --nproc_per_node="${NUM_GPUS}" --standalone \
        src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"
else
    python src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"
fi

echo "[v14] Done!"
