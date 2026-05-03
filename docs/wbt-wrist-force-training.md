# WBT Wrist-Force Training Guide (v10)

This guide covers the training-side workflow for the WBT wrist-force
experiment (`exp:g1-29dof-wbt-force`). It is scoped to training only —
deployment and sim-to-sim workflows are documented separately (Phase 8
of the plan, see `docs/plans/2026-05-03-wbt-wrist-force-v10.md`).

## What this experiment does

`g1_29dof_wbt_force` extends the baseline `g1_29dof_wbt` whole-body
tracking task with two new per-wrist signals:

* **F_cmd** — a 6-D wrist force command the policy is asked to produce,
  sampled in the body-yaw frame. F_cmd is part of both actor and critic
  observations; a trapezoidal state machine drives it through
  `COOLDOWN → RAMP_UP → HOLD → RAMP_DOWN → COOLDOWN`.
* **F_ext** — a world-frame external disturbance applied via
  `set_external_force_and_torque` to the two wrist bodies
  (`left_wrist_yaw_link`, `right_wrist_yaw_link`). F_ext is critic-only
  privileged information.

The reward term `wrist_force_position_tracking_exp` adds a virtual-
spring shift to the motion tracking target:

```
target_shifted = motion_target + (F_ext + F_cmd_world) / K_virtual
reward = exp(-||wrist_actual - target_shifted||^2.mean() / sigma^2)
```

`K_virtual` is a per-env, per-wrist constant resampled on episode reset
from `WristComplianceConfig.k_virtual_range`. In v1 this range is
`(100.0, 100.0)` (deterministic 100 N/m); v2 can widen it to, e.g.,
`(50.0, 300.0)` to train K-robust policies.

**Red line 1 (hard constraint):** external forces are only applied to
the two wrist bodies — all other `_external_force_b` entries stay 0.

## Canonical training command

Mirrors the pattern in `demo_scripts/demo_omomo_wb_tracking.sh`:

```bash
cd /path/to/holosoma
unset CONDA_ENV_NAME
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-force \
    logger:wandb \
    --training.seed 1
```

Add motion overrides as needed, e.g.:

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-force \
    logger:wandb \
    --training.seed 1 \
    --command.setup_terms.motion_command.params.motion_config.motion_file=\
holosoma/data/motions/g1_29dof/whole_body_tracking/<your_motion>.npz
```

The `source scripts/source_isaacsim_setup.sh` line is part of the
canonical command, not an assumed prerequisite — it sets
`CONDA_ENV_NAME=hssim` and activates the right conda env.

## Key configuration parameters

The `WristComplianceConfig` is stored in the command term's `params`
dict under `wrist_compliance_config`. Widen or tune it via tyro
overrides such as:

```bash
# Widen K_virtual range (v2)
--command.setup_terms.wrist_compliance_command.params.wrist_compliance_config.k_virtual_range 50.0 300.0

# Increase F_ext activation rate (default 0.01 per step)
--command.setup_terms.wrist_compliance_command.params.wrist_compliance_config.force_ext_activation_prob_per_step 0.02

# Disable debug-arrow F_total overlay
--command.setup_terms.wrist_compliance_command.params.wrist_compliance_config.debug_draw_total_arrow False
```

Defaults for the v10 experiment (see
`src/holosoma/holosoma/config_types/command.py::WristComplianceConfig`):

| Field | Default | Notes |
| ----- | ------- | ----- |
| `force_cmd_magnitude_range` | `(5.0, 30.0)` N | per episode sample |
| `force_cmd_duration_range_s` | `(1.0, 3.0)` s | includes ramps |
| `force_cmd_cooldown_range_s` | `(0.5, 2.0)` s | between episodes |
| `force_cmd_ramp_frac` | `0.25` | ramp-up + ramp-down each take this fraction of duration |
| `force_cmd_activation_prob_per_step` | `0.01` | Bernoulli roll per control step while in COOLDOWN |
| `force_ext_magnitude_range` | `(0.0, 30.0)` N | - |
| `force_ext_duration_range_s` | `(1.0, 3.0)` s | - |
| `force_ext_cooldown_range_s` | `(0.5, 2.0)` s | - |
| `force_ext_ramp_frac` | `0.25` | - |
| `force_ext_activation_prob_per_step` | `0.01` | - |
| `k_virtual_range` | `(100.0, 100.0)` | v1 deterministic |
| `debug_arrow_scale_n_per_m` | `50.0` | 30 N / 50 N/m → 0.6 m arrow |
| `debug_draw_total_arrow` | `True` | purple F_total overlay |
| `left_wrist_body_name` | `"left_wrist_yaw_link"` | - |
| `right_wrist_body_name` | `"right_wrist_yaw_link"` | - |

## Debug-draw arrows

The env subclass `WholeBodyTrackingForceInjected` overrides
`draw_debug_viz()` to render arrows at the two wrists of env 0 when
the IsaacSim viewer is open and `simulator.config.debug_viz=True`.

Legend:

* **Red (1.0, 0.0, 0.0)** — F_ext (sim-injected world-frame force)
* **Blue (0.0, 0.0, 1.0)** — F_cmd (policy-visible body-yaw-frame command,
  shown rotated into world frame for visualization)
* **Purple (0.8, 0.0, 0.8)** — F_total = F_ext + F_cmd (gated by
  `debug_draw_total_arrow`, default on)

Arrow length (meters) = `||F|| / debug_arrow_scale_n_per_m`. To shrink,
raise the scale; e.g. `scale=100.0` halves the arrow lengths.

To enable the viewer:

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-force \
    logger:wandb \
    --simulator.config.debug-viz True \
    --training.headless False
```

The `draw_debug_viz` hook is chained onto `simulator.draw_debug_viz` in
the env subclass's `__init__`, so the simulator's own debug behaviour
(e.g. `virtual_gantry`) is preserved.

## wandb monitoring

Enable with `logger:wandb`. The pipeline reuses Holosoma's existing
`update_metrics → log_dict → extras["to_log"] → LoggingHelper →
wandb Env/*` path (`agents/modules/logging_utils.py:148, 189, 339-342`);
no new pipeline is introduced.

### Force-centric keys (all prefixed `Env/`)

Command-owned (populated by `WristComplianceCommand.update_metrics`):

* `force/cmd_magnitude_l` / `force/cmd_magnitude_r` / `force/cmd_magnitude_max`
* `force/ext_magnitude_l` / `force/ext_magnitude_r` / `force/ext_magnitude_max`
* `force/cmd_ext_alignment` — cos(F_cmd_w, F_ext_w) averaged over wrists
* `force/k_virtual_l` / `force/k_virtual_r`
* `force/active_frac_cmd` / `force/active_frac_ext`
* `force/phase_ramp_up` / `force/phase_hold` / `force/phase_ramp_down`

Env-owned (populated by
`WholeBodyTrackingForceInjected._update_log_dict`):

* `force/wrist_target_shift_l` / `force/wrist_target_shift_r` —
  ||F_total/K|| in meters
* `force/wrist_pos_error_l` / `force/wrist_pos_error_r` —
  ||wrist_actual - target_shifted||
* `force/applied_f_body_l` / `force/applied_f_body_r` — norm of the
  world-frame force snapshot recorded just after injection (must match
  `ext_magnitude_{l,r}` if world→body rotation is correct)

Reward term rewards log automatically under
`Episode/rew_wrist_force_position_tracking_exp`
(`managers/reward/manager.py:200-250`).

### Recommended wandb panels

* **Force Magnitude** — 6 lines: cmd/ext × l/r/max
* **Force Activity** — `active_frac_cmd`, `active_frac_ext`, three
  phase occupancy lines
* **Virtual Spring** — `wrist_target_shift_l/r` and
  `wrist_pos_error_l/r`
* **K virtual** — `k_virtual_l/r` (v1 is a flat line at 100)
* **Reward** — `Episode/rew_wrist_force_position_tracking_exp` on its
  own axis

### Health checks (v1)

* `force/cmd_magnitude_*` ∈ [0, 30]; `force/ext_magnitude_*` ∈ [0, 30].
* `force/k_virtual_*` ≡ 100.0 — a flat line is the correct signal.
* `force/active_frac_cmd` + `force/active_frac_ext` steady-state
  ≈ 0.25–0.45 (with `activation_prob=0.01`, `dt=0.02`,
  `duration~2 s`, `cooldown~1 s`). Less than 0.05 or greater than 0.9
  indicates a misconfiguration.
* `force/applied_f_body_{l,r}` should match `ext_magnitude_{l,r}` —
  a mismatch means the world→body rotation in
  `_apply_force_in_physics_step` is broken (look at
  `quat_apply_inverse` and the wxyz quaternion convention).
* `Episode/rew_wrist_force_position_tracking_exp` should climb to
  **≥ 0.4** in the early iterations (reward is
  `exp(-error/σ²)` with σ=0.3, so 0.4 corresponds to a steady-state
  wrist error ≈ σ/2 ≈ 0.15 m).

## Debug triggers

* **Reward stalls** — try widening the MLP from `[1024, 512, 256]`
  (the v10 default) to `[2048, 1024, 512]`, or increase
  `observation.groups.actor_obs.history_length` from 10 to 15.
* **`force/active_frac_cmd` too low** — raise
  `force_cmd_activation_prob_per_step` from `0.01` to `0.02`.
* **`applied_f_body_{l,r}` ≠ `ext_magnitude_{l,r}`** — check
  `body_quat_w` is wxyz (IsaacLab convention) and that
  `quat_apply_inverse` is being applied to the right quaternion.
* **Motion tracking collapses** — drop
  `wrist_force_position_tracking_exp.weight` from `2.0` to `1.0`, or
  raise `sigma` from `0.3` to `0.5`.

## v2 roadmap

* Widen `K_VIRTUAL_RANGE_N_PER_M` in
  `src/holosoma/holosoma/config_values/wbt/g1/_k_virtual.py` —
  only that single file needs to change.
* Consider an RMA-style F_ext estimator if the critic-only privileged
  observation is insufficient in practice (v9.4 picked the CHIP
  pattern where actor and critic both see K; F_ext stays critic-only
  here).
