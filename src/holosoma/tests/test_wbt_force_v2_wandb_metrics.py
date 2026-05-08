"""Wandb metric key contract for v14 wrist-force.

v14 minimal audit set:
  1. Was F_ext generated? (force/ext_magnitude_{l,r,max})
  2. Was F_cmd_b derived from it correctly? (force/cmd_magnitude_{l,r,max},
     force/cmd_ext_alignment approx -1)
  3. Was F_ext injected? (force/applied_f_body_{l,r} -- set by env)
  4. Did the trapezoidal schedule run? (force/phase_*, force/active_frac_ext)

Policy learning signal is baseline Episode/rew_motion_tracking_* (unchanged).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.command_v2 import WristForceTrackingConfig
from holosoma.managers.command.terms.wbt_force_v2 import WristForceTrackingCommand

EXPECTED_COMMAND_METRIC_KEYS = {
    "force/ext_magnitude_l",
    "force/ext_magnitude_r",
    "force/ext_magnitude_max",
    "force/cmd_magnitude_l",
    "force/cmd_magnitude_r",
    "force/cmd_magnitude_max",
    "force/cmd_ext_alignment",
    "force/cmd_ext_alignment_active_only",
    "force/active_frac_ext",
    "force/phase_ramp_up",
    "force/phase_hold",
    "force/phase_ramp_down",
    # Force curriculum (off -> progress=1.0; on -> ramps 0..1)
    "force/curriculum_progress",
    "force/curriculum_current_hi",
}

EXPECTED_ENV_METRIC_KEYS = {
    "force/applied_f_body_l",
    "force/applied_f_body_r",
    # Q6 termination spike monitor (log-only, not in reward)
    "force/active_window_flag",
}

EXPECTED_OPTIONAL_ENV_METRIC_KEYS = {
    "force/term_rate_active_ext",
    "force/term_rate_inactive_ext",
    # Q1 closure-mechanism diagnostic (optional: requires body_incoming_wrench_b)
    "force/wrist_reaction_vs_cmd_cos_l",
    "force/wrist_reaction_vs_cmd_cos_r",
    "force/wrist_reaction_vs_cmd_mag_ratio_l",
    "force/wrist_reaction_vs_cmd_mag_ratio_r",
}

FORBIDDEN_V10_METRIC_KEYS = {
    "force/k_virtual_l",
    "force/k_virtual_r",
    "force/wrist_target_shift_l",
    "force/wrist_target_shift_r",
    "force/wrist_pos_error_l",
    "force/wrist_pos_error_r",
    "force/active_frac_cmd",  # v10 had 2 channels; v14 only has ext
}


def test_command_metrics_match_contract() -> None:
    device = torch.device("cpu")
    base_quat = torch.zeros(4, 4, device=device)
    base_quat[:, 3] = 1.0
    env = SimpleNamespace(num_envs=4, device=device, dt=0.02, base_quat=base_quat)

    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt_force_v2:WristForceTrackingCommand",
        params={"wrist_force_tracking_config": WristForceTrackingConfig()},
    )
    term = WristForceTrackingCommand(cfg, env)
    term.reset(None)
    term.step()
    term.update_metrics()

    assert set(term.metrics.keys()) == EXPECTED_COMMAND_METRIC_KEYS
    for forbidden in FORBIDDEN_V10_METRIC_KEYS:
        assert forbidden not in term.metrics
