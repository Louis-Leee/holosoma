"""v14 WBT wrist-force configuration.

v14 causality: F_ext is the sampled stochastic process (world frame); F_cmd_b
is derived as ``-R_yaw^-1(base_quat) . F_ext_w`` inside the command term. So
this config only carries the F_ext schedule. No K_virtual, no F_cmd schedule.
"""

from __future__ import annotations

from typing import Literal

from pydantic.dataclasses import dataclass

# Actor-visible F_cmd observation frame. Two variants form the ablation:
#   "yaw_only":  F_cmd_obs = -R_yaw^-1(base_quat) . F_ext_w
#                (roll/pitch kept out of obs; matches learning-compliance EE gripper force sensor)
#   "full_base": F_cmd_obs = -R^-1(base_quat) . F_ext_w
#                (full base rotation inverse; matches gentle-humanoid-training)
ForceCmdObsFrame = Literal["yaw_only", "full_base"]


@dataclass(frozen=True)
class WristForceTrackingConfig:
    """Configuration for the v14 ``WristForceTrackingCommand`` term."""

    # ---- F_ext (sim-injected disturbance, world frame) ----
    force_ext_magnitude_range: tuple[float, float] = (5.0, 30.0)
    force_ext_duration_range_s: tuple[float, float] = (1.0, 3.0)
    force_ext_cooldown_range_s: tuple[float, float] = (0.5, 2.0)
    force_ext_ramp_frac: float = 0.25
    force_ext_activation_prob_per_step: float = 0.01

    # ---- Debug arrow visual scale ----
    debug_arrow_scale_n_per_m: float = 50.0

    # ---- Enable flags + wrist body names ----
    enable_left: bool = True
    enable_right: bool = True
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"

    # ---- F_cmd observation frame (v14 ablation dimension) ----
    # "yaw_only":   learning-compliance style (default)
    # "full_base":  gentle-humanoid-training style
    force_cmd_obs_frame: ForceCmdObsFrame = "yaw_only"

    # ---- Force magnitude curriculum (v14 optional ablation toggle) ----
    # Default off: matches the current "sample directly in [5, 30] N" behavior.
    # When enabled: the peak-magnitude range of _ext_channel linearly ramps from
    # ``curriculum_initial_magnitude_range`` to ``force_ext_magnitude_range``
    # over ``curriculum_ramp_steps`` command steps. Only affects the next
    # trigger's sampled peak; an in-progress trapezoidal episode keeps its peak.
    enable_force_curriculum: bool = False
    curriculum_initial_magnitude_range: tuple[float, float] = (0.0, 5.0)
    curriculum_ramp_steps: int = 10_000

    def __post_init__(self) -> None:
        rng = self.force_ext_magnitude_range
        assert rng[0] >= 0.0, f"force_ext_magnitude_range lower bound must be >= 0, got {rng}"
        assert rng[0] <= rng[1], f"force_ext_magnitude_range must satisfy lo <= hi, got {rng}"

        for name, r in (
            ("force_ext_duration_range_s", self.force_ext_duration_range_s),
            ("force_ext_cooldown_range_s", self.force_ext_cooldown_range_s),
        ):
            assert r[0] >= 0.0, f"{name} lower bound must be >= 0, got {r}"
            assert r[0] <= r[1], f"{name} must satisfy lo <= hi, got {r}"

        assert 0.0 <= self.force_ext_ramp_frac <= 0.5, (
            f"force_ext_ramp_frac must be in [0, 0.5], got {self.force_ext_ramp_frac}"
        )
        assert 0.0 <= self.force_ext_activation_prob_per_step <= 1.0, (
            f"force_ext_activation_prob_per_step must be in [0, 1], "
            f"got {self.force_ext_activation_prob_per_step}"
        )
        assert self.debug_arrow_scale_n_per_m > 0.0, (
            f"debug_arrow_scale_n_per_m must be > 0, got {self.debug_arrow_scale_n_per_m}"
        )

        init_rng = self.curriculum_initial_magnitude_range
        assert init_rng[0] >= 0.0, (
            f"curriculum_initial_magnitude_range lower bound must be >= 0, got {init_rng}"
        )
        assert init_rng[0] <= init_rng[1], (
            f"curriculum_initial_magnitude_range must satisfy lo <= hi, got {init_rng}"
        )
        # Initial peak must not exceed final target (otherwise it is a ramp down, not a curriculum).
        target_hi = self.force_ext_magnitude_range[1]
        assert init_rng[1] <= target_hi, (
            f"curriculum_initial_magnitude_range hi ({init_rng[1]}) must not exceed "
            f"force_ext_magnitude_range hi ({target_hi})"
        )
        assert self.curriculum_ramp_steps > 0, (
            f"curriculum_ramp_steps must be > 0, got {self.curriculum_ramp_steps}"
        )
