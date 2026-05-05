"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass

from holosoma.config_values.wbt.g1._k_virtual import K_VIRTUAL_RANGE_N_PER_M


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files.
    When non-empty, takes precedence over motion_file."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)


########################################################################################################################
# Wrist compliance command configuration (WBT wrist-force, v10)
########################################################################################################################
@dataclass(frozen=True)
class WristComplianceConfig:
    """Configuration for the WBT wrist-force ``WristComplianceCommand`` term.

    Controls sampling of two per-wrist signals exposed to the reward and
    observations:

    * ``F_cmd`` (body-yaw frame): the 6-D force the policy is asked to produce.
    * ``F_ext`` (world frame): a randomly sampled external disturbance applied
      to the wrist body in the simulator.

    Both signals use the same trapezoidal state machine
    (``COOLDOWN -> RAMP_UP -> HOLD -> RAMP_DOWN -> COOLDOWN``) with independent
    magnitude/duration/cooldown ranges. ``k_virtual_range`` is per-wrist,
    per-env, resampled only on episode reset (matches GH motion-tracking kp
    convention).

    See ``docs/plans/2026-05-03-wbt-wrist-force-v10.md`` (Task 1) for the full
    rationale.
    """

    # ---- F_cmd (policy-visible, body-yaw frame) ----
    force_cmd_magnitude_range: tuple[float, float] = (5.0, 30.0)
    force_cmd_duration_range_s: tuple[float, float] = (1.0, 3.0)
    force_cmd_cooldown_range_s: tuple[float, float] = (0.5, 2.0)
    force_cmd_ramp_frac: float = 0.25
    force_cmd_activation_prob_per_step: float = 0.01

    # ---- F_ext (sim-injected disturbance, world frame) ----
    force_ext_magnitude_range: tuple[float, float] = (0.0, 30.0)
    force_ext_duration_range_s: tuple[float, float] = (1.0, 3.0)
    force_ext_cooldown_range_s: tuple[float, float] = (0.5, 2.0)
    force_ext_ramp_frac: float = 0.25
    force_ext_activation_prob_per_step: float = 0.01

    # ---- Virtual spring stiffness (reward hyperparameter, not a real K) ----
    k_virtual_range: tuple[float, float] = K_VIRTUAL_RANGE_N_PER_M

    # ---- Debug-draw visual scale (Task 3 draw_debug_viz) ----
    debug_arrow_scale_n_per_m: float = 50.0
    """Arrow length in meters per Newton. 30 N / 50 N/m = 0.6 m -> about
    wrist-reach for visual clarity. Increase to shrink arrows."""

    # ---- Enable flags + wrist body names ----
    enable_left: bool = True
    enable_right: bool = True
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"

    def __post_init__(self) -> None:
        # k_virtual must be strictly positive (reward divides by it after
        # clamp(min=1e-3), but we still want a sane config).
        assert self.k_virtual_range[0] > 0.0, f"k_virtual_range lower bound must be > 0, got {self.k_virtual_range}"
        assert self.k_virtual_range[0] <= self.k_virtual_range[1], (
            f"k_virtual_range must satisfy lo <= hi, got {self.k_virtual_range}"
        )

        # magnitude ranges: lo <= hi, lo >= 0.
        for name, rng in (
            ("force_cmd_magnitude_range", self.force_cmd_magnitude_range),
            ("force_ext_magnitude_range", self.force_ext_magnitude_range),
        ):
            assert rng[0] >= 0.0, f"{name} lower bound must be >= 0, got {rng}"
            assert rng[0] <= rng[1], f"{name} must satisfy lo <= hi, got {rng}"

        # duration + cooldown ranges: lo >= 0, lo <= hi.
        for name, rng in (
            ("force_cmd_duration_range_s", self.force_cmd_duration_range_s),
            ("force_cmd_cooldown_range_s", self.force_cmd_cooldown_range_s),
            ("force_ext_duration_range_s", self.force_ext_duration_range_s),
            ("force_ext_cooldown_range_s", self.force_ext_cooldown_range_s),
        ):
            assert rng[0] >= 0.0, f"{name} lower bound must be >= 0, got {rng}"
            assert rng[0] <= rng[1], f"{name} must satisfy lo <= hi, got {rng}"

        # ramp fractions must be in [0, 0.5] so ramp_up + ramp_down <= duration.
        for name, frac in (
            ("force_cmd_ramp_frac", self.force_cmd_ramp_frac),
            ("force_ext_ramp_frac", self.force_ext_ramp_frac),
        ):
            assert 0.0 <= frac <= 0.5, f"{name} must be in [0, 0.5], got {frac}"

        # debug arrow scale must be strictly positive (we divide by it).
        assert self.debug_arrow_scale_n_per_m > 0.0, (
            f"debug_arrow_scale_n_per_m must be > 0, got {self.debug_arrow_scale_n_per_m}"
        )
