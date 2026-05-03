"""Virtual stiffness constants for WBT wrist-force training (v10).

K_virtual is the reward-side virtual spring stiffness used in
``wrist_force_position_tracking_exp``:

    target_shifted = motion_target + (F_ext + F_cmd) / K_virtual

It is a reward hyperparameter, not a measured physical stiffness. v1 keeps
it deterministic at 100 N/m for both wrists; v2 can widen the range
(e.g. ``(50.0, 300.0)``) to train a policy that is robust to varying K.
"""

from __future__ import annotations

# v1 deterministic: both endpoints equal so per-env/per-wrist sampling
# collapses to the constant 100 N/m.
K_VIRTUAL_RANGE_N_PER_M: tuple[float, float] = (100.0, 100.0)

# Scalar alias for call sites that need a single number (e.g. inference-side
# default, docs). Must match the lower bound of the v1 range.
G1_WRIST_VIRTUAL_STIFFNESS_N_PER_M: float = K_VIRTUAL_RANGE_N_PER_M[0]
