"""CLI config for eval_agent_force.py."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ForceEvalConfig:
    """Force-aware eval CLI options for ``eval_agent_force.py``.

    Two independent flags control whether each force channel keeps its
    training-time random schedule or gets zeroed out. Combining them
    yields 4 diagnostic modes (see §3.1):
        (False, False) -> baseline-like (no force injection at all)
        (True,  False) -> F_cmd only (obs channel active; sim untouched)
        (False, True ) -> F_ext only (sim-injected ext force; obs zero)
        (True,  True ) -> full training schedule (default)
    """

    enable_force_cmd: bool = True
    """If True, keep the training-time random schedule for F_cmd
    (actor obs ``wrist_force_command``). If False, zero
    ``force_cmd_magnitude_range`` and ``force_cmd_activation_prob_per_step``
    so F_cmd stays at 0 throughout the rollout."""

    enable_force_ext: bool = True
    """If True, keep the training-time random schedule for F_ext
    (sim-injected external force on wrist bodies). If False, zero
    ``force_ext_magnitude_range`` and ``force_ext_activation_prob_per_step``
    so no external force is applied to the wrists during rollout."""
