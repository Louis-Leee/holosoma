"""Observation preset for the WBT wrist-force experiment.

Both groups use ``history_length=10`` (CHIP-style 10-step history) and
append the wrist-force terms to the baseline WBT observation set.

Expected dims on G1 29dof:

* ``actor_obs``  — 162 per step x 10 history = 1620
* ``critic_obs`` — 300 per step x 10 history = 3000

The exact dims are asserted by the isaacsim e2e test in Task 11; the
CPU regression tests here only check structure/identity.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    critic_obs_shared_terms,
    g1_29dof_wbt_observation,
)

# ---- Wrist-force observation terms (Task 4) --------------------------------
_WRIST_FORCE_MOD = "holosoma.managers.observation.terms.wbt_force"

wrist_force_command_term = ObsTermCfg(
    func=f"{_WRIST_FORCE_MOD}:wrist_force_command",
    scale=1.0,
    noise=0.0,
)

wrist_virtual_stiffness_command_term = ObsTermCfg(
    func=f"{_WRIST_FORCE_MOD}:wrist_virtual_stiffness_command",
    scale=0.01,  # 100 N/m → ~1.0 in obs space
    noise=0.0,
)

wrist_force_ext_privileged_term = ObsTermCfg(
    func=f"{_WRIST_FORCE_MOD}:wrist_force_ext_privileged",
    scale=1.0,
    noise=0.0,
)


# ---- Actor group: baseline 6 terms + 2 new wrist-force obs -----------------
_actor_terms = {
    **actor_obs_shared.terms,
    "wrist_force_command": wrist_force_command_term,
    "wrist_virtual_stiffness_command": wrist_virtual_stiffness_command_term,
}

actor_obs_force = replace(
    actor_obs_shared,
    history_length=10,
    terms=_actor_terms,
)

# ---- Critic group: baseline 10 terms + 3 wrist-force obs (incl priv) -------
_critic_terms = {
    **critic_obs_shared_terms,
    "wrist_force_command": wrist_force_command_term,
    "wrist_virtual_stiffness_command": wrist_virtual_stiffness_command_term,
    "wrist_force_ext_privileged": wrist_force_ext_privileged_term,
}

critic_obs_force = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=10,
    terms=_critic_terms,
)

# ---- Preset ----------------------------------------------------------------
g1_29dof_wbt_force_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_force,
        "critic_obs": critic_obs_force,
    },
)


# Expose the baseline identity for regression tests that need to check
# the baseline did not get mutated.
_BASELINE = g1_29dof_wbt_observation  # re-exported for clarity

__all__ = ["g1_29dof_wbt_force_observation"]
