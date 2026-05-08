"""v14 observation preset.

Actor group:  baseline + wrist_force_command_v2       (6 dims)
Critic group: baseline + wrist_force_command_v2 + wrist_force_ext_privileged_v2

history_length=10 on both groups (matches v10).
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
)

_MOD = "holosoma.managers.observation.terms.wbt_force_v2"

wrist_force_command_v2_term = ObsTermCfg(
    func=f"{_MOD}:wrist_force_command_v2",
    scale=1.0,
    noise=0.0,
)

wrist_force_ext_privileged_v2_term = ObsTermCfg(
    func=f"{_MOD}:wrist_force_ext_privileged_v2",
    scale=1.0,
    noise=0.0,
)

_actor_terms = {
    **actor_obs_shared.terms,
    "wrist_force_command_v2": wrist_force_command_v2_term,
}
actor_obs_force_v2 = replace(
    actor_obs_shared,
    history_length=10,
    terms=_actor_terms,
)

_critic_terms = {
    **critic_obs_shared_terms,
    "wrist_force_command_v2": wrist_force_command_v2_term,
    "wrist_force_ext_privileged_v2": wrist_force_ext_privileged_v2_term,
}
critic_obs_force_v2 = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=10,
    terms=_critic_terms,
)

g1_29dof_wbt_force_v2_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_force_v2,
        "critic_obs": critic_obs_force_v2,
    },
)

__all__ = ["g1_29dof_wbt_force_v2_observation"]
