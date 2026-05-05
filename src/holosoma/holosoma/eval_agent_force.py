"""Force-aware WBT evaluation entry point.

Mirrors :mod:`holosoma.eval_agent` but adds two independent CLI flags —
``--enable-force-cmd`` / ``--enable-force-ext`` — that selectively zero
each force channel in the loaded ``WristComplianceConfig``. Combining
them yields the 4 diagnostic modes described in
``docs/plans/2026-05-05-wbt-wrist-force-v11-eval.md`` §3.1.

All rollout mechanics (ckpt load, env reconstruction, ``evaluate_policy``
loop, ONNX export, wandb sync) are reused from :func:`eval_agent.run_eval_with_tyro`.
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import fields, replace
from typing import Any, Callable

import tyro
from loguru import logger as _log

from holosoma import eval_agent
from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    WristComplianceConfig,
)
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.config_types.eval_force import ForceEvalConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.managers.command.terms.wbt_force import WristComplianceCommand
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG

FORCE_AWARE_ENV_CLASS = (
    "holosoma.envs.wbt.wbt_force_injected.WholeBodyTrackingForceInjected"
)


def _resolve_command_func(func: str) -> Any:
    """Resolve a ``CommandTermCfg.func`` string to its target class or callable.

    Mirrors :meth:`holosoma.managers.command.manager.CommandManager._resolve_function`
    — command-term strings use the ``"module.path:Class"`` colon format
    rather than the pure dot path consumed by :func:`holosoma.utils.helpers.get_class`.
    """
    if ":" not in func:
        raise ValueError(f"Function string must be in format 'module:function', got: {func}")
    module_path, func_name = func.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _rehydrate_wrist_compliance_configs(exp_cfg: ExperimentConfig) -> ExperimentConfig:
    """Walk ``exp_cfg.command.{setup,reset,step}_terms`` and promote any
    ``wrist_compliance_config`` that came back from ckpt roundtrip as a plain
    ``dict`` to a real :class:`WristComplianceConfig` instance.

    Mirrors the existing hack in
    ``holosoma.managers.command.terms.wbt:MotionCommand.__init__`` (see the
    ``TODO(jchen): temporary fix for motion_config being a dict after tyro.cli``
    comment) — ``CommandTermCfg.params`` is typed ``dict[str, Any]``, so
    pydantic does not recursively rehydrate nested frozen dataclasses. We do
    this once up front so the preflight contract + rewrite logic can both
    assume real dataclass objects.

    Schema drift: dict keys that no longer exist on the current
    ``WristComplianceConfig`` (e.g. ``debug_draw_total_arrow`` from a pre-v11
    ckpt) are dropped with a warning; anything else raises naturally inside
    the dataclass constructor.
    """
    cmd = exp_cfg.command
    if cmd is None:
        return exp_cfg

    valid_fields = {f.name for f in fields(WristComplianceConfig)}

    def _rehydrate_bucket(bucket: dict[str, CommandTermCfg]) -> dict[str, CommandTermCfg]:
        term = bucket.get("wrist_compliance_command")
        if term is None:
            return bucket
        params = term.params or {}
        wcfg = params.get("wrist_compliance_config")
        if isinstance(wcfg, WristComplianceConfig) or wcfg is None:
            return bucket
        if not isinstance(wcfg, dict):
            return bucket
        filtered = {k: v for k, v in wcfg.items() if k in valid_fields}
        dropped = set(wcfg.keys()) - valid_fields
        if dropped:
            _log.warning(
                f"Dropping {len(dropped)} unknown field(s) from ckpt "
                f"WristComplianceConfig: {sorted(dropped)}"
            )
        # Tuple ranges come back as lists from YAML/torch roundtrip —
        # WristComplianceConfig's pydantic validator coerces list→tuple, so
        # we don't special-case that here.
        rehydrated = WristComplianceConfig(**filtered)
        new_params = {**params, "wrist_compliance_config": rehydrated}
        new_term = replace(term, params=new_params)
        return {**bucket, "wrist_compliance_command": new_term}

    new_cmd = replace(
        cmd,
        setup_terms=_rehydrate_bucket(cmd.setup_terms),
        reset_terms=_rehydrate_bucket(cmd.reset_terms),
        step_terms=_rehydrate_bucket(cmd.step_terms),
    )
    return dataclasses.replace(exp_cfg, command=new_cmd)


def _assert_force_aware_ckpt(saved_cfg: ExperimentConfig) -> None:
    """Raise RuntimeError with a single actionable message if saved_cfg
    is not a force-aware WBT checkpoint. Checks the whole contract
    (not just that one key exists), to catch:
      - partial registration (term in setup_terms but not reset/step)
      - stale ``func`` pointing to a different class
      - missing / wrong params type
      - wrong env class
      - actor obs missing ``wrist_force_command``
    """
    cmd = saved_cfg.command
    if cmd is None:
        raise RuntimeError(
            "Checkpoint command config is None. Expected a force-aware WBT "
            "config with a `wrist_compliance_command` term. Use eval_agent.py "
            "for baseline WBT ckpts."
        )

    # (a) The term must be registered in all three lifecycle buckets.
    for bucket_name in ("setup_terms", "reset_terms", "step_terms"):
        bucket = getattr(cmd, bucket_name)
        if "wrist_compliance_command" not in bucket:
            raise RuntimeError(
                f"Checkpoint is not force-aware: command.{bucket_name} "
                f"has no 'wrist_compliance_command' key. Use eval_agent.py "
                f"for baseline WBT ckpts."
            )

    # (b) func must resolve to WristComplianceCommand exactly.
    setup_term = cmd.setup_terms["wrist_compliance_command"]
    try:
        resolved = _resolve_command_func(setup_term.func)
    except Exception as exc:
        raise RuntimeError(
            f"wrist_compliance_command.func ({setup_term.func!r}) does not "
            f"resolve: {exc}"
        ) from exc
    if resolved is not WristComplianceCommand:
        raise RuntimeError(
            f"wrist_compliance_command.func resolves to {resolved.__name__}, "
            f"expected WristComplianceCommand."
        )

    # (c) params must carry a WristComplianceConfig.
    params = setup_term.params or {}
    wcfg = params.get("wrist_compliance_config")
    if not isinstance(wcfg, WristComplianceConfig):
        raise RuntimeError(
            "wrist_compliance_command.params['wrist_compliance_config'] is "
            f"{type(wcfg).__name__}, expected WristComplianceConfig."
        )

    # (d) env class must be force-injected.
    env_class = saved_cfg.env_class
    if not env_class.endswith("WholeBodyTrackingForceInjected"):
        raise RuntimeError(
            f"Checkpoint env_class is {env_class!r}, expected "
            f"{FORCE_AWARE_ENV_CLASS!r} (or a subclass). Is this really a "
            f"g1-29dof-wbt-force ckpt?"
        )

    # (e) actor obs must include wrist_force_command.
    observation = saved_cfg.observation
    if observation is None or "actor_obs" not in observation.groups:
        raise RuntimeError(
            "Checkpoint actor_obs has no 'wrist_force_command' term. Policy "
            "cannot be force-aware without this obs. Check training config."
        )
    actor_terms = observation.groups["actor_obs"].terms
    if "wrist_force_command" not in actor_terms:
        raise RuntimeError(
            "Checkpoint actor_obs has no 'wrist_force_command' term. Policy "
            "cannot be force-aware without this obs. Check training config."
        )


def _apply_force_eval_flags(
    cfg: WristComplianceConfig,
    *,
    enable_force_cmd: bool,
    enable_force_ext: bool,
) -> WristComplianceConfig:
    """Zero out the F_cmd and/or F_ext channel independently."""
    updates: dict = {}
    if not enable_force_cmd:
        updates.update(
            force_cmd_magnitude_range=(0.0, 0.0),
            force_cmd_activation_prob_per_step=0.0,
        )
    if not enable_force_ext:
        updates.update(
            force_ext_magnitude_range=(0.0, 0.0),
            force_ext_activation_prob_per_step=0.0,
        )
    return replace(cfg, **updates) if updates else cfg


def _rewrite_bucket(
    bucket: dict[str, CommandTermCfg],
    transform: Callable[[WristComplianceConfig], WristComplianceConfig],
) -> dict[str, CommandTermCfg]:
    """Return a new bucket dict with the ``wrist_compliance_command`` term's
    ``wrist_compliance_config`` mapped through ``transform``."""
    term = bucket["wrist_compliance_command"]
    old_params = term.params or {}
    old_wcfg = old_params["wrist_compliance_config"]
    new_wcfg = transform(old_wcfg)
    new_params = {**old_params, "wrist_compliance_config": new_wcfg}
    new_term = replace(term, params=new_params)
    return {**bucket, "wrist_compliance_command": new_term}


def _rewrite_wrist_compliance_cfg(
    exp_cfg: ExperimentConfig,
    transform: Callable[[WristComplianceConfig], WristComplianceConfig],
) -> ExperimentConfig:
    """Pure data transform: copy the command manager's 3 buckets, map each
    bucket's ``wrist_compliance_command`` term's config through ``transform``,
    and reassemble a new :class:`ExperimentConfig`.
    """
    cmd = exp_cfg.command
    assert cmd is not None, "_rewrite_wrist_compliance_cfg requires non-None command"
    new_cmd: CommandManagerCfg = replace(
        cmd,
        setup_terms=_rewrite_bucket(cmd.setup_terms, transform),
        reset_terms=_rewrite_bucket(cmd.reset_terms, transform),
        step_terms=_rewrite_bucket(cmd.step_terms, transform),
    )
    return dataclasses.replace(exp_cfg, command=new_cmd)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    eval_cbs_cfg, remaining_args = tyro.cli(
        EvalCallbacksConfig, return_unknown_args=True, add_help=False, args=remaining_args
    )
    force_eval_cfg, remaining_args = tyro.cli(
        ForceEvalConfig, return_unknown_args=True, add_help=False, args=remaining_args
    )

    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    # Rehydrate dict→WristComplianceConfig (see docstring). Must happen
    # before preflight so check (c) sees a real dataclass.
    saved_cfg = _rehydrate_wrist_compliance_configs(saved_cfg)

    # Fail fast on non-force-aware ckpts before any sim setup.
    _assert_force_aware_ckpt(saved_cfg)

    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    # tyro.cli may re-serialize nested dataclasses — rehydrate again to be
    # safe before the rewrite step assumes dataclass identities.
    overwritten_tyro_config = _rehydrate_wrist_compliance_configs(overwritten_tyro_config)

    # Rewrite only if at least one flag is False — Mode 3 (True, True) is
    # a no-op so we preserve config identity.
    if not (force_eval_cfg.enable_force_cmd and force_eval_cfg.enable_force_ext):
        overwritten_tyro_config = _rewrite_wrist_compliance_cfg(
            overwritten_tyro_config,
            lambda c: _apply_force_eval_flags(
                c,
                enable_force_cmd=force_eval_cfg.enable_force_cmd,
                enable_force_ext=force_eval_cfg.enable_force_ext,
            ),
        )

    eval_agent.run_eval_with_tyro(
        overwritten_tyro_config,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        eval_cbs_cfg=eval_cbs_cfg,
    )


if __name__ == "__main__":
    main()
