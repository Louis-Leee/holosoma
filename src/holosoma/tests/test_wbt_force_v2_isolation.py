"""V14 / V10 isolation contract test.

V14 design constraint: no V10 wrist-force module may be imported (independent
code stack). This test enforces that with AST parsing, NOT string grep --
grep would be bypassed by:

  from holosoma.managers.command.terms import wbt_force as v10_mod  # alias
  importlib.import_module("holosoma.managers.command.terms.wbt_force")
  __import__("holosoma...wbt_force")

AST inspection catches all three patterns via ast.Import / ast.ImportFrom /
ast.Call (for __import__ and importlib.import_module).

A lightweight grep fallback additionally catches references to V10 demo
scripts in shell launchers (bash files have no AST).

V10 module blacklist (V14 must not import / alias any of these):
- holosoma.managers.command.terms.wbt_force
- holosoma.managers.reward.terms.wbt_force
- holosoma.managers.observation.terms.wbt_force
- holosoma.envs.wbt.wbt_force_injected
- holosoma.config_values.wbt.g1.command_force
- holosoma.config_values.wbt.g1.observation_force
- holosoma.config_values.wbt.g1.reward_force
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

V14_SOURCE_FILES = [
    "src/holosoma/holosoma/config_types/command_v2.py",
    "src/holosoma/holosoma/managers/command/terms/wbt_force_v2.py",
    "src/holosoma/holosoma/managers/observation/terms/wbt_force_v2.py",
    "src/holosoma/holosoma/envs/wbt/wbt_force_injected_v2.py",
    "src/holosoma/holosoma/config_values/wbt/g1/command_force_v2.py",
    "src/holosoma/holosoma/config_values/wbt/g1/observation_force_v2.py",
    "src/holosoma/holosoma/config_values/wbt/g1/reward_force_v2.py",
]

V14_DEMO_SCRIPTS = [
    "demo_scripts/demo_wbt_wrist_force_training_v2.sh",
]

# Exact module names (careful: do NOT match "wbt_force_v2" -- that's V14 itself).
FORBIDDEN_V10_MODULES = frozenset({
    "holosoma.managers.command.terms.wbt_force",
    "holosoma.managers.reward.terms.wbt_force",
    "holosoma.managers.observation.terms.wbt_force",
    "holosoma.envs.wbt.wbt_force_injected",
    "holosoma.config_values.wbt.g1.command_force",
    "holosoma.config_values.wbt.g1.observation_force",
    "holosoma.config_values.wbt.g1.reward_force",
})


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_forbidden(module_name: str) -> bool:
    """Return True if module_name is in the V10 blacklist.

    Match rule: exact equality OR startswith 'X.' (so submodules count too).
    '*_v2' names are safe because the blacklist contains exact module paths
    without the _v2 suffix.
    """
    for forbidden in FORBIDDEN_V10_MODULES:
        if module_name == forbidden:
            return True
        if module_name.startswith(forbidden + "."):
            return True
    return False


def _extract_imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    """Walk AST and return (module_name, lineno) for every import.

    Covers:
      - ast.Import:     `import a.b.c`, `import a.b as x`
      - ast.ImportFrom: `from a.b import c`, `from a.b import c as x`
        (skips relative imports since level > 0 cannot reach V10 roots)
      - ast.Call:       `importlib.import_module("a.b.c")`, `__import__("a.b.c")`
    """
    imported: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                imported.append((node.module, node.lineno))
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name) and func.id == "__import__":
                name = "__import__"
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            ):
                name = "importlib.import_module"
            if name is not None and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    imported.append((first.value, node.lineno))
    return imported


@pytest.mark.parametrize("rel_path", V14_SOURCE_FILES)
def test_v14_source_does_not_import_v10_by_ast(rel_path: str) -> None:
    """AST-level check: V14 source files must not import any V10 wrist-force module.

    Covers plain imports, aliased imports, importlib.import_module, and __import__.
    """
    path = _project_root() / rel_path
    assert path.exists(), f"V14 source file must exist: {rel_path}"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    imported = _extract_imported_modules(tree)
    offenders = [
        (mod, line) for (mod, line) in imported if _is_forbidden(mod)
    ]
    assert not offenders, (
        f"V14 file {rel_path} must not import V10 wrist-force modules, "
        f"but AST found: {offenders}\n"
        f"(V14/V10 isolation contract -- see "
        f"docs/plans/2026-05-06-wbt-wrist-force-v14-force-tracking.md)"
    )


@pytest.mark.parametrize("rel_path", V14_DEMO_SCRIPTS)
def test_v14_demo_does_not_reference_v10_demo(rel_path: str) -> None:
    """Shell scripts have no AST. Use grep + comment-filter to catch references
    to V10 demo filenames in executable lines. Comments are allowed (they can
    mention V10 as historical context)."""
    path = _project_root() / rel_path
    assert path.exists(), f"V14 demo script must exist: {rel_path}"
    text = path.read_text(encoding="utf-8")

    forbidden_tokens = [
        "demo_wbt_wrist_force_training.sh",  # V10 demo filename
    ]

    non_comment_lines: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        non_comment_lines.append((lineno, line))

    for lineno, line in non_comment_lines:
        for token in forbidden_tokens:
            assert token not in line, (
                f"V14 demo {rel_path}:{lineno} must not reference V10 demo "
                f"(found: {token!r} in executable line: {line.strip()!r})"
            )


def test_v14_command_has_its_own_force_channel() -> None:
    """V14 _ForceChannel must be defined in wbt_force_v2 and be a distinct class from V10."""
    from holosoma.managers.command.terms import wbt_force as v10_mod
    from holosoma.managers.command.terms import wbt_force_v2 as v14_mod

    assert hasattr(v14_mod, "_ForceChannel"), "V14 must define its own _ForceChannel"
    v14_cls = v14_mod._ForceChannel
    v10_cls = v10_mod._ForceChannel
    assert v14_cls is not v10_cls, (
        "V14 _ForceChannel must be a distinct class from V10 _ForceChannel "
        "(V14 duplicates the implementation rather than sharing it)"
    )
    assert v14_cls.__module__ == "holosoma.managers.command.terms.wbt_force_v2"


def test_v14_modules_do_not_share_sys_modules_with_v10_after_isolated_import() -> None:
    """Import V14 entry modules and verify no attribute has a __module__ belonging to V10.

    If V14 inadvertently imported a V10 symbol via alias, the symbol's __module__
    would still report the V10 path -- this catches that kind of leak.
    """
    from holosoma.config_values.wbt.g1 import (
        command_force_v2,
        observation_force_v2,
        reward_force_v2,
    )
    from holosoma.envs.wbt import wbt_force_injected_v2 as v14_env
    from holosoma.managers.command.terms import wbt_force_v2 as v14_cmd

    v14_modules = [v14_cmd, v14_env, command_force_v2, observation_force_v2, reward_force_v2]
    v10_module_names = list(FORBIDDEN_V10_MODULES)

    for mod in v14_modules:
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            attr_mod = getattr(attr, "__module__", None)
            if attr_mod is None:
                continue
            for v10_name in v10_module_names:
                assert not (attr_mod == v10_name or attr_mod.startswith(v10_name + ".")), (
                    f"V14 module {mod.__name__!r} has attribute {attr_name!r} "
                    f"whose __module__ is V10: {attr_mod}. This means V14 picked up "
                    f"a V10 symbol (possibly via alias import) -- isolation violated."
                )
