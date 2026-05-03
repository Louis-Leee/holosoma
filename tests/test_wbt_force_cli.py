"""CLI smoke tests for the WBT wrist-force experiment (Task 10).

Verifies that:

  * ``exp:g1-29dof-wbt-force --help`` exits 0 quickly (no sim launch).
  * Baseline ``exp:g1-29dof-wbt --help`` still exits 0.
  * Help output contains ``wrist_compliance_command`` somewhere (proves
    the command preset registration + tyro exposure).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.absolute()
TRAIN_AGENT = REPO_ROOT / "src" / "holosoma" / "holosoma" / "train_agent.py"


def _run_help(exp_arg: str, timeout_s: int = 60) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src" / "holosoma") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(TRAIN_AGENT), f"exp:{exp_arg}", "--help"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
        check=False,
    )


def test_wbt_force_cli_help_exits_zero() -> None:
    result = _run_help("g1-29dof-wbt-force")
    assert result.returncode == 0, (
        f"train_agent.py --help exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )


def test_baseline_wbt_cli_help_still_works() -> None:
    result = _run_help("g1-29dof-wbt")
    assert result.returncode == 0, (
        f"baseline exp:g1-29dof-wbt --help broke (exit {result.returncode})\nSTDERR:\n{result.stderr[-2000:]}"
    )


def test_wrist_compliance_term_visible_in_help() -> None:
    result = _run_help("g1-29dof-wbt-force")
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "wrist_compliance_command" in combined or "wrist-compliance" in combined, (
        "wrist_compliance_command not found in --help output; command preset registration may be broken."
    )
