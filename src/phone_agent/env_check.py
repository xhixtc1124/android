"""Read-only setup checks. Never prints secrets."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = ROOT / "vendor" / "cookbook"
PINNED_COMMIT = "74e103ab4249d508912ccf264793e655980eead0"  # keep in sync with scripts/setup_cookbook.sh


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def run_checks() -> list[tuple[str, str, str]]:
    """Return (status, name, detail) rows. Status is OK, WARN, FAIL or INFO."""
    rows: list[tuple[str, str, str]] = []

    rows.append(("OK" if sys.version_info >= (3, 11) else "FAIL", "python", platform.python_version()))

    head = _run(["git", "-C", str(COOKBOOK), "rev-parse", "HEAD"])
    rows.append((
        "OK" if head == PINNED_COMMIT else "FAIL",
        "cookbook",
        f"{head[:8] or 'missing'} (pinned {PINNED_COMMIT[:8]})",
    ))
    try:
        import training.recipes.sft_loop  # noqa: F401
        rows.append(("OK", "sft_loop import", "ok"))
    except Exception as exc:  # report the type only; messages can contain paths or env values
        rows.append(("FAIL", "sft_loop import", type(exc).__name__))

    has_key = bool(os.environ.get("FIREWORKS_API_KEY"))
    rows.append(("OK" if has_key else "FAIL", "FIREWORKS_API_KEY", "set" if has_key else "not set"))
    opt_out = os.environ.get("TELEMETRY_OPT_OUT", "")
    rows.append((
        "OK" if opt_out == "true" else "WARN",
        "TELEMETRY_OPT_OUT",
        opt_out or "unset (run: source scripts/env.sh)",
    ))

    mem_gb = int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30
    rows.append((
        "INFO",
        "RAM",
        f"{mem_gb:.0f} GB" + ("; too small to run the 30B model locally (~32 GB needed)" if mem_gb < 32 else ""),
    ))
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    rows.append(("OK" if free_gb >= 100 else "WARN", "free disk", f"{free_gb:.0f} GB"))

    rows.append((
        "INFO",
        "sleep",
        "closing the lid sleeps the Mac and kills training; wrap long runs in `caffeinate -dims`",
    ))

    for tool in ("firectl", "adb", "emulator", "llama-server"):
        path = shutil.which(tool)
        rows.append(("OK" if path else "INFO", tool, path or "not installed (needed later)"))
    return rows


def main() -> int:
    rows = run_checks()
    for status, name, detail in rows:
        print(f"{status:5s} {name:18s} {detail}")
    return 1 if any(status == "FAIL" for status, _, _ in rows) else 0
