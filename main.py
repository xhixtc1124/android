"""Project entry point.

Usage:
    python main.py check       read-only setup checks (never prints secrets)
    python main.py sessions    read-only list of open Fireworks training sessions/jobs
    python main.py convert steps|sft ...   build data from AndroidControl (see phone_agent/convert.py)
"""

import sys

VENV_PYTHON = "vendor/cookbook/training/.venv/bin/python"


def _in_project_env() -> bool:
    """True if this Python can see the project package; otherwise explain the fix."""
    try:
        import phone_agent  # noqa: F401
        return True
    except ModuleNotFoundError:
        print(
            f"This Python ({sys.executable}) is not the project's environment.\n"
            f"Terminal: run  source scripts/env.sh  then try again.\n"
            f"VS Code:  Cmd+Shift+P > 'Python: Select Interpreter' > {VENV_PYTHON}"
        )
        return False


def main(argv: list[str]) -> int:
    if argv[:1] in (["check"], ["sessions"], ["convert"]) and not _in_project_env():
        return 1
    if argv[:1] == ["convert"]:
        from phone_agent import convert

        return convert.main(argv[1:])
    if argv[:1] == ["check"]:
        from phone_agent import env_check

        return env_check.main()
    if argv[:1] == ["sessions"]:
        from phone_agent import fireworks_status

        return fireworks_status.main()
    print(__doc__)
    return 2


if __name__ == "__main__":  # required on macOS for scripts that start worker processes
    sys.exit(main(sys.argv[1:]))
