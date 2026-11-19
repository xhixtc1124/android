"""Project entry point.

Usage:
    python main.py check    read-only setup checks (never prints secrets)
"""

import sys


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"]:
        from phone_agent import env_check

        return env_check.main()
    print(__doc__)
    return 2


if __name__ == "__main__":  # required on macOS for scripts that start worker processes
    sys.exit(main(sys.argv[1:]))
