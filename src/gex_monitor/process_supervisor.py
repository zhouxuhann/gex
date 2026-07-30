"""Small process supervisor used by the launch script.

The GEX process intentionally exits with a non-zero code when its independent
heartbeat watchdog detects a hard stall.  This supervisor starts a fresh
process without touching the bar collectors or Macro dashboard.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Restart a child process on exit")
    parser.add_argument("--restart-delay", type=float, default=10.0)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("missing child command after --")

    stop_file = Path(args.stop_file)
    stopping = False
    child: subprocess.Popen | None = None

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while not stopping and not stop_file.exists():
        print(
            f"[supervisor] starting child: {' '.join(command)}",
            flush=True,
        )
        child = subprocess.Popen(command)
        status = child.wait()
        child = None
        if stopping or stop_file.exists():
            break
        print(
            f"[supervisor] child exited status={status}; "
            f"restart in {args.restart_delay:.1f}s",
            flush=True,
        )
        deadline = time.monotonic() + max(1.0, args.restart_delay)
        while not stopping and not stop_file.exists() and time.monotonic() < deadline:
            time.sleep(0.25)

    return 0


if __name__ == "__main__":
    sys.exit(main())
