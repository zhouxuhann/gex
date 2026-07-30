"""Independent worker-heartbeat watchdog.

This thread never calls into a worker or acquires its locks.  If a worker is
stuck inside a synchronous IB/DB call, its loop heartbeat stops advancing and
the watchdog can still terminate the process for the external supervisor.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable

from .time_utils import et_now, is_market_open

log = logging.getLogger(__name__)


def stalled_workers(
    workers: Iterable,
    *,
    now_ts: float,
    threshold_seconds: float,
) -> list[str]:
    """Return descriptions for workers whose lock-free heartbeat is stale."""
    reasons = []
    for worker in workers:
        health = worker.health_snapshot()
        if not health.get("running", False):
            continue
        heartbeat = float(health.get("loop_heartbeat_ts") or 0.0)
        if heartbeat <= 0:
            continue
        age = max(0.0, now_ts - heartbeat)
        if age >= threshold_seconds:
            reasons.append(f"{health.get('symbol', '?')} loop_stale={age:.0f}s")
    return reasons


class HardStallWatchdog:
    """Monitor worker heartbeats outside the worker execution path."""

    def __init__(
        self,
        workers: Iterable,
        *,
        threshold_seconds: float,
        check_seconds: float,
        fatal_callback: Callable[[list[str]], None],
        market_open_provider: Callable[[], bool] | None = None,
    ):
        self.workers = list(workers)
        self.threshold_seconds = max(30.0, float(threshold_seconds))
        self.check_seconds = max(1.0, float(check_seconds))
        self.fatal_callback = fatal_callback
        self.market_open_provider = market_open_provider or (
            lambda: is_market_open(et_now())
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="gex-hard-stall-watchdog",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.check_seconds):
            if not self.market_open_provider():
                continue
            reasons = stalled_workers(
                self.workers,
                now_ts=time.time(),
                threshold_seconds=self.threshold_seconds,
            )
            if reasons:
                log.critical("Hard stall detected: %s", "; ".join(reasons))
                self.fatal_callback(reasons)
                return
