from types import SimpleNamespace

from gex_monitor.hard_stall_watchdog import stalled_workers


class FakeWorker:
    def __init__(self, symbol: str, heartbeat: float, running: bool = True):
        self._health = {
            "symbol": symbol,
            "running": running,
            "loop_heartbeat_ts": heartbeat,
        }

    def health_snapshot(self):
        return dict(self._health)


def test_stalled_workers_reports_only_live_stale_loops():
    workers = [
        FakeWorker("QQQ", 800.0),
        FakeWorker("SPY", 950.0),
        FakeWorker("OLD", 1.0, running=False),
    ]
    assert stalled_workers(
        workers, now_ts=1000.0, threshold_seconds=120.0
    ) == ["QQQ loop_stale=200s"]


def test_stalled_workers_ignores_uninitialized_heartbeat():
    worker = FakeWorker("QQQ", 0.0)
    assert stalled_workers(
        [worker], now_ts=1000.0, threshold_seconds=120.0
    ) == []
