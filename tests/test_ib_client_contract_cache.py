from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from gex_monitor.ib_client import IBWorker


class FakeIB:
    def __init__(self, bad_keys=None):
        self.bad_keys = set(bad_keys or [])
        self.qualify_calls = []
        self.requested = []
        self.cancelled = []

    def qualifyContracts(self, *contracts):
        self.qualify_calls.append(list(contracts))
        return [
            c for c in contracts
            if (c.lastTradeDateOrContractMonth, float(c.strike), c.right)
            not in self.bad_keys
        ]

    def reqMktData(self, contract, genericTickList="", snapshot=False):
        self.requested.append((float(contract.strike), contract.right, genericTickList))

    def cancelMktData(self, contract):
        self.cancelled.append((float(contract.strike), contract.right))

    def sleep(self, _sec):
        pass


def _worker() -> IBWorker:
    state = MagicMock()
    storage = MagicMock()
    storage.data_dir = Path("src/data")
    storage.get_previous_trading_day.return_value = None
    return IBWorker("QQQ", "QQQ", state, storage)


def test_subscribe_options_caches_invalid_contracts_after_partial_failure():
    worker = _worker()
    worker.ib = FakeIB({
        ("20260506", 709.0, "C"),
        ("20260506", 709.0, "P"),
    })

    worker._subscribe_options("20260506", [708.0, 709.0, 710.0], validate=True)

    first_attempt = [
        (float(c.strike), c.right) for c in worker.ib.qualify_calls[0]
    ]
    assert first_attempt == [
        (708.0, "C"), (708.0, "P"),
        (709.0, "C"), (709.0, "P"),
        (710.0, "C"), (710.0, "P"),
    ]
    assert ("20260506", 709.0, "C") in worker._invalid_contract_cache
    assert ("20260506", 709.0, "P") in worker._invalid_contract_cache

    worker.current_key = None
    worker._subscribe_options("20260506", [708.0, 709.0, 710.0], validate=True)

    second_attempt = [
        (float(c.strike), c.right) for c in worker.ib.qualify_calls[1]
    ]
    assert second_attempt == [
        (708.0, "C"), (708.0, "P"),
        (710.0, "C"), (710.0, "P"),
    ]


def test_subscribe_options_does_not_cache_when_qualification_mostly_fails():
    worker = _worker()
    worker.ib = FakeIB({
        ("20260506", 708.0, "C"),
        ("20260506", 708.0, "P"),
        ("20260506", 709.0, "C"),
        ("20260506", 709.0, "P"),
    })

    worker._subscribe_options("20260506", [708.0, 709.0], validate=True)

    assert worker._invalid_contract_cache == set()


def test_1102_does_not_request_reconnect():
    worker = _worker()

    worker._on_ib_error(-1, 1102, "Connectivity restored - data maintained", None)

    assert worker._reconnect_requested_reason is None


def test_cached_ticker_timestamp_is_not_counted_as_fresh_data():
    worker = _worker()
    marker_time = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    ticker = MagicMock()
    ticker.time = marker_time
    worker.ib = MagicMock()
    worker.ib.ticker.return_value = ticker
    worker.underlying = object()
    worker._last_market_data_marker = marker_time.timestamp()

    assert worker._process_tick() is False
    worker.state.update.assert_not_called()
