"""Tests for ad-hoc GEX helpers."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from gex_monitor.adhoc_gex import (
    _quality_from_tickers,
    classify_quality,
    ensure_event_loop,
    select_nearby_strikes,
)


def test_select_nearby_strikes_preserves_real_chain_spacing():
    strikes = [90, 95, 97.5, 100, 102.5, 105, 110]

    selected = select_nearby_strikes(strikes, spot=101.0, strikes_each_side=2)

    assert selected == [97.5, 100.0, 102.5, 105.0]


def test_classify_quality_thresholds():
    assert classify_quality(32, 40) == "ok"
    assert classify_quality(20, 40) == "partial"
    assert classify_quality(19, 40) == "bad"
    assert classify_quality(0, 0) == "bad"


def test_quality_from_tickers_counts_greeks_oi_and_bidask():
    def ticker(right, gamma, oi, bid=None, ask=None):
        t = MagicMock()
        t.contract = SimpleNamespace(right=right)
        t.modelGreeks = SimpleNamespace(gamma=gamma) if gamma is not None else None
        t.callOpenInterest = oi if right == "C" else None
        t.putOpenInterest = oi if right == "P" else None
        t.bid = bid
        t.ask = ask
        return t

    tickers = [
        ticker("C", 0.1, 100, bid=1.0),
        ticker("P", None, 200, ask=1.2),
        ticker("C", None, None),
    ]

    quality = _quality_from_tickers(tickers, requested_contracts=4, qualified_contracts=3)

    assert quality.greeks_count == 1
    assert quality.oi_count == 2
    assert quality.bidask_count == 2
    assert quality.quality == "bad"


def test_ensure_event_loop_in_worker_thread():
    result = []

    def worker():
        ensure_event_loop()
        result.append(asyncio.get_event_loop().is_closed())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert result == [False]
