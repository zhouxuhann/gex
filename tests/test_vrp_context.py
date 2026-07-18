from datetime import datetime
from types import SimpleNamespace

from gex_monitor.time_utils import ET
import gex_monitor.vrp_context as context_module
from gex_monitor.vrp_context import (
    VRPEventCalendar, opex_context, path_features, vix_context,
)


def test_event_calendar_labels_pre_and_post_fomc(tmp_path):
    calendar = VRPEventCalendar(tmp_path)
    pre = calendar.context(datetime(2026, 7, 29, 13, 0, tzinfo=ET))
    post = calendar.context(datetime(2026, 7, 29, 14, 30, tzinfo=ET))
    assert pre["event_flag"] == "FOMC"
    assert pre["event_phase"] == "pre"
    assert pre["minutes_to_event"] == 60
    assert post["event_phase"] == "post"
    assert post["minutes_to_event"] == -30


def test_quarterly_opex_label():
    context = opex_context(datetime(2026, 9, 18, 10, 0, tzinfo=ET))
    assert context["weekday"] == "Friday"
    assert context["opex_type"] == "quarterly"


def test_path_features_exclude_premarket_and_use_only_past_bars():
    bars = [
        {"ts": datetime(2026, 7, 16, 9, 29, tzinfo=ET), "close": 90},
        {"ts": datetime(2026, 7, 16, 9, 30, tzinfo=ET),
         "open": 100, "high": 101, "low": 99, "close": 100,
         "volume": 100, "average": 100},
        {"ts": datetime(2026, 7, 16, 9, 31, tzinfo=ET),
         "open": 100, "high": 102, "low": 100, "close": 101,
         "volume": 300, "average": 101},
        {"ts": datetime(2026, 7, 16, 9, 33, tzinfo=ET), "close": 200},
    ]
    result = path_features(bars, datetime(2026, 7, 16, 9, 32, tzinfo=ET), 101,
                           previous_close=101)
    assert result["entry_bar_count"] == 2
    assert result["session_open"] == 100
    assert abs(result["session_return_pct"] - 0.01) < 1e-9
    assert result["trend_efficiency_session"] == 1.0
    assert result["session_vwap"] == 100.75
    assert result["gap_pct"] == (100 - 101) / 101
    assert bool(result["gap_filled_before_entry"])


def test_vix_context_uses_prior_daily_history_and_shared_cache():
    class FakeIB:
        def __init__(self):
            self.snapshots = 0

        def isConnected(self):
            return True

        def qualifyContracts(self, contract):
            return [contract]

        def reqTickers(self, contract):
            self.snapshots += 1
            return [SimpleNamespace(marketPrice=lambda: 22.0)]

        def reqHistoricalData(self, *args, **kwargs):
            return [SimpleNamespace(date=f"2026-06-{day:02d}", close=float(day))
                    for day in range(1, 21)]

    context_module._VIX_LIVE_CACHE.clear()
    context_module._VIX_HISTORY_CACHE.clear()
    ib = FakeIB()
    now = datetime(2026, 7, 16, 10, 0, tzinfo=ET)
    first = vix_context(ib, now)
    second = vix_context(ib, now)
    assert first["vix"] == 22.0
    assert first["vix_previous_close"] == 20.0
    assert first["vix_ma20"] == 10.5
    assert second == first
    assert ib.snapshots == 1
