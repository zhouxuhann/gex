from datetime import datetime

from gex_monitor.time_utils import ET
from gex_monitor.vrp_context import VRPEventCalendar, opex_context, path_features


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
         "open": 100, "high": 101, "low": 99, "close": 100},
        {"ts": datetime(2026, 7, 16, 9, 31, tzinfo=ET),
         "open": 100, "high": 102, "low": 100, "close": 101},
        {"ts": datetime(2026, 7, 16, 9, 33, tzinfo=ET), "close": 200},
    ]
    result = path_features(bars, datetime(2026, 7, 16, 9, 32, tzinfo=ET), 101)
    assert result["entry_bar_count"] == 2
    assert result["session_open"] == 100
    assert abs(result["session_return_pct"] - 0.01) < 1e-9
    assert result["trend_efficiency_session"] == 1.0
