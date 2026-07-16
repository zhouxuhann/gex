import json

import pandas as pd

from gex_monitor.intraday_vrp_audit import build_vrp_daily_audit, write_vrp_daily_audit

SCHEDULE = ["09:35", "10:00", "10:30", "11:00", "12:00",
            "13:00", "14:00", "14:30", "15:00"]


def test_good_audit_has_full_coverage(tmp_path):
    quotes = pd.DataFrame({
        "scheduled_time": SCHEDULE,
        "status": ["ok"] * 9,
        "delay_seconds": [2.0] * 9,
        "quote_age_seconds": [1.0] * 9,
        "combined_spread_ratio": [0.05] * 9,
        "execution_haircut_pct": [0.025] * 9,
    })
    observations = pd.DataFrame({
        "scheduled_time": SCHEDULE,
        "rth_bar_count": [390] * 9,
    })
    report = build_vrp_daily_audit(
        symbol="QQQ", date_str="20260716", schedule=SCHEDULE,
        quotes=quotes, observations=observations,
    )
    assert report["quality"] == "good"
    assert report["coverage"] == 1.0
    assert report["settled_coverage"] == 1.0
    path = write_vrp_daily_audit(report, tmp_path)
    saved = json.loads(path.read_text())
    assert saved["symbol"] == "QQQ"
    assert saved["ohlc_complete"] is True


def test_bad_audit_identifies_missing_and_unsettled_slots():
    quotes = pd.DataFrame({
        "scheduled_time": ["09:35", "10:00", "10:30"],
        "status": ["ok", "stale_quote", "wide_spread"],
    })
    observations = pd.DataFrame({
        "scheduled_time": ["09:35"],
        "rth_bar_count": [200],
    })
    report = build_vrp_daily_audit(
        symbol="SPY", date_str="20260716", schedule=SCHEDULE,
        quotes=quotes, observations=observations,
    )
    assert report["quality"] == "bad"
    assert report["observed_slots"] == 3
    assert report["unsettled_slots"] == ["10:00", "10:30"]
    assert "11:00" in report["missing_slots"]
    assert report["status_counts"]["stale_quote"] == 1
