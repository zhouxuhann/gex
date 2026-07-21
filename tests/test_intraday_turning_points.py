from datetime import datetime, timedelta

import pandas as pd
import pytest

from gex_monitor.intraday_turning_points import (
    TurningPointConfig,
    add_lagged_features,
    add_outcome_labels,
    align_minute_gex,
    deduplicate_candidates,
    select_audit_events,
    strike_snapshot_features,
)
from gex_monitor.time_utils import ET


def _ts(minutes: int) -> datetime:
    return datetime(2026, 7, 21, 9, 30, tzinfo=ET) + timedelta(minutes=minutes)


def test_gex_alignment_is_strictly_backward_and_reports_age():
    bars = pd.DataFrame({
        "ts": [_ts(1), _ts(2)], "open": [100, 101], "high": [101, 102],
        "low": [99, 100], "close": [100, 101],
    })
    gex = pd.DataFrame({
        "ts": [_ts(0) + timedelta(seconds=50), _ts(1) + timedelta(seconds=10)],
        "spot": [100, 999], "total_gex": [1.0, 2.0], "flip": [99, 998],
    })
    aligned = align_minute_gex(bars, gex, tolerance_seconds=120)
    # 09:31 cannot see the 09:31:10 snapshot.
    assert aligned.iloc[0]["spot"] == 100
    assert aligned.iloc[0]["gex_age_seconds"] == 10
    assert aligned.iloc[1]["spot"] == 999
    assert aligned.iloc[1]["gex_ts"] <= aligned.iloc[1]["ts"]


def test_future_prices_change_labels_but_not_lagged_features():
    config = TurningPointConfig(minimum_volatility_pct=0.001)
    close = [100.0] * 20 + [99.8, 99.6, 99.4, 99.2, 99.0]
    close += [99.0 + 0.15 * index for index in range(1, 21)]
    base = pd.DataFrame({
        "ts": [_ts(i) for i in range(len(close))],
        "open": close, "high": close, "low": close, "close": close,
        "total_gex": [1e9] * len(close), "flip": [98.0] * len(close),
        "call_wall": [102.0] * len(close), "put_wall": [98.0] * len(close),
    })
    features = add_lagged_features(base, config)
    changed = base.copy()
    changed.loc[25:, ["open", "high", "low", "close"]] = 98.0
    changed_features = add_lagged_features(changed, config)
    feature_columns = [
        "return_5m", "return_15m", "rv_15m", "volatility_unit_15m",
        "dist_to_flip_vol", "total_gex_change_15m",
    ]
    pd.testing.assert_series_equal(
        features.loc[24, feature_columns],
        changed_features.loc[24, feature_columns],
        check_names=False,
    )
    original_labeled = add_outcome_labels(features, config)
    changed_labeled = add_outcome_labels(changed_features, config)
    assert original_labeled.loc[24, "outcome"] != changed_labeled.loc[24, "outcome"]
    assert not bool(original_labeled.iloc[-1]["label_complete"])
    assert not bool(original_labeled.iloc[-1]["is_candidate"])


def test_candidate_dedup_is_separate_by_direction():
    frame = pd.DataFrame({
        "ts": [_ts(0), _ts(5), _ts(6), _ts(16)],
        "is_candidate": [True] * 4,
        "gex_quality_ok": [True, False, True, True],
        "setup_direction": ["after_down", "after_down", "after_up", "after_down"],
    })
    selected = deduplicate_candidates(frame, cooldown_minutes=15)
    assert selected["ts"].tolist() == [_ts(0), _ts(6), _ts(16)]


def test_strike_features_do_not_conflict_with_legacy_spot_column():
    strikes = pd.DataFrame({
        "ts": [_ts(1), _ts(1)], "strike": [99.0, 101.0],
        "right": ["P", "C"], "gex": [-2.0, 3.0], "gamma": [0.1, 0.1],
        "oi": [10, 20], "spot": [999.0, 999.0],
    })
    spots = pd.DataFrame({"gex_ts": [_ts(0)], "spot": [100.0]})
    result = strike_snapshot_features(strikes, spots)
    assert len(result) == 1
    assert result.iloc[0]["nearest_up_strike_distance_pct"] == pytest.approx(0.01)


def test_audit_selection_is_stratified_and_bounded():
    events = pd.DataFrame({
        "event_id": [f"e{index}" for index in range(20)],
        "outcome": ["reversal_up"] * 10 + ["continuation_down"] * 10,
        "ts": [_ts(index) for index in range(20)],
    })
    selected = select_audit_events(events, count=8, seed=1)
    assert len(selected) == 8
    assert selected["outcome"].value_counts().to_dict() == {
        "reversal_up": 4, "continuation_down": 4,
    }
