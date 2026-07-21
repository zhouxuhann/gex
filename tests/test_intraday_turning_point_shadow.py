from datetime import datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from gex_monitor.config import IntradayTurningPointShadowConfig
from gex_monitor.intraday_turning_point_shadow import (
    IntradayTurningPointShadow,
    candidate_direction,
    classify_realized_outcome,
    load_shadow_model,
    score_candidate,
)
from gex_monitor.time_utils import ET

MODEL_PATH = "config/turning_point_shadow_v1.json"


class FakeStorage:
    def __init__(self):
        self.rows = {}

    def load_turning_point_shadow(self, symbol, date_str):
        values = [
            row
            for (row_symbol, row_date, _), row in self.rows.items()
            if row_symbol == symbol and row_date == date_str
        ]
        return pd.DataFrame(values)

    def persist_turning_point_shadow(self, symbol, date_str, row):
        self.rows[(symbol, date_str, row["event_id"])] = dict(row)


def _row(**values) -> pd.Series:
    base = {
        "gex_quality_ok": True,
        "return_15m": -0.02,
        "volatility_unit_15m": 0.005,
        "rr_25_change_5m": 0.0,
        "atm_iv_pct": 20.0,
        "strike_abs_gex_concentration_50bps": 0.8,
        "gex_change_5m_ratio": 0.0,
        "gex_change_15m_ratio": 0.5,
    }
    base.update(values)
    return pd.Series(base)


def _market_inputs(minutes: int, *, future_decline: bool = False):
    start = datetime(2026, 7, 22, 9, 30, tzinfo=ET)
    bars = []
    history = []
    for index in range(minutes):
        timestamp = start + timedelta(minutes=index)
        if index < 25:
            close = 100.0 + (0.01 if index % 2 else -0.01)
        else:
            close = 100.0 - 0.12 * (index - 24)
        if future_decline and index > 40:
            close -= 0.12 * (index - 40)
        bars.append(
            {
                "ts": timestamp,
                "open": close,
                "high": close + 0.01,
                "low": close - 0.01,
                "close": close,
            }
        )
        history.append(
            {
                "ts": timestamp,
                "spot": close,
                "total_gex": 1_000_000_000.0,
                "flip": 98.0,
                "call_gex": 1_500_000_000.0,
                "put_gex": -500_000_000.0,
                "atm_iv_pct": 35.0,
                "call_wall": 103.0,
                "put_wall": 97.0,
                "positive_gamma": True,
                "max_pain": 100.0,
                "rr_25": 0.02,
                "partial": False,
            }
        )
    return history, bars, []


def test_frozen_model_contains_only_cross_symbol_rules():
    model = load_shadow_model(MODEL_PATH)

    assert model.model_id == "qqq_turning_point_shadow_v1_20260722"
    assert len(model.rules) == 5
    assert {rule.direction for rule in model.rules} == {"after_down", "after_up"}


def test_observation_only_is_a_configuration_hard_lock():
    with pytest.raises(ValidationError):
        IntradayTurningPointShadowConfig(observation_only=False)


def test_same_feature_family_cannot_double_count():
    model = load_shadow_model(MODEL_PATH)
    score = score_candidate(
        _row(gex_change_5m_ratio=0.2, gex_change_15m_ratio=0.1),
        "after_up",
        model,
    )

    assert score["matched_rule_count"] == 2
    assert score["reversal_support_points"] == 1
    assert score["reversal_support_max"] == 1
    assert score["watch_level"] == "WATCH"


def test_two_independent_continuation_families_make_strong_watch():
    model = load_shadow_model(MODEL_PATH)
    score = score_candidate(
        _row(atm_iv_pct=35.0, strike_abs_gex_concentration_50bps=0.5),
        "after_down",
        model,
    )

    assert score["continuation_risk_points"] == 2
    assert score["continuation_risk_score"] == 100.0
    assert score["watch_level"] == "STRONG_WATCH"
    assert score["score_bias"] == "continuation"


def test_candidate_and_outcome_use_frozen_label_thresholds():
    model = load_shadow_model(MODEL_PATH)
    assert candidate_direction(_row(), model.candidate) == "after_down"
    assert classify_realized_outcome(
        "after_down",
        future_up=0.001,
        future_down=-0.007,
        scale=0.005,
        config=model.candidate,
    ) == "continuation_down"


def test_shadow_records_every_candidate_and_backfills_without_orders():
    storage = FakeStorage()
    shadow = IntradayTurningPointShadow(
        "QQQ",
        storage,
        IntradayTurningPointShadowConfig(enabled=True, model_path=MODEL_PATH),
    )
    initial = _market_inputs(41)
    event = shadow.on_update(
        now=datetime(2026, 7, 22, 10, 11, 5, tzinfo=ET),
        input_provider=lambda: initial,
    )

    assert event is not None
    assert event["event_id"] == "QQQ_20260722_1010"
    assert event["observation_only"] is True
    assert event["evaluation_status"] == "pending"
    assert "order" not in event

    later = _market_inputs(56, future_decline=True)
    shadow.on_update(
        now=datetime(2026, 7, 22, 10, 26, 5, tzinfo=ET),
        input_provider=lambda: later,
    )
    settled = storage.rows[("QQQ", "20260722", "QQQ_20260722_1010")]

    assert settled["evaluation_status"] == "completed"
    assert settled["future_return_5m"] < 0
    assert settled["future_return_15m"] < settled["future_return_5m"]
    assert settled["realized_outcome"] == "continuation_down"
