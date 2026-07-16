from __future__ import annotations

import pandas as pd
import pytest

from gex_monitor.synthetic_roll_monitor import (
    OptionLegSnapshot,
    SyntheticRollConfig,
    SyntheticRollSnapshot,
    evaluate_roll_need,
    _implied_financing_rate,
)


def leg(strike: float, right: str, mark: float, delta: float) -> OptionLegSnapshot:
    return OptionLegSnapshot(
        symbol="AMD",
        expiry="20270115",
        strike=strike,
        right=right,
        action="BUY",
        bid=None,
        ask=None,
        mid=None,
        mark=mark,
        mark_source="model",
        delta=delta,
        iv=0.60,
        open_interest=1000,
    )


def snap(
    spot: float,
    dte: int = 255,
    short_put_extrinsic: float | None = 20.0,
    entry_spot: float | None = 350.0,
) -> SyntheticRollSnapshot:
    return SyntheticRollSnapshot(
        ts=pd.Timestamp("2026-05-05 10:00", tz="America/New_York"),
        symbol="AMD",
        spot=spot,
        expiry="20270115",
        dte=dte,
        synthetic_strike=300.0,
        hedge_put_strike=200.0,
        quantity=1,
        entry_spot=entry_spot,
        synthetic_debit=50.0,
        implied_financing_rate=0.04,
        implied_financing_source="model/model",
        combo_mark=60.0,
        net_delta=0.90,
        short_put_extrinsic=short_put_extrinsic,
        hedge_strike_pct=200.0 / spot * 100.0,
        call=leg(300.0, "C", 110.0, 0.80),
        short_put=leg(300.0, "P", 60.0, -0.20),
        hedge_put=leg(200.0, "P", 10.0, -0.10),
    )


def cfg() -> SyntheticRollConfig:
    return SyntheticRollConfig(
        positions=[],
        upside_roll_pct=25.0,
        min_hedge_strike_pct=50.0,
    )


def test_roll_up_alert_when_spot_rises_far_above_entry():
    alert = evaluate_roll_need(snap(440.0), cfg())

    assert alert is not None
    assert "ROLL_UP" in alert.action
    assert "REANCHOR_HEDGE" in alert.action
    assert alert.priority == "watch"


def test_downside_alert_when_spot_near_short_put():
    alert = evaluate_roll_need(snap(325.0), cfg())

    assert alert is not None
    assert "DOWNSIDE_RISK" in alert.action
    assert alert.priority == "high"


def test_assignment_risk_when_short_put_itm_with_low_extrinsic():
    alert = evaluate_roll_need(snap(280.0, short_put_extrinsic=1.0), cfg())

    assert alert is not None
    assert "ASSIGNMENT_RISK" in alert.action
    assert alert.priority == "high"


def test_expiry_roll_alert_when_dte_low():
    alert = evaluate_roll_need(snap(350.0, dte=60), cfg())

    assert alert is not None
    assert "ROLL_EXPIRY" in alert.action
    assert alert.priority == "medium"


def test_no_alert_when_structure_is_inside_thresholds():
    assert evaluate_roll_need(snap(350.0), cfg()) is None


def test_implied_financing_rate_from_put_call_parity():
    call = leg(200.0, "C", 25.0, 0.55)
    put = leg(200.0, "P", 20.0, -0.45)

    rate = _implied_financing_rate(
        spot=200.0,
        strike=200.0,
        call=call,
        put=put,
        dte=365,
    )

    assert rate is not None
    assert rate == pytest.approx(0.0253, rel=1e-2)
