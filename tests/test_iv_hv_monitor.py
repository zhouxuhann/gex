from __future__ import annotations

import pandas as pd
import pytest

from gex_monitor.iv_hv_monitor import (
    IVHVRuntimeConfig,
    IVHVSnapshot,
    annualized_hv_from_closes,
    evaluate_decoupling,
)


def snap(i: int, price: float, iv: float | None, hv: float | None) -> IVHVSnapshot:
    return IVHVSnapshot(
        ts=pd.Timestamp("2026-04-30 10:00", tz="America/New_York") + pd.Timedelta(minutes=15 * i),
        symbol="AMD",
        price=price,
        atm_iv=iv,
        hv20=0.30,
        intraday_hv=hv,
        expiry="20260529",
        strike=100.0,
    )


def test_annualized_hv_from_closes_returns_positive_value():
    closes = pd.Series([100, 101, 100.5, 102, 101.7, 103])

    hv = annualized_hv_from_closes(closes, periods_per_year=252)

    assert hv is not None
    assert hv > 0


def test_decoupling_alert_when_price_up_but_iv_and_hv_down():
    cfg = IVHVRuntimeConfig(
        symbols=["AMD"],
        lookback_points=4,
        min_price_up_pct=0.5,
        iv_down_alert_pct=2.0,
        hv_down_alert_pct=5.0,
    )
    history = [
        snap(0, 100.0, 0.50, 0.40),
        snap(1, 100.4, 0.49, 0.39),
        snap(2, 100.8, 0.48, 0.37),
        snap(3, 101.2, 0.47, 0.35),
    ]

    alert = evaluate_decoupling(history, cfg)

    assert alert is not None
    assert alert.price_change_pct == pytest.approx(1.2)
    assert alert.iv_change_pct == pytest.approx(-6.0)
    assert alert.hv_change_pct == pytest.approx(-12.5)
    assert any("IV down" in reason for reason in alert.reasons)
    assert any("HV down" in reason for reason in alert.reasons)


def test_no_alert_when_price_and_iv_hv_rise_together():
    cfg = IVHVRuntimeConfig(symbols=["AMD"], lookback_points=4, min_price_up_pct=0.5)
    history = [
        snap(0, 100.0, 0.45, 0.30),
        snap(1, 100.5, 0.46, 0.31),
        snap(2, 101.0, 0.47, 0.32),
        snap(3, 101.5, 0.48, 0.33),
    ]

    assert evaluate_decoupling(history, cfg) is None


def test_no_alert_when_price_not_up_enough():
    cfg = IVHVRuntimeConfig(symbols=["AMD"], lookback_points=4, min_price_up_pct=0.5)
    history = [
        snap(0, 100.0, 0.50, 0.40),
        snap(1, 100.1, 0.48, 0.38),
        snap(2, 100.2, 0.47, 0.36),
        snap(3, 100.3, 0.46, 0.35),
    ]

    assert evaluate_decoupling(history, cfg) is None
