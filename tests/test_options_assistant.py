from __future__ import annotations

import pandas as pd

import gex_monitor.options_assistant as oa
from gex_monitor.options_assistant import (
    AssistantConfig,
    OptionQuote,
    choose_plan,
    trend_score_from_closes,
)


def q(strike: float, right: str, delta: float, mid: float = 2.0) -> OptionQuote:
    return OptionQuote(
        symbol="AMD",
        expiry="20260619",
        strike=strike,
        right=right,
        delta=delta,
        iv=0.35,
        bid=mid - 0.05,
        ask=mid + 0.05,
        mid=mid,
    )


def strong_trend(hv20: float = 0.40):
    closes = pd.Series([100 + i * 0.5 for i in range(60)])
    t = trend_score_from_closes("AMD", closes)
    assert t is not None
    t.hv20 = hv20
    return t


def weak_trend():
    closes = pd.Series([120 - i * 0.2 for i in range(60)])
    t = trend_score_from_closes("AMD", closes)
    assert t is not None
    t.hv20 = 0.40
    return t


def test_cheap_iv_prefers_long_call():
    cfg = AssistantConfig(symbols=["AMD"])
    calls = [q(125, "C", 0.60), q(135, "C", 0.30, mid=1.0)]

    plan = choose_plan(strong_trend(), atm_iv=0.38, calls=calls, puts=[], cfg=cfg)

    assert plan.action == "BUY_CALL"
    assert plan.long_leg is not None
    assert plan.long_leg.strike == 125


def test_fair_iv_prefers_call_debit_spread():
    cfg = AssistantConfig(symbols=["AMD"])
    calls = [q(125, "C", 0.60, mid=3.0), q(135, "C", 0.30, mid=1.0)]

    plan = choose_plan(strong_trend(), atm_iv=0.48, calls=calls, puts=[], cfg=cfg)

    assert plan.action == "BUY_CALL_DEBIT_SPREAD"
    assert plan.long_leg is not None
    assert plan.short_leg is not None
    assert plan.max_debit is not None


def test_expensive_iv_waits_instead_of_chasing_naked_call():
    cfg = AssistantConfig(symbols=["AMD"])
    calls = [q(125, "C", 0.60, mid=3.0), q(135, "C", 0.30, mid=1.0)]

    plan = choose_plan(strong_trend(), atm_iv=0.70, calls=calls, puts=[], cfg=cfg)

    assert plan.action == "WAIT_PREMIUM_EXPENSIVE"
    assert plan.long_leg is not None
    assert plan.short_leg is not None


def test_weak_trend_suggests_watch_or_protective_put():
    cfg = AssistantConfig(symbols=["AMD"])
    puts = [q(110, "P", -0.30, mid=1.5)]

    plan = choose_plan(weak_trend(), atm_iv=0.40, calls=[], puts=puts, cfg=cfg)

    assert plan.action == "WATCH_OR_PROTECTIVE_PUT"
    assert plan.long_leg is not None
    assert plan.long_leg.right == "P"


def test_earnings_blackout_blocks_actionable_call_signal(monkeypatch):
    monkeypatch.setattr(
        oa,
        "et_now",
        lambda: pd.Timestamp("2026-05-02 10:00", tz="America/New_York"),
    )
    cfg = AssistantConfig(
        symbols=["AMD"],
        earnings_dates={"AMD": "2026-05-05"},
        earnings_blackout_days_before=7,
        earnings_blackout_days_after=2,
    )
    calls = [q(125, "C", 0.60), q(135, "C", 0.30, mid=1.0)]

    plan = choose_plan(strong_trend(), atm_iv=0.38, calls=calls, puts=[], cfg=cfg)

    assert plan.action == "WAIT_EARNINGS_BLACKOUT"
    assert plan.earnings_blackout is True
    assert plan.earnings_date == "2026-05-05"
    assert plan.earnings_days_to == 3
