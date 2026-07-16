from datetime import timedelta
from unittest.mock import MagicMock

from gex_monitor.qqq_0dte_gex_pa_swing import (
    GexState,
    OptionQuote,
    QQQ0DTEGexPASwingTrader,
    calculate_gamma_flip,
    calculate_gex,
    et_now,
    evaluate_pa_signal,
    evaluate_pa_with_gex,
    _ticker_exposure_quantity,
)


def _pa_signal(direction="LONG"):
    if direction == "LONG":
        return {
            "direction": "LONG",
            "setup": "H2",
            "score": 72,
            "entry": 101.01,
            "stop": 100.49,
            "target": 101.79,
            "rr_ratio": 1.5,
            "reason": "H2",
        }
    return {
        "direction": "SHORT",
        "setup": "L2",
        "score": 74,
        "entry": 99.99,
        "stop": 100.51,
        "target": 99.21,
        "rr_ratio": 1.5,
        "reason": "L2",
    }


def _fresh_gex(**kwargs):
    data = {
        "ts": et_now(),
        "spot": 101.0,
        "call_wall": 103.0,
        "put_wall": 98.0,
        "gamma_flip": 100.0,
        "total_gex": -1_000_000.0,
        "positive_gamma": False,
    }
    data.update(kwargs)
    return GexState(**data)


def test_calculate_gex_nets_calls_positive_and_puts_negative():
    rows = [
        (100.0, "C", 10.0, 0.01),
        (100.0, "P", 5.0, 0.01),
    ]

    out = calculate_gex(rows, 100.0)

    assert out[100.0] == 500.0


def test_gamma_flip_uses_cumulative_interpolation():
    flip = calculate_gamma_flip({99.0: -10.0, 100.0: 5.0, 101.0: 20.0})

    assert round(flip, 2) == 100.25


def test_evaluate_accepts_long_from_pa_signal_without_gex_gate():
    plan, reason = evaluate_pa_signal(
        _pa_signal("LONG"),
        min_rr=1.2,
        fallback_rr=2.0,
        gex=_fresh_gex(),
    )

    assert reason == "accepted"
    assert plan is not None
    assert plan.right == "C"
    assert plan.target == 101.79
    assert plan.rr_ratio >= 1.2
    assert plan.gex_reason.startswith("pa_only;gex_obs:")


def test_evaluate_does_not_use_flip_as_direction_gate():
    plan, reason = evaluate_pa_signal(
        _pa_signal("SHORT"),
        min_rr=1.2,
        fallback_rr=2.0,
        gex=_fresh_gex(),
    )

    assert reason == "accepted"
    assert plan is not None
    assert plan.right == "P"


def test_evaluate_ignores_positive_gamma_for_trade_gate():
    plan, reason = evaluate_pa_signal(
        _pa_signal("LONG"),
        min_rr=1.2,
        fallback_rr=2.0,
        gex=_fresh_gex(total_gex=1_000_000.0, positive_gamma=True),
    )

    assert reason == "accepted"
    assert plan is not None
    assert "gamma=positive" in plan.gex_reason


def test_evaluate_ignores_stale_gex_for_trade_gate():
    plan, reason = evaluate_pa_signal(
        _pa_signal("LONG"),
        min_rr=1.2,
        fallback_rr=2.0,
        gex=_fresh_gex(ts=et_now() - timedelta(minutes=20)),
    )

    assert reason == "accepted"
    assert plan is not None


def test_evaluate_can_use_fallback_target_when_pa_target_missing():
    sig = _pa_signal("LONG")
    sig.pop("target")
    plan, reason = evaluate_pa_signal(
        sig,
        min_rr=1.8,
        fallback_rr=2.0,
        gex=_fresh_gex(call_wall=101.5),
    )

    assert reason == "accepted"
    assert plan is not None
    assert plan.target == 102.05
    assert plan.gex_reason.startswith("pa_only;gex_obs:")


def test_backward_compatible_evaluate_name_is_pa_only():
    plan, reason = evaluate_pa_with_gex(
        _pa_signal("LONG"),
        _fresh_gex(total_gex=1_000_000.0, positive_gamma=True),
        spot=101.0,
        max_gex_age_sec=60,
        min_rr=1.2,
        fallback_rr=2.0,
        allow_positive_gamma=False,
        require_wall=True,
    )

    assert reason == "accepted"
    assert plan is not None
    assert plan.target == 101.79


def test_ticker_exposure_quantity_uses_side_specific_open_interest():
    ticker = MagicMock()
    ticker.callOpenInterest = 123
    ticker.putOpenInterest = 456
    ticker.volume = 789

    assert _ticker_exposure_quantity(ticker, "C") == 123.0
    assert _ticker_exposure_quantity(ticker, "P") == 456.0


def test_ticker_exposure_quantity_falls_back_to_volume_when_oi_missing():
    ticker = MagicMock()
    ticker.callOpenInterest = None
    ticker.putOpenInterest = float("nan")
    ticker.volume = 789

    assert _ticker_exposure_quantity(ticker, "C") == 789.0
    assert _ticker_exposure_quantity(ticker, "P") == 789.0


def test_pa_swing_queues_and_buys_each_signal_even_with_open_position(tmp_path):
    ib = MagicMock()
    contract = MagicMock()
    contract.conId = 123
    contract.localSymbol = "QQQ   260508C00101000"
    contract.lastTradeDateOrContractMonth = "20260508"
    contract.strike = 101.0
    trader = QQQ0DTEGexPASwingTrader(
        ib,
        dry_run=True,
        state_path=tmp_path / "state.json",
        log_dir=tmp_path,
    )
    trader._in_entry_window = MagicMock(return_value=True)
    trader._pick_0dte_option = MagicMock(return_value=contract)
    trader._quote_option = MagicMock(return_value=OptionQuote(bid=1.0, ask=1.02, last=None, close=None))

    trader._handle_pa_signal(_pa_signal("LONG"), 101.0)
    trader._handle_pa_signal(_pa_signal("LONG"), 101.1)

    assert len(trader.positions) == 0
    assert len(trader._entry_queue) == 2
    trader._drain_entry_queue(101.0)

    assert len(trader.positions) == 2
    rows = (tmp_path / f"qqq_0dte_gex_pa_swing_{et_now().strftime('%Y%m%d')}.csv").read_text()
    assert rows.count(",PLAN,") == 2
    assert rows.count(",ENTER,") == 2
