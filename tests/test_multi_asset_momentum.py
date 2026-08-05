import numpy as np
import pandas as pd
import pytest

from gex_monitor.multi_asset_momentum import (
    MomentumConfig,
    calculate_momentum_signals,
    detect_asset,
)


def _bars(index: pd.DatetimeIndex, start: float = 500.0, step: float = 0.2) -> pd.DataFrame:
    close = start + np.arange(len(index)) * step
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 1_000.0),
        },
        index=index,
    )


def test_detect_asset_matches_pine_precedence_and_fallback():
    assert detect_asset("CME_MINI:MNQ1!") == "MNQ"
    assert detect_asset("BINANCE:BTCUSDT") == "BTC"
    assert detect_asset("NYSEARCA:SPY") == "SPY"
    assert detect_asset("UNKNOWN") == "QQQ"


def test_etf_session_filter_respects_new_york_dst():
    # Both timestamps are 09:30 New York, on opposite sides of the DST change.
    index = pd.DatetimeIndex(["2026-01-15 14:30:00+00:00", "2026-07-15 13:30:00+00:00"])
    result = calculate_momentum_signals(
        _bars(index), config=MomentumConfig(atr_min=0, no_trade_close=0)
    )
    assert result["in_session"].tolist() == [True, True]


def test_crypto_daily_vwap_resets_at_midnight_utc():
    index = pd.DatetimeIndex(
        ["2026-01-01 23:58:00+00:00", "2026-01-01 23:59:00+00:00", "2026-01-02 00:00:00+00:00"]
    )
    bars = _bars(index, start=100.0, step=10.0)
    result = calculate_momentum_signals(
        bars, symbol="BTCUSDT", config=MomentumConfig(asset="BTC", atr_min=0)
    )
    expected_last_hlc3 = (bars.iloc[-1]["high"] + bars.iloc[-1]["low"] + bars.iloc[-1]["close"]) / 3
    assert result.iloc[-1]["vwap"] == pytest.approx(expected_last_hlc3)


def test_calculation_is_causal_when_future_bars_are_appended():
    index = pd.date_range("2026-07-15 09:30", periods=120, freq="min", tz="America/New_York")
    bars = _bars(index)
    config = MomentumConfig(atr_min=0)
    first = calculate_momentum_signals(bars.iloc[:80], config=config)
    full = calculate_momentum_signals(bars, config=config)
    columns = ["dema_fast", "dema_slow", "vwap", "dev_z", "kalman_velocity", "atr", "score"]
    pd.testing.assert_frame_equal(first[columns], full.iloc[:80][columns])


def test_entry_event_only_fires_on_false_to_true_transition():
    index = pd.date_range("2026-07-15 09:30", periods=100, freq="min", tz="America/New_York")
    bars = _bars(index, step=0.4)
    config = MomentumConfig(
        vote_thresh=1,
        atr_min=0,
        kalman_thresh=0,
        no_trade_close=0,
        filter_time=False,
    )
    result = calculate_momentum_signals(bars, config=config)
    expected = result["long_signal"] & ~result["long_signal"].shift(fill_value=False)
    pd.testing.assert_series_equal(result["long_entry"], expected, check_names=False)
    assert result["long_entry"].any()


def test_rejects_naive_timestamps_to_prevent_session_errors():
    bars = _bars(pd.date_range("2026-07-15 09:30", periods=50, freq="min"))
    with pytest.raises(ValueError, match="timezone-aware"):
        calculate_momentum_signals(bars)
