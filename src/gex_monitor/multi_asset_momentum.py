"""Python port of the TradingView indicator "Universal v2.1 (0DTE momentum)".

The public entry point is :func:`calculate_momentum_signals`.  It accepts an
OHLCV DataFrame and returns a copy with the three component signals, filters,
and entry events appended.  Calculations are causal: no output uses a future
bar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Asset = Literal["QQQ", "SPY", "BTC", "ETH", "MNQ", "MES"]
Bias = Literal["long", "short", "both"]

_ASSETS = {"QQQ", "SPY", "BTC", "ETH", "MNQ", "MES"}
_KALMAN_THRESH = {"QQQ": 0.100, "SPY": 0.075, "MNQ": 3.70, "MES": 1.00, "BTC": 60.0, "ETH": 100.0}
_ATR_MIN = {"QQQ": 0.70, "SPY": 0.50, "MNQ": 25.0, "MES": 8.0, "BTC": 500.0, "ETH": 30.0}


@dataclass(frozen=True)
class MomentumConfig:
    """Parameters corresponding to the Pine inputs.

    ``None`` means "automatic" for asset-dependent values.  Times are
    interpreted from a timezone-aware DatetimeIndex; US market times use
    America/New_York and crypto sessions use UTC.
    """

    asset: Asset | None = None
    fast_n: int = 8
    slow_n: int = 34
    vol_mult: float | None = None
    z_lo: float = 0.6
    z_hi: float = 2.5
    z_std_win: int = 20
    crypto_vwap_reset: Literal["daily", "weekly"] = "daily"
    futures_vwap_anchor: Literal["rth", "globex"] = "rth"
    kalman_q_price: float = 0.0001
    kalman_q_vel: float = 0.00001
    kalman_r: float = 0.01
    kalman_thresh: float | None = None
    atr_len: int = 14
    atr_min: float | None = None
    vote_thresh: int = 2
    filter_time: bool = True
    no_trade_open: int | None = None
    no_trade_close: int | None = None
    futures_rth_only: bool = True
    filter_futures_dead_hour: bool = True
    crypto_asia: bool = True
    crypto_europe: bool = True
    crypto_us: bool = True
    filter_crypto_dead: bool = True
    opening_mode: bool = False
    opening_bias: Bias = "long"
    opening_end_min: int = 30
    opening_vote: int = 1
    early_confirm_bars: int = 5
    early_vol_mult: float = 1.5


def detect_asset(symbol: str) -> Asset:
    """Match Pine's ticker substring detection and fallback to QQQ."""

    ticker = symbol.upper()
    for asset in ("MNQ", "MES", "BTC", "ETH", "SPY", "QQQ"):
        if asset in ticker:
            return asset  # type: ignore[return-value]
    return "QQQ"


def _validate_config(config: MomentumConfig) -> None:
    if config.asset is not None and config.asset not in _ASSETS:
        raise ValueError(f"unsupported asset: {config.asset}")
    if config.fast_n < 2 or config.slow_n < 5:
        raise ValueError("fast_n must be >= 2 and slow_n must be >= 5")
    if config.z_std_win < 2 or config.atr_len < 1:
        raise ValueError("z_std_win must be >= 2 and atr_len must be >= 1")
    if config.vote_thresh not in (1, 2, 3) or config.opening_vote not in (1, 2, 3):
        raise ValueError("vote thresholds must be between 1 and 3")
    if config.opening_bias not in ("long", "short", "both"):
        raise ValueError("opening_bias must be long, short, or both")


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("bars must use a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware")
    if frame.index.has_duplicates:
        raise ValueError("DatetimeIndex must not contain duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted oldest to newest")
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError("OHLCV values must be numeric and non-null")
    return frame


def _pine_ema(values: pd.Series, length: int) -> pd.Series:
    """Pine-style EMA: SMA seed followed by the recursive EMA formula."""

    out = np.full(len(values), np.nan, dtype=float)
    x = values.to_numpy(dtype=float)
    if len(x) < length:
        return pd.Series(out, index=values.index)
    alpha = 2.0 / (length + 1.0)
    # General enough for the second EMA, whose input begins with NaNs.
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < length:
        return pd.Series(out, index=values.index)
    seed_end = valid[length - 1]
    seed_positions = valid[:length]
    if not np.array_equal(seed_positions, np.arange(seed_positions[0], seed_end + 1)):
        return pd.Series(out, index=values.index)
    out[seed_end] = x[seed_positions].mean()
    for i in range(seed_end + 1, len(x)):
        if np.isnan(x[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=values.index)


def _pine_rma(values: pd.Series, length: int) -> pd.Series:
    """Wilder moving average used by TradingView's ATR."""

    out = np.full(len(values), np.nan, dtype=float)
    x = values.to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < length:
        return pd.Series(out, index=values.index)
    seed_end = valid[length - 1]
    out[seed_end] = x[valid[:length]].mean()
    for i in range(seed_end + 1, len(x)):
        out[i] = out[i - 1] if np.isnan(x[i]) else (out[i - 1] * (length - 1) + x[i]) / length
    return pd.Series(out, index=values.index)


def _kalman(close: pd.Series, config: MomentumConfig) -> tuple[pd.Series, pd.Series]:
    x0 = np.nan
    x1 = 0.0
    p00, p01, p10, p11 = 1000.0, 0.0, 0.0, 1000.0
    levels = np.empty(len(close), dtype=float)
    velocities = np.empty(len(close), dtype=float)
    for i, price in enumerate(close.to_numpy(dtype=float)):
        if np.isnan(x0):
            x0 = price
        else:
            xp0, xp1 = x0 + x1, x1
            pp00 = p00 + p10 + p01 + p11 + config.kalman_q_price
            pp01 = p01 + p11
            pp10 = p10 + p11
            pp11 = p11 + config.kalman_q_vel
            innovation = price - xp0
            innovation_var = pp00 + config.kalman_r
            k0, k1 = pp00 / innovation_var, pp10 / innovation_var
            x0, x1 = xp0 + k0 * innovation, xp1 + k1 * innovation
            p00, p01 = (1.0 - k0) * pp00, (1.0 - k0) * pp01
            p10, p11 = -k1 * pp00 + pp10, -k1 * pp01 + pp11
        levels[i], velocities[i] = x0, x1
    return pd.Series(levels, index=close.index), pd.Series(velocities, index=close.index)


def _grouped_vwap(price: pd.Series, volume: pd.Series, groups: pd.Index | pd.Series) -> pd.Series:
    pv = price * volume
    cum_pv = pv.groupby(groups, sort=False).cumsum()
    cum_volume = volume.groupby(groups, sort=False).cumsum()
    return (cum_pv / cum_volume).where(cum_volume > 0, price)


def _vwap(frame: pd.DataFrame, asset: Asset, config: MomentumConfig) -> pd.Series:
    idx = frame.index
    utc = idx.tz_convert("UTC")
    ny = idx.tz_convert("America/New_York")
    hlc3 = (frame["high"] + frame["low"] + frame["close"]) / 3.0

    if asset in ("BTC", "ETH"):
        if config.crypto_vwap_reset == "weekly":
            # Monday 00:00 UTC; unlike week-number-only grouping, this is safe across years.
            groups = utc.normalize() - pd.to_timedelta(utc.weekday, unit="D")
        else:
            groups = utc.normalize()
        return _grouped_vwap(hlc3, frame["volume"], groups)

    if asset in ("MNQ", "MES") and config.futures_vwap_anchor == "rth":
        rth = (ny.hour * 60 + ny.minute >= 570) & (ny.hour * 60 + ny.minute <= 975)
        date_group = pd.Series(ny.date, index=idx)
        pv = (hlc3 * frame["volume"]).where(rth, 0.0)
        vol = frame["volume"].where(rth, 0.0)
        cum_pv = pv.groupby(date_group, sort=False).cumsum()
        cum_vol = vol.groupby(date_group, sort=False).cumsum()
        return (cum_pv / cum_vol).where(cum_vol > 0, hlc3)

    if asset in ("MNQ", "MES"):
        # CME equity-index Globex session starts at 18:00 ET.
        groups = (ny - pd.Timedelta(hours=18)).date
    else:
        groups = ny.date
    return _grouped_vwap(hlc3, frame["volume"], pd.Index(groups))


def _session_masks(
    index: pd.DatetimeIndex, asset: Asset, config: MomentumConfig
) -> tuple[np.ndarray, np.ndarray]:
    ny = index.tz_convert("America/New_York")
    utc = index.tz_convert("UTC")
    ny_minutes = np.asarray(ny.hour * 60 + ny.minute)
    utc_minutes = np.asarray(utc.hour * 60 + utc.minute)
    is_crypto = asset in ("BTC", "ETH")
    is_futures = asset in ("MNQ", "MES")
    open_delay = 0 if config.no_trade_open is None else config.no_trade_open
    close_delay = (
        (10 if is_futures else 0 if is_crypto else 15)
        if config.no_trade_close is None
        else config.no_trade_close
    )

    if not config.filter_time:
        in_session = np.ones(len(index), dtype=bool)
    elif is_crypto:
        active = (
            (config.crypto_asia & ((utc_minutes >= 0) & (utc_minutes < 480)))
            | (config.crypto_europe & ((utc_minutes >= 420) & (utc_minutes < 960)))
            | (config.crypto_us & ((utc_minutes >= 780) & (utc_minutes < 1320)))
        )
        in_session = active & ~(config.filter_crypto_dead & (utc_minutes >= 1320))
    elif is_futures:
        dead = (ny_minutes >= 1020) & (ny_minutes <= 1080)
        day = (ny_minutes >= 570 + open_delay) & (ny_minutes <= 975 - close_delay)
        night = ~dead
        in_session = ~(config.filter_futures_dead_hour & dead) & (
            day if config.futures_rth_only else night
        )
    else:
        in_session = (ny_minutes >= 570 + open_delay) & (ny_minutes <= 960 - close_delay)

    if not config.opening_mode:
        opening_window = np.zeros(len(index), dtype=bool)
    elif config.opening_end_min == 0 or is_crypto:
        # This mirrors the Pine ternary: crypto bias is active all day.
        opening_window = np.ones(len(index), dtype=bool)
    else:
        opening_window = (ny_minutes >= 570) & (ny_minutes <= 570 + config.opening_end_min)
    return in_session, opening_window


def calculate_momentum_signals(
    bars: pd.DataFrame,
    *,
    symbol: str = "QQQ",
    config: MomentumConfig | None = None,
) -> pd.DataFrame:
    """Calculate Universal v2.1 signals for chronological OHLCV bars.

    The returned DataFrame includes state columns (``long_signal`` and
    ``short_signal``) and one-bar event columns (``long_entry``,
    ``short_entry``, and the early-entry equivalents).
    """

    cfg = config or MomentumConfig()
    _validate_config(cfg)
    frame = _prepare_bars(bars)
    asset: Asset = cfg.asset or detect_asset(symbol)
    vol_mult = (
        cfg.vol_mult
        if cfg.vol_mult is not None
        else (1.5 if asset == "SPY" else 2.0 if asset in ("BTC", "ETH") else 1.8)
    )
    kalman_thresh = cfg.kalman_thresh if cfg.kalman_thresh is not None else _KALMAN_THRESH[asset]
    atr_min = cfg.atr_min if cfg.atr_min is not None else _ATR_MIN[asset]

    ema_f1 = _pine_ema(frame["close"], cfg.fast_n)
    ema_s1 = _pine_ema(frame["close"], cfg.slow_n)
    frame["dema_fast"] = 2.0 * ema_f1 - _pine_ema(ema_f1, cfg.fast_n)
    frame["dema_slow"] = 2.0 * ema_s1 - _pine_ema(ema_s1, cfg.slow_n)
    frame["volume_ma"] = frame["volume"].rolling(cfg.slow_n, min_periods=cfg.slow_n).mean()
    frame["volume_ok"] = frame["volume"] > frame["volume_ma"] * vol_mult
    frame["dema_bull"] = frame["dema_fast"] > frame["dema_slow"]
    frame["dema_bear"] = frame["dema_fast"] < frame["dema_slow"]
    previous_fast = frame["dema_fast"].shift(1)
    previous_slow = frame["dema_slow"].shift(1)
    frame["dema_cross_up"] = (
        frame["dema_bull"] & previous_fast.le(previous_slow) & frame["volume_ok"]
    )
    frame["dema_cross_down"] = (
        frame["dema_bear"] & previous_fast.ge(previous_slow) & frame["volume_ok"]
    )
    frame["sig1"] = np.select([frame["dema_bull"], frame["dema_bear"]], [1, -1], default=0).astype(
        np.int8
    )

    frame["vwap"] = _vwap(frame, asset, cfg)
    frame["deviation_std"] = (
        frame["close"].rolling(cfg.z_std_win, min_periods=cfg.z_std_win).std(ddof=0)
    )
    frame["dev_z"] = ((frame["close"] - frame["vwap"]) / frame["deviation_std"]).replace(
        [np.inf, -np.inf], 0.0
    )
    frame["vwap_upper_1"] = frame["vwap"] + frame["deviation_std"]
    frame["vwap_lower_1"] = frame["vwap"] - frame["deviation_std"]
    frame["vwap_upper_hot"] = frame["vwap"] + cfg.z_hi * frame["deviation_std"]
    frame["vwap_lower_hot"] = frame["vwap"] - cfg.z_hi * frame["deviation_std"]
    vwap_long = (frame["dev_z"] > cfg.z_lo) & (frame["dev_z"] < cfg.z_hi)
    vwap_short = (frame["dev_z"] < -cfg.z_lo) & (frame["dev_z"] > -cfg.z_hi)
    frame["sig2"] = np.select([vwap_long, vwap_short], [1, -1], default=0).astype(np.int8)

    frame["kalman_price"], frame["kalman_velocity"] = _kalman(frame["close"], cfg)
    frame["sig3"] = np.select(
        [frame["kalman_velocity"] > kalman_thresh, frame["kalman_velocity"] < -kalman_thresh],
        [1, -1],
        default=0,
    ).astype(np.int8)
    previous_velocity = frame["kalman_velocity"].shift(1)
    frame["kalman_just_long"] = (frame["kalman_velocity"] > kalman_thresh) & (
        previous_velocity <= kalman_thresh
    )
    frame["kalman_just_short"] = (frame["kalman_velocity"] < -kalman_thresh) & (
        previous_velocity >= -kalman_thresh
    )

    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            (frame["high"] - frame["low"]),
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = _pine_rma(true_range, cfg.atr_len)
    frame["atr_filter"] = frame["atr"].ge(atr_min) if atr_min > 0 else True

    in_session, opening_window = _session_masks(frame.index, asset, cfg)
    frame["in_session"] = in_session
    frame["in_opening_window"] = opening_window
    allow_long = ~opening_window | (cfg.opening_bias in ("long", "both"))
    allow_short = ~opening_window | (cfg.opening_bias in ("short", "both"))
    effective_vote = np.where(opening_window, cfg.opening_vote, cfg.vote_thresh)
    frame["effective_vote"] = effective_vote.astype(np.int8)
    frame["score"] = (frame["sig1"] + frame["sig2"] + frame["sig3"]).astype(np.int8)
    frame["long_signal"] = (
        allow_long & (frame["score"] >= effective_vote) & in_session & frame["atr_filter"]
    )
    frame["short_signal"] = (
        allow_short & (frame["score"] <= -effective_vote) & in_session & frame["atr_filter"]
    )
    frame["long_entry"] = frame["long_signal"] & ~frame["long_signal"].shift(fill_value=False)
    frame["short_entry"] = frame["short_signal"] & ~frame["short_signal"].shift(fill_value=False)

    # Pine records the most recent Kalman threshold crossing and permits a
    # DEMA/volume confirmation through N subsequent bars, inclusive.
    n = len(frame)
    long_alert = -(10**9)
    short_alert = -(10**9)
    long_window = np.zeros(n, dtype=bool)
    short_window = np.zeros(n, dtype=bool)
    just_long = frame["kalman_just_long"].to_numpy()
    just_short = frame["kalman_just_short"].to_numpy()
    for i in range(n):
        if just_long[i] and allow_long[i] and in_session[i]:
            long_alert = i
        if just_short[i] and allow_short[i] and in_session[i]:
            short_alert = i
        long_window[i] = i - long_alert <= cfg.early_confirm_bars
        short_window[i] = i - short_alert <= cfg.early_confirm_bars
    frame["early_long_window"] = long_window
    frame["early_short_window"] = short_window
    volume_surge = frame["volume"] > frame["volume_ma"] * cfg.early_vol_mult
    long_confirmed = long_window & frame["dema_bull"] & volume_surge & allow_long & in_session
    short_confirmed = short_window & frame["dema_bear"] & volume_surge & allow_short & in_session
    frame["early_long_entry"] = long_confirmed & ~long_confirmed.shift(fill_value=False)
    frame["early_short_entry"] = short_confirmed & ~short_confirmed.shift(fill_value=False)
    frame.attrs.update(
        asset=asset,
        symbol=symbol,
        vol_mult=vol_mult,
        kalman_thresh=kalman_thresh,
        atr_min=atr_min,
    )
    return frame


def _read_csv(path: Path, timestamp_column: str, timezone: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if timestamp_column not in frame.columns:
        raise ValueError(f"timestamp column not found: {timestamp_column}")
    timestamps = pd.to_datetime(frame.pop(timestamp_column), errors="raise")
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        if timezone is None:
            raise ValueError("naive CSV timestamps require --timezone")
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calculate Universal v2.1 0DTE momentum signals")
    parser.add_argument("input", type=Path, help="input OHLCV CSV")
    parser.add_argument("output", type=Path, help="output CSV with signal columns")
    parser.add_argument("--symbol", default="QQQ", help="symbol used for automatic asset detection")
    parser.add_argument(
        "--asset", choices=sorted(_ASSETS), help="override automatic asset detection"
    )
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--timezone", help="timezone for naive timestamps, e.g. America/New_York")
    parser.add_argument("--no-time-filter", action="store_true")
    args = parser.parse_args(argv)
    bars = _read_csv(args.input, args.timestamp_column, args.timezone)
    config = replace(MomentumConfig(), asset=args.asset, filter_time=not args.no_time_filter)
    result = calculate_momentum_signals(bars, symbol=args.symbol, config=config)
    result.to_csv(args.output, index=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
