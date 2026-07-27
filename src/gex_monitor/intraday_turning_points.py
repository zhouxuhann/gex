"""Offline intraday turning-point dataset builder.

This module deliberately has no connection to the live signal or execution
path.  Future bars are used only to create research labels; every feature is
computed from information available at or before the candidate timestamp.
"""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .time_utils import ET

GEX_COLUMNS = (
    "spot", "total_gex", "flip", "call_gex", "put_gex", "atm_iv_pct",
    "call_wall", "put_wall", "positive_gamma", "max_pain", "rr_25",
    "skew_slope", "rr_25_zscore", "partial", "quality_reasons",
    "volume_gamma", "gross_gex", "net_gex_ratio", "gross_volume_gamma",
    "volume_gamma_to_gex", "gex_method", "gamma_flip_status",
    "gamma_flip_reliable",
)


@dataclass(frozen=True)
class TurningPointConfig:
    lookback_minutes: int = 15
    horizon_minutes: int = 15
    volatility_window: int = 30
    impulse_sigma: float = 1.0
    reversal_sigma: float = 1.0
    adverse_sigma: float = 0.35
    continuation_sigma: float = 1.0
    minimum_volatility_pct: float = 0.001
    cooldown_minutes: int = 15
    gex_tolerance_seconds: int = 120
    strike_tolerance_seconds: int = 120
    minimum_bars_per_day: int = 200
    minimum_gex_coverage: float = 0.90


def _to_et(values: pd.Series) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce", utc=True)
    return result.dt.tz_convert(ET)


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def align_minute_gex(
    bars: pd.DataFrame,
    gex: pd.DataFrame,
    *,
    tolerance_seconds: int = 120,
) -> pd.DataFrame:
    """Backward-align the latest observable GEX snapshot to each minute bar."""
    if bars.empty or gex.empty:
        return pd.DataFrame()
    left = bars.copy()
    right = gex.copy()
    left["ts"] = _to_et(left["ts"])
    right["ts"] = _to_et(right["ts"])
    left = left.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts")
    right = right.dropna(subset=["ts"]).sort_values("ts")
    right = right.rename(columns={"ts": "gex_ts"})
    keep = ["gex_ts", *(column for column in GEX_COLUMNS if column in right)]
    aligned = pd.merge_asof(
        left,
        right[keep],
        left_on="ts",
        right_on="gex_ts",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    aligned["gex_age_seconds"] = (
        aligned["ts"] - aligned["gex_ts"]
    ).dt.total_seconds()
    return aligned


def strike_snapshot_features(strikes: pd.DataFrame, spot_by_ts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the strike surface into interpretable, timestamped features."""
    if strikes.empty or spot_by_ts.empty:
        return pd.DataFrame()
    frame = strikes.copy()
    frame["ts"] = _to_et(frame["ts"])
    spots = spot_by_ts[["gex_ts", "spot"]].dropna().drop_duplicates("gex_ts")
    spots = spots.rename(columns={"gex_ts": "ts"}).sort_values("ts")
    strike_times = frame[["ts"]].drop_duplicates().sort_values("ts")
    strike_spots = pd.merge_asof(
        strike_times,
        spots,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(minutes=2),
    ).dropna(subset=["spot"]).rename(columns={"spot": "matched_spot"})
    frame = frame.merge(strike_spots, on="ts", how="inner")
    if frame.empty:
        return pd.DataFrame()
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    frame["gex"] = pd.to_numeric(frame.get("gex"), errors="coerce").fillna(0.0)
    frame["gamma"] = pd.to_numeric(frame.get("gamma"), errors="coerce").fillna(0.0)
    frame["oi"] = pd.to_numeric(frame.get("oi"), errors="coerce").fillna(0.0)
    frame["volume"] = (
        pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        if "volume" in frame else 0.0
    )
    frame["volume_gamma"] = (
        pd.to_numeric(frame["volume_gamma"], errors="coerce").fillna(0.0)
        if "volume_gamma" in frame else 0.0
    )
    frame["distance_pct"] = frame["strike"] / frame["matched_spot"] - 1.0
    frame["abs_gex"] = frame["gex"].abs()
    frame["near"] = frame["distance_pct"].abs() <= 0.005
    frame["above_near"] = frame["near"] & (
        frame["strike"] > frame["matched_spot"]
    )
    frame["below_near"] = frame["near"] & (
        frame["strike"] < frame["matched_spot"]
    )
    rights = frame["right"].astype(str) if "right" in frame else pd.Series(
        "", index=frame.index
    )
    frame["call_near_oi"] = np.where(
        frame["near"] & rights.eq("C"), frame["oi"], 0.0
    )
    frame["put_near_oi"] = np.where(
        frame["near"] & rights.eq("P"), frame["oi"], 0.0
    )

    rows = []
    for timestamp, snapshot in frame.groupby("ts", sort=True):
        above = snapshot[snapshot["above_near"]]
        below = snapshot[snapshot["below_near"]]
        near_abs = float(snapshot.loc[snapshot["near"], "abs_gex"].sum())
        total_abs = float(snapshot["abs_gex"].sum())
        total_gex = float(snapshot["gex"].sum())
        total_oi = float(snapshot["oi"].sum())
        total_volume = float(snapshot["volume"].sum())
        gross_volume_gamma = float(snapshot["volume_gamma"].abs().sum())
        above_abs = float(above["abs_gex"].sum())
        below_abs = float(below["abs_gex"].sum())
        denominator = above_abs + below_abs
        unique_above = snapshot.loc[
            snapshot["strike"] > snapshot["matched_spot"], "strike"
        ].dropna()
        unique_below = snapshot.loc[
            snapshot["strike"] < snapshot["matched_spot"], "strike"
        ].dropna()
        spot = float(snapshot["matched_spot"].iloc[0])
        rows.append({
            "strike_ts": timestamp,
            "strike_gex_above_50bps": float(above["gex"].sum()),
            "strike_gex_below_50bps": float(below["gex"].sum()),
            "strike_abs_gex_concentration_50bps": near_abs / total_abs
            if total_abs else None,
            "strike_total_abs_gex": total_abs,
            "strike_net_gex_ratio": total_gex / total_abs if total_abs else None,
            "strike_total_volume": total_volume,
            "strike_total_oi": total_oi,
            "strike_volume_oi_ratio": total_volume / total_oi if total_oi else None,
            "strike_gross_volume_gamma": gross_volume_gamma,
            "strike_volume_gamma_to_gex": (
                gross_volume_gamma / total_abs if total_abs else None
            ),
            "strike_abs_gex_imbalance_50bps": (above_abs - below_abs) / denominator
            if denominator else None,
            "strike_gamma_above_50bps": float(above["gamma"].sum()),
            "strike_gamma_below_50bps": float(below["gamma"].sum()),
            "strike_call_oi_50bps": float(snapshot["call_near_oi"].sum()),
            "strike_put_oi_50bps": float(snapshot["put_near_oi"].sum()),
            "nearest_up_strike_distance_pct": (
                float(unique_above.min()) / spot - 1.0 if not unique_above.empty else None
            ),
            "nearest_down_strike_distance_pct": (
                float(unique_below.max()) / spot - 1.0 if not unique_below.empty else None
            ),
        })
    return pd.DataFrame(rows)


def align_strike_features(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    tolerance_seconds: int = 120,
) -> pd.DataFrame:
    if frame.empty or features.empty:
        return frame
    result = pd.merge_asof(
        frame.sort_values("ts"),
        features.sort_values("strike_ts"),
        left_on="ts",
        right_on="strike_ts",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    result["strike_age_seconds"] = (
        result["ts"] - result["strike_ts"]
    ).dt.total_seconds()
    return result


def rebuild_oi_position_gex(
    gex: pd.DataFrame,
    strikes: pd.DataFrame,
    *,
    multiplier: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild historical strike and snapshot GEX from gamma × OI only.

    Historical files without volume can still support the position channel.
    The legacy flip is deliberately cleared because it cannot be faithfully
    repriced without the original expiry metadata and full option inputs.
    """
    if gex.empty or strikes.empty:
        return pd.DataFrame(), pd.DataFrame()
    history = gex.copy()
    surface = strikes.copy()
    history["ts"] = _to_et(history["ts"])
    surface["ts"] = _to_et(surface["ts"])
    spots = history[["ts", "spot"]].copy().dropna().sort_values("ts")
    spots["spot"] = pd.to_numeric(spots["spot"], errors="coerce")
    surface = pd.merge_asof(
        surface.sort_values("ts"), spots.rename(columns={"spot": "matched_spot"}),
        on="ts", direction="backward", tolerance=pd.Timedelta(seconds=120),
    )
    for column in ("strike", "gamma", "oi", "matched_spot"):
        surface[column] = pd.to_numeric(surface.get(column), errors="coerce")
    right = surface.get("right", pd.Series("", index=surface.index)).astype(str)
    surface = surface[
        surface[["strike", "gamma", "oi", "matched_spot"]].notna().all(axis=1)
        & right.isin(["C", "P"])
    ].copy()
    if surface.empty:
        return pd.DataFrame(), pd.DataFrame()
    sign = np.where(surface["right"].astype(str).eq("C"), 1.0, -1.0)
    surface["gex"] = (
        surface["gamma"] * surface["oi"] * multiplier
        * surface["matched_spot"].pow(2) * 0.01 * sign
    )
    surface["gex_method"] = "oi_position_v2"
    surface = surface.drop(columns=["matched_spot"])

    rows = []
    for timestamp, snapshot in surface.groupby("ts", sort=True):
        calls = snapshot[snapshot["right"].astype(str).eq("C")]
        puts = snapshot[snapshot["right"].astype(str).eq("P")]
        call_gex = float(calls["gex"].sum())
        put_gex = float(puts["gex"].sum())
        gross = abs(call_gex) + abs(put_gex)
        call_by_strike = calls.groupby("strike")["gex"].sum()
        put_by_strike = puts.groupby("strike")["gex"].sum().abs()
        rows.append({
            "ts": timestamp,
            "total_gex_v2": call_gex + put_gex,
            "call_gex_v2": call_gex,
            "put_gex_v2": put_gex,
            "gross_gex_v2": gross,
            "net_gex_ratio_v2": (call_gex + put_gex) / gross if gross else None,
            "call_wall_v2": float(call_by_strike.idxmax())
            if not call_by_strike.empty else None,
            "put_wall_v2": float(put_by_strike.idxmax())
            if not put_by_strike.empty else None,
        })
    aggregates = pd.DataFrame(rows).sort_values("ts")
    metadata = history.sort_values("ts").rename(columns={"ts": "source_gex_ts"})
    rebuilt = pd.merge_asof(
        aggregates, metadata, left_on="ts", right_on="source_gex_ts",
        direction="backward", tolerance=pd.Timedelta(seconds=120),
    )
    rebuilt = rebuilt.dropna(subset=["spot"])
    rebuilt["total_gex"] = rebuilt.pop("total_gex_v2")
    rebuilt["call_gex"] = rebuilt.pop("call_gex_v2")
    rebuilt["put_gex"] = rebuilt.pop("put_gex_v2")
    rebuilt["gross_gex"] = rebuilt.pop("gross_gex_v2")
    rebuilt["net_gex_ratio"] = rebuilt.pop("net_gex_ratio_v2")
    rebuilt["call_wall"] = rebuilt.pop("call_wall_v2")
    rebuilt["put_wall"] = rebuilt.pop("put_wall_v2")
    rebuilt["positive_gamma"] = rebuilt["total_gex"] > 0
    rebuilt["gex_method"] = "oi_position_v2"
    rebuilt["flip"] = np.nan
    rebuilt["gamma_flip_status"] = "unavailable_historical_metadata"
    rebuilt["gamma_flip_reliable"] = False
    return rebuilt, surface


def add_lagged_features(frame: pd.DataFrame, config: TurningPointConfig) -> pd.DataFrame:
    """Compute feature columns using current and lagged rows only."""
    result = frame.copy().sort_values("ts").reset_index(drop=True)
    close = pd.to_numeric(result["close"], errors="coerce")
    returns = close.pct_change()
    result["return_5m"] = close.pct_change(5)
    result["return_15m"] = close.pct_change(config.lookback_minutes)
    result["return_30m"] = close.pct_change(30)
    result["rv_5m"] = returns.rolling(5, min_periods=3).std() * np.sqrt(5)
    result["rv_15m"] = returns.rolling(15, min_periods=8).std() * np.sqrt(15)
    result["volatility_unit_15m"] = (
        returns.rolling(config.volatility_window, min_periods=15).std()
        * np.sqrt(config.horizon_minutes)
    ).clip(lower=config.minimum_volatility_pct)
    result["twap_30m"] = close.rolling(30, min_periods=15).mean()
    result["twap_distance_pct"] = close / result["twap_30m"] - 1.0
    path = close.diff().abs().rolling(15, min_periods=8).sum()
    result["trend_efficiency_15m"] = close.diff(15).abs() / path

    for column in (
        "total_gex", "volume_gamma", "gross_volume_gamma",
        "flip", "call_wall", "put_wall", "rr_25",
        "strike_total_volume", "strike_gross_volume_gamma",
    ):
        if column in result:
            values = pd.to_numeric(result[column], errors="coerce")
            result[f"{column}_change_5m"] = values - values.shift(5)
            result[f"{column}_change_15m"] = values - values.shift(15)

    price_scale = close * result["volatility_unit_15m"]
    for source, target in (
        ("flip", "dist_to_flip_vol"),
        ("call_wall", "dist_to_call_wall_vol"),
        ("put_wall", "dist_to_put_wall_vol"),
        ("max_pain", "dist_to_max_pain_vol"),
    ):
        if source in result:
            result[target] = _safe_ratio(close - pd.to_numeric(
                result[source], errors="coerce"
            ), price_scale)
    if "call_gex" in result and "put_gex" in result:
        call = pd.to_numeric(result["call_gex"], errors="coerce")
        put = pd.to_numeric(result["put_gex"], errors="coerce")
        result["call_put_gex_imbalance"] = _safe_ratio(call + put, call.abs() + put.abs())
    gross_source = None
    if "strike_total_abs_gex" in result:
        gross_source = result["strike_total_abs_gex"]
    elif "gross_gex" in result:
        gross_source = result["gross_gex"]
    gross = (
        pd.to_numeric(gross_source, errors="coerce")
        if gross_source is not None else None
    )
    if "total_gex" in result and gross is not None:
        result["net_gex_ratio"] = _safe_ratio(
            pd.to_numeric(result["total_gex"], errors="coerce"), gross
        )
    for minutes in (5, 15):
        gex_change = result.get(f"total_gex_change_{minutes}m")
        if gex_change is not None and gross is not None:
            result[f"gex_change_{minutes}m_gross_ratio"] = _safe_ratio(
                pd.to_numeric(gex_change, errors="coerce"), gross
            )
        volume_change = result.get(f"strike_gross_volume_gamma_change_{minutes}m")
        if volume_change is not None and gross is not None:
            result[f"volume_gamma_change_{minutes}m_gross_ratio"] = _safe_ratio(
                pd.to_numeric(volume_change, errors="coerce"), gross
            )
    return result


def add_outcome_labels(frame: pd.DataFrame, config: TurningPointConfig) -> pd.DataFrame:
    """Use future bars only for labels and realized outcome measurements."""
    result = frame.copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    high = pd.to_numeric(result.get("high", result["close"]), errors="coerce")
    low = pd.to_numeric(result.get("low", result["close"]), errors="coerce")
    future_up = pd.concat([
        high.shift(-offset) / close - 1.0
        for offset in range(1, config.horizon_minutes + 1)
    ], axis=1).max(axis=1)
    future_down = pd.concat([
        low.shift(-offset) / close - 1.0
        for offset in range(1, config.horizon_minutes + 1)
    ], axis=1).min(axis=1)
    scale = result["volatility_unit_15m"]
    prior = result["return_15m"]
    after_down = prior <= -config.impulse_sigma * scale
    after_up = prior >= config.impulse_sigma * scale
    up_reversal = (
        after_down
        & (future_up >= config.reversal_sigma * scale)
        & (future_down >= -config.adverse_sigma * scale)
    )
    down_continuation = (
        after_down
        & (future_down <= -config.continuation_sigma * scale)
        & (future_up <= config.adverse_sigma * scale)
    )
    down_reversal = (
        after_up
        & (future_down <= -config.reversal_sigma * scale)
        & (future_up <= config.adverse_sigma * scale)
    )
    up_continuation = (
        after_up
        & (future_up >= config.continuation_sigma * scale)
        & (future_down >= -config.adverse_sigma * scale)
    )
    result["setup_direction"] = np.select(
        [after_down, after_up], ["after_down", "after_up"], default="none"
    )
    result["outcome"] = np.select(
        [up_reversal, down_reversal, down_continuation, up_continuation],
        ["reversal_up", "reversal_down", "continuation_down", "continuation_up"],
        default="ambiguous",
    )
    result["future_mfe_up_pct"] = future_up
    result["future_mae_down_pct"] = future_down
    result["future_return_5m"] = close.shift(-5) / close - 1.0
    result["future_return_10m"] = close.shift(-10) / close - 1.0
    result["future_return_15m"] = close.shift(-15) / close - 1.0
    result["label_complete"] = close.shift(-config.horizon_minutes).notna()
    result["is_candidate"] = (after_down | after_up) & result["label_complete"]
    return result


def deduplicate_candidates(frame: pd.DataFrame, cooldown_minutes: int = 15) -> pd.DataFrame:
    """Keep the first observable candidate per setup direction during cooldown."""
    if frame.empty:
        return frame
    eligible = frame["is_candidate"].fillna(False)
    if "gex_quality_ok" in frame:
        eligible &= frame["gex_quality_ok"].fillna(False)
    candidates = frame[eligible].sort_values("ts")
    selected = []
    last_by_direction: dict[str, pd.Timestamp] = {}
    for index, row in candidates.iterrows():
        direction = str(row["setup_direction"])
        timestamp = pd.Timestamp(row["ts"])
        last = last_by_direction.get(direction)
        if last is not None and timestamp - last < pd.Timedelta(minutes=cooldown_minutes):
            continue
        selected.append(index)
        last_by_direction[direction] = timestamp
    return candidates.loc[selected].copy().reset_index(drop=True)


def prepare_day(
    bars: pd.DataFrame,
    gex: pd.DataFrame,
    strikes: pd.DataFrame | None,
    *,
    symbol: str,
    date_str: str,
    config: TurningPointConfig,
) -> pd.DataFrame:
    aligned = align_minute_gex(
        bars, gex, tolerance_seconds=config.gex_tolerance_seconds
    )
    if aligned.empty:
        return aligned
    aligned = aligned[
        (aligned["ts"].dt.time >= pd.Timestamp("09:30").time())
        & (aligned["ts"].dt.time <= pd.Timestamp("16:00").time())
    ].copy()
    if strikes is not None and not strikes.empty:
        strike_features = strike_snapshot_features(strikes, aligned)
        aligned = align_strike_features(
            aligned, strike_features, tolerance_seconds=config.strike_tolerance_seconds
        )
    aligned["symbol"] = symbol
    aligned["trading_date"] = date_str
    aligned["gex_quality_ok"] = aligned["total_gex"].notna()
    if "partial" in aligned:
        aligned["gex_quality_ok"] &= ~aligned["partial"].fillna(False).astype(bool)
    aligned = add_lagged_features(aligned, config)
    return add_outcome_labels(aligned, config)


def build_symbol_dataset(
    data_dir: Path | str,
    symbol: str,
    config: TurningPointConfig | None = None,
    *,
    gex_method: str = "stored",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Build full minute features and deduplicated candidate events."""
    config = config or TurningPointConfig()
    data_dir = Path(data_dir)
    minute_frames = []
    day_frames: dict[str, pd.DataFrame] = {}
    rejected: dict[str, str] = {}
    for bars_path in sorted(data_dir.glob(f"ohlc_{symbol}_*.parquet")):
        date_str = bars_path.stem.rsplit("_", 1)[-1]
        gex_path = data_dir / f"gex_{symbol}_{date_str}.parquet"
        strikes_path = data_dir / f"strikes_{symbol}_{date_str}.parquet"
        if not gex_path.exists():
            rejected[date_str] = "missing_gex"
            continue
        bars = pd.read_parquet(bars_path)
        if len(bars) < config.minimum_bars_per_day:
            rejected[date_str] = f"insufficient_bars:{len(bars)}"
            continue
        gex = pd.read_parquet(gex_path)
        strikes = pd.read_parquet(strikes_path) if strikes_path.exists() else None
        if gex_method == "oi_position_v2":
            if strikes is None:
                rejected[date_str] = "missing_strikes_for_oi_position_v2"
                continue
            gex, strikes = rebuild_oi_position_gex(gex, strikes)
            if gex.empty or strikes.empty:
                rejected[date_str] = "oi_position_v2_rebuild_failed"
                continue
        elif gex_method != "stored":
            raise ValueError(f"unsupported gex_method: {gex_method}")
        frame = prepare_day(
            bars, gex, strikes, symbol=symbol, date_str=date_str, config=config
        )
        coverage = float(frame["total_gex"].notna().mean()) if not frame.empty else 0.0
        if coverage < config.minimum_gex_coverage:
            rejected[date_str] = f"gex_coverage:{coverage:.3f}"
            continue
        frame["event_id"] = [f"{symbol}_{date_str}_{i:04d}" for i in range(len(frame))]
        minute_frames.append(frame)
        day_frames[date_str] = frame
    minutes = pd.concat(minute_frames, ignore_index=True) if minute_frames else pd.DataFrame()
    event_frames = []
    for date_str, frame in day_frames.items():
        events = deduplicate_candidates(frame, config.cooldown_minutes)
        events["event_id"] = [
            f"{symbol}_{date_str}_{pd.Timestamp(ts).strftime('%H%M')}"
            for ts in events["ts"]
        ]
        event_frames.append(events)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    summary = {
        "schema_version": 1,
        "symbol": symbol,
        "gex_method": gex_method,
        "config": config.__dict__,
        "usable_days": len(day_frames),
        "rejected_days": rejected,
        "minute_rows": len(minutes),
        "candidate_events": len(events),
        "turning_events": int(events["outcome"].isin([
            "reversal_up", "reversal_down"
        ]).sum()) if not events.empty else 0,
        "continuation_events": int(events["outcome"].isin([
            "continuation_up", "continuation_down"
        ]).sum()) if not events.empty else 0,
        "outcome_counts": events["outcome"].value_counts().to_dict()
        if not events.empty else {},
        "setup_counts": events["setup_direction"].value_counts().to_dict()
        if not events.empty else {},
    }
    return minutes, events, day_frames, summary


def _polyline(values: list[float], x0: float, y0: float, width: float,
              height: float) -> str:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return ""
    lo, hi = min(finite), max(finite)
    if hi == lo:
        hi = lo + 1.0
    points = []
    count = max(1, len(values) - 1)
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        x = x0 + width * index / count
        y = y0 + height * (hi - value) / (hi - lo)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_event_svg(event: pd.Series, day: pd.DataFrame, output: Path) -> Path:
    """Render a dependency-free event audit chart as SVG."""
    timestamp = pd.Timestamp(event["ts"])
    window = day[
        (day["ts"] >= timestamp - pd.Timedelta(minutes=30))
        & (day["ts"] <= timestamp + pd.Timedelta(minutes=30))
    ].copy()
    width, height = 960, 520
    left, plot_width = 70, 850
    price_top, price_height = 55, 285
    gex_top, gex_height = 385, 90
    close = pd.to_numeric(window["close"], errors="coerce").tolist()
    gex = pd.to_numeric(window["total_gex"], errors="coerce").tolist()
    price_points = _polyline(close, left, price_top, plot_width, price_height)
    gex_points = _polyline(gex, left, gex_top, plot_width, gex_height)
    event_index = int((window["ts"] <= timestamp).sum()) - 1
    event_x = left + plot_width * max(0, event_index) / max(1, len(window) - 1)
    price_values = [value for value in close if math.isfinite(value)]
    price_lo = min(price_values) if price_values else 0.0
    price_hi = max(price_values) if price_values else 1.0
    if price_hi == price_lo:
        price_hi += 1.0

    level_lines = []
    for column, color, label in (
        ("flip", "#f59e0b", "Flip"),
        ("call_wall", "#ef4444", "Call Wall"),
        ("put_wall", "#22c55e", "Put Wall"),
    ):
        value = _finite(event.get(column))
        if value is None or not (price_lo <= value <= price_hi):
            continue
        y = price_top + price_height * (price_hi - value) / (price_hi - price_lo)
        level_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
            f'y2="{y:.1f}" stroke="{color}" stroke-dasharray="7 5"/>'
            f'<text x="{left + 5}" y="{y - 4:.1f}" fill="{color}" '
            f'font-size="12">{label} {value:.2f}</text>'
        )
    title = html.escape(
        f"{event['event_id']}  {event['outcome']}  "
        f"prior15={event.get('return_15m', float('nan')):.2%}  "
        f"future15={event.get('future_return_15m', float('nan')):.2%}"
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#0f172a"/>\n'
        f'<text x="{left}" y="28" fill="#e2e8f0" font-size="17" '
        f'font-family="monospace">{title}</text>\n'
        f'<rect x="{left}" y="{price_top}" width="{plot_width}" '
        f'height="{price_height}" fill="#111827" stroke="#334155"/>\n'
        f'{"".join(level_lines)}\n'
        f'<polyline points="{price_points}" fill="none" stroke="#38bdf8" '
        'stroke-width="2.4"/>\n'
        f'<line x1="{event_x:.1f}" y1="{price_top}" x2="{event_x:.1f}" '
        f'y2="{gex_top + gex_height}" stroke="#f8fafc" stroke-width="1.5"/>\n'
        f'<text x="{left}" y="{price_top + 18}" fill="#94a3b8" '
        f'font-size="12">Price: {price_lo:.2f} – {price_hi:.2f}</text>\n'
        f'<rect x="{left}" y="{gex_top}" width="{plot_width}" '
        f'height="{gex_height}" fill="#111827" stroke="#334155"/>\n'
        f'<polyline points="{gex_points}" fill="none" stroke="#a78bfa" '
        'stroke-width="2"/>\n'
        f'<text x="{left}" y="{gex_top - 8}" fill="#a78bfa" '
        'font-size="12">Total GEX</text>\n'
        f'<text x="{left}" y="505" fill="#94a3b8" font-size="12">'
        '30 minutes before / after; white line = candidate timestamp</text>\n'
        '</svg>'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def select_audit_events(events: pd.DataFrame, count: int = 50,
                        seed: int = 20260721) -> pd.DataFrame:
    if events.empty or count <= 0:
        return events.iloc[0:0]
    pieces = []
    groups = list(events.groupby("outcome", sort=True))
    per_group = max(1, count // max(1, len(groups)))
    for _, group in groups:
        pieces.append(group.sample(min(per_group, len(group)), random_state=seed))
    selected = pd.concat(pieces).drop_duplicates("event_id")
    remaining = events[~events["event_id"].isin(selected["event_id"])]
    if len(selected) < count and not remaining.empty:
        selected = pd.concat([
            selected,
            remaining.sample(min(count - len(selected), len(remaining)), random_state=seed + 1),
        ])
    return selected.head(count).sort_values("ts").reset_index(drop=True)


def write_audit_gallery(
    selected: pd.DataFrame,
    day_frames: dict[str, pd.DataFrame],
    output_dir: Path | str,
    summary: dict,
) -> Path:
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figs" / f"turning_points_{summary['symbol']}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for old_chart in figure_dir.glob("*.svg"):
        old_chart.unlink()
    cards = []
    for _, event in selected.iterrows():
        date_str = str(event["trading_date"])
        filename = f"{event['event_id']}_{event['outcome']}.svg"
        path = render_event_svg(event, day_frames[date_str], figure_dir / filename)
        relative = path.relative_to(output_dir)
        cards.append(
            f'<section><h3>{html.escape(str(event["event_id"]))} — '
            f'{html.escape(str(event["outcome"]))}</h3>'
            f'<img loading="lazy" src="{relative.as_posix()}"/></section>'
        )
    report = output_dir / f"turning_point_audit_{summary['symbol']}.html"
    report.write_text(
        "<!doctype html><meta charset='utf-8'><title>Turning-point audit</title>"
        "<style>body{background:#020617;color:#e2e8f0;font-family:system-ui;"
        "max-width:1100px;margin:auto}section{margin:28px 0;padding:14px;"
        "background:#0f172a;border-radius:10px}img{width:100%}pre{white-space:pre-wrap}</style>"
        f"<h1>{summary['symbol']} intraday turning-point label audit</h1>"
        f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>"
        + "".join(cards),
        encoding="utf-8",
    )
    return report
