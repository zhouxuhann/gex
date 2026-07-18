"""Deterministic context features for intraday VRP observations."""
from __future__ import annotations

import csv
import threading
import time
from calendar import monthcalendar, FRIDAY
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .time_utils import ET


_VIX_LOCK = threading.Lock()
_VIX_LIVE_CACHE: dict = {}
_VIX_HISTORY_CACHE: dict = {}


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def opex_context(now: datetime) -> dict:
    """Return weekday/monthly/quarterly expiry labels without market data calls."""
    now = now.astimezone(ET)
    day = now.date()
    fridays = [week[FRIDAY] for week in monthcalendar(day.year, day.month)
               if week[FRIDAY]]
    third_friday = fridays[2]
    is_monthly = day.weekday() == FRIDAY and day.day == third_friday
    is_quarterly = is_monthly and day.month in {3, 6, 9, 12}
    return {
        "weekday": day.strftime("%A"),
        "weekday_number": day.weekday(),
        "is_friday": day.weekday() == FRIDAY,
        "opex_type": ("quarterly" if is_quarterly else
                      "monthly" if is_monthly else
                      "weekly" if day.weekday() == FRIDAY else "none"),
    }


class VRPEventCalendar:
    """Small, auditable ET event calendar; a data-dir CSV can override package data."""

    def __init__(self, data_dir: Path | str):
        override = Path(data_dir) / "vrp_events.csv"
        packaged = Path(__file__).with_name("vrp_events_2026.csv")
        self.path = override if override.exists() else packaged
        self.events = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    ts = datetime.fromisoformat(row["event_ts_et"]).replace(tzinfo=ET)
                except (KeyError, ValueError):
                    continue
                rows.append({"ts": ts, "type": row.get("event_type", "unknown"),
                             "source": row.get("source", "local_calendar")})
        return sorted(rows, key=lambda item: item["ts"])

    def context(self, now: datetime) -> dict:
        now = now.astimezone(ET)
        same_day = [event for event in self.events if event["ts"].date() == now.date()]
        if not same_day:
            return {"event_flag": "none", "event_phase": "none",
                    "minutes_to_event": None, "event_ts_et": None,
                    "event_source": None}
        event = min(same_day, key=lambda item: abs((item["ts"] - now).total_seconds()))
        minutes = (event["ts"] - now).total_seconds() / 60.0
        return {"event_flag": event["type"],
                "event_phase": "pre" if minutes > 0 else "post",
                "minutes_to_event": minutes, "event_ts_et": event["ts"],
                "event_source": event["source"]}


def vix_context(ib, now: datetime, cache_seconds: int = 300) -> dict:
    """Fetch a throttled VIX snapshot and daily context from IB only.

    The module-level cache is shared by QQQ/SPY workers, so nine VRP slots do
    not turn into duplicate VIX subscriptions or repeated daily-history calls.
    """
    empty = {"vix": None, "vix_previous_close": None, "vix_change_pct": None,
             "vix_ma20": None, "vix_ma20_ratio": None,
             "vix_20d_percentile": None, "vix_asof": None,
             "vix_source": "unavailable"}
    if ib is None or not ib.isConnected():
        return empty
    now = now.astimezone(ET)
    date_str = now.strftime("%Y%m%d")
    with _VIX_LOCK:
        cached = _VIX_LIVE_CACHE.get("value")
        if cached and time.monotonic() - cached[0] <= max(30, cache_seconds):
            return dict(cached[1])
        try:
            from ib_insync import Index

            contract = Index("VIX", "CBOE", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                return dict(cached[1]) if cached else empty
            ticker_rows = ib.reqTickers(qualified[0])
            ticker = ticker_rows[0] if ticker_rows else None
            value = _finite(ticker.marketPrice()) if ticker is not None else None

            history = _VIX_HISTORY_CACHE.get(date_str)
            if history is None:
                bars = ib.reqHistoricalData(
                    qualified[0], endDateTime="", durationStr="40 D",
                    barSizeSetting="1 day", whatToShow="TRADES", useRTH=True,
                    formatDate=2, keepUpToDate=False, chartOptions=[],
                )
                history = []
                for bar in bars or []:
                    close = _finite(getattr(bar, "close", None))
                    try:
                        bar_date = pd.Timestamp(getattr(bar, "date")).date()
                    except (TypeError, ValueError):
                        continue
                    if close is not None and bar_date < now.date():
                        history.append((bar_date, close))
                _VIX_HISTORY_CACHE.clear()
                _VIX_HISTORY_CACHE[date_str] = history
            closes = [item[1] for item in history]
            previous = closes[-1] if closes else None
            history20 = closes[-20:]
            ma20 = float(np.mean(history20)) if history20 else None
            percentile = None
            if value is not None and history20:
                percentile = float(np.mean(np.asarray(history20) <= value))
            result = {
                "vix": value,
                "vix_previous_close": previous,
                "vix_change_pct": (value - previous) / previous
                if value is not None and previous else None,
                "vix_ma20": ma20,
                "vix_ma20_ratio": value / ma20 if value is not None and ma20 else None,
                "vix_20d_percentile": percentile,
                "vix_asof": now,
                "vix_source": "ib_snapshot+daily_history",
            }
            _VIX_LIVE_CACHE["value"] = (time.monotonic(), result)
            return dict(result)
        except Exception:
            return dict(cached[1]) if cached else empty


def path_features(bars, now: datetime, spot: float,
                  previous_close: float | None = None) -> dict:
    """Compute only features observable at entry from already-collected 1m bars."""
    empty = {
        "session_open": None, "session_return_pct": None,
        "twap_distance_pct": None, "session_vwap": None,
        "vwap_distance_pct": None, "vwap_source": "unavailable",
        "previous_close": previous_close, "gap_pct": None,
        "gap_direction": "unknown", "gap_filled_before_entry": None,
        "minutes_to_gap_fill": None,
        "rv_5m": None, "rv_15m": None,
        "rv_30m": None, "rv_60m": None, "range_15m_pct": None,
        "range_30m_pct": None, "range_60m_pct": None,
        "trend_efficiency_session": None, "trend_efficiency_30m": None,
        "entry_bar_count": 0,
    }
    if bars is None:
        return empty
    df = pd.DataFrame(bars).copy()
    if df.empty or "ts" not in df or "close" not in df:
        return empty
    ts = pd.to_datetime(df["ts"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ET)
    else:
        ts = ts.dt.tz_convert(ET)
    df["ts"] = ts
    now = now.astimezone(ET)
    df = df[(df["ts"].dt.date == now.date()) &
            (df["ts"] <= pd.Timestamp(now).floor("min"))]
    df = df[df["ts"].dt.time >= datetime.strptime("09:30", "%H:%M").time()]
    df = df.dropna(subset=["close"]).drop_duplicates("ts", keep="last").sort_values("ts")
    if df.empty:
        return empty
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        return empty
    session_open = _finite(df.iloc[0].get("open")) or float(close.iloc[0])

    def rv(minutes: int) -> float | None:
        values = close.tail(minutes + 1)
        if len(values) < 2:
            return None
        return float(np.sqrt(np.square(np.log(values).diff().dropna()).sum()))

    def window_range(minutes: int) -> float | None:
        window = df.tail(minutes)
        highs = pd.to_numeric(window.get("high", window["close"]), errors="coerce")
        lows = pd.to_numeric(window.get("low", window["close"]), errors="coerce")
        return float((highs.max() - lows.min()) / spot) if spot > 0 else None

    def efficiency(values: pd.Series) -> float | None:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) < 2:
            return None
        distance = abs(float(values.iloc[-1] - values.iloc[0]))
        path = float(values.diff().abs().sum())
        return distance / path if path > 0 else 0.0

    twap = float(close.mean())
    vwap = None
    vwap_source = "unavailable"
    if "volume" in df:
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        if volume.sum() > 0:
            average = pd.to_numeric(df.get("average"), errors="coerce") \
                if "average" in df else pd.Series(index=df.index, dtype=float)
            if average.notna().any():
                price = average.fillna(pd.to_numeric(df["close"], errors="coerce"))
                vwap_source = "ib_bar_average_volume"
            else:
                high = pd.to_numeric(df.get("high", df["close"]), errors="coerce")
                low = pd.to_numeric(df.get("low", df["close"]), errors="coerce")
                price = (high + low + pd.to_numeric(df["close"], errors="coerce")) / 3
                vwap_source = "typical_price_volume"
            vwap = float((price * volume).sum() / volume.sum())

    gap_pct = None
    gap_direction = "unknown"
    gap_filled = None
    minutes_to_fill = None
    if previous_close is not None and previous_close > 0 and session_open is not None:
        gap_pct = (session_open - previous_close) / previous_close
        gap_direction = "up" if gap_pct > 0 else "down" if gap_pct < 0 else "flat"
        if gap_direction == "flat":
            gap_filled, minutes_to_fill = True, 0
        else:
            highs = pd.to_numeric(df.get("high", df["close"]), errors="coerce")
            lows = pd.to_numeric(df.get("low", df["close"]), errors="coerce")
            hit = lows <= previous_close if gap_direction == "up" else highs >= previous_close
            gap_filled = bool(hit.any())
            if gap_filled:
                first_ts = df.loc[hit, "ts"].iloc[0]
                minutes_to_fill = int((first_ts - df.iloc[0]["ts"]).total_seconds() // 60)
    result = dict(empty)
    result.update({
        "session_open": session_open,
        "session_return_pct": (spot - session_open) / session_open if session_open else None,
        "twap_distance_pct": (spot - twap) / twap if twap else None,
        "session_vwap": vwap,
        "vwap_distance_pct": (spot - vwap) / vwap if vwap else None,
        "vwap_source": vwap_source,
        "previous_close": previous_close,
        "gap_pct": gap_pct,
        "gap_direction": gap_direction,
        "gap_filled_before_entry": gap_filled,
        "minutes_to_gap_fill": minutes_to_fill,
        "entry_bar_count": len(df),
        "trend_efficiency_session": efficiency(close),
        "trend_efficiency_30m": efficiency(close.tail(31)),
    })
    for minutes in (5, 15, 30, 60):
        result[f"rv_{minutes}m"] = rv(minutes)
    for minutes in (15, 30, 60):
        result[f"range_{minutes}m_pct"] = window_range(minutes)
    return result
