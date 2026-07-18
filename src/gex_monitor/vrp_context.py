"""Deterministic context features for intraday VRP observations."""
from __future__ import annotations

import csv
from calendar import monthcalendar, FRIDAY
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .time_utils import ET


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


def path_features(bars, now: datetime, spot: float) -> dict:
    """Compute only features observable at entry from already-collected 1m bars."""
    empty = {
        "session_open": None, "session_return_pct": None,
        "twap_distance_pct": None, "rv_5m": None, "rv_15m": None,
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
    result = dict(empty)
    result.update({
        "session_open": session_open,
        "session_return_pct": (spot - session_open) / session_open if session_open else None,
        "twap_distance_pct": (spot - twap) / twap if twap else None,
        "entry_bar_count": len(df),
        "trend_efficiency_session": efficiency(close),
        "trend_efficiency_30m": efficiency(close.tail(31)),
    })
    for minutes in (5, 15, 30, 60):
        result[f"rv_{minutes}m"] = rv(minutes)
    for minutes in (15, 30, 60):
        result[f"range_{minutes}m_pct"] = window_range(minutes)
    return result
