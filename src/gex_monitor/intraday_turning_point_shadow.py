"""Forward-only shadow scorer for intraday turning-point candidates.

The scorer observes completed minute bars, records every impulse candidate and
backfills its realized path.  It has no broker or order interface by design.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import IntradayTurningPointShadowConfig
from .intraday_turning_points import (
    TurningPointConfig,
    add_lagged_features,
    align_minute_gex,
    align_strike_features,
    strike_snapshot_features,
)
from .storage import StorageManager
from .time_utils import ET


@dataclass(frozen=True)
class ShadowRule:
    rule_id: str
    direction: str
    target: str
    family: str
    feature: str
    lower_exclusive: float | None
    upper_inclusive: float | None
    validation_lift: float
    test_lift: float
    external_lift: float

    def matches(self, row: pd.Series) -> bool:
        value = _finite(row.get(self.feature))
        if value is None:
            return False
        if self.lower_exclusive is not None and value <= self.lower_exclusive:
            return False
        return self.upper_inclusive is None or value <= self.upper_inclusive


@dataclass(frozen=True)
class ShadowModel:
    model_id: str
    label_schema: str
    candidate: TurningPointConfig
    rules: tuple[ShadowRule, ...]


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_shadow_model(path: Path | str) -> ShadowModel:
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported turning-point shadow model schema")
    if payload.get("selection_policy") != "stable_cross_symbol_only":
        raise ValueError("shadow model must contain only cross-symbol candidates")
    candidate = TurningPointConfig(**payload["candidate"])
    rules = tuple(
        ShadowRule(
            rule_id=item["id"],
            direction=item["direction"],
            target=item["target"],
            family=item["family"],
            feature=item["feature"],
            lower_exclusive=_finite(item.get("lower_exclusive")),
            upper_inclusive=_finite(item.get("upper_inclusive")),
            validation_lift=float(item["validation_lift"]),
            test_lift=float(item["test_lift"]),
            external_lift=float(item["external_lift"]),
        )
        for item in payload["rules"]
    )
    if not rules:
        raise ValueError("turning-point shadow model contains no rules")
    return ShadowModel(
        model_id=payload["model_id"],
        label_schema=payload["label_schema"],
        candidate=candidate,
        rules=rules,
    )


def build_realtime_feature_frame(
    history: list[dict],
    bars: list[dict],
    strikes: list[dict],
    config: TurningPointConfig,
) -> pd.DataFrame:
    """Recreate the offline feature path without adding any future labels."""
    if not history or not bars:
        return pd.DataFrame()
    aligned = align_minute_gex(
        pd.DataFrame(bars),
        pd.DataFrame(history),
        tolerance_seconds=config.gex_tolerance_seconds,
    )
    if aligned.empty:
        return aligned
    aligned = aligned[
        (aligned["ts"].dt.time >= pd.Timestamp("09:30").time())
        & (aligned["ts"].dt.time <= pd.Timestamp("16:00").time())
    ].copy()
    if strikes:
        strike_features = strike_snapshot_features(pd.DataFrame(strikes), aligned)
        aligned = align_strike_features(
            aligned,
            strike_features,
            tolerance_seconds=config.strike_tolerance_seconds,
        )
    aligned["gex_quality_ok"] = aligned["total_gex"].notna()
    if "partial" in aligned:
        aligned["gex_quality_ok"] &= ~aligned["partial"].fillna(False).astype(bool)
    result = add_lagged_features(aligned, config)
    total = pd.to_numeric(result.get("total_gex"), errors="coerce").abs()
    scale = total.replace(0, np.nan)
    for minutes in (5, 15):
        change = pd.to_numeric(
            result.get(f"total_gex_change_{minutes}m"), errors="coerce"
        )
        result[f"gex_change_{minutes}m_ratio"] = change / scale
    return result


def candidate_direction(row: pd.Series, config: TurningPointConfig) -> str | None:
    prior = _finite(row.get("return_15m"))
    scale = _finite(row.get("volatility_unit_15m"))
    if prior is None or scale is None or not bool(row.get("gex_quality_ok", False)):
        return None
    if prior <= -config.impulse_sigma * scale:
        return "after_down"
    if prior >= config.impulse_sigma * scale:
        return "after_up"
    return None


def score_candidate(
    row: pd.Series,
    direction: str,
    model: ShadowModel,
) -> dict:
    applicable = [rule for rule in model.rules if rule.direction == direction]
    matched = [rule for rule in applicable if rule.matches(row)]
    target_families = {
        target: {rule.family for rule in applicable if rule.target == target}
        for target in ("reversal", "continuation")
    }
    matched_families = {
        target: {rule.family for rule in matched if rule.target == target}
        for target in ("reversal", "continuation")
    }

    def points(target: str) -> int:
        return len(matched_families[target])

    def maximum(target: str) -> int:
        return len(target_families[target])

    reversal_points = points("reversal")
    continuation_points = points("continuation")
    reversal_max = maximum("reversal")
    continuation_max = maximum("continuation")
    reversal_score = 100.0 * reversal_points / reversal_max if reversal_max else 0.0
    continuation_score = (
        100.0 * continuation_points / continuation_max if continuation_max else 0.0
    )
    if reversal_points and continuation_points:
        watch_level, bias = "CONFLICT", "conflict"
    elif max(reversal_points, continuation_points) >= 2:
        watch_level = "STRONG_WATCH"
        bias = "reversal" if reversal_points else "continuation"
    elif reversal_points or continuation_points:
        watch_level = "WATCH"
        bias = "reversal" if reversal_points else "continuation"
    else:
        watch_level, bias = "IGNORE", "neutral"
    return {
        "reversal_support_points": reversal_points,
        "reversal_support_max": reversal_max,
        "reversal_support_score": reversal_score,
        "continuation_risk_points": continuation_points,
        "continuation_risk_max": continuation_max,
        "continuation_risk_score": continuation_score,
        "watch_level": watch_level,
        "score_bias": bias,
        "matched_rule_ids": json.dumps(
            [rule.rule_id for rule in matched], ensure_ascii=False
        ),
        "matched_rule_count": len(matched),
    }


def classify_realized_outcome(
    direction: str,
    future_up: float,
    future_down: float,
    scale: float,
    config: TurningPointConfig,
) -> str:
    if direction == "after_down":
        if (
            future_up >= config.reversal_sigma * scale
            and future_down >= -config.adverse_sigma * scale
        ):
            return "reversal_up"
        if (
            future_down <= -config.continuation_sigma * scale
            and future_up <= config.adverse_sigma * scale
        ):
            return "continuation_down"
    elif direction == "after_up":
        if (
            future_down <= -config.reversal_sigma * scale
            and future_up <= config.adverse_sigma * scale
        ):
            return "reversal_down"
        if (
            future_up >= config.continuation_sigma * scale
            and future_down >= -config.adverse_sigma * scale
        ):
            return "continuation_up"
    return "ambiguous"


class IntradayTurningPointShadow:
    """Observe, score and settle minute-level candidates without execution."""

    def __init__(
        self,
        symbol: str,
        storage: StorageManager,
        config: IntradayTurningPointShadowConfig,
    ):
        if config.observation_only is not True:
            raise ValueError("turning-point shadow scorer must be observation-only")
        self.symbol = symbol
        self.storage = storage
        self.config = config
        loaded_model = load_shadow_model(config.model_path)
        self.model = replace(
            loaded_model,
            candidate=replace(
                loaded_model.candidate,
                gex_tolerance_seconds=config.gex_tolerance_seconds,
                strike_tolerance_seconds=config.strike_tolerance_seconds,
            ),
        )
        self._last_poll_minute: pd.Timestamp | None = None
        self._processed_bar_ts: pd.Timestamp | None = None
        self._date: str | None = None
        self._events: dict[str, dict] = {}
        self._last_candidate_by_direction: dict[str, pd.Timestamp] = {}

    def on_update(
        self,
        *,
        now: datetime,
        input_provider: Callable[[], tuple[list[dict], list[dict], list[dict]]],
    ) -> dict | None:
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize(ET)
        else:
            now_ts = now_ts.tz_convert(ET)
        minute = now_ts.floor("min")
        if self._last_poll_minute == minute:
            return None
        self._last_poll_minute = minute
        date_str = minute.strftime("%Y%m%d")
        self._ensure_date(date_str)
        history, bars, strikes = input_provider()
        frame = build_realtime_feature_frame(history, bars, strikes, self.model.candidate)
        if frame.empty:
            return None
        completed = frame[frame["ts"] < minute].sort_values("ts")
        if completed.empty:
            return None
        self._backfill(completed, date_str)
        latest = completed.iloc[-1]
        bar_ts = pd.Timestamp(latest["ts"])
        if self._processed_bar_ts is not None and bar_ts <= self._processed_bar_ts:
            return None
        self._processed_bar_ts = bar_ts
        close_time = bar_ts + pd.Timedelta(minutes=1)
        lag_seconds = (now_ts - close_time).total_seconds()
        if lag_seconds < 0 or lag_seconds > self.config.max_candidate_lag_seconds:
            return None
        latest_clock = datetime.strptime(
            self.config.latest_candidate_time_et, "%H:%M"
        ).time()
        if bar_ts.time() > latest_clock:
            return None
        direction = candidate_direction(latest, self.model.candidate)
        if direction is None:
            return None
        last = self._last_candidate_by_direction.get(direction)
        if last is not None and bar_ts - last < pd.Timedelta(
            minutes=self.config.cooldown_minutes
        ):
            return None
        event_id = f"{self.symbol}_{date_str}_{bar_ts.strftime('%H%M')}"
        if event_id in self._events:
            return None
        score = score_candidate(latest, direction, self.model)
        row = self._event_row(latest, now_ts, date_str, event_id, direction, score)
        self._events[event_id] = row
        self._last_candidate_by_direction[direction] = bar_ts
        self.storage.persist_turning_point_shadow(self.symbol, date_str, row)
        return dict(row)

    def _ensure_date(self, date_str: str) -> None:
        if self._date == date_str:
            return
        self._date = date_str
        self._events = {}
        self._last_candidate_by_direction = {}
        existing = self.storage.load_turning_point_shadow(self.symbol, date_str)
        for record in existing.to_dict("records"):
            event_id = str(record["event_id"])
            self._events[event_id] = record
            direction = str(record.get("setup_direction"))
            timestamp = pd.Timestamp(record["ts"])
            previous = self._last_candidate_by_direction.get(direction)
            if previous is None or timestamp > previous:
                self._last_candidate_by_direction[direction] = timestamp

    def _event_row(
        self,
        feature: pd.Series,
        observed_at: pd.Timestamp,
        date_str: str,
        event_id: str,
        direction: str,
        score: dict,
    ) -> dict:
        columns = (
            "close",
            "return_5m",
            "return_15m",
            "rv_15m",
            "volatility_unit_15m",
            "trend_efficiency_15m",
            "twap_distance_pct",
            "total_gex",
            "total_gex_change_5m",
            "total_gex_change_15m",
            "gex_change_5m_ratio",
            "gex_change_15m_ratio",
            "flip",
            "dist_to_flip_vol",
            "call_wall",
            "put_wall",
            "dist_to_call_wall_vol",
            "dist_to_put_wall_vol",
            "call_put_gex_imbalance",
            "atm_iv_pct",
            "rr_25",
            "rr_25_change_5m",
            "rr_25_change_15m",
            "strike_abs_gex_concentration_50bps",
            "strike_abs_gex_imbalance_50bps",
            "gex_age_seconds",
            "strike_age_seconds",
        )
        row = {
            "schema_version": 1,
            "model_id": self.model.model_id,
            "label_schema": self.model.label_schema,
            "symbol": self.symbol,
            "trading_date": date_str,
            "event_id": event_id,
            "ts": feature["ts"],
            "observed_at": observed_at,
            "setup_direction": direction,
            "evaluation_status": "pending",
            "realized_outcome": None,
            "future_return_5m": None,
            "future_return_10m": None,
            "future_return_15m": None,
            "future_mfe_up_pct": None,
            "future_mae_down_pct": None,
            "observation_only": True,
            **score,
        }
        row.update({column: _finite(feature.get(column)) for column in columns})
        return row

    def _backfill(self, frame: pd.DataFrame, date_str: str) -> None:
        if not self._events:
            return
        latest_ts = pd.Timestamp(frame["ts"].max())
        for event_id, event in list(self._events.items()):
            event_ts = pd.Timestamp(event["ts"])
            changed = False
            for minutes in (5, 10, 15):
                column = f"future_return_{minutes}m"
                if _finite(event.get(column)) is not None:
                    continue
                target = event_ts + pd.Timedelta(minutes=minutes)
                match = frame[frame["ts"].eq(target)]
                if not match.empty:
                    event[column] = float(match.iloc[-1]["close"] / event["close"] - 1.0)
                    changed = True
            horizon = event_ts + pd.Timedelta(minutes=self.model.candidate.horizon_minutes)
            if event.get("evaluation_status") != "completed" and latest_ts >= horizon:
                future = frame[(frame["ts"] > event_ts) & (frame["ts"] <= horizon)]
                if len(future) >= self.model.candidate.horizon_minutes:
                    future_up = float(future["high"].max() / event["close"] - 1.0)
                    future_down = float(future["low"].min() / event["close"] - 1.0)
                    event["future_mfe_up_pct"] = future_up
                    event["future_mae_down_pct"] = future_down
                    event["realized_outcome"] = classify_realized_outcome(
                        str(event["setup_direction"]),
                        future_up,
                        future_down,
                        float(event["volatility_unit_15m"]),
                        self.model.candidate,
                    )
                    event["evaluation_status"] = "completed"
                    changed = True
            if changed:
                self.storage.persist_turning_point_shadow(self.symbol, date_str, event)
