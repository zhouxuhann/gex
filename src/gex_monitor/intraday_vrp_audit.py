"""日内 VRP 采集与结算的日终质量审计。"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .time_utils import ET


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df or df.empty:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def build_vrp_daily_audit(*, symbol: str, date_str: str, schedule: list[str] | tuple[str, ...],
                          quotes: pd.DataFrame, observations: pd.DataFrame,
                          mtm: pd.DataFrame | None = None,
                          iron_fly_mtm: pd.DataFrame | None = None,
                          min_rth_bars: int = 389) -> dict:
    """生成一份可机器读取的 VRP 日终审计报告。"""
    expected_slots = list(schedule)
    actual_slots = (set(quotes.get("scheduled_time", pd.Series(dtype=str)).astype(str))
                    if not quotes.empty else set())
    settled_slots = (set(observations.get("scheduled_time", pd.Series(dtype=str)).astype(str))
                     if not observations.empty else set())
    status_counts = Counter(
        quotes.get("status", pd.Series(dtype=str)).fillna("missing_status").astype(str)
    )
    expected = len(expected_slots)
    observed = len(actual_slots.intersection(expected_slots))
    settled = len(settled_slots.intersection(actual_slots))
    ok_count = int(status_counts.get("ok", 0))
    rth_bars = 0
    if not observations.empty and "rth_bar_count" in observations:
        bars = pd.to_numeric(observations["rth_bar_count"], errors="coerce").dropna()
        rth_bars = int(bars.max()) if not bars.empty else 0

    coverage = observed / expected if expected else 1.0
    settled_coverage = settled / observed if observed else 0.0
    ok_ratio = ok_count / observed if observed else 0.0
    missing_slots = [slot for slot in expected_slots if slot not in actual_slots]
    unsettled_slots = sorted(actual_slots - settled_slots)
    problems = []
    feature_columns = [
        "straddle_gamma", "straddle_theta", "straddle_vega",
        "rv_15m", "rv_30m", "trend_efficiency_session",
        "dist_to_flip_im", "weekday", "opex_type", "event_flag",
    ]
    feature_coverage = {}
    for column in feature_columns:
        feature_coverage[column] = (
            float(quotes[column].notna().mean()) if column in quotes and not quotes.empty else 0.0
        )
    if missing_slots:
        problems.append(f"missing_slots={','.join(missing_slots)}")
    if unsettled_slots:
        problems.append(f"unsettled_slots={','.join(unsettled_slots)}")
    for name in ("missing_pair", "invalid_nbbo", "stale_quote", "zero_bid",
                 "not_true_0dte", "wide_spread"):
        if status_counts.get(name):
            problems.append(f"{name}={status_counts[name]}")
    if rth_bars < min_rth_bars:
        problems.append(f"rth_bars={rth_bars}<{min_rth_bars}")

    if coverage >= 0.95 and settled_coverage == 1.0 and rth_bars >= min_rth_bars:
        quality = "good" if ok_ratio >= 0.90 else "warning"
    elif coverage >= 0.80 and settled_coverage >= 0.80:
        quality = "warning"
    else:
        quality = "bad"

    return {
        "schema_version": 2,
        "symbol": symbol,
        "trading_date": date_str,
        "generated_at": datetime.now(ET),
        "quality": quality,
        "expected_slots": expected,
        "observed_slots": observed,
        "settled_slots": settled,
        "coverage": coverage,
        "settled_coverage": settled_coverage,
        "ok_quotes": ok_count,
        "ok_quote_ratio": ok_ratio,
        "status_counts": dict(sorted(status_counts.items())),
        "missing_slots": missing_slots,
        "unsettled_slots": unsettled_slots,
        "mean_delay_seconds": _mean(quotes, "delay_seconds"),
        "mean_quote_age_seconds": _mean(quotes, "quote_age_seconds"),
        "mean_combined_spread_ratio": _mean(quotes, "combined_spread_ratio"),
        "mean_execution_haircut_pct": _mean(quotes, "execution_haircut_pct"),
        "rth_bar_count": rth_bars,
        "ohlc_complete": rth_bars >= min_rth_bars,
        "feature_coverage": feature_coverage,
        "mtm_rows": 0 if mtm is None else len(mtm),
        "iron_fly_mtm_rows": 0 if iron_fly_mtm is None else len(iron_fly_mtm),
        "problems": problems,
    }


def write_vrp_daily_audit(report: dict, data_dir: Path | str) -> Path:
    """原子写入日终 JSON 报告。"""
    path = Path(data_dir) / (
        f"vrp_quality_{report['symbol']}_{report['trading_date']}.json"
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path
