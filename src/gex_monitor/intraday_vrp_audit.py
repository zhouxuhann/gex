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


def _execution_funnel(frame: pd.DataFrame | None) -> dict:
    if frame is None or frame.empty:
        return {"eligible": 0, "submitted": 0, "filled": 0, "cancelled": 0,
                "skipped": 0, "blocked": 0, "submit_errors": 0,
                "fill_rate": None, "mean_fill_delay_seconds": None,
                "mean_credit_slippage": None}
    statuses = frame.get("status", pd.Series("missing", index=frame.index)).astype(str)
    skipped = statuses.str.startswith("SKIPPED_")
    blocked = statuses.eq("BLOCKED_SAFETY")
    submit_errors = statuses.eq("SUBMIT_ERROR")
    non_submitted = skipped | blocked | submit_errors | statuses.eq("INTENT_RECORDED")
    submitted = ~non_submitted
    filled = statuses.isin(["Filled", "EXPIRED"])
    cancelled = statuses.isin(["Cancelled", "ApiCancelled", "Inactive"])
    delays = pd.Series(dtype=float)
    if "submitted_at" in frame and "filled_at" in frame:
        submitted_at = pd.to_datetime(frame["submitted_at"], errors="coerce", utc=True)
        filled_at = pd.to_datetime(frame["filled_at"], errors="coerce", utc=True)
        delays = (filled_at - submitted_at).dt.total_seconds().dropna()
    slippage = pd.to_numeric(
        frame.get("credit_slippage", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    submitted_count = int(submitted.sum())
    filled_count = int(filled.sum())
    return {
        "eligible": len(frame), "submitted": submitted_count,
        "filled": filled_count, "cancelled": int(cancelled.sum()),
        "skipped": int(skipped.sum()), "blocked": int(blocked.sum()),
        "submit_errors": int(submit_errors.sum()),
        "fill_rate": filled_count / submitted_count if submitted_count else None,
        "mean_fill_delay_seconds": float(delays.mean()) if not delays.empty else None,
        "mean_credit_slippage": float(slippage.mean()) if not slippage.empty else None,
    }


def build_vrp_daily_audit(*, symbol: str, date_str: str, schedule: list[str] | tuple[str, ...],
                          quotes: pd.DataFrame, observations: pd.DataFrame,
                          mtm: pd.DataFrame | None = None,
                          iron_fly_mtm: pd.DataFrame | None = None,
                          paper_orders: pd.DataFrame | None = None,
                          paper_mtm: pd.DataFrame | None = None,
                          paper_straddle_orders: pd.DataFrame | None = None,
                          paper_straddle_mtm: pd.DataFrame | None = None,
                          minute_nodes: pd.DataFrame | None = None,
                          cone_nodes: pd.DataFrame | None = None,
                          expected_minute_nodes: int = 0,
                          expected_cone_nodes: int = 0,
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
        "gap_pct", "session_vwap", "vix", "vix_ma20_ratio",
        "vix1d", "minutes_to_close", "tau_session", "tau_calendar_years",
        "put_25_iv", "call_25_iv", "iv_1dte", "iv_2dte", "iv_5dte",
        "surface_term_spread_iv", "butterfly_25",
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
        core_quality = "good" if ok_ratio >= 0.90 else "warning"
    elif coverage >= 0.80 and settled_coverage >= 0.80:
        core_quality = "warning"
    else:
        core_quality = "bad"
    critical_context = ("vix", "vix_ma20_ratio", "surface_term_spread_iv")
    critical_values = [feature_coverage[name] for name in critical_context]
    if all(value >= 0.80 for value in critical_values):
        context_quality = "good"
    elif any(value > 0 for value in critical_values):
        context_quality = "warning"
    else:
        context_quality = "bad"
    quality = core_quality
    if core_quality == "good" and context_quality != "good":
        quality = "warning"
        missing = [name for name in critical_context if feature_coverage[name] < 0.80]
        problems.append(f"critical_context_incomplete={','.join(missing)}")

    fly_funnel = _execution_funnel(paper_orders)
    straddle_funnel = _execution_funnel(paper_straddle_orders)
    combined_frames = [frame.assign(strategy=name) for frame, name in (
        (paper_orders, "iron_fly"), (paper_straddle_orders, "short_straddle")
    ) if frame is not None and not frame.empty]
    combined_funnel = _execution_funnel(
        pd.concat(combined_frames, ignore_index=True) if combined_frames else None
    )

    return {
        "schema_version": 3,
        "symbol": symbol,
        "trading_date": date_str,
        "generated_at": datetime.now(ET),
        "quality": quality,
        "core_quality": core_quality,
        "context_quality": context_quality,
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
        "mtm_status_counts": {} if mtm is None or "status" not in mtm else
        dict(sorted(Counter(mtm["status"].fillna("missing_status").astype(str)).items())),
        "iron_fly_mtm_status_counts": {}
        if iron_fly_mtm is None or "status" not in iron_fly_mtm else
        dict(sorted(Counter(
            iron_fly_mtm["status"].fillna("missing_status").astype(str)
        ).items())),
        "paper_order_rows": 0 if paper_orders is None else len(paper_orders),
        "paper_order_status_counts": {}
        if paper_orders is None or "status" not in paper_orders else
        dict(sorted(Counter(
            paper_orders["status"].fillna("missing_status").astype(str)
        ).items())),
        "paper_mtm_rows": 0 if paper_mtm is None else len(paper_mtm),
        "paper_straddle_order_rows": (
            0 if paper_straddle_orders is None else len(paper_straddle_orders)
        ),
        "paper_straddle_order_status_counts": {}
        if paper_straddle_orders is None or "status" not in paper_straddle_orders else
        dict(sorted(Counter(
            paper_straddle_orders["status"].fillna("missing_status").astype(str)
        ).items())),
        "paper_straddle_mtm_rows": (
            0 if paper_straddle_mtm is None else len(paper_straddle_mtm)
        ),
        "paper_execution_funnel": {
            "combined": combined_funnel,
            "iron_fly": fly_funnel,
            "short_straddle": straddle_funnel,
        },
        "minute_node_rows": 0 if minute_nodes is None else len(minute_nodes),
        "minute_node_coverage": (
            len(minute_nodes) / expected_minute_nodes
            if minute_nodes is not None and expected_minute_nodes else None
        ),
        "cone_node_rows": 0 if cone_nodes is None else len(cone_nodes),
        "cone_node_coverage": (
            len(cone_nodes) / expected_cone_nodes
            if cone_nodes is not None and expected_cone_nodes else None
        ),
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
