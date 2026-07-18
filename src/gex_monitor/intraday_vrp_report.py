"""Cross-day VRP statistics with date-cluster bootstrap and safe sizing gates."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import _atomic_write_parquet
from .time_utils import ET


def cluster_bootstrap_mean(values: pd.Series, dates: pd.Series, *,
                           samples: int = 1000, seed: int = 1729) -> tuple[float, float]:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"),
                          "date": dates.astype(str)}).dropna()
    unique = frame["date"].unique()
    if len(unique) < 2:
        return (float("nan"), float("nan"))
    clusters = {date: frame.loc[frame["date"] == date, "value"].to_numpy()
                for date in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        means[index] = np.concatenate([clusters[date] for date in selected]).mean()
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def _sample_status(days: int) -> str:
    if days < 20:
        return "raw_only"
    if days < 60:
        return "exploratory"
    if days < 90:
        return "comparison_ready"
    return "sizing_eligible"


def _group_rows(frame: pd.DataFrame, strategy: str, *, bounded: bool) -> list[dict]:
    if frame.empty:
        return []
    dimensions = [("overall", pd.Series("all", index=frame.index))]
    for column in ("scheduled_time", "target_wing_width", "positive_gamma",
                   "event_flag", "opex_type"):
        if column in frame:
            dimensions.append((column, frame[column].fillna("missing").astype(str)))
    if "target_wing_width" in frame and "scheduled_time" in frame:
        dimensions.append((
            "entry_structure",
            frame["scheduled_time"].astype(str) + "|wing=" +
            frame["target_wing_width"].astype(str),
        ))
    if "vix_ma20_ratio" in frame:
        ratio = pd.to_numeric(frame["vix_ma20_ratio"], errors="coerce")
        dimensions.append(("vix_ma20_regime", pd.cut(
            ratio, [-np.inf, 0.9, 1.1, np.inf],
            labels=["below_0.9", "0.9_to_1.1", "above_1.1"]
        ).astype(str)))

    rows = []
    for dimension, labels in dimensions:
        for label in labels.dropna().unique():
            group = frame[labels == label].copy()
            pnl = pd.to_numeric(group["pnl_dollars"], errors="coerce").dropna()
            group = group.loc[pnl.index]
            if pnl.empty:
                continue
            days = group["trading_date"].astype(str).nunique()
            ci_low, ci_high = cluster_bootstrap_mean(
                pnl, group["trading_date"], samples=1000
            )
            losses = -pnl
            row = {
                "strategy": strategy, "dimension": dimension, "bucket": str(label),
                "rows": len(group), "trading_days": days,
                "sample_status": _sample_status(days),
                "mean_pnl_dollars": float(pnl.mean()),
                "median_pnl_dollars": float(pnl.median()),
                "win_rate": float((pnl > 0).mean()),
                "p95_loss_dollars": max(0.0, float(losses.quantile(0.95))),
                "p99_loss_dollars": max(0.0, float(losses.quantile(0.99))),
                "mean_ci95_low": ci_low, "mean_ci95_high": ci_high,
                "kelly_fraction_raw": None, "kelly_fraction_quarter": None,
                "p99_risk_cap_fraction": None, "recommended_fraction": None,
                "sizing_reason": "unbounded_strategy" if not bounded else "insufficient_days",
            }
            if bounded and "return_on_max_risk" in group:
                returns = pd.to_numeric(group["return_on_max_risk"], errors="coerce").dropna()
                if len(returns) >= 2:
                    mean, variance = float(returns.mean()), float(returns.var(ddof=1))
                    raw = max(0.0, mean / variance) if variance > 0 else 0.0
                    quarter = min(raw / 4.0, 1.0)
                    p99_loss_return = max(0.0, float((-returns).quantile(0.99)))
                    risk_cap = min(1.0, 0.02 / p99_loss_return) \
                        if p99_loss_return > 0 else 1.0
                    row.update({"kelly_fraction_raw": raw,
                                "kelly_fraction_quarter": quarter,
                                "p99_risk_cap_fraction": risk_cap})
                    if days >= 90 and mean > 0:
                        row["recommended_fraction"] = min(quarter, risk_cap, 0.25)
                        row["sizing_reason"] = "eligible_1q_kelly_p99_capped"
                    elif mean <= 0:
                        row["sizing_reason"] = "nonpositive_net_edge"
            rows.append(row)
    return rows


def _load_many(data_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob(pattern)):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_vrp_report(data_dir: Path | str, symbol: str) -> pd.DataFrame:
    """Regenerate an idempotent all-history report for one symbol."""
    data_dir = Path(data_dir)
    straddles = _load_many(data_dir, f"vrp_observations_{symbol}_*.parquet")
    flies = _load_many(data_dir, f"vrp_iron_fly_observations_{symbol}_*.parquet")
    paper = _load_many(data_dir, f"vrp_paper_orders_{symbol}_*.parquet")
    rows = []
    if not straddles.empty and "pnl_executable" in straddles:
        if "status" in straddles:
            straddles = straddles[straddles["status"] == "ok"].copy()
        straddles["pnl_dollars"] = pd.to_numeric(
            straddles["pnl_executable"], errors="coerce"
        ) * 100
        rows.extend(_group_rows(straddles, "short_straddle", bounded=False))
    if not flies.empty and "iron_fly_pnl_dollars" in flies:
        if "status" in flies:
            flies = flies[flies["status"] == "ok"].copy()
        flies["pnl_dollars"] = pd.to_numeric(
            flies["iron_fly_pnl_dollars"], errors="coerce"
        )
        rows.extend(_group_rows(flies, "iron_fly", bounded=True))
    if not paper.empty and "paper_realized_pnl_dollars" in paper:
        paper = paper[paper["status"] == "EXPIRED"].copy()
        paper["pnl_dollars"] = pd.to_numeric(
            paper["paper_realized_pnl_dollars"], errors="coerce"
        )
        paper["return_on_max_risk"] = pd.to_numeric(
            paper.get("paper_return_on_max_risk"), errors="coerce"
        )
        rows.extend(_group_rows(paper, "paper_iron_fly", bounded=True))
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report.insert(0, "schema_version", 1)
    report.insert(1, "symbol", symbol)
    report.insert(2, "generated_at", datetime.now(ET))
    _atomic_write_parquet(report, data_dir / f"vrp_report_{symbol}.parquet")
    summary = {
        "schema_version": 1, "symbol": symbol,
        "generated_at": datetime.now(ET).isoformat(),
        "rows": len(report),
        "overall": report[report["dimension"] == "overall"].replace(
            {np.nan: None}
        ).to_dict("records"),
        "sizing_gate": "Iron fly only; >=90 independent days; positive net edge; "
                       "min(1/4 Kelly, 2% p99 cap, 25% hard cap).",
    }
    path = data_dir / f"vrp_report_{symbol}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)
    return report
