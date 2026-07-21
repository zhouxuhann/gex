"""Conditional statistics for the offline intraday turning-point dataset."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "abs_dist_to_flip_vol",
    "relevant_wall_distance_vol",
    "abs_relevant_wall_distance_vol",
    "gex_change_5m_ratio",
    "gex_change_15m_ratio",
    "call_put_gex_imbalance",
    "rr_25",
    "rr_25_change_5m",
    "rr_25_change_15m",
    "atm_iv_pct",
    "trend_efficiency_15m",
    "twap_distance_pct",
    "rv_15m",
    "strike_abs_gex_concentration_50bps",
    "strike_abs_gex_imbalance_50bps",
)

CATEGORICAL_FEATURES = ("positive_gamma", "time_bucket")

FEATURE_LABELS = {
    "abs_dist_to_flip_vol": "距 Gamma Flip（绝对波动单位）",
    "relevant_wall_distance_vol": "相关墙内侧距离（波动单位）",
    "abs_relevant_wall_distance_vol": "距相关 Call/Put Wall（绝对波动单位）",
    "gex_change_5m_ratio": "Total GEX 5分钟相对变化",
    "gex_change_15m_ratio": "Total GEX 15分钟相对变化",
    "call_put_gex_imbalance": "Call/Put GEX不平衡",
    "rr_25": "RR25",
    "rr_25_change_5m": "RR25 5分钟变化",
    "rr_25_change_15m": "RR25 15分钟变化",
    "atm_iv_pct": "ATM IV",
    "trend_efficiency_15m": "15分钟趋势效率",
    "twap_distance_pct": "距30分钟TWAP",
    "rv_15m": "15分钟实现波动",
    "strike_abs_gex_concentration_50bps": "Spot附近50bps Gamma集中度",
    "strike_abs_gex_imbalance_50bps": "Spot上下50bps Gamma不平衡",
    "positive_gamma": "正/负 GEX",
    "time_bucket": "日内时间段",
}


@dataclass(frozen=True)
class ResearchConfig:
    train_days: int = 40
    validation_days: int = 10
    quantile_bins: int = 4
    minimum_train_bin_rows: int = 15
    minimum_train_bin_days: int = 5
    minimum_validation_rows: int = 8
    minimum_validation_days: int = 3
    minimum_test_rows: int = 8
    minimum_test_days: int = 4
    minimum_external_rows: int = 10
    minimum_external_days: int = 5
    minimum_confirmed_lift: float = 1.05
    bootstrap_samples: int = 800
    random_seed: int = 20260722


def prepare_research_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only contemporaneous derived features and outcome target columns."""
    result = frame.copy()
    result["trading_date"] = result["trading_date"].astype(str)
    result["ts"] = pd.to_datetime(result["ts"], errors="coerce", utc=True)
    result["is_reversal"] = result["outcome"].isin(
        ["reversal_up", "reversal_down"]
    ).astype(int)
    result["is_continuation"] = result["outcome"].isin(
        ["continuation_up", "continuation_down"]
    ).astype(int)
    result["abs_dist_to_flip_vol"] = pd.to_numeric(
        result.get("dist_to_flip_vol"), errors="coerce"
    ).abs()
    after_down = result["setup_direction"].eq("after_down")
    put_distance = pd.to_numeric(result.get("dist_to_put_wall_vol"), errors="coerce")
    call_distance = pd.to_numeric(result.get("dist_to_call_wall_vol"), errors="coerce")
    # Positive means spot remains inside the relevant wall: above Put Wall after
    # a decline, or below Call Wall after a rise.
    result["relevant_wall_distance_vol"] = np.where(
        after_down, put_distance, -call_distance
    )
    result["abs_relevant_wall_distance_vol"] = pd.to_numeric(
        result["relevant_wall_distance_vol"], errors="coerce"
    ).abs()
    total = pd.to_numeric(result.get("total_gex"), errors="coerce").abs()
    scale = total.replace(0, np.nan)
    for minutes in (5, 15):
        change = pd.to_numeric(
            result.get(f"total_gex_change_{minutes}m"), errors="coerce"
        )
        result[f"gex_change_{minutes}m_ratio"] = change / scale
    hour = result["ts"].dt.tz_convert("America/New_York").dt.hour
    minute = result["ts"].dt.tz_convert("America/New_York").dt.minute
    clock = hour * 60 + minute
    result["time_bucket"] = pd.cut(
        clock,
        bins=[-np.inf, 11 * 60, 14 * 60, np.inf],
        labels=["open_to_1100", "1100_to_1400", "after_1400"],
        right=False,
    ).astype("object")
    result["positive_gamma"] = result.get("positive_gamma").astype("boolean").astype(
        "object"
    )
    return result


def chronological_date_splits(
    frame: pd.DataFrame, config: ResearchConfig
) -> dict[str, list[str]]:
    dates = sorted(frame["trading_date"].dropna().astype(str).unique())
    if len(dates) <= config.train_days:
        raise ValueError(
            f"need more than {config.train_days} dates for chronological validation"
        )
    train_end = config.train_days
    validation_end = min(len(dates), train_end + config.validation_days)
    return {
        "train": dates[:train_end],
        "validation": dates[train_end:validation_end],
        "test": dates[validation_end:],
    }


def fit_quantile_edges(values: pd.Series, bins: int = 4) -> list[float] | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.nunique() < 2:
        return None
    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    inner = sorted(set(float(value) for value in clean.quantile(quantiles)))
    if not inner:
        return None
    return [-np.inf, *inner, np.inf]


def apply_numeric_bins(values: pd.Series, edges: list[float]) -> pd.Series:
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        edges,
        labels=labels,
        include_lowest=True,
    ).astype("object")


def cluster_rate_interval(
    frame: pd.DataFrame,
    target: str,
    *,
    samples: int = 800,
    seed: int = 20260722,
) -> tuple[float | None, float | None]:
    clean = frame.dropna(subset=["trading_date", target])
    if clean.empty:
        return None, None
    by_date = clean.groupby("trading_date")[target].agg(["sum", "count"])
    if len(by_date) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    sums = by_date["sum"].to_numpy(dtype=float)
    counts = by_date["count"].to_numpy(dtype=float)
    draws = rng.integers(0, len(by_date), size=(samples, len(by_date)))
    rates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def _group_statistics(
    frame: pd.DataFrame,
    *,
    target: str,
    feature: str,
    split: str,
    direction: str,
    base_rate: float | None,
    config: ResearchConfig,
) -> list[dict]:
    if "feature_bin" not in frame.columns:
        return []
    rows = []
    for bin_name, group in frame.dropna(subset=["feature_bin"]).groupby(
        "feature_bin", observed=True
    ):
        rate = float(group[target].mean()) if not group.empty else None
        low, high = cluster_rate_interval(
            group, target, samples=config.bootstrap_samples,
            seed=config.random_seed,
        )
        rows.append({
            "direction": direction,
            "target": target,
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "feature_bin": str(bin_name),
            "split": split,
            "n": len(group),
            "days": int(group["trading_date"].nunique()),
            "rate": rate,
            "base_rate": base_rate,
            "lift": rate / base_rate if rate is not None and base_rate else None,
            "ci_low": low,
            "ci_high": high,
        })
    return rows


def run_feature_research(
    primary: pd.DataFrame,
    external: pd.DataFrame | None = None,
    config: ResearchConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Fit bins on primary train dates, then validate without refitting."""
    config = config or ResearchConfig()
    primary = prepare_research_frame(primary)
    external = prepare_research_frame(external) if external is not None else None
    splits = chronological_date_splits(primary, config)
    primary["split"] = "excluded"
    for name, dates in splits.items():
        primary.loc[primary["trading_date"].isin(dates), "split"] = name

    targets = ("is_reversal", "is_continuation")
    bin_rows: list[dict] = []
    ranking_rows: list[dict] = []
    base_rows: list[dict] = []
    bin_definitions: dict[str, dict] = {}
    for direction in ("after_down", "after_up"):
        direction_frame = primary[primary["setup_direction"].eq(direction)].copy()
        train = direction_frame[direction_frame["split"].eq("train")]
        external_direction = (
            external[external["setup_direction"].eq(direction)].copy()
            if external is not None
            else direction_frame.iloc[0:0].copy()
        )
        for target in targets:
            for split_name in ("train", "validation", "test"):
                split_frame = direction_frame[direction_frame["split"].eq(split_name)]
                base_rows.append({
                    "direction": direction, "target": target, "split": split_name,
                    "n": len(split_frame),
                    "days": int(split_frame["trading_date"].nunique()),
                    "rate": float(split_frame[target].mean())
                    if not split_frame.empty else None,
                })
            base_rows.append({
                "direction": direction, "target": target, "split": "external",
                "n": len(external_direction),
                "days": int(external_direction["trading_date"].nunique())
                if not external_direction.empty else 0,
                "rate": float(external_direction[target].mean())
                if not external_direction.empty else None,
            })

        for feature in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES):
            if feature not in train:
                continue
            if feature in NUMERIC_FEATURES:
                edges = fit_quantile_edges(train[feature], config.quantile_bins)
                if edges is None:
                    continue
                bin_definitions[f"{direction}:{feature}"] = {"edges": edges}

                def assign(source: pd.DataFrame) -> pd.DataFrame:
                    result = source.copy()
                    result["feature_bin"] = apply_numeric_bins(result[feature], edges)
                    return result
            else:
                categories = sorted(str(value) for value in train[feature].dropna().unique())
                bin_definitions[f"{direction}:{feature}"] = {"categories": categories}

                def assign(source: pd.DataFrame) -> pd.DataFrame:
                    result = source.copy()
                    result["feature_bin"] = result[feature].map(
                        lambda value: str(value) if pd.notna(value) else None
                    )
                    return result

            assigned = assign(direction_frame)
            assigned_external = assign(external_direction)
            for target in targets:
                stats_for_feature = []
                for split_name in ("train", "validation", "test"):
                    split_frame = assigned[assigned["split"].eq(split_name)]
                    base_rate = (
                        float(split_frame[target].mean())
                        if not split_frame.empty
                        else None
                    )
                    stats_for_feature.extend(
                        _group_statistics(
                            split_frame,
                            target=target,
                            feature=feature,
                            split=split_name,
                            direction=direction,
                            base_rate=base_rate,
                            config=config,
                        )
                    )
                external_base = (
                    float(assigned_external[target].mean())
                    if not assigned_external.empty
                    else None
                )
                stats_for_feature.extend(
                    _group_statistics(
                        assigned_external,
                        target=target,
                        feature=feature,
                        split="external",
                        direction=direction,
                        base_rate=external_base,
                        config=config,
                    )
                )
                bin_rows.extend(stats_for_feature)
                stats = pd.DataFrame(stats_for_feature)
                train_stats = stats[
                    (stats["split"] == "train")
                    & (stats["n"] >= config.minimum_train_bin_rows)
                    & (stats["days"] >= config.minimum_train_bin_days)
                ]
                if train_stats.empty:
                    continue
                selected = train_stats.sort_values(
                    ["lift", "n"], ascending=[False, False]
                ).iloc[0]
                selected_bin = str(selected["feature_bin"])
                definition = bin_definitions[f"{direction}:{feature}"]
                row = {
                    "direction": direction, "target": target,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "selected_bin": selected_bin,
                    "selected_condition": _selected_condition(
                        feature, selected_bin, definition
                    ),
                }
                for split_name in ("train", "validation", "test", "external"):
                    match = stats[
                        (stats["split"] == split_name)
                        & (stats["feature_bin"] == selected["feature_bin"])
                    ]
                    for column in ("n", "days", "rate", "lift", "ci_low", "ci_high"):
                        row[f"{split_name}_{column}"] = (
                            match.iloc[0][column] if not match.empty else None
                        )
                qqq_confirmed = (
                    _finite_or_zero(row.get("validation_n"))
                    >= config.minimum_validation_rows
                    and _finite_or_zero(row.get("validation_days"))
                    >= config.minimum_validation_days
                    and _finite_or_zero(row.get("test_n"))
                    >= config.minimum_test_rows
                    and _finite_or_zero(row.get("test_days"))
                    >= config.minimum_test_days
                    and _finite_or_zero(row.get("validation_lift"))
                    >= config.minimum_confirmed_lift
                    and _finite_or_zero(row.get("test_lift"))
                    >= config.minimum_confirmed_lift
                )
                external_confirmed = (
                    _finite_or_zero(row.get("external_n"))
                    >= config.minimum_external_rows
                    and _finite_or_zero(row.get("external_days"))
                    >= config.minimum_external_days
                    and _finite_or_zero(row.get("external_lift"))
                    >= config.minimum_confirmed_lift
                )
                row["status"] = (
                    "stable_cross_symbol" if qqq_confirmed and external_confirmed
                    else "stable_qqq" if qqq_confirmed
                    else "not_confirmed"
                )
                ranking_rows.append(row)

    bins = pd.DataFrame(bin_rows)
    rankings = pd.DataFrame(ranking_rows)
    bases = pd.DataFrame(base_rows).drop_duplicates(
        ["direction", "target", "split"]
    )
    summary = {
        "schema_version": 1,
        "config": config.__dict__,
        "date_splits": splits,
        "bin_definitions": bin_definitions,
        "primary_rows": len(primary),
        "external_rows": len(external) if external is not None else 0,
        "stable_cross_symbol": int(rankings["status"].eq(
            "stable_cross_symbol"
        ).sum()) if not rankings.empty else 0,
        "stable_qqq": int(rankings["status"].eq("stable_qqq").sum())
        if not rankings.empty else 0,
    }
    return bins, rankings, bases, summary


def _finite_or_zero(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _format_boundary(value: float) -> str:
    if np.isneginf(value):
        return "-∞"
    if np.isposinf(value):
        return "+∞"
    return f"{value:.6g}"


def _selected_condition(
    feature: str,
    selected_bin: str,
    definition: dict,
) -> str:
    edges = definition.get("edges")
    if edges and selected_bin.startswith("Q"):
        index = int(selected_bin[1:]) - 1
        if 0 <= index < len(edges) - 1:
            lower = _format_boundary(float(edges[index]))
            upper = _format_boundary(float(edges[index + 1]))
            return f"{lower} < {feature} ≤ {upper}"
    return f"{feature} = {selected_bin}"


def write_research_report(
    rankings: pd.DataFrame,
    bases: pd.DataFrame,
    summary: dict,
    output: Path | str,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    display_columns = [
        "direction", "target", "feature_label", "selected_condition", "status",
        "train_n", "train_rate", "train_lift",
        "validation_n", "validation_rate", "validation_lift",
        "test_n", "test_rate", "test_lift", "test_ci_low", "test_ci_high",
        "external_n", "external_rate", "external_lift", "external_ci_low",
        "external_ci_high",
    ]
    if rankings.empty:
        ordered = rankings
    else:
        status_order = {
            "stable_cross_symbol": 0,
            "stable_qqq": 1,
            "not_confirmed": 2,
        }
        ordered = (
            rankings.assign(
                _status_order=rankings["status"].map(status_order).fillna(3)
            )
            .sort_values(
                ["_status_order", "test_lift", "external_lift"],
                ascending=[True, False, False],
                na_position="last",
            )
            .drop(columns="_status_order")
        )
    stable = ordered[ordered["status"].ne("not_confirmed")]

    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "<p>No rows.</p>"
        view = frame[[column for column in display_columns if column in frame]].copy()
        for column in view.columns:
            if column.endswith(("_rate", "_lift", "_low", "_high")):
                view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
        return view.to_html(index=False, border=0, classes="data")

    output.write_text(
        "<!doctype html><meta charset='utf-8'><title>Turning-point research</title>"
        "<style>body{font-family:system-ui;max-width:1500px;margin:24px auto;"
        "color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{padding:7px;border-bottom:1px solid #dbe2ea;text-align:right}"
        "th:nth-child(-n+5),td:nth-child(-n+5){text-align:left}"
        "th{position:sticky;top:0;background:#eef2ff}pre{background:#f1f5f9;"
        "padding:14px;white-space:pre-wrap}.note{background:#fff7ed;padding:12px}</style>"
        "<h1>QQQ intraday turning-point conditional research</h1>"
        "<p class='note'>Bins and selected conditions are fitted on the first 40 QQQ dates. "
        "Validation, test and SPY external results are not used for selection. This is a "
        "research ranking, not a trading signal.</p>"
        "<h2>Baseline rates</h2>"
        + bases.to_html(index=False, border=0, classes="data")
        + "<h2>Replicated candidate conditions</h2>" + table(stable)
        + "<h2>All feature candidates</h2>" + table(ordered)
        + "<h2>Research metadata</h2><pre>"
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return output
