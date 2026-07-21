from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from gex_monitor.intraday_turning_point_research import (
    ResearchConfig,
    apply_numeric_bins,
    chronological_date_splits,
    cluster_rate_interval,
    fit_quantile_edges,
    run_feature_research,
)


def _research_events(days: int, *, start_day: int = 0) -> pd.DataFrame:
    rows = []
    origin = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    for day_offset in range(start_day, start_day + days):
        day = origin + timedelta(days=day_offset)
        for direction in ("after_down", "after_up"):
            for value, outcome in (
                (0.1, "reversal_up"),
                (0.2, "reversal_up"),
                (0.8, "continuation_down"),
                (0.9, "continuation_down"),
            ):
                rows.append(
                    {
                        "trading_date": day.strftime("%Y%m%d"),
                        "ts": day,
                        "setup_direction": direction,
                        "outcome": outcome,
                        "dist_to_flip_vol": value,
                        "dist_to_put_wall_vol": value,
                        "dist_to_call_wall_vol": -value,
                        "total_gex": 100.0,
                        "total_gex_change_5m": 0.0,
                        "total_gex_change_15m": 0.0,
                        "positive_gamma": True,
                    }
                )
    return pd.DataFrame(rows)


def test_chronological_splits_are_disjoint_and_ordered():
    frame = _research_events(12)
    config = ResearchConfig(train_days=6, validation_days=3)
    splits = chronological_date_splits(frame, config)

    assert [len(splits[name]) for name in ("train", "validation", "test")] == [6, 3, 3]
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["validation"]).isdisjoint(splits["test"])
    assert max(splits["train"]) < min(splits["validation"]) < min(splits["test"])


def test_numeric_bins_use_fixed_training_edges_for_external_values():
    edges = fit_quantile_edges(pd.Series([1.0, 2.0, 3.0, 4.0]), bins=4)
    assert edges is not None
    assigned = apply_numeric_bins(pd.Series([-100.0, 100.0]), edges)

    assert assigned.tolist() == ["Q1", "Q4"]


def test_cluster_interval_resamples_whole_dates():
    frame = pd.DataFrame(
        {
            "trading_date": ["a"] * 20 + ["b"] * 20,
            "target": [1] * 20 + [0] * 20,
        }
    )
    low, high = cluster_rate_interval(frame, "target", samples=2_000, seed=3)

    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)


def test_feature_selection_is_fit_on_train_and_confirmed_out_of_sample():
    primary = _research_events(60)
    external = _research_events(12, start_day=100)
    config = ResearchConfig(
        train_days=40,
        validation_days=10,
        bootstrap_samples=50,
    )

    bins, rankings, bases, summary = run_feature_research(primary, external, config)
    row = rankings[
        rankings["direction"].eq("after_down")
        & rankings["target"].eq("is_reversal")
        & rankings["feature"].eq("abs_dist_to_flip_vol")
    ].iloc[0]

    assert not bins.empty
    assert not bases.empty
    assert row["selected_bin"] in {"Q1", "Q2"}
    assert "abs_dist_to_flip_vol" in row["selected_condition"]
    assert row["validation_lift"] > 1.0
    assert row["test_lift"] > 1.0
    assert row["external_lift"] > 1.0
    assert row["status"] == "stable_cross_symbol"
    assert summary["date_splits"]["train"] == sorted(
        primary["trading_date"].unique()
    )[:40]


def test_research_can_run_without_external_frame():
    _, rankings, _, summary = run_feature_research(
        _research_events(12),
        config=ResearchConfig(
            train_days=6,
            validation_days=3,
            minimum_train_bin_rows=3,
            minimum_train_bin_days=2,
            bootstrap_samples=20,
        ),
    )

    assert not rankings.empty
    assert summary["external_rows"] == 0
    assert rankings["external_n"].fillna(0).eq(0).all()
