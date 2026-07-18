import pandas as pd

from gex_monitor.intraday_vrp_report import (
    cluster_bootstrap_mean,
    generate_vrp_report,
)


def test_cluster_bootstrap_requires_multiple_dates():
    low, high = cluster_bootstrap_mean(
        pd.Series([1.0, 2.0]), pd.Series(["20260101", "20260101"])
    )
    assert pd.isna(low) and pd.isna(high)


def test_report_gates_naked_straddle_and_sizes_mature_iron_fly(tmp_path):
    dates = pd.date_range("2026-01-02", periods=90, freq="B").strftime("%Y%m%d")
    straddles = pd.DataFrame({
        "symbol": "QQQ", "trading_date": dates, "scheduled_time": "10:00",
        "status": "ok", "pnl_executable": [0.10] * 89 + [-0.50],
        "positive_gamma": True, "event_flag": "none", "opex_type": "none",
    })
    flies = pd.DataFrame({
        "symbol": "QQQ", "trading_date": dates, "scheduled_time": "10:00",
        "target_wing_width": 3.0, "status": "ok",
        "iron_fly_pnl_dollars": [10.0] * 89 + [-50.0],
        "return_on_max_risk": [0.10] * 89 + [-0.50],
        "positive_gamma": True, "event_flag": "none", "opex_type": "none",
    })
    straddles.to_parquet(tmp_path / "vrp_observations_QQQ_20260101.parquet")
    flies.to_parquet(tmp_path / "vrp_iron_fly_observations_QQQ_20260101.parquet")
    report = generate_vrp_report(tmp_path, "QQQ")
    overall = report[report["dimension"] == "overall"].set_index("strategy")
    assert overall.loc["short_straddle", "sizing_reason"] == "unbounded_strategy"
    assert pd.isna(overall.loc["short_straddle", "recommended_fraction"])
    assert overall.loc["iron_fly", "sample_status"] == "sizing_eligible"
    assert overall.loc["iron_fly", "recommended_fraction"] > 0
    assert (tmp_path / "vrp_report_QQQ.parquet").exists()
    assert (tmp_path / "vrp_report_QQQ.json").exists()
