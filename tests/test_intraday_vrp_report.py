import pandas as pd

from gex_monitor.intraday_vrp_report import (
    cluster_bootstrap_mean,
    generate_vrp_cone_report,
    generate_vrp_report,
)


def test_cluster_bootstrap_requires_multiple_dates():
    low, high = cluster_bootstrap_mean(
        pd.Series([1.0, 2.0]), pd.Series(["20260101", "20260101"])
    )
    assert pd.isna(low) and pd.isna(high)


def test_report_gates_naked_straddle_and_sizes_mature_iron_fly(tmp_path):
    dates = pd.date_range("2024-01-02", periods=500, freq="B").strftime("%Y%m%d")
    straddles = pd.DataFrame({
        "symbol": "QQQ", "trading_date": dates, "scheduled_time": "10:00",
        "status": "ok", "pnl_executable": [0.10] * 499 + [-0.50],
        "positive_gamma": True, "event_flag": "none", "opex_type": "none",
    })
    flies = pd.DataFrame({
        "symbol": "QQQ", "trading_date": dates, "scheduled_time": "10:00",
        "target_wing_width": 3.0, "status": "ok",
        "iron_fly_pnl_dollars": [10.0] * 499 + [-50.0],
        "return_on_max_risk": [0.10] * 499 + [-0.50],
        "positive_gamma": True, "event_flag": "none", "opex_type": "none",
    })
    straddles.to_parquet(tmp_path / "vrp_observations_QQQ_20260101.parquet")
    flies.to_parquet(tmp_path / "vrp_iron_fly_observations_QQQ_20260101.parquet")
    report = generate_vrp_report(tmp_path, "QQQ")
    overall = report[report["dimension"] == "overall"].set_index("strategy")
    assert overall.loc["short_straddle", "sizing_reason"] == "unbounded_strategy"
    assert pd.isna(overall.loc["short_straddle", "recommended_fraction"])
    assert overall.loc["iron_fly", "sample_status"] == "q10_q90_ready"
    assert overall.loc["iron_fly", "recommended_fraction"] > 0
    assert (tmp_path / "vrp_report_QQQ.parquet").exists()
    assert (tmp_path / "vrp_report_QQQ.json").exists()


def test_cone_report_counts_days_not_intraday_rows(tmp_path):
    frame = pd.DataFrame({
        "symbol": ["SPY"] * 4,
        "trading_date": ["20260102", "20260102", "20260105", "20260105"],
        "scheduled_time": ["10:00"] * 4,
        "observed_at": pd.to_datetime([
            "2026-01-02T15:00:00Z", "2026-01-02T15:00:02Z",
            "2026-01-05T15:00:00Z", "2026-01-05T15:00:02Z",
        ]),
        "status": ["ok"] * 4,
        "event_flag": ["none"] * 4,
        "atm_iv_decimal": [0.20, 0.21, 0.30, 0.31],
    })
    frame.to_parquet(tmp_path / "vrp_cone_observations_SPY_20260105.parquet")
    report = generate_vrp_cone_report(tmp_path, "SPY")
    row = report[
        (report["condition"] == "all")
        & (report["metric"] == "atm_iv_decimal")
    ].iloc[0]
    assert row["trading_days"] == 2
    assert row["non_null_days"] == 2
    assert row["q50"] == 0.26
