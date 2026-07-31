from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from gex_monitor.config import IntradayVRPConfig
from gex_monitor.intraday_vrp_monitor import IntradayVRPMonitor, vrp_gex_advisory
from gex_monitor.storage import StorageManager
from gex_monitor.time_utils import ET


class FakeIB:
    def __init__(self, tickers):
        self.tickers = tickers

    def ticker(self, contract):
        return self.tickers[contract]


class FakeContract:
    def __init__(self, strike, right):
        self.strike = strike
        self.right = right
        self.lastTradeDateOrContractMonth = "20260716"


def contract(strike, right):
    return FakeContract(strike, right)


def ticker(bid, ask, delta, ts, gamma=0.02, theta=-0.08, vega=0.03, iv=0.25):
    return SimpleNamespace(
        bid=bid, ask=ask, bidSize=10, askSize=12, time=ts,
        modelGreeks=SimpleNamespace(delta=delta, gamma=gamma, theta=theta,
                                    vega=vega, impliedVol=iv),
    )


def test_vrp_gex_context_is_advisory_and_requires_new_oi_method():
    valid = vrp_gex_advisory({
        "gex_method": "oi_position_v2", "total_gex": -200.0,
        "gross_gex": 1000.0, "gross_volume_gamma": 250.0,
        "partial": False,
    })
    assert valid["gex_vrp_regime"] == "short_gamma"
    assert valid["gex_strategy_preference"] == "defined_risk_preferred"
    assert valid["volume_gamma_to_gex"] == 0.25
    assert valid["gex_hard_gate_enabled"] is False

    legacy = vrp_gex_advisory({
        "gex_method": "legacy_mixed", "total_gex": 200.0,
        "gross_gex": 1000.0,
    })
    assert legacy["gex_vrp_regime"] == "unknown"
    assert legacy["gex_strategy_preference"] == "observe_only"


def test_due_slot_window_and_persisted_dedup(tmp_path):
    storage = StorageManager(tmp_path)
    monitor = IntradayVRPMonitor("QQQ", storage, IntradayVRPConfig(enabled=True))
    now = datetime(2026, 7, 16, 9, 35, 30, tzinfo=ET)
    assert monitor.due_slot(now)[0] == "09:35"
    storage.persist_vrp_quote("QQQ", "20260716", {
        "symbol": "QQQ", "trading_date": "20260716",
        "scheduled_time": "09:35", "observed_at": now,
    })
    monitor2 = IntradayVRPMonitor("QQQ", storage, IntradayVRPConfig(enabled=True))
    assert monitor2.due_slot(now) is None
    storage.shutdown()


def test_mtm_targets_are_generated_every_five_minutes_to_1555(tmp_path):
    storage = StorageManager(tmp_path)
    monitor = IntradayVRPMonitor(
        "QQQ", storage,
        IntradayVRPConfig(enabled=True, mtm_checkpoints_minutes=[],
                         mtm_interval_minutes=5, mtm_fixed_times_et=["15:30"]),
    )
    entry = datetime(2026, 7, 16, 14, 0, tzinfo=ET)
    targets = monitor._checkpoint_targets({"observed_at": entry})
    assert targets[0][0] == "+5m"
    assert targets[-1][0] == "+115m"
    assert targets[-1][1].strftime("%H:%M") == "15:55"
    assert "15:30" in {name for name, _ in targets}
    storage.shutdown()


def test_collects_same_strike_pair_and_executable_credit(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(enabled=True, max_quote_age_seconds=10)
    monitor = IntradayVRPMonitor("QQQ", storage, config)
    now = datetime(2026, 7, 16, 9, 35, 2, tzinfo=ET)
    contracts = [contract(724, "C"), contract(724, "P"),
                 contract(725, "C"), contract(725, "P"),
                 contract(726, "C"), contract(726, "P")]
    tickers = {
        contracts[0]: ticker(1.9, 2.0, 0.55, now.astimezone(timezone.utc)),
        contracts[1]: ticker(1.7, 1.8, -0.45, now.astimezone(timezone.utc)),
        contracts[2]: ticker(1.4, 1.5, 0.48, now.astimezone(timezone.utc)),
        contracts[3]: ticker(2.1, 2.2, -0.52, now.astimezone(timezone.utc)),
        contracts[4]: ticker(0.9, 1.0, 0.35, now.astimezone(timezone.utc)),
        contracts[5]: ticker(2.8, 2.9, -0.65, now.astimezone(timezone.utc)),
    }
    bars = [{"ts": datetime(2026, 7, 16, 9, 34, tzinfo=ET),
             "open": 724.0, "high": 724.7, "low": 723.9, "close": 724.5}]
    assert monitor.on_gex_update(
        FakeIB(tickers), contracts, now=now, spot=724.6, expiry="20260716",
        is_true_0dte=True,
        gex_state={"total_gex": 1e9, "gamma_flip": 720,
                   "positive_gamma": True, "regime_tags": {},
                   "rr_25": 0.04, "skew_slope": 0.2,
                   "put_25_iv": 0.31, "call_25_iv": 0.27,
                   "rr_25_zscore": 1.1, "drr_25": 0.005,
                   "drr_25_zscore": 0.7},
        intraday_bars_provider=lambda: bars,
    )
    row = storage.load_vrp_quotes("QQQ", "20260716").iloc[0]
    assert row["status"] == "ok"
    assert row["strike"] == 725
    assert row["sell_credit_bid"] == 3.5
    assert row["schema_version"] == 5
    assert row["rr_25"] == 0.04
    assert row["drr_25"] == 0.005
    assert row["put_25_iv"] == 0.31
    assert row["call_25_iv"] == 0.27
    assert abs(row["minutes_to_close"] - (385 - 2 / 60)) < 1e-9
    assert abs(row["tau_session"] - row["minutes_to_close"] / 390) < 1e-9
    assert abs(row["tau_years"] - row["minutes_to_close"] / (252 * 390)) < 1e-9
    assert abs(
        row["tau_calendar_years"]
        - row["minutes_to_close"] / (365 * 24 * 60)
    ) < 1e-9
    assert abs(row["straddle_gamma"] - 0.04) < 1e-9
    assert abs(row["straddle_theta"] + 0.16) < 1e-9
    assert row["entry_bar_count"] == 1
    assert row["weekday"] == "Thursday"
    assert row["opex_type"] == "none"
    assert row["event_flag"] == "none"
    assert abs(row["dist_to_flip_im"] - (724.6 - 720) / 3.6) < 1e-9
    wings = pd.read_parquet(tmp_path / "vrp_wing_quotes_QQQ_20260716.parquet")
    assert len(wings) == 6
    assert set(wings["status"]) == {"ok"}
    flies = pd.read_parquet(tmp_path / "vrp_iron_flies_QQQ_20260716.parquet")
    assert len(flies) == 1
    fly = flies.iloc[0]
    assert fly["target_wing_width"] == 1
    assert abs(fly["gross_net_credit"] - 0.7) < 1e-9
    assert abs(fly["net_credit_after_fees"] - 0.674) < 1e-9
    assert abs(fly["max_loss_dollars"] - 32.6) < 1e-9
    minute = storage.load_vrp_minute_nodes("QQQ", "20260716")
    cone = storage.load_vrp_cone_nodes("QQQ", "20260716")
    assert len(minute) == 1
    assert len(cone) == 1
    assert minute.iloc[0]["sample_kind"] == "raw_1m"
    assert cone.iloc[0]["sample_kind"] == "cone_5m"
    assert minute.iloc[0]["entry_bar_count"] == 1
    assert "rv_session_to_now" in minute.columns
    assert "rv_annualized_to_now" in minute.columns
    assert "rv_5m_annualized" in minute.columns
    storage.shutdown()


def test_captures_executable_straddle_and_iron_fly_mtm(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(enabled=True, mtm_checkpoints_minutes=[5],
                               mtm_fixed_times_et=[])
    monitor = IntradayVRPMonitor("QQQ", storage, config)
    entry_time = datetime(2026, 7, 16, 9, 35, 2, tzinfo=ET)
    contracts = [contract(724, "C"), contract(724, "P"),
                 contract(725, "C"), contract(725, "P"),
                 contract(726, "C"), contract(726, "P")]
    entry_tickers = {
        contracts[0]: ticker(1.9, 2.0, 0.55, entry_time),
        contracts[1]: ticker(1.7, 1.8, -0.45, entry_time),
        contracts[2]: ticker(1.4, 1.5, 0.48, entry_time),
        contracts[3]: ticker(2.1, 2.2, -0.52, entry_time),
        contracts[4]: ticker(0.9, 1.0, 0.35, entry_time),
        contracts[5]: ticker(2.8, 2.9, -0.65, entry_time),
    }
    assert monitor.on_gex_update(
        FakeIB(entry_tickers), contracts, now=entry_time, spot=724.6,
        expiry="20260716", is_true_0dte=True,
        gex_state={"gamma_flip": 720, "regime_tags": {}},
    )
    mark_time = datetime(2026, 7, 16, 9, 40, 3, tzinfo=ET)
    mark_tickers = {
        contracts[0]: ticker(1.0, 1.1, 0.55, mark_time),
        contracts[1]: ticker(0.5, 0.6, -0.45, mark_time),
        contracts[2]: ticker(1.1, 1.2, 0.48, mark_time),
        contracts[3]: ticker(1.6, 1.7, -0.52, mark_time),
        contracts[4]: ticker(0.4, 0.5, 0.35, mark_time),
        contracts[5]: ticker(2.0, 2.1, -0.65, mark_time),
    }
    invalid_tickers = dict(mark_tickers)
    invalid_tickers[contracts[2]] = ticker(1.1, None, 0.48, mark_time)
    assert not monitor.on_gex_update(
        FakeIB(invalid_tickers), contracts, now=mark_time, spot=725.0,
        expiry="20260716", is_true_0dte=True,
        gex_state={"gamma_flip": 720, "regime_tags": {}},
    )
    failed = storage.load_vrp_mtm("QQQ", "20260716").iloc[0]
    assert failed["status"] == "invalid_nbbo"
    assert failed["retry_count"] == 1
    mark_time = datetime(2026, 7, 16, 9, 40, 4, tzinfo=ET)
    for item in mark_tickers.values():
        item.time = mark_time
    assert not monitor.on_gex_update(
        FakeIB(mark_tickers), contracts, now=mark_time, spot=725.0,
        expiry="20260716", is_true_0dte=True,
        gex_state={"gamma_flip": 720, "regime_tags": {}},
    )
    mtm = storage.load_vrp_mtm("QQQ", "20260716").iloc[0]
    assert mtm["checkpoint"] == "+5m"
    assert mtm["status"] == "ok"
    assert mtm["retry_count"] == 2
    assert abs(mtm["close_cost_ask"] - 2.9) < 1e-9
    assert abs(mtm["pnl_executable_roundtrip"] - 0.574) < 1e-9
    fly = storage.load_vrp_iron_fly_mtm("QQQ", "20260716").iloc[0]
    assert fly["checkpoint"] == "+5m"
    assert abs(fly["close_debit_executable"] - 2.0) < 1e-9
    storage.shutdown()


def test_settlement_uses_actual_strike_and_cleans_cross_day_bars(tmp_path):
    storage = StorageManager(tmp_path)
    monitor = IntradayVRPMonitor(
        "QQQ", storage, IntradayVRPConfig(enabled=True, commission_per_straddle=1.30)
    )
    observed = datetime(2026, 7, 15, 15, 0, tzinfo=ET)
    storage.persist_vrp_quote("QQQ", "20260715", {
        "schema_version": 1, "symbol": "QQQ", "trading_date": "20260715",
        "scheduled_time": "15:00", "observed_at": observed, "spot": 724.6,
        "strike": 725.0, "straddle_mid": 2.0, "sell_credit_bid": 1.8,
        "call_implied_vol": 0.20, "put_implied_vol": 0.20,
        "tau_calendar_years": 1 / (365 * 24),
        "status": "ok",
    })
    storage.persist_vrp_iron_flies("QQQ", "20260715", [{
        "schema_version": 1, "symbol": "QQQ", "trading_date": "20260715",
        "scheduled_time": "15:00", "observed_at": observed,
        "target_wing_width": 3.0, "atm_strike": 725.0,
        "lower_put_strike": 722.0, "upper_call_strike": 728.0,
        "downside_wing_width": 3.0, "upside_wing_width": 3.0,
        "net_credit_after_fees": 1.2,
        "lower_breakeven": 723.8, "upper_breakeven": 726.2,
        "max_loss_points": 1.8, "max_loss_dollars": 180.0,
        "status": "ok",
    }])
    ts = pd.date_range("2026-07-15 09:30", periods=390, freq="min", tz=ET)
    bars = pd.DataFrame({"ts": ts, "open": 724.0, "high": 726.0,
                         "low": 723.0, "close": 724.0})
    polluted = pd.DataFrame({"ts": [pd.Timestamp("2026-07-14 15:59", tz=ET)],
                             "open": [700.0], "high": [700.0], "low": [700.0],
                             "close": [700.0]})
    pd.concat([polluted, bars]).to_parquet(tmp_path / "ohlc_QQQ_20260715.parquet",
                                           index=False)
    assert monitor.settle_date("20260715") == 1
    row = storage.load_vrp_observations("QQQ", "20260715").iloc[0]
    assert row["terminal_payoff"] == 1.0
    assert abs(row["pnl_executable"] - 0.787) < 1e-9
    assert row["rth_bar_count"] == 390
    assert row["implied_variance_remaining"] > 0
    assert row["realized_variance_remaining"] == 0
    assert row["ex_post_variance_spread"] > 0
    assert bool(row["breakeven_breached"])
    assert row["path_high"] == 726.0
    fly = storage.load_vrp_iron_fly_observations("QQQ", "20260715").iloc[0]
    assert fly["settlement_price"] == 724.0
    assert fly["terminal_fly_payoff"] == 1.0
    assert abs(fly["iron_fly_pnl_dollars"] - 20.0) < 1e-9
    assert abs(fly["return_on_max_risk"] - (20 / 180)) < 1e-9
    assert not bool(fly["hit_max_loss"])
    assert bool(fly["expired_inside_breakeven"])
    assert bool(fly["breakeven_breached_intraday"])
    assert not bool(fly["wing_touched_intraday"])
    assert fly["worst_path_pnl_dollars"] == -80.0
    storage.shutdown()


def test_entry_bars_merge_fresher_state_over_lagging_official_file(tmp_path):
    storage = StorageManager(tmp_path)
    monitor = IntradayVRPMonitor("QQQ", storage, IntradayVRPConfig(enabled=True))
    official = pd.DataFrame({
        "ts": pd.date_range("2026-07-16 09:30", periods=2, freq="min", tz=ET),
        "close": [100.0, 101.0],
        "volume": [10, 20],
    })
    official.to_parquet(
        tmp_path / "official_ohlc_QQQ_20260716.parquet", index=False
    )
    fallback = [
        {"ts": datetime(2026, 7, 16, 9, 31, tzinfo=ET), "close": 100.5},
        {"ts": datetime(2026, 7, 16, 9, 32, tzinfo=ET), "close": 102.0},
    ]
    merged = monitor._entry_bars("20260716", fallback)
    assert len(merged) == 3
    assert merged.iloc[-1]["close"] == 102.0
    assert merged.loc[
        merged["ts"] == pd.Timestamp("2026-07-16 09:31", tz=ET), "close"
    ].iloc[0] == 101.0
    storage.shutdown()


def test_settlement_defers_until_complete_rth_bars(tmp_path):
    storage = StorageManager(tmp_path)
    monitor = IntradayVRPMonitor("QQQ", storage, IntradayVRPConfig(enabled=True))
    observed = datetime(2026, 7, 16, 10, 0, tzinfo=ET)
    storage.persist_vrp_quote("QQQ", "20260716", {
        "symbol": "QQQ", "trading_date": "20260716",
        "scheduled_time": "10:00", "observed_at": observed,
        "spot": 100.0, "strike": 100.0, "status": "ok",
    })
    bars = pd.DataFrame({
        "ts": pd.date_range("2026-07-16 09:30", periods=200, freq="min", tz=ET),
        "close": 100.0,
    })
    bars.to_parquet(tmp_path / "ohlc_QQQ_20260716.parquet", index=False)
    assert monitor.settle_date("20260716") == 0
    assert storage.load_vrp_observations("QQQ", "20260716").empty
    storage.shutdown()


def test_iron_fly_settlement_caps_payoff_at_wing(tmp_path):
    storage = StorageManager(tmp_path)
    monitor = IntradayVRPMonitor("SPY", storage, IntradayVRPConfig(enabled=True))
    storage.persist_vrp_iron_flies("SPY", "20260715", [{
        "schema_version": 1, "symbol": "SPY", "trading_date": "20260715",
        "scheduled_time": "14:00", "target_wing_width": 2.0,
        "atm_strike": 750.0, "downside_wing_width": 2.0,
        "upside_wing_width": 2.0, "net_credit_after_fees": 1.25,
        "lower_breakeven": 748.75, "upper_breakeven": 751.25,
        "max_loss_points": 0.75, "max_loss_dollars": 75.0,
    }])
    assert monitor._settle_iron_flies(
        date_str="20260715", settlement_price=755.0,
        settlement_source="test", rth_bar_count=390,
    ) == 1
    fly = storage.load_vrp_iron_fly_observations("SPY", "20260715").iloc[0]
    assert fly["terminal_fly_payoff"] == 2.0
    assert fly["iron_fly_pnl_dollars"] == -75.0
    assert fly["return_on_max_risk"] == -1.0
    assert bool(fly["hit_max_loss"])
    assert not bool(fly["expired_inside_breakeven"])
    storage.shutdown()
