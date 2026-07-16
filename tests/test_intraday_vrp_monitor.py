from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from gex_monitor.config import IntradayVRPConfig
from gex_monitor.intraday_vrp_monitor import IntradayVRPMonitor
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


def ticker(bid, ask, delta, ts):
    return SimpleNamespace(
        bid=bid, ask=ask, bidSize=10, askSize=12, time=ts,
        modelGreeks=SimpleNamespace(delta=delta),
    )


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


def test_collects_same_strike_pair_and_executable_credit(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(enabled=True, max_quote_age_seconds=10)
    monitor = IntradayVRPMonitor("QQQ", storage, config)
    now = datetime(2026, 7, 16, 9, 35, 2, tzinfo=ET)
    contracts = [contract(724, "C"), contract(724, "P"),
                 contract(725, "C"), contract(725, "P")]
    tickers = {
        contracts[0]: ticker(1.9, 2.0, 0.55, now.astimezone(timezone.utc)),
        contracts[1]: ticker(1.7, 1.8, -0.45, now.astimezone(timezone.utc)),
        contracts[2]: ticker(1.4, 1.5, 0.48, now.astimezone(timezone.utc)),
        contracts[3]: ticker(2.1, 2.2, -0.52, now.astimezone(timezone.utc)),
    }
    assert monitor.on_gex_update(
        FakeIB(tickers), contracts, now=now, spot=724.6, expiry="20260716",
        is_true_0dte=True,
        gex_state={"total_gex": 1e9, "gamma_flip": 720,
                   "positive_gamma": True, "regime_tags": {}},
    )
    row = storage.load_vrp_quotes("QQQ", "20260716").iloc[0]
    assert row["status"] == "ok"
    assert row["strike"] == 725
    assert row["sell_credit_bid"] == 3.5
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
        "status": "ok",
    })
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
    storage.shutdown()
