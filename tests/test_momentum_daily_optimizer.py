import json

import numpy as np
import pandas as pd
import pytest

import gex_monitor.momentum_daily_optimizer as optimizer
from gex_monitor.momentum_0dte_paper_trader import DailyBarJournal, Momentum0DTEPaperTrader
from gex_monitor.multi_asset_momentum import MomentumConfig


def _bars(days: int = 1) -> pd.DataFrame:
    frames = []
    for day in range(days):
        index = pd.date_range(
            pd.Timestamp("2026-07-06 09:30", tz="America/New_York") + pd.Timedelta(days=day),
            periods=10,
            freq="min",
        )
        close = 500.0 + np.arange(len(index)) * 0.1
        frames.append(
            pd.DataFrame(
                {
                    "open": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1_000.0,
                },
                index=index,
            )
        )
    return pd.concat(frames)


def test_replay_executes_close_signal_at_next_bar_open(monkeypatch):
    bars = _bars()
    result = bars.copy()
    result["long_signal"] = [False, True, True, False, False, False, False, False, False, False]
    result["short_signal"] = False
    result["long_entry"] = [False, True, False, False, False, False, False, False, False, False]
    result["short_entry"] = False
    # Signal is seen on row 1, so entry is row 2 open. It disappears on row 3,
    # so exit is row 4 open.
    result.loc[result.index[2], "open"] = 501.0
    result.loc[result.index[4], "open"] = 503.0
    monkeypatch.setattr(optimizer, "calculate_momentum_signals", lambda *args, **kwargs: result)

    metrics, trades = optimizer.replay_signals(
        bars, "QQQ", MomentumConfig(asset="QQQ"), round_trip_cost_points=0.0
    )

    assert metrics.trades == 1
    assert trades[0].entry_price == 501.0
    assert trades[0].exit_price == 503.0
    assert trades[0].pnl_points == 2.0


def test_daily_journal_replaces_current_session_atomically(tmp_path):
    result = _bars()
    for column in DailyBarJournal.columns:
        if column not in result:
            result[column] = False if "signal" in column or column.startswith("in_") else 0.0
    journal = DailyBarJournal(tmp_path)

    path = journal.write_current_session("QQQ", result)
    result.iloc[-1, result.columns.get_loc("close")] = 777.0
    journal.write_current_session("QQQ", result)

    saved = pd.read_csv(path)
    assert len(saved) == len(result)
    assert saved.iloc[-1]["close"] == 777.0
    assert not path.with_suffix(".csv.tmp").exists()


def test_optimizer_holds_baseline_until_enough_days():
    outcome = optimizer.optimize_symbol(_bars(days=2), "QQQ", minimum_days=5)
    assert outcome["status"] == "insufficient_data"
    assert outcome["days"] == 2


def test_shadow_alignment_rewards_continuation_direction():
    trade = optimizer.ReplayTrade(
        session="2026-07-06",
        direction="long",
        entry_ts="2026-07-06T14:05:00+00:00",
        exit_ts="2026-07-06T14:10:00+00:00",
        entry_price=500.0,
        exit_price=501.0,
        pnl_points=1.0,
    )
    events = pd.DataFrame(
        [
            {
                "ts": "2026-07-06T14:00:00+00:00",
                "setup_direction": "after_up",
                "score_bias": "continuation",
                "realized_outcome": "continuation_up",
            }
        ]
    )

    metrics = optimizer.align_trades_with_shadow([trade], events)

    assert metrics.aligned_trades == 1
    assert metrics.bias_agreement_rate == 1.0
    assert metrics.realized_direction_accuracy == 1.0
    assert metrics.reversal_conflict_rate == 0.0


def test_official_bars_are_used_before_daily_journal_override(tmp_path):
    official_dir = tmp_path / "official"
    journal_dir = tmp_path / "journal"
    official_dir.mkdir()
    journal_dir.mkdir()
    frame = _bars()
    official = frame.reset_index(names="ts")
    official.to_parquet(official_dir / "official_ohlc_QQQ_20260706.parquet")
    journal = frame.copy()
    journal.iloc[-1, journal.columns.get_loc("close")] = 999.0
    journal.index.name = "timestamp"
    journal.to_csv(journal_dir / "QQQ_20260706.csv")

    loaded = optimizer.load_symbol_bars(journal_dir, "QQQ", 20, official_data_dir=official_dir)

    assert loaded.iloc[-1]["close"] == 999.0


def test_daily_run_writes_latest_json_and_markdown(tmp_path):
    bars_dir = tmp_path / "bars"
    output_dir = tmp_path / "output"
    bars_dir.mkdir()
    day = _bars()
    for symbol in ("QQQ", "SPY"):
        frame = day.copy()
        frame.index.name = "timestamp"
        frame.to_csv(bars_dir / f"{symbol}_20260706.csv")

    payload = optimizer.run_daily_optimization(
        bar_data_dir=bars_dir,
        output_dir=output_dir,
        trade_log_dir=tmp_path,
        minimum_days=5,
        as_of=pd.Timestamp("2026-07-06 16:10", tz="America/New_York").to_pydatetime(),
    )

    assert payload["recommendations"]["QQQ"]["status"] == "insufficient_data"
    assert (output_dir / "latest_recommendations.json").exists()
    assert "每日复盘与优化" in (output_dir / "latest_review.md").read_text()


def test_trader_only_loads_recommendation_when_auto_apply_enabled(tmp_path):
    config_path = tmp_path / "recommendations.json"
    config_path.write_text(
        json.dumps(
            {
                "recommendations": {
                    "QQQ": {
                        "status": "recommended",
                        "params": {"fast_n": 10, "vote_thresh": 3, "unknown": 999},
                    }
                }
            }
        )
    )
    common = {
        "ib_port": 4002,
        "symbols": ("QQQ", "SPY"),
        "state_path": tmp_path / "state.json",
        "trade_log_path": tmp_path / "trades.csv",
        "bar_data_dir": tmp_path / "journal",
        "optimization_config_path": config_path,
    }
    disabled = Momentum0DTEPaperTrader(
        pytest.importorskip("ib_insync").IB(), auto_apply_optimization=False, **common
    )
    enabled = Momentum0DTEPaperTrader(
        pytest.importorskip("ib_insync").IB(), auto_apply_optimization=True, **common
    )

    assert disabled.indicator_configs["QQQ"].fast_n == 8
    assert enabled.indicator_configs["QQQ"].fast_n == 10
    assert enabled.indicator_configs["QQQ"].vote_thresh == 3
