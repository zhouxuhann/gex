from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

import gex_monitor.ddput_paper_trader as dp
from gex_monitor.ddput_paper_trader import Alert, DdputPaperTrader


DAY = "20260501"


def _make_trader(tmp_path: Path) -> DdputPaperTrader:
    return DdputPaperTrader(
        ib=MagicMock(),
        data_dir=tmp_path,
        symbol="QQQ",
        trade_symbol="TQQQ",
        qty=200,
        min_time=10.5,
        max_entry_time=15.0,
        dry_run=True,
        state_path=tmp_path / "state.json",
        csv_path=tmp_path / "trades.csv",
        signal_csv_path=tmp_path / "signals.csv",
    )


def _write_alert(tmp_path: Path, ts: str, z_score: float = 2.0) -> None:
    df = pd.DataFrame([{
        "ts": pd.Timestamp(ts, tz="America/New_York"),
        "direction": "+",
        "strength": "strong" if z_score >= 2.5 else "mild",
        "z_score": z_score,
        "spot": 100.0,
    }])
    df.to_parquet(tmp_path / f"signals_live_QQQ_{DAY}.parquet")


def _write_gex(
    tmp_path: Path,
    ts: str,
    *,
    spot: float = 100.0,
    flip: float = 99.0,
    positive_gamma: bool = True,
    call_wall: float = 101.0,
    put_wall: float = 98.0,
    status: str = "in_window",
    reliable: bool = True,
) -> None:
    df = pd.DataFrame([{
        "ts": pd.Timestamp(ts, tz="America/New_York"),
        "spot": spot,
        "flip": flip,
        "total_gex": 1_000_000_000.0,
        "positive_gamma": positive_gamma,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip_status": status,
        "gamma_flip_reliable": reliable,
    }])
    df.to_parquet(tmp_path / f"gex_QQQ_{DAY}.parquet")


def test_ddput_paper_accepts_signal_below_flip_when_wall_room_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00")
    _write_gex(tmp_path, "2026-05-01 11:00", flip=101.0, call_wall=102.0)

    trader = _make_trader(tmp_path)

    assert trader._next_alert() is not None
    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "ACCEPT"
    assert rows.iloc[-1]["reason"] == "ddput_gex_regime_ok"


def test_ddput_paper_ignores_unreliable_flip_for_trading(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00")
    _write_gex(
        tmp_path,
        "2026-05-01 11:00",
        flip=101.0,
        call_wall=102.0,
        status="all_negative",
        reliable=False,
    )

    trader = _make_trader(tmp_path)
    trader.require_spot_above_flip = True

    assert trader._next_alert() is not None
    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "ACCEPT"
    assert rows.iloc[-1]["reason"] == "ddput_gex_regime_ok"


def test_ddput_paper_rejects_signal_too_close_to_call_wall(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00")
    _write_gex(tmp_path, "2026-05-01 11:00", flip=99.0, call_wall=100.1)

    trader = _make_trader(tmp_path)

    assert trader._next_alert() is None
    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "REJECT"
    assert rows.iloc[-1]["reason"] == "too_close_to_call_wall"


def test_ddput_paper_accepts_signal_with_gex_regime_room(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00", z_score=2.8)
    _write_gex(tmp_path, "2026-05-01 11:00", flip=99.0, call_wall=101.0)

    trader = _make_trader(tmp_path)
    alert = trader._next_alert()

    assert alert is not None
    assert alert.z_score == 2.8
    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "ACCEPT"
    assert rows.iloc[-1]["reason"] == "ddput_gex_regime_ok"


def test_ddput_paper_emails_on_enter(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "trading_date_str", lambda: DAY)
    notifier = MagicMock()
    notifier.send_alert.return_value = True
    trader = _make_trader(tmp_path)
    trader.email_notifier = notifier
    trader._current_price = lambda: 66.12

    alert = Alert(
        ts=pd.Timestamp("2026-05-01 11:00", tz="America/New_York"),
        direction="+",
        strength="strong",
        z_score=2.8,
        spot=100.0,
    )
    trader._enter(alert)

    notifier.send_alert.assert_called_once()
    subject, body = notifier.send_alert.call_args.args
    assert "TQQQ DRY-RUN ENTER" in subject
    assert "已记录虚拟交易" in body
    assert "交易标的:     TQQQ" in body
    assert "数量:         200" in body


def test_ddput_paper_pyramids_repeated_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "trading_date_str", lambda: DAY)
    trader = _make_trader(tmp_path)
    prices = iter([66.0, 66.0, 68.0, 68.0])
    trader._current_price = lambda: next(prices)
    alert1 = Alert(
        ts=pd.Timestamp("2026-05-01 11:00", tz="America/New_York"),
        direction="+",
        strength="mild",
        z_score=2.0,
        spot=100.0,
    )
    alert2 = Alert(
        ts=pd.Timestamp("2026-05-01 11:05", tz="America/New_York"),
        direction="+",
        strength="strong",
        z_score=2.8,
        spot=101.0,
    )

    trader._enter(alert1)
    trader._enter(alert2)

    assert trader.position.in_position is True
    assert trader.position.qty == 400
    assert trader.position.entry_price == 67.0
    rows = pd.read_csv(tmp_path / "trades.csv")
    assert list(rows["event"]) == ["ENTER", "ENTER"]
    assert list(rows["qty"]) == [200, 200]


def test_ddput_paper_respects_max_position_qty(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "trading_date_str", lambda: DAY)
    trader = _make_trader(tmp_path)
    trader.max_position_qty = 350
    prices = iter([66.0, 66.0, 67.0, 67.0])
    trader._current_price = lambda: next(prices)
    alert = Alert(
        ts=pd.Timestamp("2026-05-01 11:00", tz="America/New_York"),
        direction="+",
        strength="mild",
        z_score=2.0,
        spot=100.0,
    )

    trader._enter(alert)
    trader._enter(alert)
    trader._enter(alert)

    assert trader.position.qty == 350
    rows = pd.read_csv(tmp_path / "trades.csv")
    assert list(rows["qty"]) == [200, 150]


def test_account_gate_paper_requires_du_account(tmp_path):
    trader = _make_trader(tmp_path)
    trader.ib.managedAccounts.return_value = ["U1234567"]

    assert trader.verify_account() is False


def test_account_gate_live_accepts_non_du_account(tmp_path):
    trader = _make_trader(tmp_path)
    trader.account_mode = "live"
    trader.ib.managedAccounts.return_value = ["U1234567"]

    assert trader.verify_account() is True
