from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

import gex_monitor.qqq_0dte_option_scalper as sc
from gex_monitor.qqq_0dte_option_scalper import QQQ0DTEOptionScalper


DAY = "20260501"


def _make_scalper(tmp_path: Path) -> QQQ0DTEOptionScalper:
    return QQQ0DTEOptionScalper(
        ib=MagicMock(),
        data_dir=tmp_path,
        qty=1,
        min_z=1.5,
        dry_run=True,
        state_path=tmp_path / "state.json",
        trade_csv_path=tmp_path / "trades.csv",
        signal_csv_path=tmp_path / "signals.csv",
    )


def _write_alert(tmp_path: Path, ts: str, z_score: float = 2.8) -> None:
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


def test_0dte_scalper_accepts_strong_ddput_with_good_gex(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00", z_score=2.8)
    _write_gex(
        tmp_path,
        "2026-05-01 11:00",
        call_wall=101.0,
        status="all_negative",
        reliable=False,
    )

    scalper = _make_scalper(tmp_path)
    alert = scalper._next_alert()

    assert alert is not None
    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "ACCEPT"
    assert rows.iloc[-1]["reason"] == "ddput_gex_0dte_call_ok"


def test_0dte_scalper_rejects_negative_gamma(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00", z_score=2.8)
    _write_gex(tmp_path, "2026-05-01 11:00", positive_gamma=False)

    scalper = _make_scalper(tmp_path)
    assert scalper._next_alert() is None

    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "REJECT"
    assert rows.iloc[-1]["reason"] == "not_positive_gamma"


def test_0dte_scalper_rejects_too_close_to_call_wall(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00", z_score=2.8)
    _write_gex(tmp_path, "2026-05-01 11:00", call_wall=100.1)

    scalper = _make_scalper(tmp_path)
    assert scalper._next_alert() is None

    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "REJECT"
    assert rows.iloc[-1]["reason"] == "too_close_to_call_wall"


def test_0dte_scalper_accepts_mild_signal_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "trading_date_str", lambda: DAY)
    _write_alert(tmp_path, "2026-05-01 11:00", z_score=1.8)
    _write_gex(tmp_path, "2026-05-01 11:00")

    scalper = _make_scalper(tmp_path)

    assert scalper._next_alert() is not None
    rows = pd.read_csv(tmp_path / "signals.csv")
    assert rows.iloc[-1]["decision"] == "ACCEPT"
