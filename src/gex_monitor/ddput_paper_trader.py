"""
ddput paper trader.

Consumes live ddput alert parquet files produced by GEX Monitor and submits
long-only stock orders to an IB Paper account. The trader can pyramid repeated
signals into one aggregate stock position, with a paper-account gate, fixed
hold, stop loss, EOD flatten, and CSV audit trail.

Usage:
    python -m gex_monitor.ddput_paper_trader --qty 10 --port 4002
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, Stock

from .email_notifier import EmailConfig, EmailNotifier
from .time_utils import ET, et_now, is_market_open, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_LOG_DIR = ROOT_DIR / "logs"

log = logging.getLogger("ddput_paper_trader")


@dataclass
class PositionState:
    in_position: bool = False
    qty: int = 0
    entry_ts: str | None = None
    entry_price: float | None = None
    entry_signal_ts: str | None = None
    entry_z: float | None = None
    entry_strength: str | None = None
    hold_until_ts: str | None = None
    trade_symbol: str = "QQQ"


@dataclass
class Alert:
    ts: pd.Timestamp
    direction: str
    strength: str
    z_score: float
    spot: float | None


@dataclass
class GexContext:
    ts: pd.Timestamp
    spot: float
    flip: float | None
    total_gex: float | None
    positive_gamma: bool
    call_wall: float | None
    put_wall: float | None
    gamma_flip_status: str | None
    gamma_flip_reliable: bool
    age_sec: float = 0.0

    @property
    def spot_minus_flip(self) -> float | None:
        if self.flip is None:
            return None
        return self.spot - self.flip

    @property
    def call_wall_room_pct(self) -> float | None:
        if self.call_wall is None or self.spot <= 0:
            return None
        return (self.call_wall - self.spot) / self.spot * 100.0

    @property
    def put_wall_room_pct(self) -> float | None:
        if self.put_wall is None or self.spot <= 0:
            return None
        return (self.spot - self.put_wall) / self.spot * 100.0


@dataclass
class SignalDecision:
    allowed: bool
    reason: str
    context: GexContext | None


class CSVTradeLogger:
    fields = [
        "ts", "event", "symbol", "side", "qty", "signal_ts", "strength",
        "z_score", "price", "fill_price", "pnl_bps", "reason", "dry_run",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def write(self, **row) -> None:
        out = {k: row.get(k, "") for k in self.fields}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(out)


class CSVSignalLogger:
    fields = [
        "ts", "event", "symbol", "trade_symbol", "signal_ts", "direction",
        "strength", "z_score", "signal_spot", "decision", "reason",
        "gex_ts", "gex_age_sec", "gex_spot", "flip", "spot_minus_flip",
        "total_gex", "positive_gamma", "call_wall", "put_wall",
        "call_wall_room_pct", "put_wall_room_pct", "gamma_flip_status",
        "gamma_flip_reliable", "dry_run",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def write(
        self,
        *,
        alert: Alert,
        symbol: str,
        trade_symbol: str,
        decision: SignalDecision,
        dry_run: bool,
        event: str = "SIGNAL",
    ) -> None:
        ctx = decision.context
        row = {
            "ts": et_now().isoformat(),
            "event": event,
            "symbol": symbol,
            "trade_symbol": trade_symbol,
            "signal_ts": alert.ts.isoformat(),
            "direction": alert.direction,
            "strength": alert.strength,
            "z_score": alert.z_score,
            "signal_spot": alert.spot,
            "decision": "ACCEPT" if decision.allowed else "REJECT",
            "reason": decision.reason,
            "dry_run": dry_run,
        }
        if ctx is not None:
            row.update({
                "gex_ts": ctx.ts.isoformat(),
                "gex_age_sec": ctx.age_sec,
                "gex_spot": ctx.spot,
                "flip": ctx.flip,
                "spot_minus_flip": ctx.spot_minus_flip,
                "total_gex": ctx.total_gex,
                "positive_gamma": ctx.positive_gamma,
                "call_wall": ctx.call_wall,
                "put_wall": ctx.put_wall,
                "call_wall_room_pct": ctx.call_wall_room_pct,
                "put_wall_room_pct": ctx.put_wall_room_pct,
                "gamma_flip_status": ctx.gamma_flip_status,
                "gamma_flip_reliable": ctx.gamma_flip_reliable,
            })
        out = {k: row.get(k, "") for k in self.fields}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(out)


class DdputPaperTrader:
    def __init__(
        self,
        ib: IB,
        data_dir: Path,
        symbol: str = "QQQ",
        trade_symbol: str = "QQQ",
        qty: int = 10,
        max_position_qty: int = 0,
        hold_min: int = 20,
        stop_bps: float = 8.0,
        min_z: float = 1.5,
        min_time: float = 10.5,
        max_entry_time: float = 15.0,
        flatten_time: str = "15:55",
        poll_sec: float = 5.0,
        dry_run: bool = False,
        account_mode: str = "paper",
        require_gex_filter: bool = True,
        require_spot_above_flip: bool = False,
        min_call_wall_room_pct: float = 0.15,
        min_put_wall_room_pct: float = 0.0,
        max_gex_age_sec: float = 120.0,
        exit_on_regime_break: bool = True,
        exit_on_flip_break: bool = False,
        state_path: Path | None = None,
        csv_path: Path | None = None,
        signal_csv_path: Path | None = None,
        email_notifier: EmailNotifier | None = None,
    ):
        self.ib = ib
        self.data_dir = data_dir
        self.symbol = symbol
        self.trade_symbol = trade_symbol
        self.qty = qty
        self.max_position_qty = max(0, max_position_qty)
        self.hold_min = hold_min
        self.stop_bps = stop_bps
        self.min_z = min_z
        self.min_time = min_time
        self.max_entry_time = max_entry_time
        self.flatten_time = self._parse_hhmm(flatten_time)
        self.poll_sec = poll_sec
        self.dry_run = dry_run
        self.account_mode = account_mode
        self.require_gex_filter = require_gex_filter
        self.require_spot_above_flip = require_spot_above_flip
        self.min_call_wall_room_pct = min_call_wall_room_pct
        self.min_put_wall_room_pct = min_put_wall_room_pct
        self.max_gex_age_sec = max_gex_age_sec
        self.exit_on_regime_break = exit_on_regime_break
        self.exit_on_flip_break = exit_on_flip_break
        self.state_path = state_path or DEFAULT_LOG_DIR / f"ddput_paper_state_{trade_symbol}.json"
        self.csv_logger = CSVTradeLogger(
            csv_path or DEFAULT_LOG_DIR / f"ddput_paper_trades_{trading_date_str()}.csv"
        )
        self.signal_logger = CSVSignalLogger(
            signal_csv_path or DEFAULT_LOG_DIR / f"ddput_paper_signals_{trading_date_str()}.csv"
        )
        self.email_notifier = email_notifier

        self.contract = Stock(trade_symbol, "SMART", "USD")
        self.ticker = None
        self.position = self._load_state()
        self.last_signal_ts: pd.Timestamp | None = self._load_last_signal_ts()
        self._running = True

    @staticmethod
    def _parse_hhmm(value: str) -> dtime:
        hh, mm = value.split(":")
        return dtime(int(hh), int(mm))

    def _load_last_signal_ts(self) -> pd.Timestamp | None:
        raw = None
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text()).get("last_signal_ts")
            except Exception:
                raw = None
        if not raw:
            return None
        ts = pd.Timestamp(raw)
        return ts.tz_convert(ET) if ts.tz is not None else ts.tz_localize(ET)

    def _load_state(self) -> PositionState:
        if not self.state_path.exists():
            return PositionState(trade_symbol=self.trade_symbol)
        try:
            data = json.loads(self.state_path.read_text())
            pos = PositionState(**data.get("position", {}))
            pos.trade_symbol = self.trade_symbol
            return pos
        except Exception as e:
            log.warning("State load failed, starting flat: %s", e)
            return PositionState(trade_symbol=self.trade_symbol)

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "position": asdict(self.position),
            "open_qty": self._open_qty(),
            "max_position_qty": self.max_position_qty,
            "last_signal_ts": self.last_signal_ts.isoformat() if self.last_signal_ts is not None else None,
            "updated": et_now().isoformat(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def stop(self) -> None:
        self._running = False

    def verify_account(self) -> bool:
        accounts = self.ib.managedAccounts()
        if not accounts:
            log.error("No IB accounts found")
            return False
        paper_accounts = [str(a) for a in accounts if str(a).startswith("DU")]
        live_accounts = [str(a) for a in accounts if not str(a).startswith("DU")]
        if self.account_mode == "paper" and paper_accounts:
            log.info("Paper account verified: %s", accounts)
            return True
        if self.account_mode == "live" and live_accounts:
            log.warning("LIVE account mode verified: accounts=%s", accounts)
            return True
        if self.account_mode == "any":
            log.warning("Account mode 'any' accepted: accounts=%s", accounts)
            return True
        log.error("BLOCKED: account_mode=%s Accounts=%s", self.account_mode, accounts)
        return False

    def _ib_position_qty(self) -> float:
        try:
            positions = self.ib.positions()
        except Exception as e:
            log.warning("IB positions query failed: %s", e)
            return 0.0
        for p in positions:
            if p.contract.symbol == self.trade_symbol and p.contract.secType == "STK":
                return float(p.position)
        return 0.0

    def _sync_position_or_halt(self) -> bool:
        if self.dry_run:
            return True
        ib_qty = self._ib_position_qty()
        expected = self.position.qty if self.position.in_position else 0
        if abs(ib_qty - expected) < 0.5:
            log.info("Position sync OK: IB=%s local=%s", ib_qty, expected)
            return True
        log.error(
            "Position mismatch. IB=%s local=%s. Refusing to start; flatten or fix state first.",
            ib_qty, expected,
        )
        return False

    def start(self) -> None:
        self.ib.qualifyContracts(self.contract)
        self.ticker = self.ib.reqMktData(self.contract, "", False, False)
        if not self.dry_run and not self.verify_account():
            return
        if not self._sync_position_or_halt():
            return

        log.info(
            "Started ddput paper trader: signal=%s trade=%s qty=%s max_position_qty=%s hold=%smin stop=%.1fbps "
            "gex_filter=%s call_room>=%.3f%% put_room>=%.3f%% "
            "exit_on_regime_break=%s dry_run=%s",
            self.symbol, self.trade_symbol, self.qty, self.max_position_qty or "unlimited",
            self.hold_min, self.stop_bps,
            self.require_gex_filter, self.min_call_wall_room_pct,
            self.min_put_wall_room_pct, self.exit_on_regime_break, self.dry_run,
        )

        while self._running:
            try:
                self._tick()
            except Exception:
                log.exception("Loop error")
            self.ib.sleep(self.poll_sec)

    def _tick(self) -> None:
        now = et_now()
        if self.position.in_position:
            self._maybe_exit(now)

        if not is_market_open(now):
            return
        if not self._in_entry_window(now):
            self._consume_alerts_without_entry()
            return
        if self._available_qty() <= 0:
            self._consume_alerts_without_entry(reason="max_position_qty_reached")
            return

        alert = self._next_alert()
        if alert is None:
            return
        self._enter(alert)

    def _in_entry_window(self, now: datetime) -> bool:
        hour = now.hour + now.minute / 60.0
        return self.min_time <= hour < self.max_entry_time

    def _today_alert_path(self) -> Path:
        return self.data_dir / f"signals_live_{self.symbol}_{trading_date_str()}.parquet"

    def _today_gex_path(self) -> Path:
        return self.data_dir / f"gex_{self.symbol}_{trading_date_str()}.parquet"

    def _read_alerts(self) -> pd.DataFrame:
        path = self._today_alert_path()
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
            return pd.DataFrame()
        if df.empty or "ts" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize(ET)
        else:
            df["ts"] = df["ts"].dt.tz_convert(ET)
        return df.sort_values("ts")

    def _eligible_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df[(df["direction"] == "+") & (df["z_score"] >= self.min_z)].copy()
        if self.last_signal_ts is not None:
            out = out[out["ts"] > self.last_signal_ts]
        if out.empty:
            return out
        hour = out["ts"].dt.hour + out["ts"].dt.minute / 60.0
        return out[(hour >= self.min_time) & (hour < self.max_entry_time)]

    def _consume_alerts_without_entry(self, reason: str = "not_in_entry_state") -> None:
        df = self._read_alerts()
        if df.empty:
            return
        df = self._eligible_alerts(df)
        if df.empty:
            return
        for _, row in df.iterrows():
            alert = self._alert_from_row(row)
            self.signal_logger.write(
                alert=alert,
                symbol=self.symbol,
                trade_symbol=self.trade_symbol,
                decision=SignalDecision(False, reason, self._gex_context_for_alert(alert.ts)),
                dry_run=self.dry_run,
            )
        newest = df["ts"].max()
        if self.last_signal_ts is None or newest > self.last_signal_ts:
            self.last_signal_ts = newest
            self._save_state()

    def _next_alert(self) -> Alert | None:
        df = self._eligible_alerts(self._read_alerts())
        if df.empty:
            return None
        accepted: Alert | None = None
        newest = self.last_signal_ts
        for _, row in df.iterrows():
            alert = self._alert_from_row(row)
            decision = self._evaluate_alert(alert)
            self.signal_logger.write(
                alert=alert,
                symbol=self.symbol,
                trade_symbol=self.trade_symbol,
                decision=decision,
                dry_run=self.dry_run,
            )
            newest = alert.ts if newest is None or alert.ts > newest else newest
            if decision.allowed and accepted is None:
                accepted = alert
                break
            log.info(
                "Rejected ddput signal %s z=%.2f strength=%s reason=%s",
                alert.ts.strftime("%H:%M"), alert.z_score, alert.strength, decision.reason,
            )
        if newest is not None:
            self.last_signal_ts = newest
            self._save_state()
        return accepted

    def _alert_from_row(self, row: pd.Series) -> Alert:
        return Alert(
            ts=row["ts"],
            direction=str(row["direction"]),
            strength=str(row.get("strength", "")),
            z_score=float(row["z_score"]),
            spot=float(row["spot"]) if "spot" in row and pd.notna(row["spot"]) else None,
        )

    def _read_gex(self) -> pd.DataFrame:
        path = self._today_gex_path()
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
            return pd.DataFrame()
        if df.empty or "ts" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize(ET)
        else:
            df["ts"] = df["ts"].dt.tz_convert(ET)
        return df.sort_values("ts")

    @staticmethod
    def _float_or_none(value) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None

    def _context_from_row(self, row: pd.Series, age_sec: float = 0.0) -> GexContext:
        flip = self._float_or_none(row.get("flip"))
        if flip is None:
            flip = self._float_or_none(row.get("gamma_flip_flow"))
        return GexContext(
            ts=row["ts"],
            spot=float(row["spot"]),
            flip=flip,
            total_gex=self._float_or_none(row.get("total_gex")),
            positive_gamma=bool(row.get("positive_gamma", False)),
            call_wall=self._float_or_none(row.get("call_wall")),
            put_wall=self._float_or_none(row.get("put_wall")),
            gamma_flip_status=str(row.get("gamma_flip_status")) if pd.notna(row.get("gamma_flip_status")) else None,
            gamma_flip_reliable=bool(row.get("gamma_flip_reliable", False)),
            age_sec=age_sec,
        )

    def _gex_context_for_alert(self, alert_ts: pd.Timestamp) -> GexContext | None:
        df = self._read_gex()
        if df.empty:
            return None
        idx = (df["ts"] - alert_ts).abs().idxmin()
        row = df.loc[idx]
        age_sec = abs((row["ts"] - alert_ts).total_seconds())
        return self._context_from_row(row, age_sec=age_sec)

    def _latest_gex_context(self) -> GexContext | None:
        df = self._read_gex()
        if df.empty:
            return None
        row = df.iloc[-1]
        age_sec = abs((et_now() - row["ts"]).total_seconds())
        return self._context_from_row(row, age_sec=age_sec)

    def _evaluate_alert(self, alert: Alert) -> SignalDecision:
        if not self.require_gex_filter:
            return SignalDecision(True, "ddput_only", None)
        ctx = self._gex_context_for_alert(alert.ts)
        if ctx is None:
            return SignalDecision(False, "missing_gex_context", None)
        if ctx.age_sec > self.max_gex_age_sec:
            return SignalDecision(False, f"stale_gex_context:{ctx.age_sec:.0f}s", ctx)
        if not ctx.positive_gamma:
            return SignalDecision(False, "not_positive_gamma", ctx)
        if ctx.call_wall is None:
            return SignalDecision(False, "missing_call_wall", ctx)
        if ctx.call_wall_room_pct is None or ctx.call_wall_room_pct < self.min_call_wall_room_pct:
            return SignalDecision(False, "too_close_to_call_wall", ctx)
        if ctx.put_wall is None:
            return SignalDecision(False, "missing_put_wall", ctx)
        if ctx.put_wall_room_pct is None or ctx.put_wall_room_pct < self.min_put_wall_room_pct:
            return SignalDecision(False, "below_or_too_close_to_put_wall", ctx)
        return SignalDecision(True, "ddput_gex_regime_ok", ctx)

    def _current_price(self) -> float | None:
        if self.ticker is None:
            return None
        price = self.ticker.marketPrice()
        if price is None or not np.isfinite(price) or price <= 0:
            price = self.ticker.last or self.ticker.close
        if price is None or not np.isfinite(price) or price <= 0:
            return None
        return float(price)

    def _open_qty(self) -> int:
        return int(self.position.qty) if self.position.in_position else 0

    def _available_qty(self) -> int:
        if self.max_position_qty <= 0:
            return self.qty
        return max(0, self.max_position_qty - self._open_qty())

    def _make_order(self, side: str, qty: int | None = None) -> MarketOrder:
        order_qty = self.qty if qty is None else qty
        order = MarketOrder(side, order_qty)
        order.tif = "IOC"
        order.outsideRth = False
        return order

    def _wait_fill(self, trade, timeout: float = 30.0) -> float | None:
        start = time.time()
        while time.time() - start < timeout:
            self.ib.sleep(0.25)
            if trade.isDone() or trade.orderStatus.status == "Filled":
                break
        total_filled = sum(f.execution.shares for f in trade.fills)
        if total_filled <= 0:
            return None
        avg = trade.orderStatus.avgFillPrice
        if avg and avg > 0:
            return float(avg)
        value = sum(f.execution.shares * f.execution.price for f in trade.fills)
        return float(value / total_filled) if total_filled else None

    def _submit(self, side: str, qty: int | None = None) -> float | None:
        order_qty = self.qty if qty is None else qty
        price = self._current_price()
        if self.dry_run:
            log.info("[DRY RUN] %s %s %s @ %s", side, order_qty, self.trade_symbol, price)
            return price
        order = self._make_order(side, order_qty)
        trade = self.ib.placeOrder(self.contract, order)
        fill = self._wait_fill(trade)
        if fill is None:
            log.error("%s order not filled; cancelling", side)
            try:
                self.ib.cancelOrder(order)
            except Exception:
                pass
        return fill

    def _enter(self, alert: Alert) -> None:
        entry_qty = min(self.qty, self._available_qty())
        if entry_qty <= 0:
            log.info("Skip %s entry: max_position_qty reached (%s/%s)", self.trade_symbol, self._open_qty(), self.max_position_qty)
            return
        price = self._current_price()
        fill = self._submit("BUY", entry_qty)
        if fill is None:
            return
        now = et_now()
        hold_until = now + pd.Timedelta(minutes=self.hold_min)
        if self.position.in_position and self.position.entry_price is not None:
            old_qty = int(self.position.qty)
            new_qty = old_qty + entry_qty
            avg_entry = ((float(self.position.entry_price) * old_qty) + (fill * entry_qty)) / new_qty
            old_hold_until = pd.Timestamp(self.position.hold_until_ts) if self.position.hold_until_ts else hold_until
            if old_hold_until.tz is None:
                old_hold_until = old_hold_until.tz_localize(ET)
            else:
                old_hold_until = old_hold_until.tz_convert(ET)
            self.position.qty = new_qty
            self.position.entry_price = avg_entry
            self.position.hold_until_ts = max(old_hold_until, hold_until).isoformat()
            self.position.entry_signal_ts = alert.ts.isoformat()
            self.position.entry_z = alert.z_score
            self.position.entry_strength = alert.strength
        else:
            self.position = PositionState(
                in_position=True,
                qty=entry_qty,
                entry_ts=now.isoformat(),
                entry_price=fill,
                entry_signal_ts=alert.ts.isoformat(),
                entry_z=alert.z_score,
                entry_strength=alert.strength,
                hold_until_ts=hold_until.isoformat(),
                trade_symbol=self.trade_symbol,
            )
        self._save_state()
        log.info(
            "ENTER long %s qty=%s fill=%.2f open_qty=%s signal=%s z=%.2f strength=%s",
            self.trade_symbol, entry_qty, fill, self._open_qty(), alert.ts.strftime("%H:%M"),
            alert.z_score, alert.strength,
        )
        self.csv_logger.write(
            ts=now.isoformat(), event="ENTER", symbol=self.trade_symbol, side="BUY",
            qty=entry_qty, signal_ts=alert.ts.isoformat(), strength=alert.strength,
            z_score=alert.z_score, price=price, fill_price=fill, reason="ddput+",
            dry_run=self.dry_run,
        )
        self._send_trade_email(
            event="ENTER",
            side="BUY",
            qty=entry_qty,
            fill=fill,
            signal_ts=alert.ts,
            strength=alert.strength,
            z_score=alert.z_score,
            reason="ddput_gex_regime_ok",
        )

    def _maybe_exit(self, now: datetime) -> None:
        price = self._current_price()
        if price is None or self.position.entry_price is None:
            return
        reason = None
        entry = float(self.position.entry_price)
        ret_bps = (price / entry - 1.0) * 10000
        hold_until = pd.Timestamp(self.position.hold_until_ts)
        if hold_until.tz is None:
            hold_until = hold_until.tz_localize(ET)
        else:
            hold_until = hold_until.tz_convert(ET)

        regime_reason = self._regime_exit_reason()
        if self.exit_on_regime_break and regime_reason is not None:
            reason = regime_reason
        elif ret_bps <= -self.stop_bps:
            reason = "stop"
        elif now >= hold_until:
            reason = "time_exit"
        elif now.time() >= self.flatten_time:
            reason = "eod_flatten"

        if reason:
            self._exit(reason, price, ret_bps)

    def _regime_exit_reason(self) -> str | None:
        ctx = self._latest_gex_context()
        if ctx is None or ctx.age_sec > self.max_gex_age_sec:
            return None
        if not ctx.positive_gamma:
            return "gex_exit:not_positive_gamma"
        if ctx.put_wall is not None and ctx.spot <= ctx.put_wall:
            return "gex_exit:below_put_wall"
        return None

    def _exit(self, reason: str, mark_price: float, mark_ret_bps: float) -> None:
        exit_qty = self._open_qty()
        fill = self._submit("SELL", exit_qty)
        if fill is None:
            return
        entry = float(self.position.entry_price or fill)
        pnl_bps = (fill / entry - 1.0) * 10000
        now = et_now()
        log.info(
            "EXIT long %s qty=%s fill=%.2f pnl=%.2fbps reason=%s",
            self.trade_symbol, exit_qty, fill, pnl_bps, reason,
        )
        self.csv_logger.write(
            ts=now.isoformat(), event="EXIT", symbol=self.trade_symbol, side="SELL",
            qty=exit_qty, signal_ts=self.position.entry_signal_ts,
            strength=self.position.entry_strength, z_score=self.position.entry_z,
            price=mark_price, fill_price=fill, pnl_bps=pnl_bps,
            reason=reason, dry_run=self.dry_run,
        )
        self._send_trade_email(
            event="EXIT",
            side="SELL",
            qty=exit_qty,
            fill=fill,
            signal_ts=self.position.entry_signal_ts,
            strength=self.position.entry_strength,
            z_score=self.position.entry_z,
            reason=reason,
            pnl_bps=pnl_bps,
        )
        self.position = PositionState(trade_symbol=self.trade_symbol)
        self._save_state()

    def _send_trade_email(
        self,
        *,
        event: str,
        side: str,
        qty: int,
        fill: float,
        signal_ts,
        strength: str | None,
        z_score: float | None,
        reason: str,
        pnl_bps: float | None = None,
    ) -> None:
        if self.email_notifier is None:
            return
        ts = et_now().strftime("%Y-%m-%d %H:%M:%S ET")
        signal_text = ""
        if signal_ts:
            signal = pd.Timestamp(signal_ts)
            signal = signal.tz_convert(ET) if signal.tz is not None else signal.tz_localize(ET)
            signal_text = signal.strftime("%Y-%m-%d %H:%M:%S ET")
        pnl_line = f"PnL:          {pnl_bps:+.2f} bps\n" if pnl_bps is not None else ""
        if self.dry_run:
            mode_text = "DRY RUN"
        elif self.account_mode == "live":
            mode_text = "IB Live"
        else:
            mode_text = "IB Paper"
        dry_line = f"模式:         {mode_text}\n"
        subject_prefix = "DRY-RUN" if self.dry_run else self.account_mode.upper()
        action_text = "已记录虚拟交易" if self.dry_run else "已执行真实订单"
        subject = f"{self.trade_symbol} {subject_prefix} {event} {side} {qty} @ {fill:.2f}"
        position_cap = str(self.max_position_qty) if self.max_position_qty > 0 else "unlimited"
        body = (
            f"ddput trader {action_text}: {event}\n\n"
            f"交易标的:     {self.trade_symbol}\n"
            f"信号标的:     {self.symbol}\n"
            f"方向:         {side}\n"
            f"数量:         {qty}\n"
            f"当前总持仓:   {self._open_qty()} / {position_cap}\n"
            f"成交价:       {fill:.2f}\n"
            f"{pnl_line}"
            f"原因:         {reason}\n"
            f"信号时间:     {signal_text}\n"
            f"信号强度:     {strength or ''}\n"
            f"Z-score:      {z_score if z_score is not None else ''}\n"
            f"通知时间:     {ts}\n"
            f"{dry_line}"
        )
        sent = self.email_notifier.send_alert(subject, body)
        if not sent:
            log.warning("Trade email not sent: event=%s side=%s symbol=%s", event, side, self.trade_symbol)


def configure_logging() -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DEFAULT_LOG_DIR / f"ddput_paper_{trading_date_str()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    log.info("Log file: %s", log_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ddput long-only IB Paper trader")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002, help="IB Gateway paper port, usually 4002")
    p.add_argument("--client-id", type=int, default=76)
    p.add_argument("--symbol", default="QQQ", help="Signal symbol, reads signals_live_SYMBOL_DATE.parquet")
    p.add_argument("--trade-symbol", default="QQQ", help="Stock symbol to trade")
    p.add_argument("--qty", type=int, default=10)
    p.add_argument("--max-position-qty", type=int, default=0, help="Max aggregate stock position; 0 means unlimited")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--hold-min", type=int, default=20)
    p.add_argument("--stop-bps", type=float, default=8.0)
    p.add_argument("--min-z", type=float, default=1.5)
    p.add_argument("--min-time", type=float, default=10.5, help="ET decimal hour, default 10.5 = 10:30")
    p.add_argument("--max-entry-time", type=float, default=15.0, help="ET decimal hour, default 15.0")
    p.add_argument("--flatten-time", default="15:55")
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true", help="Do not submit orders")
    p.add_argument("--account-mode", choices=["paper", "live", "any"], default="paper")
    p.add_argument("--ib-readonly", action="store_true", help="Connect IB API in readonly mode")
    p.add_argument("--disable-gex-filter", action="store_true", help="Trade ddput signals without GEX regime filters")
    p.add_argument("--require-spot-above-flip", action="store_true", help="Deprecated; flip is logged only and ignored for trading")
    p.add_argument("--min-call-wall-room-pct", type=float, default=0.15)
    p.add_argument("--min-put-wall-room-pct", type=float, default=0.0)
    p.add_argument("--max-gex-age-sec", type=float, default=120.0)
    p.add_argument("--disable-regime-exit", action="store_true", help="Do not exit on flip/put wall/gamma regime breaks")
    p.add_argument("--exit-on-flip-break", action="store_true", help="Deprecated; flip is logged only and ignored for trading")
    p.add_argument("--email-enabled", action="store_true", help="Send email on paper trade enter/exit")
    p.add_argument("--email-sender", default="fzhouxu615@gmail.com")
    p.add_argument("--email-password-env", default="GMAIL_APP_PASSWORD")
    p.add_argument("--email-recipients", default="", help="Comma-separated recipients")
    p.add_argument("--email-subject-prefix", default="[GEX]")
    return p.parse_args()


def _email_notifier_from_args(args: argparse.Namespace) -> EmailNotifier | None:
    recipients = [r.strip() for r in str(args.email_recipients).split(",") if r.strip()]
    if not args.email_enabled:
        return None
    return EmailNotifier(EmailConfig(
        enabled=True,
        sender=args.email_sender,
        password_env=args.email_password_env,
        recipients=recipients,
        only_strong=False,
        cooldown_sec=0,
        subject_prefix=args.email_subject_prefix,
    ))


def main() -> None:
    args = parse_args()
    configure_logging()
    ib = IB()
    trader = None

    def _handle_signal(*_):
        if trader is not None:
            trader.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        readonly = args.ib_readonly or args.dry_run
        log.info(
            "Connecting IB %s:%s clientId=%s readonly=%s",
            args.host, args.port, args.client_id, readonly,
        )
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=20, readonly=readonly)
        log.info("Connected. Accounts=%s", ib.managedAccounts())
        trader = DdputPaperTrader(
            ib=ib,
            data_dir=args.data_dir,
            symbol=args.symbol,
            trade_symbol=args.trade_symbol,
            qty=args.qty,
            max_position_qty=args.max_position_qty,
            hold_min=args.hold_min,
            stop_bps=args.stop_bps,
            min_z=args.min_z,
            min_time=args.min_time,
            max_entry_time=args.max_entry_time,
            flatten_time=args.flatten_time,
            poll_sec=args.poll_sec,
            dry_run=args.dry_run,
            account_mode=args.account_mode,
            require_gex_filter=not args.disable_gex_filter,
            require_spot_above_flip=args.require_spot_above_flip,
            min_call_wall_room_pct=args.min_call_wall_room_pct,
            min_put_wall_room_pct=args.min_put_wall_room_pct,
            max_gex_age_sec=args.max_gex_age_sec,
            exit_on_regime_break=not args.disable_regime_exit,
            exit_on_flip_break=args.exit_on_flip_break,
            email_notifier=_email_notifier_from_args(args),
        )
        trader.start()
    finally:
        if ib.isConnected():
            ib.disconnect()
        log.info("Stopped")


if __name__ == "__main__":
    main()
