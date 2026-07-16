"""QQQ 0DTE ATM option paper scalper.

Consumes live QQQ ddput alerts, checks the latest GEX regime, and trades ATM
same-day QQQ calls in an IB Paper account. This is intentionally paper-first
and audit-heavy: every accepted/rejected signal and every order is written to
CSV.
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
from ib_insync import IB, LimitOrder, Option, Stock

from .email_notifier import EmailConfig, EmailNotifier
from .time_utils import ET, et_now, is_market_open, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_LOG_DIR = ROOT_DIR / "logs"

log = logging.getLogger("qqq_0dte_option_scalper")


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
        return None if self.flip is None else self.spot - self.flip

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


@dataclass
class OptionQuote:
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last if self.last is not None and self.last > 0 else self.close

    @property
    def spread_pct(self) -> float | None:
        mid = self.mid
        if mid is None or mid <= 0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / mid * 100.0


@dataclass
class PositionState:
    in_position: bool = False
    qty: int = 0
    entry_ts: str | None = None
    entry_price: float | None = None
    entry_underlying: float | None = None
    entry_signal_ts: str | None = None
    entry_z: float | None = None
    entry_strength: str | None = None
    hold_until_ts: str | None = None
    con_id: int | None = None
    local_symbol: str | None = None
    expiry: str | None = None
    strike: float | None = None
    right: str = "C"


class CSVSignalLogger:
    fields = [
        "ts", "event", "symbol", "right", "signal_ts", "direction", "strength",
        "z_score", "signal_spot", "decision", "reason", "gex_ts", "gex_age_sec",
        "gex_spot", "flip", "spot_minus_flip", "total_gex", "positive_gamma",
        "call_wall", "put_wall", "call_wall_room_pct", "put_wall_room_pct",
        "gamma_flip_status", "gamma_flip_reliable", "dry_run",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def write(self, *, alert: Alert, symbol: str, right: str, decision: SignalDecision, dry_run: bool) -> None:
        ctx = decision.context
        row = {
            "ts": et_now().isoformat(),
            "event": "SIGNAL",
            "symbol": symbol,
            "right": right,
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
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow({k: row.get(k, "") for k in self.fields})


class CSVTradeLogger:
    fields = [
        "ts", "event", "symbol", "local_symbol", "right", "expiry", "strike",
        "side", "qty", "signal_ts", "strength", "z_score", "underlying",
        "bid", "ask", "mid", "limit_price", "fill_price", "pnl_pct",
        "reason", "dry_run",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def write(self, **row) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow({k: row.get(k, "") for k in self.fields})


class QQQ0DTEOptionScalper:
    def __init__(
        self,
        ib: IB,
        data_dir: Path,
        symbol: str = "QQQ",
        right: str = "C",
        qty: int = 1,
        max_position_qty: int = 1,
        hold_min: int = 10,
        stop_pct: float = 20.0,
        take_profit_pct: float = 25.0,
        min_z: float = 1.5,
        min_time: float = 10.5,
        max_entry_time: float = 14.5,
        flatten_time: str = "15:45",
        poll_sec: float = 5.0,
        dry_run: bool = False,
        max_gex_age_sec: float = 120.0,
        min_call_wall_room_pct: float = 0.15,
        min_put_wall_room_pct: float = 0.0,
        max_spread_pct: float = 8.0,
        exit_on_regime_break: bool = True,
        exit_on_flip_break: bool = False,
        account_mode: str = "paper",
        state_path: Path | None = None,
        trade_csv_path: Path | None = None,
        signal_csv_path: Path | None = None,
        email_notifier: EmailNotifier | None = None,
    ):
        self.ib = ib
        self.data_dir = Path(data_dir)
        self.symbol = symbol.upper()
        self.right = right.upper()
        self.qty = qty
        self.max_position_qty = max_position_qty
        self.hold_min = hold_min
        self.stop_pct = stop_pct
        self.take_profit_pct = take_profit_pct
        self.min_z = min_z
        self.min_time = min_time
        self.max_entry_time = max_entry_time
        self.flatten_time = self._parse_hhmm(flatten_time)
        self.poll_sec = poll_sec
        self.dry_run = dry_run
        self.max_gex_age_sec = max_gex_age_sec
        self.min_call_wall_room_pct = min_call_wall_room_pct
        self.min_put_wall_room_pct = min_put_wall_room_pct
        self.max_spread_pct = max_spread_pct
        self.exit_on_regime_break = exit_on_regime_break
        self.exit_on_flip_break = exit_on_flip_break
        self.account_mode = account_mode
        self.email_notifier = email_notifier
        self.underlying = Stock(self.symbol, "SMART", "USD")
        self.underlying_ticker = None
        suffix = f"{self.symbol}_0dte_{self.right}"
        self.state_path = state_path or DEFAULT_LOG_DIR / f"{suffix}_state.json"
        self.trade_logger = CSVTradeLogger(
            trade_csv_path or DEFAULT_LOG_DIR / f"{suffix.lower()}_trades_{trading_date_str()}.csv"
        )
        self.signal_logger = CSVSignalLogger(
            signal_csv_path or DEFAULT_LOG_DIR / f"{suffix.lower()}_signals_{trading_date_str()}.csv"
        )
        self.positions = self._load_positions()
        self.last_signal_ts = self._load_last_signal_ts()
        self._running = True

    @staticmethod
    def _parse_hhmm(value: str) -> dtime:
        hh, mm = value.split(":")
        return dtime(int(hh), int(mm))

    @staticmethod
    def _round_price(value: float) -> float:
        return round(max(value, 0.01), 2)

    def _load_positions(self) -> list[PositionState]:
        if not self.state_path.exists():
            return []
        try:
            data = json.loads(self.state_path.read_text())
            raw_positions = data.get("positions")
            if raw_positions is None:
                raw_position = data.get("position", {})
                raw_positions = [raw_position] if raw_position.get("in_position") else []
            positions = []
            today = trading_date_str()
            for raw in raw_positions:
                pos = PositionState(**raw)
                pos.right = self.right
                if not pos.in_position or pos.qty <= 0:
                    continue
                if pos.expiry and pos.expiry != today:
                    log.warning("Dropping stale 0DTE position from state: %s %s", pos.local_symbol, pos.expiry)
                    continue
                positions.append(pos)
            return positions
        except Exception as e:
            log.warning("State load failed, starting flat: %s", e)
            return []

    def _load_last_signal_ts(self) -> pd.Timestamp | None:
        if not self.state_path.exists():
            return None
        try:
            raw = json.loads(self.state_path.read_text()).get("last_signal_ts")
        except Exception:
            raw = None
        if not raw:
            return None
        ts = pd.Timestamp(raw)
        return ts.tz_convert(ET) if ts.tz is not None else ts.tz_localize(ET)

    def _save_state(self) -> None:
        primary_position = self.positions[0] if self.positions else PositionState(right=self.right)
        payload = {
            "position": asdict(primary_position),
            "positions": [asdict(pos) for pos in self.positions],
            "open_qty": self._open_qty(),
            "max_position_qty": self.max_position_qty,
            "last_signal_ts": self.last_signal_ts.isoformat() if self.last_signal_ts is not None else None,
            "updated": et_now().isoformat(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def stop(self) -> None:
        self._running = False

    def verify_account(self) -> bool:
        accounts = self.ib.managedAccounts()
        paper_accounts = [str(a) for a in accounts if str(a).startswith("DU")]
        if self.account_mode == "paper" and paper_accounts:
            log.info("Paper account verified: %s", accounts)
            return True
        if self.account_mode == "any" and accounts:
            log.warning("Account mode 'any' accepted: accounts=%s", accounts)
            return True
        log.error("BLOCKED: 0DTE scalper requires paper account. account_mode=%s accounts=%s", self.account_mode, accounts)
        return False

    def start(self) -> None:
        self.ib.qualifyContracts(self.underlying)
        self.underlying_ticker = self.ib.reqMktData(self.underlying, "", False, False)
        if not self.dry_run and not self.verify_account():
            return
        log.info(
            "Started QQQ 0DTE option scalper: symbol=%s right=%s qty=%s max_position_qty=%s hold=%smin "
            "stop=%.1f%% take_profit=%.1f%% min_z=%.2f window=%.2f-%.2f "
            "call_room>=%.3f%% max_spread=%.1f%% dry_run=%s",
            self.symbol, self.right, self.qty, self.max_position_qty, self.hold_min, self.stop_pct,
            self.take_profit_pct, self.min_z, self.min_time, self.max_entry_time,
            self.min_call_wall_room_pct, self.max_spread_pct, self.dry_run,
        )
        while self._running:
            try:
                self._tick()
            except Exception:
                log.exception("Loop error")
            self.ib.sleep(self.poll_sec)

    def _tick(self) -> None:
        now = et_now()
        if self.positions:
            self._maybe_exit_all(now)
        if not is_market_open(now):
            return
        if not self._in_entry_window(now):
            self._consume_alerts_without_entry()
            return
        while self._available_qty() >= self.qty:
            alert = self._next_alert()
            if alert is None:
                return
            self._enter(alert)
        self._consume_alerts_without_entry("max_position_qty_reached")

    def _open_qty(self) -> int:
        return sum(max(0, int(pos.qty)) for pos in self.positions)

    def _available_qty(self) -> int:
        return max(0, self.max_position_qty - self._open_qty())

    def _in_entry_window(self, now: datetime) -> bool:
        hour = now.hour + now.minute / 60.0
        return self.min_time <= hour < self.max_entry_time

    def _alert_path(self) -> Path:
        return self.data_dir / f"signals_live_{self.symbol}_{trading_date_str()}.parquet"

    def _gex_path(self) -> Path:
        return self.data_dir / f"gex_{self.symbol}_{trading_date_str()}.parquet"

    def _read_alerts(self) -> pd.DataFrame:
        path = self._alert_path()
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

    def _alert_from_row(self, row: pd.Series) -> Alert:
        return Alert(
            ts=row["ts"],
            direction=str(row["direction"]),
            strength=str(row.get("strength", "")),
            z_score=float(row["z_score"]),
            spot=float(row["spot"]) if "spot" in row and pd.notna(row["spot"]) else None,
        )

    def _consume_alerts_without_entry(self, reason: str = "not_in_entry_state") -> None:
        df = self._eligible_alerts(self._read_alerts())
        if df.empty:
            return
        for _, row in df.iterrows():
            alert = self._alert_from_row(row)
            self.signal_logger.write(
                alert=alert,
                symbol=self.symbol,
                right=self.right,
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
        accepted = None
        newest = self.last_signal_ts
        for _, row in df.iterrows():
            alert = self._alert_from_row(row)
            decision = self._evaluate_alert(alert)
            self.signal_logger.write(alert=alert, symbol=self.symbol, right=self.right, decision=decision, dry_run=self.dry_run)
            newest = alert.ts if newest is None or alert.ts > newest else newest
            if decision.allowed and accepted is None:
                accepted = alert
                break
            log.info("Rejected 0DTE signal %s z=%.2f reason=%s", alert.ts.strftime("%H:%M"), alert.z_score, decision.reason)
        if newest is not None:
            self.last_signal_ts = newest
            self._save_state()
        return accepted

    def _read_gex(self) -> pd.DataFrame:
        path = self._gex_path()
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

    def _context_from_row(self, row: pd.Series, age_sec: float) -> GexContext:
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
        return self._context_from_row(row, abs((row["ts"] - alert_ts).total_seconds()))

    def _latest_gex_context(self) -> GexContext | None:
        df = self._read_gex()
        if df.empty:
            return None
        row = df.iloc[-1]
        return self._context_from_row(row, abs((et_now() - row["ts"]).total_seconds()))

    def _evaluate_alert(self, alert: Alert) -> SignalDecision:
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
        return SignalDecision(True, "ddput_gex_0dte_call_ok", ctx)

    def _underlying_price(self) -> float | None:
        if self.underlying_ticker is None:
            return None
        price = self.underlying_ticker.marketPrice()
        if price is None or not np.isfinite(price) or price <= 0:
            price = self.underlying_ticker.last or self.underlying_ticker.close
        return float(price) if price is not None and np.isfinite(price) and price > 0 else None

    def _pick_0dte_atm_option(self, spot: float) -> Option | None:
        expiry = trading_date_str()
        chains = self.ib.reqSecDefOptParams(self.symbol, "", self.underlying.secType, self.underlying.conId)
        chain = (
            next((c for c in chains if c.exchange == "SMART" and c.tradingClass == self.symbol), None)
            or next((c for c in chains if c.exchange == "SMART"), None)
            or (chains[0] if chains else None)
        )
        if chain is None:
            log.warning("No option chain for %s", self.symbol)
            return None
        if expiry not in {str(e) for e in chain.expirations}:
            log.warning("No 0DTE expiry %s for %s", expiry, self.symbol)
            return None
        strikes = [float(s) for s in chain.strikes if s and s > 0]
        if not strikes:
            return None
        strike = min(strikes, key=lambda s: abs(s - spot))
        contract = Option(self.symbol, expiry, strike, self.right, "SMART", tradingClass=chain.tradingClass)
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else None

    def _quote_option(self, contract: Option, wait_sec: float = 1.5) -> OptionQuote:
        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(wait_sec)
        quote = OptionQuote(
            bid=float(ticker.bid) if ticker.bid is not None and np.isfinite(ticker.bid) and ticker.bid > 0 else None,
            ask=float(ticker.ask) if ticker.ask is not None and np.isfinite(ticker.ask) and ticker.ask > 0 else None,
            last=float(ticker.last) if ticker.last is not None and np.isfinite(ticker.last) and ticker.last > 0 else None,
            close=float(ticker.close) if ticker.close is not None and np.isfinite(ticker.close) and ticker.close > 0 else None,
        )
        self.ib.cancelMktData(contract)
        return quote

    def _limit_for_side(self, side: str, quote: OptionQuote) -> float | None:
        if side == "BUY":
            return self._round_price(quote.ask or quote.mid or 0)
        return self._round_price(quote.bid or quote.mid or 0)

    def _wait_fill(self, trade, timeout: float = 20.0) -> float | None:
        start = time.time()
        while time.time() - start < timeout:
            self.ib.sleep(0.25)
            if trade.isDone() or trade.orderStatus.status == "Filled":
                break
        total = sum(f.execution.shares for f in trade.fills)
        if total <= 0:
            return None
        avg = trade.orderStatus.avgFillPrice
        if avg and avg > 0:
            return float(avg)
        value = sum(f.execution.shares * f.execution.price for f in trade.fills)
        return float(value / total) if total else None

    def _submit(self, contract: Option, side: str, quote: OptionQuote, qty: int | None = None) -> tuple[float | None, float | None]:
        limit_price = self._limit_for_side(side, quote)
        if limit_price is None or limit_price <= 0:
            return None, None
        order_qty = int(qty or self.qty)
        if self.dry_run:
            log.info("[DRY RUN] %s %s %s @ %.2f", side, order_qty, contract.localSymbol, limit_price)
            return limit_price, limit_price
        order = LimitOrder(side, order_qty, limit_price)
        order.tif = "IOC"
        order.outsideRth = False
        trade = self.ib.placeOrder(contract, order)
        fill = self._wait_fill(trade)
        if fill is None:
            log.error("%s %s not filled at %.2f; cancelling", side, contract.localSymbol, limit_price)
            try:
                self.ib.cancelOrder(order)
            except Exception:
                pass
        return limit_price, fill

    def _enter(self, alert: Alert) -> None:
        underlying = self._underlying_price() or alert.spot
        if underlying is None:
            log.warning("Cannot enter: missing underlying price")
            return
        contract = self._pick_0dte_atm_option(underlying)
        if contract is None:
            return
        quote = self._quote_option(contract)
        if quote.mid is None:
            log.warning("Cannot enter %s: missing option quote", contract.localSymbol)
            return
        if quote.spread_pct is None or quote.spread_pct > self.max_spread_pct:
            log.info("Rejected %s: spread %.2f%% > %.2f%%", contract.localSymbol, quote.spread_pct or -1, self.max_spread_pct)
            return
        entry_qty = min(self.qty, self._available_qty())
        if entry_qty <= 0:
            log.info("Rejected entry: open_qty=%s max_position_qty=%s", self._open_qty(), self.max_position_qty)
            return
        limit_price, fill = self._submit(contract, "BUY", quote, qty=entry_qty)
        if fill is None:
            return
        now = et_now()
        hold_until = now + pd.Timedelta(minutes=self.hold_min)
        position = PositionState(
            in_position=True,
            qty=entry_qty,
            entry_ts=now.isoformat(),
            entry_price=fill,
            entry_underlying=underlying,
            entry_signal_ts=alert.ts.isoformat(),
            entry_z=alert.z_score,
            entry_strength=alert.strength,
            hold_until_ts=hold_until.isoformat(),
            con_id=contract.conId,
            local_symbol=contract.localSymbol,
            expiry=contract.lastTradeDateOrContractMonth,
            strike=float(contract.strike),
            right=contract.right,
        )
        self.positions.append(position)
        self._save_state()
        self.trade_logger.write(
            ts=now.isoformat(), event="ENTER", symbol=self.symbol, local_symbol=contract.localSymbol,
            right=contract.right, expiry=contract.lastTradeDateOrContractMonth, strike=contract.strike,
            side="BUY", qty=entry_qty, signal_ts=alert.ts.isoformat(), strength=alert.strength,
            z_score=alert.z_score, underlying=underlying, bid=quote.bid, ask=quote.ask, mid=quote.mid,
            limit_price=limit_price, fill_price=fill, reason="ddput_gex_0dte_call_ok", dry_run=self.dry_run,
        )
        self._send_trade_email("ENTER", "BUY", contract, fill, alert.ts, alert.strength, alert.z_score, "ddput_gex_0dte_call_ok", qty=entry_qty)

    def _contract_from_position(self, position: PositionState) -> Option | None:
        if not position.expiry or position.strike is None:
            return None
        contract = Option(self.symbol, position.expiry, position.strike, position.right, "SMART")
        if position.con_id:
            contract.conId = int(position.con_id)
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else contract

    def _maybe_exit_all(self, now: datetime) -> None:
        for position in list(self.positions):
            self._maybe_exit(position, now)

    def _maybe_exit(self, position: PositionState, now: datetime) -> None:
        contract = self._contract_from_position(position)
        if contract is None or position.entry_price is None:
            return
        quote = self._quote_option(contract, wait_sec=1.0)
        mark = quote.mid
        if mark is None:
            return
        pnl_pct = (mark / float(position.entry_price) - 1.0) * 100.0
        hold_until = pd.Timestamp(position.hold_until_ts)
        hold_until = hold_until.tz_convert(ET) if hold_until.tz is not None else hold_until.tz_localize(ET)
        reason = None
        regime_reason = self._regime_exit_reason()
        if self.exit_on_regime_break and regime_reason is not None:
            reason = regime_reason
        elif pnl_pct <= -self.stop_pct:
            reason = "stop"
        elif pnl_pct >= self.take_profit_pct:
            reason = "take_profit"
        elif now >= hold_until:
            reason = "time_exit"
        elif now.time() >= self.flatten_time:
            reason = "eod_flatten"
        if reason:
            self._exit(position, contract, reason, quote, pnl_pct)

    def _regime_exit_reason(self) -> str | None:
        ctx = self._latest_gex_context()
        if ctx is None or ctx.age_sec > self.max_gex_age_sec:
            return None
        if not ctx.positive_gamma:
            return "gex_exit:not_positive_gamma"
        if ctx.put_wall is not None and ctx.spot <= ctx.put_wall:
            return "gex_exit:below_put_wall"
        return None

    def _exit(self, position: PositionState, contract: Option, reason: str, quote: OptionQuote, mark_pnl_pct: float) -> None:
        exit_qty = max(1, int(position.qty))
        limit_price, fill = self._submit(contract, "SELL", quote, qty=exit_qty)
        if fill is None:
            return
        entry = float(position.entry_price or fill)
        pnl_pct = (fill / entry - 1.0) * 100.0
        now = et_now()
        underlying = self._underlying_price()
        self.trade_logger.write(
            ts=now.isoformat(), event="EXIT", symbol=self.symbol, local_symbol=contract.localSymbol,
            right=contract.right, expiry=contract.lastTradeDateOrContractMonth, strike=contract.strike,
            side="SELL", qty=exit_qty, signal_ts=position.entry_signal_ts,
            strength=position.entry_strength, z_score=position.entry_z,
            underlying=underlying, bid=quote.bid, ask=quote.ask, mid=quote.mid,
            limit_price=limit_price, fill_price=fill, pnl_pct=pnl_pct,
            reason=reason, dry_run=self.dry_run,
        )
        self._send_trade_email("EXIT", "SELL", contract, fill, position.entry_signal_ts, position.entry_strength, position.entry_z, reason, pnl_pct, qty=exit_qty)
        try:
            self.positions.remove(position)
        except ValueError:
            self.positions = [pos for pos in self.positions if pos.entry_ts != position.entry_ts]
        self._save_state()

    def _send_trade_email(
        self,
        event: str,
        side: str,
        contract: Option,
        fill: float,
        signal_ts,
        strength: str | None,
        z_score: float | None,
        reason: str,
        pnl_pct: float | None = None,
        qty: int | None = None,
    ) -> None:
        if self.email_notifier is None:
            return
        mode_text = "DRY RUN" if self.dry_run else "IB Paper"
        subject_prefix = "DRY-RUN" if self.dry_run else "PAPER"
        pnl_line = f"PnL:          {pnl_pct:+.2f}%\n" if pnl_pct is not None else ""
        order_qty = int(qty or self.qty)
        signal_text = ""
        if signal_ts:
            ts = pd.Timestamp(signal_ts)
            ts = ts.tz_convert(ET) if ts.tz is not None else ts.tz_localize(ET)
            signal_text = ts.strftime("%Y-%m-%d %H:%M:%S ET")
        subject = f"{self.symbol} 0DTE {subject_prefix} {event} {side} {order_qty} {contract.localSymbol} @ {fill:.2f}"
        body = (
            f"QQQ 0DTE option scalper: {event}\n\n"
            f"合约:         {contract.localSymbol}\n"
            f"方向:         {side}\n"
            f"数量:         {order_qty}\n"
            f"当前总持仓:   {self._open_qty()} / {self.max_position_qty}\n"
            f"成交价:       {fill:.2f}\n"
            f"{pnl_line}"
            f"原因:         {reason}\n"
            f"信号时间:     {signal_text}\n"
            f"信号强度:     {strength or ''}\n"
            f"Z-score:      {z_score if z_score is not None else ''}\n"
            f"模式:         {mode_text}\n"
            f"通知时间:     {et_now().strftime('%Y-%m-%d %H:%M:%S ET')}\n"
        )
        if not self.email_notifier.send_alert(subject, body):
            log.warning("0DTE trade email not sent: %s %s", event, contract.localSymbol)


def configure_logging() -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DEFAULT_LOG_DIR / f"qqq_0dte_option_scalper_{trading_date_str()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )
    log.info("Log file: %s", log_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QQQ 0DTE ATM option paper scalper")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=78)
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--right", choices=["C"], default="C")
    p.add_argument("--qty", type=int, default=1)
    p.add_argument("--max-position-qty", type=int, default=1, help="Maximum open option contracts across all active scalp lots")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--hold-min", type=int, default=10)
    p.add_argument("--stop-pct", type=float, default=20.0)
    p.add_argument("--take-profit-pct", type=float, default=25.0)
    p.add_argument("--min-z", type=float, default=1.5)
    p.add_argument("--min-time", type=float, default=10.5)
    p.add_argument("--max-entry-time", type=float, default=14.5)
    p.add_argument("--flatten-time", default="15:45")
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ib-readonly", action="store_true")
    p.add_argument("--account-mode", choices=["paper", "any"], default="paper")
    p.add_argument("--max-gex-age-sec", type=float, default=120.0)
    p.add_argument("--min-call-wall-room-pct", type=float, default=0.15)
    p.add_argument("--min-put-wall-room-pct", type=float, default=0.0)
    p.add_argument("--max-spread-pct", type=float, default=8.0)
    p.add_argument("--disable-regime-exit", action="store_true")
    p.add_argument("--exit-on-flip-break", action="store_true", help="Deprecated; flip is logged only and ignored for trading")
    p.add_argument("--email-enabled", action="store_true")
    p.add_argument("--email-sender", default="fzhouxu615@gmail.com")
    p.add_argument("--email-password-env", default="GMAIL_APP_PASSWORD")
    p.add_argument("--email-recipients", default="")
    p.add_argument("--email-subject-prefix", default="[GEX]")
    return p.parse_args()


def _email_notifier_from_args(args: argparse.Namespace) -> EmailNotifier | None:
    if not args.email_enabled:
        return None
    recipients = [r.strip() for r in str(args.email_recipients).split(",") if r.strip()]
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
    if args.ib_readonly and not args.dry_run:
        log.error("Refusing to trade while --ib-readonly is set. Use --dry-run or disable readonly.")
        return
    ib = IB()
    scalper = None

    def _handle_signal(*_):
        if scalper is not None:
            scalper.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        readonly = args.ib_readonly or args.dry_run
        log.info("Connecting IB %s:%s clientId=%s readonly=%s", args.host, args.port, args.client_id, readonly)
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=20, readonly=readonly)
        log.info("Connected. Accounts=%s", ib.managedAccounts())
        scalper = QQQ0DTEOptionScalper(
            ib=ib,
            data_dir=args.data_dir,
            symbol=args.symbol,
            right=args.right,
            qty=args.qty,
            max_position_qty=args.max_position_qty,
            hold_min=args.hold_min,
            stop_pct=args.stop_pct,
            take_profit_pct=args.take_profit_pct,
            min_z=args.min_z,
            min_time=args.min_time,
            max_entry_time=args.max_entry_time,
            flatten_time=args.flatten_time,
            poll_sec=args.poll_sec,
            dry_run=args.dry_run,
            max_gex_age_sec=args.max_gex_age_sec,
            min_call_wall_room_pct=args.min_call_wall_room_pct,
            min_put_wall_room_pct=args.min_put_wall_room_pct,
            max_spread_pct=args.max_spread_pct,
            exit_on_regime_break=not args.disable_regime_exit,
            exit_on_flip_break=args.exit_on_flip_break,
            account_mode=args.account_mode,
            email_notifier=_email_notifier_from_args(args),
        )
        scalper.start()
    finally:
        if ib.isConnected():
            ib.disconnect()
        log.info("Stopped")


if __name__ == "__main__":
    main()
