"""QQQ 0DTE passive scalper.

Paper-first experiment for buying a 0DTE OTM option passively and selling it
back at a small fixed credit. It never places an opening SELL order: the sell
limit is only submitted after a BUY fill exists.

NOTE on "grid": this bot holds at most ONE position at a time (the tick loop
manages an open position to completion before looking for a new entry). It is
a single-shot passive scalper, not a layered grid. `max_open_qty` is only a
safety cap used to detect/flatten unmanaged positions, not a layering target.
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
from ib_insync import IB, LimitOrder, Option, Stock

from .email_notifier import EmailConfig, EmailNotifier
from .time_utils import ET, et_now, is_market_open, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT_DIR / "logs"

log = logging.getLogger("qqq_0dte_grid_scalper")


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
        if self.last is not None and self.last > 0:
            return self.last
        return self.close if self.close is not None and self.close > 0 else None

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float | None:
        mid = self.mid
        if mid is None or mid <= 0 or self.spread is None:
            return None
        return self.spread / mid * 100.0


@dataclass
class PositionState:
    in_position: bool = False
    qty: int = 0
    con_id: int | None = None
    local_symbol: str | None = None
    expiry: str | None = None
    strike: float | None = None
    right: str = "C"
    entry_ts: str | None = None
    entry_price: float | None = None
    entry_underlying: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    exit_order_id: int | None = None


@dataclass
class DailyStats:
    date: str
    trades: int = 0
    realized_pnl: float = 0.0
    wins: int = 0
    losses: int = 0


class CSVLogger:
    fields = [
        "ts",
        "event",
        "symbol",
        "local_symbol",
        "right",
        "expiry",
        "strike",
        "side",
        "qty",
        "underlying",
        "bid",
        "ask",
        "mid",
        "spread_pct",
        "limit_price",
        "fill_price",
        "entry_price",
        "target_price",
        "stop_price",
        "pnl",
        "pnl_pct",
        "reason",
        "dry_run",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def write(self, **row) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(
                {k: row.get(k, "") for k in self.fields}
            )


class QQQ0DTEGridScalper:
    def __init__(
        self,
        ib: IB,
        symbol: str = "QQQ",
        right: str = "C",
        qty: int = 1,
        max_open_qty: int = 5,
        max_trades_per_day: int = 300,
        max_daily_loss: float = 250.0,
        otm_offset: float = 1.0,
        entry_edge: float = 0.10,
        target_credit: float = 0.20,
        stop_debit: float = 0.15,
        max_hold_sec: int = 120,
        order_timeout_sec: float = 8.0,
        poll_sec: float = 1.0,
        entry_start: str = "09:45",
        entry_end: str = "15:12",
        flatten_time: str = "15:45",
        min_option_price: float = 0.25,
        max_option_price: float = 4.00,
        max_spread_pct: float = 18.0,
        flatten_unmanaged: bool = True,
        dry_run: bool = True,
        state_path: Path | None = None,
        csv_path: Path | None = None,
        email_notifier: EmailNotifier | None = None,
    ):
        self.ib = ib
        self.symbol = symbol.upper()
        self.right = right.upper()
        self.qty = int(qty)
        self.max_open_qty = int(max_open_qty)
        self.max_trades_per_day = int(max_trades_per_day)
        self.max_daily_loss = float(max_daily_loss)
        self.otm_offset = float(otm_offset)
        self.entry_edge = float(entry_edge)
        self.target_credit = float(target_credit)
        self.stop_debit = float(stop_debit)
        self.max_hold_sec = int(max_hold_sec)
        self.order_timeout_sec = float(order_timeout_sec)
        self.poll_sec = float(poll_sec)
        # Entry window and flatten time are all HH:MM strings now (was a mix of
        # decimal-hours and HH:MM, which is an easy way to set the wrong time).
        self.entry_start = self._parse_hhmm(entry_start)
        self.entry_end = self._parse_hhmm(entry_end)
        self.flatten_time = self._parse_hhmm(flatten_time)
        self.min_option_price = float(min_option_price)
        self.max_option_price = float(max_option_price)
        self.max_spread_pct = float(max_spread_pct)
        self.flatten_unmanaged = bool(flatten_unmanaged)
        self.dry_run = bool(dry_run)
        self.email_notifier = email_notifier
        self.underlying = Stock(self.symbol, "SMART", "USD")
        self.underlying_ticker = None
        suffix = f"{self.symbol.lower()}_0dte_grid_{self.right.lower()}"
        self.state_path = state_path or DEFAULT_LOG_DIR / f"{suffix}_state.json"
        self.csv_logger = CSVLogger(csv_path or DEFAULT_LOG_DIR / f"{suffix}_{trading_date_str()}.csv")
        self.position = PositionState(right=self.right)
        self.stats = DailyStats(date=trading_date_str())
        self.exit_trade = None
        self.last_cycle_ts: datetime | None = None
        self._after_close_logged = False
        self._running = True
        self._load_state()

    @staticmethod
    def _parse_hhmm(value: str) -> dtime:
        # Reject decimal-hour style values loudly instead of mis-parsing them.
        if ":" not in str(value):
            raise ValueError(
                f"time value {value!r} must be HH:MM (e.g. '15:12'), not decimal hours"
            )
        hh, mm = str(value).split(":")
        return dtime(int(hh), int(mm))

    @staticmethod
    def _clean_price(value) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) and out > 0 else None

    @staticmethod
    def _round_price(value: float) -> float:
        return round(max(0.01, float(value)), 2)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("State file unreadable, starting fresh: %s", e)
            return
        # Restore stats and position independently so a problem with one does
        # not silently drop the other.
        try:
            stats = raw.get("stats") or {}
            if stats.get("date") == trading_date_str():
                self.stats = DailyStats(**stats)
        except Exception as e:
            log.warning("Stats restore failed, using fresh stats: %s", e)
        try:
            pos = raw.get("position") or {}
            if pos.get("in_position") and pos.get("expiry") == trading_date_str():
                self.position = PositionState(**pos)
                log.info(
                    "Restored open position from state: %s strike=%s qty=%s entry=%s",
                    self.position.local_symbol,
                    self.position.strike,
                    self.position.qty,
                    self.position.entry_price,
                )
        except Exception as e:
            log.error(
                "POSITION restore FAILED - there may be an untracked live position "
                "in IB for today. Check the account manually. Error: %s",
                e,
            )

    def _save_state(self) -> None:
        payload = {
            "position": asdict(self.position),
            "stats": asdict(self.stats),
            "updated": et_now().isoformat(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def stop(self) -> None:
        self._running = False
        if self.position.in_position:
            log.warning(
                "STOP requested while a position is OPEN: %s qty=%s entry=%s. "
                "The position is left open and any resting target order may still "
                "be live. Reconcile manually, or restart the bot to resume "
                "management (it will re-attach the resting target).",
                self.position.local_symbol,
                self.position.qty,
                self.position.entry_price,
            )

    def verify_paper_account(self) -> bool:
        accounts = [str(a) for a in self.ib.managedAccounts()]
        paper_accounts = [a for a in accounts if a.startswith("DU")]
        if paper_accounts:
            log.info("Paper account verified: %s", accounts)
            return True
        log.error("BLOCKED: grid scalper requires IB Paper account. accounts=%s", accounts)
        return False

    def start(self) -> None:
        if not self.dry_run and not self.verify_paper_account():
            return
        self.ib.qualifyContracts(self.underlying)
        self.underlying_ticker = self.ib.reqMktData(self.underlying, "", False, False)
        # Reconcile any orders/positions left over from a previous run BEFORE the
        # loop starts, so a restored position cannot be double-exited by a stale
        # resting SELL we no longer have a handle on.
        self._reconcile_open_orders()
        if self.position.in_position and self.exit_trade is None:
            log.warning(
                "Restored position has no resting target order; re-arming target exit"
            )
            restored_contract = self._contract_from_position()
            if restored_contract is not None:
                self._submit_target_exit(restored_contract)
        log.info(
            "Started %s 0DTE scalper right=%s qty=%s max_open=%s max_trades=%s "
            "entry_edge=%.2f target_credit=%.2f stop_debit=%.2f otm_offset=%.2f "
            "window=%s-%s flatten=%s flatten_unmanaged=%s dry_run=%s",
            self.symbol,
            self.right,
            self.qty,
            self.max_open_qty,
            self.max_trades_per_day,
            self.entry_edge,
            self.target_credit,
            self.stop_debit,
            self.otm_offset,
            self.entry_start.strftime("%H:%M"),
            self.entry_end.strftime("%H:%M"),
            self.flatten_time.strftime("%H:%M"),
            self.flatten_unmanaged,
            self.dry_run,
        )
        while self._running:
            try:
                self._tick()
            except Exception:
                log.exception("Loop error")
            self.ib.sleep(self.poll_sec)

    def _reconcile_open_orders(self) -> None:
        """Cancel stray resting orders for this symbol/expiry left from a prior
        run. If a restored position has a matching resting SELL, re-attach it as
        the managed exit_trade instead of cancelling it."""
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(0.5)
        except Exception as e:
            log.warning("reqAllOpenOrders failed during reconcile: %s", e)
        expiry = trading_date_str()
        stray = []
        for trade in list(self.ib.openTrades()):
            contract = trade.contract
            if getattr(contract, "symbol", "") != self.symbol:
                continue
            if getattr(contract, "secType", "") != "OPT":
                continue
            if str(getattr(contract, "lastTradeDateOrContractMonth", "")) != expiry:
                continue
            if trade.isDone():
                continue
            is_our_target = (
                self.position.in_position
                and self._position_matches_local(contract)
                and getattr(trade.order, "action", "") == "SELL"
            )
            if is_our_target:
                self.exit_trade = trade
                self.position.exit_order_id = int(getattr(trade.order, "orderId", 0)) or None
                log.info(
                    "Re-attached resting target order id=%s for restored position %s",
                    getattr(trade.order, "orderId", "?"),
                    getattr(contract, "localSymbol", ""),
                )
                continue
            stray.append(trade)
        for trade in stray:
            log.warning(
                "Cancelling stray open order: %s id=%s %s",
                getattr(trade.order, "action", "?"),
                getattr(trade.order, "orderId", "?"),
                getattr(trade.contract, "localSymbol", ""),
            )
            if not self.dry_run:
                try:
                    self.ib.cancelOrder(trade.order)
                except Exception as e:
                    log.warning("Cancel stray order failed: %s", e)
        if self.exit_trade is not None:
            self._save_state()

    def _tick(self) -> None:
        now = et_now()
        self._sync_local_position_with_ib()
        if self.position.in_position:
            self._manage_position(now)
            return
        if not self._can_open(now):
            return
        self._try_entry_cycle(now)

    def _can_open(self, now: datetime) -> bool:
        if not is_market_open(now):
            return False
        if now.time() >= self.flatten_time:
            return False
        if now.time() < self.entry_start or now.time() >= self.entry_end:
            return False
        if self.stats.date != trading_date_str():
            self.stats = DailyStats(date=trading_date_str())
        if self.stats.trades >= self.max_trades_per_day:
            return False
        if self.stats.realized_pnl <= -abs(self.max_daily_loss):
            return False
        if self.last_cycle_ts is not None and (now - self.last_cycle_ts).total_seconds() < self.order_timeout_sec:
            return False
        if self.max_open_qty <= 0:
            return False
        ib_qty = self._ib_open_qty()
        if ib_qty >= self.max_open_qty:
            return False
        if ib_qty > 0 and not self.position.in_position:
            if self.flatten_unmanaged:
                self._flatten_unmanaged_positions("unmanaged_position_before_entry")
            else:
                log.error("BLOCKED: unmanaged %s %s 0DTE position qty=%s", self.symbol, self.right, ib_qty)
            return False
        return True

    def _ib_positions_for_side(self):
        expiry = trading_date_str()
        out = []
        for position in self.ib.positions():
            contract = position.contract
            if getattr(contract, "symbol", "") != self.symbol:
                continue
            if getattr(contract, "secType", "") != "OPT":
                continue
            if str(getattr(contract, "lastTradeDateOrContractMonth", "")) != expiry:
                continue
            if getattr(contract, "right", "") != self.right:
                continue
            if not position.position:
                continue
            out.append(position)
        return out

    def _sync_local_position_with_ib(self) -> None:
        if not self.position.in_position:
            return
        matches = [
            position
            for position in self._ib_positions_for_side()
            if self._position_matches_local(position.contract)
        ]
        if not matches:
            contract = self._contract_from_position() or Option(
                self.symbol,
                self.position.expiry or trading_date_str(),
                self.position.strike or 0,
                self.position.right,
                "SMART",
            )
            self._log_event(
                "LOCAL_POSITION_GONE",
                contract,
                OptionQuote(None, None, None, None),
                qty=self.position.qty,
                underlying=self._underlying_price(),
                entry_price=self.position.entry_price,
                target_price=self.position.target_price,
                stop_price=self.position.stop_price,
                reason="ib_position_missing",
            )
            self.position = PositionState(right=self.right)
            self.exit_trade = None
            self._save_state()
            return
        if self.flatten_unmanaged:
            self._flatten_unmanaged_positions("unmanaged_position_while_managing")
        actual_qty = int(sum(abs(float(position.position)) for position in matches))
        if actual_qty != self.position.qty:
            log.warning(
                "Adjusting local qty from %s to IB actual qty %s for %s %s",
                self.position.qty,
                actual_qty,
                self.symbol,
                self.right,
            )
            self.position.qty = actual_qty
            self._save_state()

    def _ib_open_qty(self) -> int:
        return int(sum(abs(float(position.position)) for position in self._ib_positions_for_side()))

    def _ib_qty_for_local(self) -> int:
        """IB position quantity that matches the bot's current local position."""
        total = 0
        for position in self._ib_positions_for_side():
            if self._position_matches_local(position.contract):
                total += int(abs(float(position.position)))
        return total

    def _safe_exit_qty(self) -> int:
        """Never sell more contracts than IB actually shows for this position.
        This is the guardrail against turning a partial fill / stale state into
        a naked short."""
        return max(0, min(int(self.position.qty), self._ib_qty_for_local()))

    def _position_matches_local(self, contract: Option) -> bool:
        if not self.position.in_position:
            return False
        if self.position.con_id and getattr(contract, "conId", None) == self.position.con_id:
            return True
        return (
            str(getattr(contract, "lastTradeDateOrContractMonth", "")) == str(self.position.expiry)
            and float(getattr(contract, "strike", -1)) == float(self.position.strike or -2)
            and getattr(contract, "right", "") == self.position.right
        )

    def _flatten_unmanaged_positions(self, reason: str) -> None:
        positions = self._ib_positions_for_side()
        for position in positions:
            contract = position.contract
            if self._position_matches_local(contract):
                continue
            qty = int(abs(position.position))
            if qty <= 0:
                continue
            side = "SELL" if position.position > 0 else "BUY"
            if self.dry_run:
                self._log_event(
                    "UNMANAGED_POSITION",
                    contract,
                    OptionQuote(None, None, None, None),
                    side,
                    qty,
                    self._underlying_price(),
                    reason=reason,
                )
                continue
            self._submit_market_flatten(contract, side, qty, reason)

    def _submit_market_flatten(self, contract: Option, side: str, qty: int, reason: str) -> None:
        from ib_insync import MarketOrder

        contract.exchange = "SMART"
        order = MarketOrder(side, qty)
        order.tif = "DAY"
        order.outsideRth = False
        self._log_event(
            "UNMANAGED_FLATTEN_SUBMIT",
            contract,
            OptionQuote(None, None, None, None),
            side,
            qty,
            self._underlying_price(),
            reason=reason,
        )
        trade = self.ib.placeOrder(contract, order)
        fill, _ = self._wait_fill(trade, timeout=10.0)
        self._log_event(
            "UNMANAGED_FLATTEN",
            contract,
            OptionQuote(None, None, None, None),
            side,
            qty,
            self._underlying_price(),
            fill_price=fill,
            reason=reason if fill is not None else f"{reason}:not_filled",
        )

    def _underlying_price(self) -> float | None:
        if self.underlying_ticker is None:
            return None
        price = self._clean_price(self.underlying_ticker.marketPrice())
        if price is None:
            price = self._clean_price(self.underlying_ticker.last)
        if price is None:
            price = self._clean_price(self.underlying_ticker.close)
        return price

    def _pick_0dte_otm_option(self, spot: float) -> Option | None:
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
        strikes = sorted(float(s) for s in chain.strikes if s and s > 0)
        if not strikes:
            return None
        target = spot + self.otm_offset if self.right == "C" else spot - self.otm_offset
        if self.right == "C":
            candidates = [s for s in strikes if s >= spot]
        else:
            candidates = [s for s in strikes if s <= spot]
        if not candidates:
            candidates = strikes
        strike = min(candidates, key=lambda s: abs(s - target))
        contract = Option(self.symbol, expiry, strike, self.right, "SMART", tradingClass=chain.tradingClass)
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else None

    def _contract_from_position(self) -> Option | None:
        if not self.position.expiry or self.position.strike is None:
            return None
        contract = Option(self.symbol, self.position.expiry, self.position.strike, self.position.right, "SMART")
        if self.position.con_id:
            contract.conId = int(self.position.con_id)
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else contract

    def _quote_option(self, contract: Option, wait_sec: float = 0.8) -> OptionQuote:
        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(wait_sec)
        quote = OptionQuote(
            bid=self._clean_price(ticker.bid),
            ask=self._clean_price(ticker.ask),
            last=self._clean_price(ticker.last),
            close=self._clean_price(ticker.close),
        )
        self.ib.cancelMktData(contract)
        return quote

    def _try_entry_cycle(self, now: datetime) -> None:
        self.last_cycle_ts = now
        spot = self._underlying_price()
        if spot is None:
            log.info("Skip entry: missing underlying price")
            return
        contract = self._pick_0dte_otm_option(spot)
        if contract is None:
            return
        quote = self._quote_option(contract)
        mid = quote.mid
        if mid is None:
            self._log_event("SKIP", contract, quote, underlying=spot, reason="missing_option_mid")
            return
        if mid < self.min_option_price or mid > self.max_option_price:
            self._log_event("SKIP", contract, quote, underlying=spot, reason="option_price_out_of_range")
            return
        if quote.spread_pct is None or quote.spread_pct > self.max_spread_pct:
            self._log_event("SKIP", contract, quote, underlying=spot, reason="spread_too_wide")
            return
        order_qty = min(self.qty, self.max_open_qty)
        if order_qty <= 0:
            return
        buy_limit = self._round_price(mid - self.entry_edge)
        self._log_event("BUY_SUBMIT", contract, quote, "BUY", order_qty, spot, buy_limit, reason="passive_entry")
        if self.dry_run:
            return
        order = LimitOrder("BUY", order_qty, buy_limit)
        order.tif = "DAY"
        order.outsideRth = False
        trade = self.ib.placeOrder(contract, order)
        fill, filled_qty = self._wait_fill(trade, timeout=self.order_timeout_sec)
        # If we did not get a complete fill, cancel whatever is still working and
        # re-check (catches a cancel/fill race). _wait_fill never cancels, so a
        # partial fill would otherwise leave working quantity live AND build the
        # position at the wrong size.
        if filled_qty < order_qty:
            race_fill, race_qty = self._cancel_and_check_fill(trade, order, wait_sec=2.0)
            if race_qty > filled_qty:
                fill, filled_qty = race_fill, race_qty
            elif race_fill is not None and fill is None:
                fill = race_fill
        if filled_qty <= 0 or fill is None:
            self._log_event("BUY_CANCEL", contract, quote, "BUY", order_qty, spot, buy_limit, reason="entry_timeout")
            return
        if filled_qty < order_qty:
            log.warning(
                "Entry partial fill: %s of %s contracts filled - managing actual filled qty",
                filled_qty,
                order_qty,
            )
        # Manage exactly what was actually filled, never what was requested.
        self._on_entry_fill(contract, quote, spot, buy_limit, fill, filled_qty)

    def _wait_fill(self, trade, timeout: float) -> tuple[float | None, int]:
        """Wait up to `timeout` for fills. Returns (avg_fill_price, filled_qty)."""
        start = time.time()
        while time.time() - start < timeout:
            self.ib.sleep(0.20)
            if trade.isDone() or trade.orderStatus.status == "Filled":
                break
        return self._trade_fill_price(trade), self._trade_filled_qty(trade)

    def _cancel_and_check_fill(self, trade, order, wait_sec: float = 2.0) -> tuple[float | None, int]:
        """Cancel an order, then re-check fills to catch cancel/fill races.
        Returns (avg_fill_price, filled_qty)."""
        try:
            self.ib.cancelOrder(order)
        except Exception:
            pass
        start = time.time()
        while time.time() - start < wait_sec:
            self.ib.sleep(0.20)
            status = getattr(trade.orderStatus, "status", "")
            if status in {"Cancelled", "ApiCancelled", "Inactive", "Filled"}:
                break
        return self._trade_fill_price(trade), self._trade_filled_qty(trade)

    @staticmethod
    def _trade_filled_qty(trade) -> int:
        try:
            return int(sum(f.execution.shares for f in trade.fills))
        except Exception:
            return 0

    def _trade_fill_price(self, trade) -> float | None:
        total = self._trade_filled_qty(trade)
        if total <= 0:
            return None
        avg = self._clean_price(trade.orderStatus.avgFillPrice)
        if avg is not None:
            return avg
        value = sum(f.execution.shares * f.execution.price for f in trade.fills)
        return float(value / total) if total else None

    def _on_entry_fill(self, contract: Option, quote: OptionQuote, spot: float, limit_price: float, fill: float, qty: int) -> None:
        qty = int(qty)
        target = self._round_price(fill + self.target_credit)
        stop = self._round_price(fill - self.stop_debit)
        now = et_now()
        self.position = PositionState(
            in_position=True,
            qty=qty,
            con_id=contract.conId,
            local_symbol=contract.localSymbol,
            expiry=contract.lastTradeDateOrContractMonth,
            strike=float(contract.strike),
            right=contract.right,
            entry_ts=now.isoformat(),
            entry_price=fill,
            entry_underlying=spot,
            target_price=target,
            stop_price=stop,
        )
        self.stats.trades += 1
        self._save_state()
        self._log_event(
            "ENTER",
            contract,
            quote,
            "BUY",
            qty,
            spot,
            limit_price,
            fill_price=fill,
            entry_price=fill,
            target_price=target,
            stop_price=stop,
            reason="buy_filled",
        )
        self._send_email("ENTER", "BUY", contract, fill, "buy_filled", qty=qty)
        self._submit_target_exit(contract)

    def _submit_target_exit(self, contract: Option) -> None:
        if not self.position.in_position or self.position.target_price is None:
            return
        sell_qty = self._safe_exit_qty()
        if sell_qty <= 0:
            # IB does not (yet) show the position; do not place a SELL that could
            # become naked. _sync_local_position_with_ib owns the reconciliation.
            log.warning(
                "Target exit not submitted: IB shows 0 qty for local position "
                "(will retry once IB position feed catches up)"
            )
            return
        order = LimitOrder("SELL", sell_qty, self.position.target_price)
        order.tif = "DAY"
        order.outsideRth = False
        self.exit_trade = self.ib.placeOrder(contract, order)
        self.position.exit_order_id = int(order.orderId)
        self._save_state()
        self._log_event(
            "TARGET_SUBMIT",
            contract,
            OptionQuote(None, None, None, None),
            "SELL",
            sell_qty,
            self._underlying_price(),
            self.position.target_price,
            entry_price=self.position.entry_price,
            target_price=self.position.target_price,
            stop_price=self.position.stop_price,
            reason="target_after_entry",
        )

    def _manage_position(self, now: datetime) -> None:
        # After the close, do not spin force-exit IOC retries into a market with
        # no bids. 0DTE positions settle at expiration; just stop managing.
        if not is_market_open(now):
            if not self._after_close_logged:
                log.warning(
                    "Market closed with a position still open: %s qty=%s. Not "
                    "force-exiting until the market reopens; a 0DTE option will "
                    "settle at expiration.",
                    self.position.local_symbol,
                    self.position.qty,
                )
                self._after_close_logged = True
            return
        self._after_close_logged = False

        contract = self._contract_from_position()
        if contract is None:
            return

        # Resting target order check. Handle partial fills explicitly so we never
        # wipe the local position while contracts are still held.
        if self.exit_trade is not None:
            tfill = self._trade_fill_price(self.exit_trade)
            if tfill is not None:
                tqty = self._trade_filled_qty(self.exit_trade)
                if tqty >= self.position.qty:
                    self._complete_exit(contract, "target_filled", tfill, self.position.qty)
                else:
                    self._book_partial_exit(contract, tfill, tqty)
                return

        quote = self._quote_option(contract, wait_sec=0.5)
        mark = quote.bid or quote.mid
        if mark is None:
            return
        reason = None
        if self.position.stop_price is not None and mark <= self.position.stop_price:
            reason = "stop"
        elif self.position.entry_ts is not None:
            entry_ts = datetime.fromisoformat(self.position.entry_ts)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=ET)
            if (now - entry_ts).total_seconds() >= self.max_hold_sec:
                reason = "max_hold"
        # Do not let the EOD check overwrite a stop/max_hold reason in the log.
        if reason is None and now.time() >= self.flatten_time:
            reason = "eod_flatten"
        if reason is not None:
            self._force_exit(contract, quote, reason)

    def _book_partial_exit(self, contract: Option, fill: float, filled_qty: int) -> None:
        """Book a partial fill of the resting target, cancel whatever is left of
        that order, and reduce the local position so the remainder is managed
        (and eventually force-exited) at the correct size."""
        if self.exit_trade is not None and not self.exit_trade.isDone():
            try:
                self.ib.cancelOrder(self.exit_trade.order)
            except Exception:
                pass
            self.ib.sleep(0.25)
        final_qty = filled_qty
        final_fill = fill
        if self.exit_trade is not None:
            final_qty = max(final_qty, self._trade_filled_qty(self.exit_trade))
            final_fill = self._trade_fill_price(self.exit_trade) or fill
        if final_qty >= self.position.qty:
            self._complete_exit(contract, "target_filled", final_fill, self.position.qty)
            return
        entry = float(self.position.entry_price or final_fill)
        pnl = (final_fill - entry) * final_qty * 100.0
        pnl_pct = (final_fill / entry - 1.0) * 100.0 if entry > 0 else None
        self.stats.realized_pnl += pnl
        self._log_event(
            "PARTIAL_EXIT",
            contract,
            OptionQuote(None, None, None, None),
            "SELL",
            final_qty,
            self._underlying_price(),
            self.position.target_price,
            fill_price=final_fill,
            entry_price=entry,
            target_price=self.position.target_price,
            stop_price=self.position.stop_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason="target_partial_fill",
        )
        log.warning(
            "Target partially filled %s/%s; remainder %s will be managed and "
            "force-exited (no resting target re-armed)",
            final_qty,
            self.position.qty,
            self.position.qty - final_qty,
        )
        self.position.qty -= final_qty
        self.exit_trade = None
        self.position.exit_order_id = None
        self._save_state()

    def _force_exit(self, contract: Option, quote: OptionQuote, reason: str) -> None:
        # 1) Cancel any still-resting target order, catching a fill during the
        #    cancel. A partial fill is booked; a full fill completes the exit.
        if self.exit_trade is not None and not self.exit_trade.isDone():
            try:
                self.ib.cancelOrder(self.exit_trade.order)
            except Exception:
                pass
            self.ib.sleep(0.25)
            target_fill = self._trade_fill_price(self.exit_trade)
            if target_fill is not None:
                tqty = self._trade_filled_qty(self.exit_trade)
                if tqty >= self.position.qty:
                    self._complete_exit(contract, "target_filled_during_cancel", target_fill, self.position.qty)
                    return
                self._book_partial_exit(contract, target_fill, tqty)
                if not self.position.in_position:
                    return
            else:
                self.exit_trade = None
        else:
            self.exit_trade = None

        # 2) Never sell more than IB actually shows for this position.
        sell_qty = self._safe_exit_qty()
        if sell_qty <= 0:
            log.warning(
                "Force-exit skipped this tick: IB shows 0 qty for the local "
                "position (reason=%s). _sync_local_position_with_ib will clear "
                "stale state if the position is genuinely gone.",
                reason,
            )
            return

        # 3) An exit needs a real quote to be priced. With no quote, retry next
        #    tick rather than firing an effectively-market order. The market-
        #    closed guard above prevents this from spinning forever.
        exit_mark = quote.bid or quote.mid
        if exit_mark is None or exit_mark <= 0:
            self._log_event(
                "EXIT_NO_QUOTE",
                contract,
                quote,
                "SELL",
                sell_qty,
                self._underlying_price(),
                entry_price=self.position.entry_price,
                target_price=self.position.target_price,
                stop_price=self.position.stop_price,
                reason=f"{reason}:no_quote",
            )
            return

        limit_price = self._round_price(exit_mark)
        self._log_event(
            "EXIT_SUBMIT",
            contract,
            quote,
            "SELL",
            sell_qty,
            self._underlying_price(),
            limit_price,
            entry_price=self.position.entry_price,
            target_price=self.position.target_price,
            stop_price=self.position.stop_price,
            reason=reason,
        )
        if self.dry_run:
            return
        order = LimitOrder("SELL", sell_qty, limit_price)
        order.tif = "IOC"
        order.outsideRth = False
        trade = self.ib.placeOrder(contract, order)
        fill, filled_qty = self._wait_fill(trade, timeout=5.0)
        # Re-read after the wait to catch a cancel/fill race on the IOC.
        if filled_qty <= 0 or fill is None:
            fill = self._trade_fill_price(trade)
            filled_qty = self._trade_filled_qty(trade)
        if fill is not None and filled_qty > 0:
            if filled_qty < self.position.qty:
                # IOC only took part of it; book that and keep managing the rest.
                self._book_ioc_partial(contract, fill, filled_qty, reason)
            else:
                self._complete_exit(contract, reason, fill, self.position.qty)
        else:
            self._log_event(
                "EXIT_NOT_FILLED",
                contract,
                quote,
                "SELL",
                sell_qty,
                self._underlying_price(),
                limit_price,
                reason=reason,
            )

    def _book_ioc_partial(self, contract: Option, fill: float, filled_qty: int, reason: str) -> None:
        """A force-exit IOC only partially filled. Book the filled part and
        reduce the position so the next tick force-exits the remainder."""
        filled_qty = min(int(filled_qty), int(self.position.qty))
        entry = float(self.position.entry_price or fill)
        pnl = (fill - entry) * filled_qty * 100.0
        pnl_pct = (fill / entry - 1.0) * 100.0 if entry > 0 else None
        self.stats.realized_pnl += pnl
        self._log_event(
            "EXIT_PARTIAL",
            contract,
            OptionQuote(None, None, None, None),
            "SELL",
            filled_qty,
            self._underlying_price(),
            fill_price=fill,
            entry_price=entry,
            target_price=self.position.target_price,
            stop_price=self.position.stop_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=f"{reason}:partial",
        )
        log.warning(
            "Force-exit IOC partially filled %s/%s; remainder will be retried next tick",
            filled_qty,
            self.position.qty,
        )
        self.position.qty -= filled_qty
        self._save_state()

    def _complete_exit(self, contract: Option, reason: str, fill: float, qty: int | None = None) -> None:
        entry = float(self.position.entry_price or fill)
        qty = int(qty if qty is not None else (self.position.qty or self.qty))
        pnl = (fill - entry) * qty * 100.0
        pnl_pct = (fill / entry - 1.0) * 100.0 if entry > 0 else None
        self.stats.realized_pnl += pnl
        if pnl >= 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1
        self._log_event(
            "EXIT",
            contract,
            OptionQuote(None, None, None, None),
            "SELL",
            qty,
            self._underlying_price(),
            self.position.target_price,
            fill_price=fill,
            entry_price=entry,
            target_price=self.position.target_price,
            stop_price=self.position.stop_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )
        self._send_email("EXIT", "SELL", contract, fill, reason, pnl=pnl, pnl_pct=pnl_pct, qty=qty)
        self.position = PositionState(right=self.right)
        self.exit_trade = None
        self._save_state()

    def _log_event(
        self,
        event: str,
        contract: Option,
        quote: OptionQuote,
        side: str = "",
        qty: int | None = None,
        underlying: float | None = None,
        limit_price: float | None = None,
        fill_price: float | None = None,
        entry_price: float | None = None,
        target_price: float | None = None,
        stop_price: float | None = None,
        pnl: float | None = None,
        pnl_pct: float | None = None,
        reason: str = "",
    ) -> None:
        self.csv_logger.write(
            ts=et_now().isoformat(),
            event=event,
            symbol=self.symbol,
            local_symbol=getattr(contract, "localSymbol", None),
            right=getattr(contract, "right", self.right),
            expiry=getattr(contract, "lastTradeDateOrContractMonth", None),
            strike=getattr(contract, "strike", None),
            side=side,
            qty=qty,
            underlying=underlying,
            bid=quote.bid,
            ask=quote.ask,
            mid=quote.mid,
            spread_pct=quote.spread_pct,
            limit_price=limit_price,
            fill_price=fill_price,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            dry_run=self.dry_run,
        )
        log.info("%s %s %s limit=%s fill=%s reason=%s", event, side, getattr(contract, "localSymbol", ""), limit_price, fill_price, reason)

    def _send_email(
        self,
        event: str,
        side: str,
        contract: Option,
        fill: float,
        reason: str,
        pnl: float | None = None,
        pnl_pct: float | None = None,
        qty: int | None = None,
    ) -> None:
        if self.email_notifier is None:
            return
        mode = "DRY" if self.dry_run else "PAPER"
        qty = int(qty if qty is not None else self.position.qty or self.qty)
        pnl_line = ""
        if pnl is not None:
            pnl_line = f"PnL:          ${pnl:+.2f} ({pnl_pct:+.2f}%)\n"
        subject = f"{self.symbol} 0DTE GRID {mode} {event} {side} {contract.localSymbol} @ {fill:.2f}"
        body = (
            f"QQQ 0DTE grid scalper: {event}\n\n"
            f"合约:         {contract.localSymbol}\n"
            f"方向:         {side}\n"
            f"数量:         {qty}\n"
            f"成交价:       {fill:.2f}\n"
            f"{pnl_line}"
            f"原因:         {reason}\n"
            f"今日交易数:   {self.stats.trades} / {self.max_trades_per_day}\n"
            f"今日PnL:      ${self.stats.realized_pnl:+.2f}\n"
            f"模式:         {mode}\n"
            f"时间:         {et_now().strftime('%Y-%m-%d %H:%M:%S ET')}\n"
        )
        if not self.email_notifier.send_alert(subject, body):
            log.warning("Grid scalper email not sent: %s %s", event, contract.localSymbol)


def configure_logging() -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DEFAULT_LOG_DIR / f"qqq_0dte_grid_scalper_{trading_date_str()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )
    log.info("Log file: %s", log_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QQQ 0DTE OTM passive scalper")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=278)
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--right", choices=["C", "P"], default="C")
    p.add_argument("--qty", type=int, default=1)
    p.add_argument("--max-open-qty", type=int, default=5)
    p.add_argument("--max-trades-per-day", type=int, default=300)
    p.add_argument("--max-daily-loss", type=float, default=250.0)
    p.add_argument("--otm-offset", type=float, default=1.0)
    p.add_argument("--entry-edge", type=float, default=0.10)
    p.add_argument("--target-credit", type=float, default=0.20)
    p.add_argument("--stop-debit", type=float, default=0.15)
    p.add_argument("--max-hold-sec", type=int, default=120)
    p.add_argument("--order-timeout-sec", type=float, default=8.0)
    p.add_argument("--poll-sec", type=float, default=1.0)
    # CHANGED: these were decimal hours (e.g. 15.20 == 15:12, an easy footgun).
    # They are now HH:MM strings, consistent with --flatten-time. Old-style
    # decimal values will now fail loudly at startup instead of mis-parsing.
    p.add_argument("--entry-start", default="09:45", help="Earliest entry time HH:MM ET")
    p.add_argument("--entry-end", default="15:12", help="Latest entry time HH:MM ET")
    p.add_argument("--flatten-time", default="15:45", help="Force-flatten time HH:MM ET")
    p.add_argument("--min-option-price", type=float, default=0.25)
    p.add_argument("--max-option-price", type=float, default=4.00)
    p.add_argument("--max-spread-pct", type=float, default=18.0)
    p.add_argument(
        "--no-flatten-unmanaged",
        action="store_true",
        help="Do not auto-flatten QQQ 0DTE positions that exist in IB but not in local state",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ib-readonly", action="store_true")
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
        scalper = QQQ0DTEGridScalper(
            ib=ib,
            symbol=args.symbol,
            right=args.right,
            qty=args.qty,
            max_open_qty=args.max_open_qty,
            max_trades_per_day=args.max_trades_per_day,
            max_daily_loss=args.max_daily_loss,
            otm_offset=args.otm_offset,
            entry_edge=args.entry_edge,
            target_credit=args.target_credit,
            stop_debit=args.stop_debit,
            max_hold_sec=args.max_hold_sec,
            order_timeout_sec=args.order_timeout_sec,
            poll_sec=args.poll_sec,
            entry_start=args.entry_start,
            entry_end=args.entry_end,
            flatten_time=args.flatten_time,
            min_option_price=args.min_option_price,
            max_option_price=args.max_option_price,
            max_spread_pct=args.max_spread_pct,
            flatten_unmanaged=not args.no_flatten_unmanaged,
            dry_run=args.dry_run,
            email_notifier=_email_notifier_from_args(args),
        )
        scalper.start()
    finally:
        if ib.isConnected():
            ib.disconnect()
        log.info("Stopped")


if __name__ == "__main__":
    main()
