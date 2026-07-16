"""QQQ 0DTE Price Action intraday swing trader with optional GEX diagnostics.

This script combines the Brooks-style PA engine in ``pa_signal.py`` with a
live same-day QQQ option GEX snapshot.  PA defines the entry trigger, direction,
stop, and target.  GEX is logged as diagnostics only and is not used as an entry
or exit gate; options execution is dry-run by default.

Trading model:
  - Feed closed 5-minute QQQ bars into PASignalEngine.
  - Accept H2/H1 spike/H3 long setups when PA risk/reward allows it.
  - Accept L2/L1 spike/L3 short setups when PA risk/reward allows it.
  - Buy a same-day slightly ITM QQQ call/put immediately when PA emits a signal.
  - Exit on underlying stop/target, option stop/take-profit, max hold time,
    or end-of-day flatten.

Run dry:
    python -m gex_monitor.qqq_0dte_gex_pa_swing --dry-run

Paper trading:
    python -m gex_monitor.qqq_0dte_gex_pa_swing --no-dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
from ib_insync import IB, LimitOrder, Option, Stock

from .pa_signal import PASignalEngine
from .time_utils import ET, et_now, is_market_open, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT_DIR / "logs"

log = logging.getLogger("qqq_0dte_gex_pa_swing")


@dataclass
class GexState:
    ts: datetime | None = None
    spot: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    gamma_flip: float | None = None
    total_gex: float = 0.0
    positive_gamma: bool = True
    gex_by_strike: dict[float, float] = field(default_factory=dict)

    @property
    def age_sec(self) -> float | None:
        if self.ts is None:
            return None
        return (et_now() - self.ts).total_seconds()

    def is_fresh(self, max_age_sec: float) -> bool:
        age = self.age_sec
        return age is not None and age <= max_age_sec

    def bias_direction(self, spot: float) -> str | None:
        """Diagnostic flip-side bias only; flip is not used as a trade gate."""
        if self.gamma_flip is None:
            return None
        return "LONG" if spot > self.gamma_flip else "SHORT"

    def wall_for(self, direction: str, spot: float) -> float | None:
        if direction == "LONG" and self.call_wall is not None and self.call_wall > spot:
            return self.call_wall
        if direction == "SHORT" and self.put_wall is not None and self.put_wall < spot:
            return self.put_wall
        return None


@dataclass
class SwingPlan:
    ts: datetime
    direction: str
    setup: str
    score: int
    entry: float
    stop: float
    target: float
    rr_ratio: float
    right: str
    expires_at: datetime
    reason: str
    gex_reason: str

    def triggered(self, spot: float) -> bool:
        return spot >= self.entry if self.direction == "LONG" else spot <= self.entry

    def invalidated(self, spot: float) -> bool:
        return spot <= self.stop if self.direction == "LONG" else spot >= self.stop


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
        return self.close

    @property
    def spread_pct(self) -> float | None:
        mid = self.mid
        if mid is None or mid <= 0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / mid * 100.0


def _clean_positive_number(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _ticker_exposure_quantity(ticker, right: str) -> float | None:
    """Return contract quantity for live GEX: OI first, intraday volume fallback."""
    oi_attr = "callOpenInterest" if str(right).upper() == "C" else "putOpenInterest"
    oi = _clean_positive_number(getattr(ticker, oi_attr, None))
    if oi is not None:
        return oi
    return _clean_positive_number(getattr(ticker, "volume", None))


@dataclass
class PositionState:
    in_position: bool = False
    qty: int = 0
    direction: str | None = None
    right: str | None = None
    entry_ts: str | None = None
    entry_price: float | None = None
    entry_underlying: float | None = None
    underlying_stop: float | None = None
    underlying_target: float | None = None
    setup: str | None = None
    score: int | None = None
    con_id: int | None = None
    local_symbol: str | None = None
    expiry: str | None = None
    strike: float | None = None


class CSVLogger:
    fields = [
        "ts", "event", "symbol", "direction", "setup", "score", "decision",
        "reason", "gex_reason", "spot", "entry", "stop", "target", "rr",
        "right", "local_symbol", "strike", "bid", "ask", "mid",
        "limit_price", "fill_price", "pnl_pct", "dry_run",
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


def calculate_gex(rows: Iterable[tuple[float, str, float, float]], spot: float) -> dict[float, float]:
    """Return strike -> net GEX from rows of (strike, right, open_interest, gamma)."""
    out: dict[float, float] = {}
    for strike, right, oi, gamma in rows:
        if oi <= 0 or not np.isfinite(gamma):
            continue
        val = float(gamma) * float(oi) * 100.0 * (spot**2) * 0.01
        if right.upper() == "C":
            out[float(strike)] = out.get(float(strike), 0.0) + val
        else:
            out[float(strike)] = out.get(float(strike), 0.0) - val
    return out


def calculate_gamma_flip(gex_by_strike: dict[float, float]) -> float | None:
    if not gex_by_strike:
        return None
    strikes = sorted(gex_by_strike)
    total = 0.0
    cumulative: list[float] = []
    for strike in strikes:
        total += gex_by_strike[strike]
        cumulative.append(total)
    for i in range(len(strikes) - 1):
        c0 = cumulative[i]
        c1 = cumulative[i + 1]
        if c0 <= 0 <= c1 or c0 >= 0 >= c1:
            if abs(c1 - c0) < 1e-12:
                return strikes[i]
            t = -c0 / (c1 - c0)
            return strikes[i] + t * (strikes[i + 1] - strikes[i])
    return None


def analyze_gex(gex_by_strike: dict[float, float], spot: float) -> GexState:
    pos = {s: v for s, v in gex_by_strike.items() if v > 0}
    neg = {s: v for s, v in gex_by_strike.items() if v < 0}
    total = float(sum(gex_by_strike.values()))
    return GexState(
        ts=et_now(),
        spot=spot,
        call_wall=max(pos, key=pos.get) if pos else None,
        put_wall=min(neg, key=neg.get) if neg else None,
        gamma_flip=calculate_gamma_flip(gex_by_strike),
        total_gex=total,
        positive_gamma=total > 0,
        gex_by_strike=dict(gex_by_strike),
    )


def _gex_observation_reason(gex: GexState | None) -> str:
    if gex is None or gex.ts is None:
        return "pa_only;gex_obs:missing"
    flip_text = f"{gex.gamma_flip:.2f}" if gex.gamma_flip is not None else "N/A"
    gamma_text = "positive" if gex.positive_gamma else "negative"
    return (
        "pa_only;"
        f"gex_obs:gamma={gamma_text};"
        f"flip={flip_text};"
        f"call_wall={gex.call_wall};"
        f"put_wall={gex.put_wall};"
        f"age_sec={gex.age_sec:.0f}"
    )


def evaluate_pa_signal(
    pa_signal: dict,
    *,
    min_rr: float,
    fallback_rr: float,
    gex: GexState | None = None,
) -> tuple[SwingPlan | None, str]:
    """Convert a PA signal into a tradable swing plan without GEX gating."""
    direction = str(pa_signal["direction"])
    entry = float(pa_signal["entry"])
    stop = float(pa_signal["stop"])
    risk = abs(entry - stop)
    if risk <= 0:
        return None, "invalid_risk"

    target_value = pa_signal.get("target")
    try:
        target = float(target_value)
    except (TypeError, ValueError):
        target = entry + risk * fallback_rr if direction == "LONG" else entry - risk * fallback_rr

    reward = abs(target - entry)
    rr = reward / risk
    if rr < min_rr:
        return None, f"rr_too_low:{rr:.2f}"

    right = "C" if direction == "LONG" else "P"
    expires_at = et_now() + timedelta(minutes=15)
    plan = SwingPlan(
        ts=et_now(),
        direction=direction,
        setup=str(pa_signal["setup"]),
        score=int(pa_signal["score"]),
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        rr_ratio=round(rr, 2),
        right=right,
        expires_at=expires_at,
        reason=str(pa_signal.get("reason", "")),
        gex_reason=_gex_observation_reason(gex),
    )
    return plan, "accepted"


# Backward-compatible name for older tests/manual imports.  The implementation
# is now PA-only; GEX arguments are intentionally ignored for trade gating.
def evaluate_pa_with_gex(
    pa_signal: dict,
    gex: GexState,
    *,
    spot: float,
    max_gex_age_sec: float,
    min_rr: float,
    fallback_rr: float,
    allow_positive_gamma: bool,
    require_wall: bool,
) -> tuple[SwingPlan | None, str]:
    return evaluate_pa_signal(
        pa_signal,
        min_rr=min_rr,
        fallback_rr=fallback_rr,
        gex=gex,
    )


class QQQ0DTEGexPASwingTrader:
    def __init__(
        self,
        ib: IB,
        *,
        symbol: str = "QQQ",
        qty: int = 1,
        dry_run: bool = True,
        account_mode: str = "paper",
        bar_size: str = "5 mins",
        gex_refresh_sec: float = 300.0,
        max_gex_age_sec: float = 420.0,
        num_strikes: int = 10,
        greeks_timeout: float = 4.0,
        max_spread_pct: float = 12.0,
        min_rr: float = 1.2,
        fallback_rr: float = 2.0,
        allow_positive_gamma: bool = False,
        require_wall: bool = False,
        pending_expiry_min: int = 15,
        max_hold_min: int = 75,
        option_stop_pct: float = 35.0,
        option_take_profit_pct: float = 70.0,
        min_time: float = 9.75,
        max_entry_time: float = 14.75,
        flatten_time: str = "15:45",
        poll_sec: float = 1.0,
        log_dir: Path = DEFAULT_LOG_DIR,
        state_path: Path | None = None,
    ):
        self.ib = ib
        self.symbol = symbol.upper()
        self.qty = qty
        self.dry_run = dry_run
        self.account_mode = account_mode
        self.bar_size = bar_size
        self.gex_refresh_sec = gex_refresh_sec
        self.max_gex_age_sec = max_gex_age_sec
        self.num_strikes = num_strikes
        self.greeks_timeout = greeks_timeout
        self.max_spread_pct = max_spread_pct
        self.min_rr = min_rr
        self.fallback_rr = fallback_rr
        self.allow_positive_gamma = allow_positive_gamma
        self.require_wall = require_wall
        self.pending_expiry_min = pending_expiry_min
        self.max_hold_min = max_hold_min
        self.option_stop_pct = option_stop_pct
        self.option_take_profit_pct = option_take_profit_pct
        self.min_time = min_time
        self.max_entry_time = max_entry_time
        self.flatten_time = self._parse_hhmm(flatten_time)
        self.poll_sec = poll_sec
        self.underlying = Stock(self.symbol, "SMART", "USD")
        self.underlying_ticker = None
        self.pa = PASignalEngine(min_score=60, min_rr=1.2)
        self.gex = GexState()
        self.pending: SwingPlan | None = None
        self._entry_queue: list[SwingPlan] = []
        self.positions: list[PositionState] = []
        self._last_gex_refresh = 0.0
        self._last_processed_bar = None
        self._running = True
        suffix = f"{self.symbol.lower()}_0dte_gex_pa_swing_{trading_date_str()}"
        self.logger = CSVLogger(Path(log_dir) / f"{suffix}.csv")
        self.state_path = state_path or Path(log_dir) / f"{suffix}_state.json"

    @staticmethod
    def _parse_hhmm(value: str) -> dtime:
        hh, mm = value.split(":")
        return dtime(int(hh), int(mm))

    @staticmethod
    def _round_price(value: float) -> float:
        return round(max(value, 0.01), 2)

    def stop(self) -> None:
        self._running = False

    def verify_account(self) -> bool:
        accounts = [str(a) for a in self.ib.managedAccounts()]
        if self.account_mode == "paper" and any(a.startswith("DU") for a in accounts):
            return True
        if self.account_mode == "any" and accounts:
            log.warning("Account mode 'any' accepted: %s", accounts)
            return True
        log.error("Blocked: paper account required. accounts=%s", accounts)
        return False

    def start(self) -> None:
        self.ib.qualifyContracts(self.underlying)
        self.underlying_ticker = self.ib.reqMktData(self.underlying, "", False, False)
        if not self.dry_run and not self.verify_account():
            return
        self._warmup_pa()
        self._subscribe_live_bars()
        log.info(
            "Started %s 0DTE PA-only swing: dry_run=%s qty=%s rr>=%.1f hold<=%sm entry=queued multi_position=unlimited",
            self.symbol, self.dry_run, self.qty, self.min_rr, self.max_hold_min,
        )
        while self._running:
            try:
                self._refresh_gex_if_needed()
                self._poll_pending_and_position()
            except Exception:
                log.exception("Loop error")
            self.ib.sleep(self.poll_sec)

    def _warmup_pa(self) -> None:
        bars = self.ib.reqHistoricalData(
            self.underlying,
            endDateTime="",
            durationStr="2 D",
            barSizeSetting=self.bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        for bar in bars[-80:]:
            self.pa.on_bar(bar.open, bar.high, bar.low, bar.close, bar.volume)
        log.info("PA warmup done: state=%s", self.pa.state)

    def _subscribe_live_bars(self) -> None:
        bars_live = self.ib.reqHistoricalData(
            self.underlying,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting=self.bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=True,
        )
        bars_live.updateEvent += self._on_bar_update
        log.info("Live %s bars subscribed", self.bar_size)

    def _on_bar_update(self, bars, has_new_bar: bool) -> None:
        if not has_new_bar or len(bars) < 2:
            return
        ib_bar = bars[-2]
        if ib_bar.date == self._last_processed_bar:
            return
        self._last_processed_bar = ib_bar.date
        if hasattr(ib_bar.date, "time") and not (dtime(9, 30) <= ib_bar.date.time() <= dtime(16, 0)):
            return

        pa_signal = self.pa.on_bar(ib_bar.open, ib_bar.high, ib_bar.low, ib_bar.close, ib_bar.volume)
        log.info(
            "Bar %s close=%.2f PA=%s pending=%s positions=%s",
            ib_bar.date, ib_bar.close, self.pa.state,
            self.pending.setup if self.pending else None, len(self.positions),
        )
        if pa_signal:
            self._handle_pa_signal(pa_signal, float(ib_bar.close))

    def _handle_pa_signal(self, pa_signal: dict, spot: float) -> None:
        if not self._in_entry_window(et_now()):
            self._log_signal(pa_signal, "REJECT", "outside_entry_window", spot)
            return
        plan, reason = evaluate_pa_signal(
            pa_signal,
            min_rr=self.min_rr,
            fallback_rr=self.fallback_rr,
            gex=self.gex,
        )
        if plan is None:
            self._log_signal(pa_signal, "REJECT", reason, spot)
            return
        self.logger.write(
            ts=et_now().isoformat(), event="PLAN", symbol=self.symbol,
            direction=plan.direction, setup=plan.setup, score=plan.score,
            decision="ACCEPT", reason=plan.reason, gex_reason=plan.gex_reason,
            spot=spot, entry=plan.entry, stop=plan.stop, target=plan.target,
            rr=plan.rr_ratio, right=plan.right, dry_run=self.dry_run,
        )
        log.info(
            "PA signal %s %s queued: spot %.2f trigger %.2f stop %.2f target %.2f rr %.2f",
            plan.direction, plan.setup, spot, plan.entry, plan.stop, plan.target, plan.rr_ratio,
        )
        self._entry_queue.append(plan)
        self._save_state()

    def _log_signal(self, pa_signal: dict, decision: str, reason: str, spot: float) -> None:
        self.logger.write(
            ts=et_now().isoformat(), event="SIGNAL", symbol=self.symbol,
            direction=pa_signal.get("direction"), setup=pa_signal.get("setup"),
            score=pa_signal.get("score"), decision=decision, reason=reason,
            spot=spot, entry=pa_signal.get("entry"), stop=pa_signal.get("stop"),
            target=pa_signal.get("target"), rr=pa_signal.get("rr_ratio"),
            dry_run=self.dry_run,
        )
        log.info("Rejected PA signal %s reason=%s", pa_signal, reason)

    def _in_entry_window(self, now: datetime) -> bool:
        hour = now.hour + now.minute / 60.0
        return self.min_time <= hour < self.max_entry_time

    def _underlying_price(self) -> float | None:
        if self.underlying_ticker is None:
            return None
        price = self.underlying_ticker.marketPrice()
        if price is None or not np.isfinite(price) or price <= 0:
            price = self.underlying_ticker.last or self.underlying_ticker.close
        return float(price) if price is not None and np.isfinite(price) and price > 0 else None

    def _poll_pending_and_position(self) -> None:
        spot = self._underlying_price()
        if spot is None:
            return
        now = et_now()
        if self._entry_queue and is_market_open(now):
            self._drain_entry_queue(spot)
        if self.pending is not None:
            if now >= self.pending.expires_at:
                self._cancel_pending("pending_expired", spot)
            elif self.pending.invalidated(spot):
                self._cancel_pending("pending_invalidated_before_trigger", spot)
            elif self.pending.triggered(spot) and is_market_open(now):
                self._enter_from_pending(self.pending, spot)
        for position in list(self.positions):
            self._maybe_exit_position(position, spot, now)

    def _drain_entry_queue(self, spot: float) -> None:
        queued = self._entry_queue
        self._entry_queue = []
        now = et_now()
        for plan in queued:
            if now >= plan.expires_at:
                self._log_plan_cancel(plan, "queued_plan_expired", spot)
                continue
            if plan.invalidated(spot):
                self._log_plan_cancel(plan, "queued_plan_invalidated", spot)
                continue
            self._enter_from_plan(plan, spot)
        self._save_state()

    def _cancel_pending(self, reason: str, spot: float) -> None:
        plan = self.pending
        if plan is None:
            return
        self.logger.write(
            ts=et_now().isoformat(), event="CANCEL_PLAN", symbol=self.symbol,
            direction=plan.direction, setup=plan.setup, score=plan.score,
            decision="CANCEL", reason=reason, gex_reason=plan.gex_reason,
            spot=spot, entry=plan.entry, stop=plan.stop, target=plan.target,
            rr=plan.rr_ratio, right=plan.right, dry_run=self.dry_run,
        )
        log.info("Cancelled pending plan: %s spot=%.2f", reason, spot)
        self.pending = None
        self._save_state()

    def _enter_from_pending(self, plan: SwingPlan, spot: float) -> None:
        if self._enter_from_plan(plan, spot):
            self.pending = None
            self._save_state()

    def _enter_from_plan(self, plan: SwingPlan, spot: float) -> bool:
        contract = self._pick_0dte_option(spot, plan.right)
        if contract is None:
            self._log_plan_cancel(plan, "missing_option_contract", spot)
            return False
        quote = self._quote_option(contract)
        if quote.mid is None:
            self._log_plan_cancel(plan, "missing_option_quote", spot)
            return False
        if quote.spread_pct is None or quote.spread_pct > self.max_spread_pct:
            self._log_plan_cancel(plan, f"spread_too_wide:{quote.spread_pct}", spot)
            return False
        limit_price, fill = self._submit(contract, "BUY", quote)
        if fill is None:
            self._log_plan_cancel(plan, "entry_not_filled", spot)
            return False
        now = et_now()
        position = PositionState(
            in_position=True,
            qty=self.qty,
            direction=plan.direction,
            right=plan.right,
            entry_ts=now.isoformat(),
            entry_price=fill,
            entry_underlying=spot,
            underlying_stop=plan.stop,
            underlying_target=plan.target,
            setup=plan.setup,
            score=plan.score,
            con_id=contract.conId,
            local_symbol=contract.localSymbol,
            expiry=contract.lastTradeDateOrContractMonth,
            strike=float(contract.strike),
        )
        self.positions.append(position)
        self._save_state()
        self.logger.write(
            ts=now.isoformat(), event="ENTER", symbol=self.symbol,
            direction=plan.direction, setup=plan.setup, score=plan.score,
            decision="FILLED", reason=plan.reason, gex_reason=plan.gex_reason,
            spot=spot, entry=plan.entry, stop=plan.stop, target=plan.target,
            rr=plan.rr_ratio, right=plan.right, local_symbol=contract.localSymbol,
            strike=contract.strike, bid=quote.bid, ask=quote.ask, mid=quote.mid,
            limit_price=limit_price, fill_price=fill, dry_run=self.dry_run,
        )
        log.info("Entered %s %s @ %.2f open_positions=%s", contract.localSymbol, plan.direction, fill, len(self.positions))
        return True

    def _log_plan_cancel(self, plan: SwingPlan, reason: str, spot: float) -> None:
        self.logger.write(
            ts=et_now().isoformat(), event="CANCEL_PLAN", symbol=self.symbol,
            direction=plan.direction, setup=plan.setup, score=plan.score,
            decision="CANCEL", reason=reason, gex_reason=plan.gex_reason,
            spot=spot, entry=plan.entry, stop=plan.stop, target=plan.target,
            rr=plan.rr_ratio, right=plan.right, dry_run=self.dry_run,
        )
        log.info("Cancelled PA plan: %s spot=%.2f", reason, spot)

    def _maybe_exit_position(self, position: PositionState, spot: float, now: datetime) -> None:
        contract = self._contract_from_position(position)
        if contract is None or position.entry_price is None:
            return
        quote = self._quote_option(contract, wait_sec=1.0)
        mark = quote.mid
        if mark is None:
            return
        pnl_pct = (mark / float(position.entry_price) - 1.0) * 100.0
        reason = self._exit_reason(position, spot, now, pnl_pct)
        if reason is not None:
            self._exit(position, contract, quote, reason, pnl_pct, spot)

    def _exit_reason(self, position: PositionState, spot: float, now: datetime, pnl_pct: float) -> str | None:
        direction = position.direction
        if direction == "LONG":
            if position.underlying_stop is not None and spot <= position.underlying_stop:
                return "underlying_stop"
            if position.underlying_target is not None and spot >= position.underlying_target:
                return "underlying_target"
        elif direction == "SHORT":
            if position.underlying_stop is not None and spot >= position.underlying_stop:
                return "underlying_stop"
            if position.underlying_target is not None and spot <= position.underlying_target:
                return "underlying_target"
        if pnl_pct <= -self.option_stop_pct:
            return "option_stop"
        if pnl_pct >= self.option_take_profit_pct:
            return "option_take_profit"
        if position.entry_ts:
            entry_ts = datetime.fromisoformat(position.entry_ts)
            if now - entry_ts >= timedelta(minutes=self.max_hold_min):
                return "max_hold"
        if now.time() >= self.flatten_time:
            return "eod_flatten"
        return None

    def _exit(self, position: PositionState, contract: Option, quote: OptionQuote, reason: str, mark_pnl_pct: float, spot: float) -> None:
        limit_price, fill = self._submit(contract, "SELL", quote)
        if fill is None:
            return
        entry = float(position.entry_price or fill)
        pnl_pct = (fill / entry - 1.0) * 100.0
        self.logger.write(
            ts=et_now().isoformat(), event="EXIT", symbol=self.symbol,
            direction=position.direction, setup=position.setup,
            score=position.score, decision="FILLED", reason=reason,
            spot=spot, stop=position.underlying_stop,
            target=position.underlying_target, right=position.right,
            local_symbol=contract.localSymbol, strike=contract.strike,
            bid=quote.bid, ask=quote.ask, mid=quote.mid,
            limit_price=limit_price, fill_price=fill, pnl_pct=pnl_pct,
            dry_run=self.dry_run,
        )
        log.info("Exited %s reason=%s pnl=%.1f%%", contract.localSymbol, reason, pnl_pct)
        self.positions = [p for p in self.positions if p is not position]
        self._save_state()

    def _refresh_gex_if_needed(self) -> None:
        now = time.time()
        if now - self._last_gex_refresh < self.gex_refresh_sec:
            return
        market_now = et_now().time()
        if not (dtime(9, 25) <= market_now <= dtime(16, 5)):
            return
        spot = self._underlying_price()
        if spot is None:
            return
        try:
            state = self._fetch_live_gex(spot)
        finally:
            self._last_gex_refresh = now
        if state is not None:
            self.gex = state
            log.info(
                "GEX spot=%.2f total=%.2fM flip=%s call_wall=%s put_wall=%s positive=%s",
                spot,
                self.gex.total_gex / 1e6,
                f"{self.gex.gamma_flip:.2f}" if self.gex.gamma_flip is not None else "N/A",
                self.gex.call_wall,
                self.gex.put_wall,
                self.gex.positive_gamma,
            )

    def _fetch_live_gex(self, spot: float) -> GexState | None:
        expiry = trading_date_str()
        chains = self.ib.reqSecDefOptParams(
            self.symbol, "", self.underlying.secType, self.underlying.conId
        )
        chain = (
            next((c for c in chains if c.exchange == "SMART" and c.tradingClass == self.symbol), None)
            or next((c for c in chains if c.exchange == "SMART"), None)
            or (chains[0] if chains else None)
        )
        if chain is None or expiry not in {str(e) for e in chain.expirations}:
            log.warning("No 0DTE option chain for %s expiry=%s", self.symbol, expiry)
            return None
        strikes = sorted(float(s) for s in chain.strikes if s and s > 0)
        if not strikes:
            return None
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        selected = strikes[max(0, atm_idx - self.num_strikes): atm_idx + self.num_strikes + 1]
        contracts = [
            Option(self.symbol, expiry, strike, right, "SMART", tradingClass=chain.tradingClass)
            for strike in selected
            for right in ("C", "P")
        ]
        qualified = [c for c in self.ib.qualifyContracts(*contracts) if c.conId]
        rows: list[tuple[float, str, float, float]] = []
        for contract in qualified:
            data = self._fetch_greek_row(contract)
            if data is not None:
                rows.append(data)
        if not rows:
            log.warning("No Greeks/OI received for GEX refresh")
            return None
        return analyze_gex(calculate_gex(rows, spot), spot)

    def _fetch_greek_row(self, contract: Option) -> tuple[float, str, float, float] | None:
        ticker = self.ib.reqMktData(contract, "100,101,104,106", False, False)
        start = time.time()
        try:
            while time.time() - start < self.greeks_timeout:
                self.ib.sleep(0.2)
                greeks = ticker.modelGreeks
                qty = _ticker_exposure_quantity(ticker, contract.right)
                if greeks and greeks.gamma is not None and qty is not None and qty > 0:
                    gamma = float(greeks.gamma)
                    if np.isfinite(gamma):
                        return (float(contract.strike), contract.right, float(qty), gamma)
            return None
        finally:
            self.ib.cancelMktData(contract)

    def _pick_0dte_option(self, spot: float, right: str) -> Option | None:
        expiry = trading_date_str()
        chains = self.ib.reqSecDefOptParams(
            self.symbol, "", self.underlying.secType, self.underlying.conId
        )
        chain = (
            next((c for c in chains if c.exchange == "SMART" and c.tradingClass == self.symbol), None)
            or next((c for c in chains if c.exchange == "SMART"), None)
            or (chains[0] if chains else None)
        )
        if chain is None or expiry not in {str(e) for e in chain.expirations}:
            return None
        strikes = sorted(float(s) for s in chain.strikes if s and s > 0)
        if not strikes:
            return None
        if right == "C":
            candidates = [s for s in strikes if s <= spot] or strikes
        else:
            candidates = [s for s in strikes if s >= spot] or strikes
        strike = min(candidates, key=lambda s: abs(s - spot))
        contract = Option(self.symbol, expiry, strike, right, "SMART", tradingClass=chain.tradingClass)
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else None

    def _quote_option(self, contract: Option, wait_sec: float = 1.5) -> OptionQuote:
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

    @staticmethod
    def _clean_price(value) -> float | None:
        return _clean_positive_number(value)

    def _limit_for_side(self, side: str, quote: OptionQuote) -> float | None:
        if side == "BUY":
            return self._round_price(quote.ask or quote.mid or 0.0)
        return self._round_price(quote.bid or quote.mid or 0.0)

    def _submit(self, contract: Option, side: str, quote: OptionQuote) -> tuple[float | None, float | None]:
        limit_price = self._limit_for_side(side, quote)
        if limit_price is None or limit_price <= 0:
            return None, None
        if self.dry_run:
            log.info("[DRY] %s %s %s @ %.2f", side, self.qty, contract.localSymbol, limit_price)
            return limit_price, quote.mid or limit_price
        order = LimitOrder(side, self.qty, limit_price)
        order.tif = "IOC"
        order.outsideRth = False
        trade = self.ib.placeOrder(contract, order)
        fill = self._wait_fill(trade)
        if fill is None:
            try:
                self.ib.cancelOrder(order)
            except Exception:
                pass
        return limit_price, fill

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

    def _contract_from_position(self, position: PositionState) -> Option | None:
        if not position.expiry or position.strike is None or not position.right:
            return None
        contract = Option(self.symbol, position.expiry, position.strike, position.right, "SMART")
        if position.con_id:
            contract.conId = int(position.con_id)
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else contract

    def _save_state(self) -> None:
        def _plan_payload(plan: SwingPlan) -> dict:
            payload = asdict(plan)
            payload["ts"] = plan.ts.isoformat()
            payload["expires_at"] = plan.expires_at.isoformat()
            return payload

        pending = None
        if self.pending is not None:
            pending = _plan_payload(self.pending)
        payload = {
            "updated": et_now().isoformat(),
            "pending": pending,
            "entry_queue": [_plan_payload(plan) for plan in self._entry_queue],
            "position": asdict(self.positions[0]) if self.positions else asdict(PositionState()),
            "positions": [asdict(position) for position in self.positions],
            "open_positions": len(self.positions),
            "gex": {
                "ts": self.gex.ts.isoformat() if self.gex.ts else None,
                "spot": self.gex.spot,
                "call_wall": self.gex.call_wall,
                "put_wall": self.gex.put_wall,
                "gamma_flip": self.gex.gamma_flip,
                "total_gex": self.gex.total_gex,
                "positive_gamma": self.gex.positive_gamma,
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_logging() -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DEFAULT_LOG_DIR / f"qqq_0dte_gex_pa_swing_{trading_date_str()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )
    log.info("Log file: %s", log_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QQQ 0DTE PA intraday swing trader with GEX diagnostics")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=88)
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--qty", type=int, default=1)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    p.add_argument("--ib-readonly", action="store_true")
    p.add_argument("--account-mode", choices=["paper", "any"], default="paper")
    p.add_argument("--bar-size", default="5 mins")
    p.add_argument("--gex-refresh-sec", type=float, default=300.0)
    p.add_argument("--max-gex-age-sec", type=float, default=420.0)
    p.add_argument("--num-strikes", type=int, default=10)
    p.add_argument("--max-spread-pct", type=float, default=12.0)
    p.add_argument("--min-rr", type=float, default=1.2)
    p.add_argument("--fallback-rr", type=float, default=2.0)
    p.add_argument("--allow-positive-gamma", action="store_true")
    p.add_argument("--require-wall", action="store_true")
    p.add_argument("--pending-expiry-min", type=int, default=15)
    p.add_argument("--max-hold-min", type=int, default=75)
    p.add_argument("--option-stop-pct", type=float, default=35.0)
    p.add_argument("--option-take-profit-pct", type=float, default=70.0)
    p.add_argument("--min-time", type=float, default=9.75)
    p.add_argument("--max-entry-time", type=float, default=14.75)
    p.add_argument("--flatten-time", default="15:45")
    p.add_argument("--poll-sec", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.ib_readonly and not args.dry_run:
        log.error("Refusing to trade with --ib-readonly and --no-dry-run together")
        return
    ib = IB()
    trader: QQQ0DTEGexPASwingTrader | None = None

    def _handle_signal(*_) -> None:
        if trader is not None:
            trader.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        readonly = args.ib_readonly or args.dry_run
        log.info("Connecting IB %s:%s clientId=%s readonly=%s", args.host, args.port, args.client_id, readonly)
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=20, readonly=readonly)
        trader = QQQ0DTEGexPASwingTrader(
            ib,
            symbol=args.symbol,
            qty=args.qty,
            dry_run=args.dry_run,
            account_mode=args.account_mode,
            bar_size=args.bar_size,
            gex_refresh_sec=args.gex_refresh_sec,
            max_gex_age_sec=args.max_gex_age_sec,
            num_strikes=args.num_strikes,
            max_spread_pct=args.max_spread_pct,
            min_rr=args.min_rr,
            fallback_rr=args.fallback_rr,
            allow_positive_gamma=args.allow_positive_gamma,
            require_wall=args.require_wall,
            pending_expiry_min=args.pending_expiry_min,
            max_hold_min=args.max_hold_min,
            option_stop_pct=args.option_stop_pct,
            option_take_profit_pct=args.option_take_profit_pct,
            min_time=args.min_time,
            max_entry_time=args.max_entry_time,
            flatten_time=args.flatten_time,
            poll_sec=args.poll_sec,
        )
        trader.start()
    finally:
        if ib.isConnected():
            ib.disconnect()
        log.info("Stopped")


if __name__ == "__main__":
    main()
