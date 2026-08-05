"""Trade QQQ/SPY 0DTE option triads from Universal v2.1 momentum signals.

This runner is deliberately paper-only.  It consumes completed one-minute
bars from IB Gateway, calculates the indicator, buys calls on a new long
arrow or puts on a new short arrow, and closes the group when that arrow is
no longer present.  Each group targets three strikes: about $1 ITM, ATM, and
about $1 OTM.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import time as dtime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from ib_insync import IB, LimitOrder, Option, Stock

from .multi_asset_momentum import MomentumConfig, calculate_momentum_signals
from .time_utils import ET, et_now, is_market_open, trading_date_str

log = logging.getLogger("momentum_0dte_paper_trader")

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT_DIR / "logs"
PAPER_PORTS = {4002, 7497}
ORDER_REF_PREFIX = "MV21"


@dataclass(frozen=True)
class StrikeChoice:
    role: Literal["ITM", "ATM", "OTM"]
    strike: float


@dataclass
class OptionLeg:
    role: str
    con_id: int
    local_symbol: str
    expiry: str
    strike: float
    right: str
    qty: int
    entry_price: float | None = None


@dataclass
class SymbolPosition:
    symbol: str
    direction: Literal["long", "short"]
    entry_bar: str
    legs: list[OptionLeg] = field(default_factory=list)


@dataclass(frozen=True)
class Quote:
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2.0
        return self.last or self.close

    @property
    def spread_pct(self) -> float | None:
        mid = self.mid
        if mid is None or mid <= 0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / mid * 100.0


def select_triad_strikes(
    strikes: list[float] | tuple[float, ...], spot: float, right: str
) -> list[StrikeChoice]:
    """Select distinct strikes nearest spot-$1, spot, and spot+$1.

    Roles are from the option holder's perspective: lower strike is ITM for
    calls and OTM for puts; upper strike is the reverse.
    """

    available = sorted({float(value) for value in strikes if value and value > 0})
    if spot <= 0 or len(available) < 3:
        return []
    atm = min(available, key=lambda value: (abs(value - spot), value))
    lower = [value for value in available if value < atm]
    upper = [value for value in available if value > atm]
    if not lower or not upper:
        return []
    low = min(lower, key=lambda value: (abs(value - (spot - 1.0)), -value))
    high = min(upper, key=lambda value: (abs(value - (spot + 1.0)), value))
    if right.upper() == "C":
        return [StrikeChoice("ITM", low), StrikeChoice("ATM", atm), StrikeChoice("OTM", high)]
    if right.upper() == "P":
        return [StrikeChoice("ITM", high), StrikeChoice("ATM", atm), StrikeChoice("OTM", low)]
    raise ValueError("right must be C or P")


def arrow_direction(row: pd.Series) -> Literal["long", "short"] | None:
    """Return the persistent full-vote arrow direction for one bar."""

    if bool(row["long_signal"]):
        return "long"
    if bool(row["short_signal"]):
        return "short"
    return None


class TradeCSVLogger:
    fields = [
        "ts",
        "event",
        "symbol",
        "direction",
        "bar_ts",
        "score",
        "role",
        "local_symbol",
        "right",
        "expiry",
        "strike",
        "side",
        "requested_qty",
        "filled_qty",
        "bid",
        "ask",
        "limit_price",
        "fill_price",
        "pnl_pct",
        "pnl_usd",
        "reason",
        "dry_run",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def write(self, **values) -> None:
        row = {name: values.get(name, "") for name in self.fields}
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)


class DailyBarJournal:
    """Persist the complete current RTH session for deterministic replay."""

    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dema_fast",
        "dema_slow",
        "vwap",
        "dev_z",
        "kalman_velocity",
        "atr",
        "sig1",
        "sig2",
        "sig3",
        "score",
        "effective_vote",
        "long_signal",
        "short_signal",
        "long_entry",
        "short_entry",
        "early_long_entry",
        "early_short_entry",
        "in_session",
        "atr_filter",
    ]

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_current_session(self, symbol: str, result: pd.DataFrame) -> Path:
        if result.empty:
            raise ValueError("cannot journal an empty signal frame")
        et_index = result.index.tz_convert(ET)
        session_date = et_index[-1].date()
        mask = np.asarray(et_index.date == session_date)
        available = [column for column in self.columns if column in result.columns]
        output = result.loc[mask, available].copy()
        output.index = output.index.tz_convert(ET)
        output.index.name = "timestamp"
        path = self.directory / f"{symbol}_{session_date:%Y%m%d}.csv"
        temporary = path.with_suffix(".csv.tmp")
        output.to_csv(temporary)
        temporary.replace(path)
        return path


class Momentum0DTEPaperTrader:
    """IB event-driven paper trader for QQQ and SPY."""

    def __init__(
        self,
        ib: IB,
        *,
        ib_port: int,
        symbols: tuple[str, ...] = ("QQQ", "SPY"),
        qty_per_leg: int = 1,
        dry_run: bool = True,
        bar_size: str = "1 min",
        max_spread_pct: float = 30.0,
        order_timeout_sec: float = 15.0,
        flatten_time: str = "15:50",
        state_path: Path | None = None,
        trade_log_path: Path | None = None,
        bar_data_dir: Path | None = None,
        optimization_config_path: Path | None = None,
        auto_apply_optimization: bool = False,
    ):
        if qty_per_leg < 1:
            raise ValueError("qty_per_leg must be >= 1")
        self.ib = ib
        self.ib_port = int(ib_port)
        self.symbols = tuple(symbol.upper() for symbol in symbols)
        self.qty_per_leg = int(qty_per_leg)
        self.dry_run = bool(dry_run)
        self.bar_size = bar_size
        self.max_spread_pct = float(max_spread_pct)
        self.order_timeout_sec = float(order_timeout_sec)
        self.flatten_time = dtime.fromisoformat(flatten_time)
        state_name = (
            "momentum_0dte_dry_run_state.json" if self.dry_run else "momentum_0dte_paper_state.json"
        )
        log_name = (
            f"momentum_0dte_dry_run_trades_{trading_date_str()}.csv"
            if self.dry_run
            else f"momentum_0dte_paper_trades_{trading_date_str()}.csv"
        )
        self.state_path = state_path or DEFAULT_LOG_DIR / state_name
        self.trade_logger = TradeCSVLogger(trade_log_path or DEFAULT_LOG_DIR / log_name)
        self.bar_journal = DailyBarJournal(bar_data_dir or DEFAULT_LOG_DIR / "momentum_0dte_bars")
        self.optimization_config_path = (
            optimization_config_path
            or DEFAULT_LOG_DIR / "momentum_optimization" / "latest_recommendations.json"
        )
        self.auto_apply_optimization = bool(auto_apply_optimization)
        self.indicator_configs = self._load_indicator_configs()
        self.account: str | None = None
        self.underlyings: dict[str, Stock] = {}
        self.live_bars: dict[str, object] = {}
        self.last_processed_bar: dict[str, pd.Timestamp] = {}
        self.blocked_symbols: set[str] = set()
        self.positions: dict[str, SymbolPosition] = self._load_state()
        self._running = True

    def _load_indicator_configs(self) -> dict[str, MomentumConfig]:
        configs = {symbol: MomentumConfig(asset=symbol) for symbol in self.symbols}
        if not self.auto_apply_optimization:
            return configs
        try:
            payload = json.loads(self.optimization_config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            log.warning("No optimization config yet: %s", self.optimization_config_path)
            return configs
        except Exception:
            log.exception("Invalid optimization config; using v2.1 defaults")
            return configs
        allowed = {
            "fast_n",
            "slow_n",
            "z_lo",
            "z_hi",
            "z_std_win",
            "kalman_q_price",
            "kalman_q_vel",
            "kalman_r",
            "kalman_thresh",
            "atr_len",
            "atr_min",
            "vote_thresh",
            "no_trade_open",
            "no_trade_close",
            "early_confirm_bars",
            "early_vol_mult",
        }
        for symbol in self.symbols:
            recommendation = payload.get("recommendations", {}).get(symbol, {})
            if recommendation.get("status") != "recommended":
                continue
            params = {
                key: value
                for key, value in recommendation.get("params", {}).items()
                if key in allowed
            }
            try:
                candidate = MomentumConfig(asset=symbol, **params)
                if (
                    candidate.fast_n < 2
                    or candidate.slow_n < 5
                    or candidate.fast_n >= candidate.slow_n
                    or candidate.z_lo <= 0
                    or candidate.z_hi <= candidate.z_lo
                    or (candidate.kalman_thresh is not None and candidate.kalman_thresh < 0)
                    or (candidate.atr_min is not None and candidate.atr_min < 0)
                    or candidate.vote_thresh not in (1, 2, 3)
                ):
                    raise ValueError("optimized parameters failed safety constraints")
                configs[symbol] = candidate
                log.warning("Applied optimized %s parameters: %s", symbol, params)
            except (TypeError, ValueError):
                log.exception("Rejected optimized parameters for %s", symbol)
        return configs

    def _load_state(self) -> dict[str, SymbolPosition]:
        if not self.state_path.exists():
            return {}
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            positions: dict[str, SymbolPosition] = {}
            for symbol, item in raw.get("positions", {}).items():
                legs = [OptionLeg(**leg) for leg in item.get("legs", [])]
                positions[symbol] = SymbolPosition(
                    symbol=item["symbol"],
                    direction=item["direction"],
                    entry_bar=item["entry_bar"],
                    legs=legs,
                )
            return positions
        except Exception:
            log.exception(
                "Could not load state %s; order submission will be blocked", self.state_path
            )
            self.blocked_symbols = set(self.symbols)
            return {}

    def _save_state(self) -> None:
        payload = {
            "updated": et_now().isoformat(),
            "account": self.account,
            "positions": {symbol: asdict(position) for symbol, position in self.positions.items()},
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def verify_paper_account(self) -> bool:
        accounts = [str(account) for account in (self.ib.managedAccounts() or [])]
        if self.ib_port not in PAPER_PORTS:
            log.error(
                "BLOCKED: expected IB paper port %s, got %s", sorted(PAPER_PORTS), self.ib_port
            )
            return False
        if len(accounts) != 1 or not accounts[0].startswith("DU"):
            log.error("BLOCKED: exactly one DU paper account is required; accounts=%s", accounts)
            return False
        self.account = accounts[0]
        return True

    def start(self) -> None:
        # Dry run may inspect any connected account; actual orders never can.
        if not self.dry_run and not self.verify_paper_account():
            raise RuntimeError("paper account guard failed")
        if self.dry_run:
            accounts = [str(account) for account in (self.ib.managedAccounts() or [])]
            self.account = accounts[0] if len(accounts) == 1 else None
        self._qualify_underlyings()
        self._cancel_own_stale_orders()
        self._reconcile_positions()
        for symbol in self.symbols:
            self._subscribe(symbol)
        log.info(
            "Started momentum 0DTE trader: symbols=%s qty/leg=%s dry_run=%s account=%s",
            self.symbols,
            self.qty_per_leg,
            self.dry_run,
            self.account,
        )
        while self._running:
            try:
                self._safety_tick()
            except Exception:
                log.exception("Safety loop error")
            self.ib.sleep(1.0)

    def stop(self) -> None:
        self._running = False

    def _qualify_underlyings(self) -> None:
        raw = [Stock(symbol, "SMART", "USD") for symbol in self.symbols]
        qualified = list(self.ib.qualifyContracts(*raw))
        by_symbol = {contract.symbol.upper(): contract for contract in qualified}
        missing = [symbol for symbol in self.symbols if symbol not in by_symbol]
        if missing:
            raise RuntimeError(f"could not qualify underlyings: {missing}")
        self.underlyings = by_symbol

    def _cancel_own_stale_orders(self) -> None:
        for trade in self.ib.openTrades():
            order_ref = str(getattr(trade.order, "orderRef", "") or "")
            if order_ref.startswith(ORDER_REF_PREFIX):
                log.warning("Cancelling stale strategy order %s", order_ref)
                self.ib.cancelOrder(trade.order)

    def _broker_option_positions(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for position in self.ib.positions():
            contract = position.contract
            if (
                contract.secType == "OPT"
                and contract.symbol.upper() in self.symbols
                and int(position.position) != 0
            ):
                result[int(contract.conId)] = int(position.position)
        return result

    def _reconcile_positions(self) -> None:
        broker = self._broker_option_positions()
        managed_ids = {leg.con_id for position in self.positions.values() for leg in position.legs}
        unmanaged_ids = set(broker).difference(managed_ids)
        if unmanaged_ids:
            # Never adopt or flatten an option position that this strategy
            # cannot prove it owns.
            for position in self.ib.positions():
                if int(position.contract.conId or 0) in unmanaged_ids:
                    self.blocked_symbols.add(position.contract.symbol.upper())
            log.error(
                "Unmanaged QQQ/SPY option positions found; blocked symbols=%s", self.blocked_symbols
            )

        changed = False
        for symbol, position in list(self.positions.items()):
            live_legs: list[OptionLeg] = []
            for leg in position.legs:
                broker_qty = broker.get(leg.con_id, 0)
                if broker_qty > 0:
                    leg.qty = min(leg.qty, broker_qty)
                    live_legs.append(leg)
            if live_legs:
                position.legs = live_legs
            else:
                self.positions.pop(symbol)
            changed = True
        if changed:
            self._save_state()

    def _subscribe(self, symbol: str) -> None:
        bars = self.ib.reqHistoricalData(
            self.underlyings[symbol],
            endDateTime="",
            durationStr="2 D",
            barSizeSetting=self.bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            keepUpToDate=True,
        )
        bars.updateEvent += lambda updated, has_new, s=symbol: self._on_bar_update(
            s, updated, has_new
        )
        self.live_bars[symbol] = bars
        log.info("Subscribed %s %s completed bars", symbol, self.bar_size)

    @staticmethod
    def _bars_to_frame(bars) -> pd.DataFrame:
        rows = [
            {
                "timestamp": pd.Timestamp(bar.date),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in bars
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        index = pd.DatetimeIndex(frame.pop("timestamp"))
        if index.tz is None:
            index = index.tz_localize(ET)
        frame.index = index
        return frame[~frame.index.duplicated(keep="last")].sort_index()

    def _on_bar_update(self, symbol: str, bars, has_new_bar: bool) -> None:
        if not has_new_bar or len(bars) < 2:
            return
        completed = pd.Timestamp(bars[-2].date)
        if completed == self.last_processed_bar.get(symbol):
            return
        self.last_processed_bar[symbol] = completed
        try:
            frame = self._bars_to_frame(bars[:-1])
            if frame.empty:
                return
            result = calculate_momentum_signals(
                frame,
                symbol=symbol,
                config=self.indicator_configs[symbol],
            )
            self.bar_journal.write_current_session(symbol, result)
            row = result.iloc[-1]
            self._handle_completed_bar(symbol, result.index[-1], row)
        except Exception:
            log.exception("Failed to process completed %s bar %s", symbol, completed)

    def _handle_completed_bar(self, symbol: str, bar_ts: pd.Timestamp, row: pd.Series) -> None:
        direction = arrow_direction(row)
        position = self.positions.get(symbol)
        log.info(
            "%s %s close=%.2f score=%s arrow=%s position=%s",
            symbol,
            bar_ts,
            float(row["close"]),
            int(row["score"]),
            direction,
            position.direction if position else None,
        )

        if position is not None and direction != position.direction:
            reason = "arrow_disappeared" if direction is None else "arrow_reversed"
            if not self._exit_group(position, bar_ts, int(row["score"]), reason):
                return
            position = None

        is_new_long = bool(row["long_entry"])
        is_new_short = bool(row["short_entry"])
        entry_direction = "long" if is_new_long else "short" if is_new_short else None
        if position is None and entry_direction is not None:
            if entry_direction != direction:
                return
            self._enter_group(
                symbol,
                entry_direction,
                bar_ts,
                float(row["close"]),
                int(row["score"]),
            )

    def _option_chain(self, symbol: str):
        underlying = self.underlyings[symbol]
        chains = self.ib.reqSecDefOptParams(symbol, "", underlying.secType, underlying.conId)
        return next(
            (
                chain
                for chain in chains
                if chain.exchange == "SMART" and chain.tradingClass == symbol
            ),
            None,
        ) or next((chain for chain in chains if chain.tradingClass == symbol), None)

    def _pick_contracts(self, symbol: str, spot: float, right: str) -> list[tuple[str, Option]]:
        chain = self._option_chain(symbol)
        expiry = trading_date_str()
        if chain is None or expiry not in {str(value) for value in chain.expirations}:
            log.warning("No same-day %s option chain for %s", expiry, symbol)
            return []
        choices = select_triad_strikes(tuple(chain.strikes), spot, right)
        raw = [
            Option(
                symbol,
                expiry,
                choice.strike,
                right,
                "SMART",
                currency="USD",
                tradingClass=chain.tradingClass,
            )
            for choice in choices
        ]
        qualified = list(self.ib.qualifyContracts(*raw)) if raw else []
        by_strike = {float(contract.strike): contract for contract in qualified if contract.conId}
        return [
            (choice.role, by_strike[choice.strike])
            for choice in choices
            if choice.strike in by_strike
        ]

    @staticmethod
    def _clean_price(value) -> float | None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if np.isfinite(price) and price > 0 else None

    def _quote(self, contract: Option, wait_sec: float = 1.25) -> Quote:
        ticker = self.ib.reqMktData(contract, "", False, False)
        try:
            self.ib.sleep(wait_sec)
            return Quote(
                bid=self._clean_price(ticker.bid),
                ask=self._clean_price(ticker.ask),
                last=self._clean_price(ticker.last),
                close=self._clean_price(ticker.close),
            )
        finally:
            self.ib.cancelMktData(contract)

    @staticmethod
    def _round_option_price(value: float) -> float:
        # QQQ/SPY option minimum tick is normally $0.01 in this price range.
        return round(max(0.01, value) + 1e-9, 2)

    def _submit(
        self, contract: Option, side: str, qty: int, quote: Quote, order_ref: str
    ) -> tuple[int, float | None, float | None]:
        reference = quote.ask or quote.mid if side == "BUY" else quote.bid or quote.mid
        if reference is None:
            return 0, None, None
        limit_price = self._round_option_price(reference)
        if self.dry_run:
            log.info("[DRY] %s %s %s @ %.2f", side, qty, contract.localSymbol, limit_price)
            return qty, limit_price, limit_price
        order = LimitOrder(
            side,
            qty,
            limit_price,
            tif="IOC",
            outsideRth=False,
            account=self.account,
            orderRef=order_ref,
        )
        trade = self.ib.placeOrder(contract, order)
        started = time.monotonic()
        while time.monotonic() - started < self.order_timeout_sec:
            self.ib.sleep(0.2)
            if trade.isDone() or trade.orderStatus.status == "Filled":
                break
        filled_qty = int(round(sum(float(fill.execution.shares) for fill in trade.fills)))
        if not trade.isDone():
            self.ib.cancelOrder(order)
        if filled_qty <= 0:
            return 0, limit_price, None
        avg = self._clean_price(trade.orderStatus.avgFillPrice)
        if avg is None:
            value = sum(
                float(fill.execution.shares) * float(fill.execution.price) for fill in trade.fills
            )
            avg = value / filled_qty
        return filled_qty, limit_price, avg

    def _entry_quote_ok(self, quote: Quote) -> bool:
        spread = quote.spread_pct
        return quote.ask is not None and spread is not None and spread <= self.max_spread_pct

    def _enter_group(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        bar_ts: pd.Timestamp,
        spot: float,
        score: int,
    ) -> None:
        if symbol in self.blocked_symbols or not is_market_open(et_now()):
            log.warning("Entry blocked for %s", symbol)
            return
        right = "C" if direction == "long" else "P"
        contracts = self._pick_contracts(symbol, spot, right)
        if len(contracts) != 3:
            log.error(
                "Entry aborted: expected 3 qualified %s contracts, got %s", symbol, len(contracts)
            )
            return

        # Preflight all quotes so one obviously untradeable leg does not
        # create an avoidable partial group.
        quoted: list[tuple[str, Option, Quote]] = []
        for role, contract in contracts:
            quote = self._quote(contract)
            if not self._entry_quote_ok(quote):
                log.warning(
                    "Entry aborted: bad quote %s bid=%s ask=%s spread=%s",
                    contract.localSymbol,
                    quote.bid,
                    quote.ask,
                    quote.spread_pct,
                )
                return
            quoted.append((role, contract, quote))

        position = SymbolPosition(symbol=symbol, direction=direction, entry_bar=bar_ts.isoformat())
        for role, contract, quote in quoted:
            order_ref = f"{ORDER_REF_PREFIX}-{symbol}-{direction}-{role}-{trading_date_str()}"
            filled, limit_price, fill_price = self._submit(
                contract, "BUY", self.qty_per_leg, quote, order_ref
            )
            self.trade_logger.write(
                ts=et_now().isoformat(),
                event="ENTER",
                symbol=symbol,
                direction=direction,
                bar_ts=bar_ts.isoformat(),
                score=score,
                role=role,
                local_symbol=contract.localSymbol,
                right=right,
                expiry=contract.lastTradeDateOrContractMonth,
                strike=contract.strike,
                side="BUY",
                requested_qty=self.qty_per_leg,
                filled_qty=filled,
                bid=quote.bid,
                ask=quote.ask,
                limit_price=limit_price,
                fill_price=fill_price,
                reason="new_full_arrow",
                dry_run=self.dry_run,
            )
            if filled > 0:
                position.legs.append(
                    OptionLeg(
                        role=role,
                        con_id=int(contract.conId),
                        local_symbol=contract.localSymbol,
                        expiry=contract.lastTradeDateOrContractMonth,
                        strike=float(contract.strike),
                        right=right,
                        qty=filled,
                        entry_price=fill_price,
                    )
                )
                self.positions[symbol] = position
                self._save_state()
        if not position.legs:
            log.error("No %s entry legs filled", symbol)
        elif len(position.legs) != 3:
            log.error(
                "Partial %s group entered (%s/3 legs); it remains managed",
                symbol,
                len(position.legs),
            )

    def _contract_for_leg(self, symbol: str, leg: OptionLeg) -> Option | None:
        contract = Option(symbol, leg.expiry, leg.strike, leg.right, "SMART", currency="USD")
        contract.conId = leg.con_id
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else None

    def _exit_group(
        self, position: SymbolPosition, bar_ts: pd.Timestamp, score: int, reason: str
    ) -> bool:
        remaining: list[OptionLeg] = []
        for leg in position.legs:
            contract = self._contract_for_leg(position.symbol, leg)
            if contract is None:
                remaining.append(leg)
                continue
            quote = self._quote(contract, wait_sec=0.75)
            order_ref = f"{ORDER_REF_PREFIX}-{position.symbol}-EXIT-{leg.role}-{trading_date_str()}"
            filled, limit_price, fill_price = self._submit(
                contract, "SELL", leg.qty, quote, order_ref
            )
            pnl_pct = None
            pnl_usd = None
            exit_reason = reason
            if fill_price is not None and leg.entry_price is not None and leg.entry_price > 0:
                pnl_pct = (fill_price / leg.entry_price - 1.0) * 100.0
                pnl_usd = (fill_price - leg.entry_price) * 100.0 * filled
                outcome = "take_profit" if pnl_usd >= 0 else "stop_loss"
                exit_reason = f"{reason}:{outcome}"
            self.trade_logger.write(
                ts=et_now().isoformat(),
                event="EXIT",
                symbol=position.symbol,
                direction=position.direction,
                bar_ts=bar_ts.isoformat(),
                score=score,
                role=leg.role,
                local_symbol=leg.local_symbol,
                right=leg.right,
                expiry=leg.expiry,
                strike=leg.strike,
                side="SELL",
                requested_qty=leg.qty,
                filled_qty=filled,
                bid=quote.bid,
                ask=quote.ask,
                limit_price=limit_price,
                fill_price=fill_price,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                reason=exit_reason,
                dry_run=self.dry_run,
            )
            residual = leg.qty - filled
            if residual > 0:
                leg.qty = residual
                remaining.append(leg)
        if remaining:
            position.legs = remaining
            self.positions[position.symbol] = position
            self._save_state()
            log.error("Partial exit for %s; %s legs remain", position.symbol, len(remaining))
            return False
        self.positions.pop(position.symbol, None)
        self._save_state()
        return True

    def _safety_tick(self) -> None:
        now = et_now()
        if now.time() < self.flatten_time:
            return
        for position in list(self.positions.values()):
            self._exit_group(position, pd.Timestamp(now), 0, "eod_flatten")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IB paper-only QQQ/SPY Universal v2.1 0DTE option trader"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway paper port")
    parser.add_argument("--client-id", type=int, default=71)
    parser.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    parser.add_argument("--qty-per-leg", type=int, default=1)
    parser.add_argument("--max-spread-pct", type=float, default=30.0)
    parser.add_argument("--flatten-time", default="15:50")
    parser.add_argument(
        "--execute-paper",
        action="store_true",
        help="submit orders; without this flag the runner is dry-run only",
    )
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--trade-log-path", type=Path)
    parser.add_argument("--bar-data-dir", type=Path)
    parser.add_argument("--optimization-config-path", type=Path)
    parser.add_argument(
        "--auto-apply-optimization",
        action="store_true",
        help="load validated latest optimizer recommendations at process start",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    symbols = tuple(symbol.upper() for symbol in args.symbols)
    unsupported = sorted(set(symbols).difference({"QQQ", "SPY"}))
    if unsupported:
        raise SystemExit(f"only QQQ and SPY are allowed: {unsupported}")
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)
    trader = Momentum0DTEPaperTrader(
        ib,
        ib_port=args.port,
        symbols=symbols,
        qty_per_leg=args.qty_per_leg,
        dry_run=not args.execute_paper,
        max_spread_pct=args.max_spread_pct,
        flatten_time=args.flatten_time,
        state_path=args.state_path,
        trade_log_path=args.trade_log_path,
        bar_data_dir=args.bar_data_dir,
        optimization_config_path=args.optimization_config_path,
        auto_apply_optimization=args.auto_apply_optimization,
    )
    signal.signal(signal.SIGINT, lambda *_: trader.stop())
    signal.signal(signal.SIGTERM, lambda *_: trader.stop())
    try:
        trader.start()
    finally:
        ib.disconnect()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
