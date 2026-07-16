"""Flatten a single stock position in an IB Paper account.

This utility is intentionally narrow: it verifies a Paper account, reads the
current stock position, then submits the opposite market order when --execute is
provided. Without --execute it only logs what it would do.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from ib_insync import IB, MarketOrder, Stock

from .time_utils import et_now, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"

log = logging.getLogger("flatten_position")


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"flatten_position_{trading_date_str()}.log"
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


def verify_paper_account(ib: IB) -> bool:
    accounts = ib.managedAccounts()
    if not accounts:
        log.error("No IB accounts found")
        return False
    if any(str(account).startswith("DU") for account in accounts):
        log.info("Paper account verified: %s", accounts)
        return True
    log.error("BLOCKED: not a Paper account. Accounts=%s", accounts)
    return False


def stock_position_qty(ib: IB, symbol: str) -> float:
    positions = ib.positions()
    for position in positions:
        contract = position.contract
        if contract.secType == "STK" and contract.symbol.upper() == symbol.upper():
            return float(position.position)
    return 0.0


def wait_fill(ib: IB, trade, timeout_sec: float) -> float | None:
    start = time.time()
    while time.time() - start < timeout_sec:
        ib.sleep(0.25)
        if trade.isDone() or trade.orderStatus.status == "Filled":
            break

    filled = sum(fill.execution.shares for fill in trade.fills)
    if filled <= 0:
        return None
    avg = trade.orderStatus.avgFillPrice
    if avg and avg > 0:
        return float(avg)
    value = sum(fill.execution.shares * fill.execution.price for fill in trade.fills)
    return float(value / filled) if filled else None


def flatten_position(
    ib: IB,
    symbol: str,
    execute: bool,
    timeout_sec: float,
    reset_state_path: Path | None = None,
) -> int:
    qty = stock_position_qty(ib, symbol)
    log.info("Current %s position: %s", symbol, qty)
    if abs(qty) < 0.5:
        log.info("%s already flat", symbol)
        if execute and reset_state_path is not None:
            reset_ddput_state(reset_state_path, symbol)
        return 0

    side = "SELL" if qty > 0 else "BUY"
    shares = abs(int(qty))
    contract = Stock(symbol.upper(), "SMART", "USD")
    ib.qualifyContracts(contract)

    if not execute:
        log.info("[DRY RUN] Would %s %s %s to flatten", side, shares, symbol.upper())
        return 0

    order = MarketOrder(side, shares)
    order.tif = "DAY"
    order.outsideRth = False
    log.warning("Submitting flatten order: %s %s %s", side, shares, symbol.upper())
    trade = ib.placeOrder(contract, order)
    fill_price = wait_fill(ib, trade, timeout_sec)
    if fill_price is None:
        log.error("Flatten order not filled; cancelling")
        try:
            ib.cancelOrder(order)
        except Exception:
            pass
        return 2

    log.info("Flatten filled: %s %s %s @ %.4f", side, shares, symbol.upper(), fill_price)
    remaining = stock_position_qty(ib, symbol)
    log.info("Remaining %s position: %s", symbol.upper(), remaining)
    if abs(remaining) < 0.5:
        if reset_state_path is not None:
            reset_ddput_state(reset_state_path, symbol)
        return 0
    return 3


def reset_ddput_state(path: Path, symbol: str) -> None:
    """Reset ddput paper trader state to a flat local position."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "position": {
            "in_position": False,
            "qty": 0,
            "entry_ts": None,
            "entry_price": None,
            "entry_signal_ts": None,
            "entry_z": None,
            "entry_strength": None,
            "hold_until_ts": None,
            "trade_symbol": symbol.upper(),
        },
        "last_signal_ts": None,
        "updated": et_now().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Reset ddput state to flat: %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten one stock position in IB Paper")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=77)
    parser.add_argument("--symbol", default="TQQQ")
    parser.add_argument("--execute", action="store_true", help="Submit the flatten order")
    parser.add_argument("--reset-state-path", type=Path, default=None,
                        help="Reset ddput paper trader state to flat after successful flatten")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    ib = IB()
    try:
        log.info("Connecting IB %s:%s clientId=%s", args.host, args.port, args.client_id)
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)
        if not verify_paper_account(ib):
            return
        raise SystemExit(flatten_position(
            ib, args.symbol, args.execute, args.timeout_sec, args.reset_state_path
        ))
    finally:
        if ib.isConnected():
            ib.disconnect()
        log.info("Stopped")


if __name__ == "__main__":
    main()
