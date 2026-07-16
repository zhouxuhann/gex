"""Scan LEAPS calls from IB for stock-replacement candidates."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from ib_insync import IB, Option, Stock


DEFAULT_STRIKES = "300,320,340,350,360,380,400,420,450,500,550,600"
DEFAULT_EXPIRIES = "20270115,20280121"


@dataclass
class OptionRow:
    expiry: str
    strike: float
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None
    model: float | None
    price: float | None
    price_source: str
    delta: float | None
    iv: float | None
    theta: float | None
    vega: float | None
    dte: int
    intrinsic: float | None
    extrinsic: float | None
    breakeven: float | None
    profit_at_target: float | None
    contracts_for_share_delta: float | None
    position_delta_shares: float | None
    position_cost: float | None
    position_profit_at_target: float | None


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _positive(value: float | None) -> float | None:
    if value is None or isinstance(value, float) and math.isnan(value) or value <= 0:
        return None
    return float(value)


def _fmt_money(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _choose_price(bid: float | None, ask: float | None, last: float | None, close: float | None, model: float | None) -> tuple[float | None, str]:
    if bid is not None and ask is not None:
        return (bid + ask) / 2, "bid/ask"
    if model is not None:
        return model, "model"
    if last is not None:
        return last, "last"
    if close is not None:
        return close, "close"
    return None, "none"


def _get_spot(ib: IB, symbol: str, wait_sec: float) -> tuple[Stock, float]:
    stock = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(stock)
    ticker = ib.reqMktData(stock, "", False, False)
    ib.sleep(wait_sec)
    spot = _positive(ticker.marketPrice()) or _positive(ticker.last) or _positive(ticker.close)
    spot = spot or _positive(ticker.bid) or _positive(ticker.ask)
    if spot is None:
        raise RuntimeError(f"Could not get a valid spot price for {symbol}")
    return stock, spot


def _qualify_options(ib: IB, symbol: str, expiries: Iterable[str], strikes: Iterable[float]) -> list[Option]:
    qualified: list[Option] = []
    for expiry in expiries:
        for strike in strikes:
            contract = Option(symbol, expiry, strike, "C", "SMART", currency="USD")
            try:
                matches = ib.qualifyContracts(contract)
            except Exception as exc:
                print(f"skip unknown contract: {symbol} {expiry} {strike:g}C ({exc})", file=sys.stderr)
                continue
            if matches:
                qualified.extend(matches)
    return qualified


def scan(args: argparse.Namespace) -> tuple[float, list[OptionRow]]:
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True, timeout=args.timeout)
    try:
        ib.reqMarketDataType(args.market_data_type)
        _, spot = _get_spot(ib, args.symbol, args.spot_wait_sec)
        contracts = _qualify_options(ib, args.symbol, args.expiries, args.strikes)
        if not contracts:
            raise RuntimeError("No option contracts qualified")

        tickers = [ib.reqMktData(contract, "106", False, False) for contract in contracts]
        ib.sleep(args.option_wait_sec)

        today = date.today()
        rows: list[OptionRow] = []
        for ticker in tickers:
            contract = ticker.contract
            greeks = ticker.modelGreeks
            bid = _positive(ticker.bid)
            ask = _positive(ticker.ask)
            last = _positive(ticker.last)
            close = _positive(ticker.close)
            model = _positive(getattr(greeks, "optPrice", None)) if greeks else None
            price, price_source = _choose_price(bid, ask, last, close, model)
            delta = getattr(greeks, "delta", None) if greeks else None
            iv = getattr(greeks, "impliedVol", None) if greeks else None
            theta = getattr(greeks, "theta", None) if greeks else None
            vega = getattr(greeks, "vega", None) if greeks else None
            expiry_date = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d").date()
            dte = (expiry_date - today).days

            intrinsic = extrinsic = breakeven = profit_at_target = None
            contracts_for_share_delta = position_delta_shares = position_cost = position_profit = None
            if price is not None:
                intrinsic = max(spot - float(contract.strike), 0.0)
                extrinsic = price - intrinsic
                breakeven = float(contract.strike) + price
                profit_at_target = max(args.target - float(contract.strike), 0.0) - price
                position_cost = args.contracts * price * 100
                position_profit = args.contracts * profit_at_target * 100
            if delta is not None and delta > 0:
                contracts_for_share_delta = args.shares / (delta * 100)
                position_delta_shares = args.contracts * delta * 100

            rows.append(
                OptionRow(
                    expiry=contract.lastTradeDateOrContractMonth,
                    strike=float(contract.strike),
                    bid=bid,
                    ask=ask,
                    last=last,
                    close=close,
                    model=model,
                    price=price,
                    price_source=price_source,
                    delta=delta,
                    iv=iv,
                    theta=theta,
                    vega=vega,
                    dte=dte,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    breakeven=breakeven,
                    profit_at_target=profit_at_target,
                    contracts_for_share_delta=contracts_for_share_delta,
                    position_delta_shares=position_delta_shares,
                    position_cost=position_cost,
                    position_profit_at_target=position_profit,
                )
            )
        return spot, sorted(rows, key=lambda row: (row.expiry, row.strike))
    finally:
        ib.disconnect()


def print_table(symbol: str, spot: float, rows: list[OptionRow], args: argparse.Namespace) -> None:
    print(
        f"{symbol} spot={spot:.2f} target={args.target:.2f} "
        f"replacement_shares={args.shares:g} contracts_view={args.contracts:g}"
    )
    print(
        "expiry     K     src      bid      ask      px    delta    IV    theta    "
        "extrinsic     BE   P@target  ctr@delta  pos_delta  pos_cost  pos_P@target"
    )
    for row in rows:
        print(
            f"{row.expiry} "
            f"{row.strike:5.0f} "
            f"{row.price_source:7s} "
            f"{_fmt_money(row.bid):>8s} "
            f"{_fmt_money(row.ask):>8s} "
            f"{_fmt_money(row.price):>8s} "
            f"{_fmt_num(row.delta, 3):>7s} "
            f"{_fmt_pct(row.iv):>7s} "
            f"{_fmt_num(row.theta, 3):>8s} "
            f"{_fmt_money(row.extrinsic):>10s} "
            f"{_fmt_money(row.breakeven):>8s} "
            f"{_fmt_money(row.profit_at_target):>10s} "
            f"{_fmt_num(row.contracts_for_share_delta, 2):>9s} "
            f"{_fmt_num(row.position_delta_shares, 0):>9s} "
            f"{_fmt_money(row.position_cost):>9s} "
            f"{_fmt_money(row.position_profit_at_target):>12s}"
        )


def write_csv(path: Path, rows: list[OptionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OptionRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan IB LEAPS calls for stock-replacement candidates")
    parser.add_argument("--symbol", default="AMD")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=341)
    parser.add_argument("--market-data-type", type=int, default=1, choices=[1, 2, 3, 4], help="IB market data type: 1=live, 2=frozen, 3=delayed, 4=delayed frozen")
    parser.add_argument("--expiries", type=_parse_csv_strings, default=_parse_csv_strings(DEFAULT_EXPIRIES))
    parser.add_argument("--strikes", type=_parse_csv_floats, default=_parse_csv_floats(DEFAULT_STRIKES))
    parser.add_argument("--target", type=float, default=600.0)
    parser.add_argument("--shares", type=float, default=600.0)
    parser.add_argument("--contracts", type=float, default=6.0, help="Position size to show in cost/P&L columns")
    parser.add_argument("--spot-wait-sec", type=float, default=2.0)
    parser.add_argument("--option-wait-sec", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spot, rows = scan(args)
    print_table(args.symbol, spot, rows, args)
    if args.csv:
        write_csv(args.csv, rows)
        print(f"csv_written={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
