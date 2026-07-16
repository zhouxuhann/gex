"""Collect Big 7 0DTE/1DTE option-flow snapshots for research.

This collector stores repeated intraday snapshots from IB. Option volume is
usually day-cumulative, so the useful research field is ``volume_delta`` versus
the previous local snapshot for the same symbol/expiry/strike/right.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Option, Stock

from gex_monitor.time_utils import et_now, is_market_open, trading_date_str


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "big7_short_dated_flow"
BIG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


def _clean(value, *, positive: bool = False) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if positive and out <= 0:
        return None
    return out


def _parse_csv_strings(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_dtes(value: str | None) -> list[int]:
    return [int(item) for item in _parse_csv_strings(value)]


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):,.{digits}f}"


def _connect(args: argparse.Namespace) -> IB:
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True, timeout=args.timeout)
    ib.reqMarketDataType(args.market_data_type)
    return ib


def _stock_and_spot(ib: IB, symbol: str, wait_sec: float) -> tuple[Stock, float]:
    stock = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(stock)
    ticker = ib.reqMktData(stock, "", False, False)
    ib.sleep(wait_sec)
    spot = (
        _clean(ticker.marketPrice(), positive=True)
        or _clean(ticker.last, positive=True)
        or _clean(ticker.close, positive=True)
        or _clean(ticker.bid, positive=True)
        or _clean(ticker.ask, positive=True)
    )
    ib.cancelMktData(stock)
    if spot is None:
        raise RuntimeError(f"Could not get valid spot for {symbol}")
    return stock, spot


def _option_params(ib: IB, stock: Stock, symbol: str) -> tuple[list[str], list[float]]:
    chains = ib.reqSecDefOptParams(symbol, "", stock.secType, stock.conId)
    expiries = sorted({e for chain in chains for e in chain.expirations})
    strikes = sorted({float(s) for chain in chains for s in chain.strikes})
    return expiries, strikes


def _dte(expiry: str, asof: date | None = None) -> int:
    asof = asof or et_now().date()
    return (datetime.strptime(expiry, "%Y%m%d").date() - asof).days


def _choose_expiries(expiries: list[str], dtes: list[int]) -> list[tuple[str, int]]:
    wanted = set(dtes)
    out: list[tuple[str, int]] = []
    for expiry in expiries:
        try:
            dte = _dte(expiry)
        except ValueError:
            continue
        if dte in wanted:
            out.append((expiry, dte))
    return out


def _choose_strikes(strikes: list[float], spot: float, args: argparse.Namespace) -> list[float]:
    filtered = [s for s in strikes if args.min_moneyness <= s / spot <= args.max_moneyness]
    if not filtered:
        return []
    if len(filtered) <= args.max_strikes:
        return filtered
    atm = min(filtered, key=lambda s: abs(s - spot))
    selected = sorted(filtered, key=lambda s: abs(s - spot))[: args.max_strikes]
    if atm not in selected:
        selected.append(atm)
    return sorted(selected)


def _qualify_options(
    ib: IB,
    symbol: str,
    expiries: list[tuple[str, int]],
    strikes: list[float],
    rights: list[str],
) -> tuple[list[Option], list[dict]]:
    dte_by_expiry = dict(expiries)
    contracts: list[Option] = []
    meta: list[dict] = []
    for expiry, _ in expiries:
        for strike in strikes:
            for right in rights:
                contract = Option(symbol, expiry, strike, right, "SMART", currency="USD")
                try:
                    matches = ib.qualifyContracts(contract)
                except Exception:
                    matches = []
                if matches:
                    contracts.append(matches[0])
                    meta.append(
                        {
                            "symbol": symbol,
                            "expiry": expiry,
                            "dte": dte_by_expiry[expiry],
                            "strike": float(strike),
                            "right": right,
                        }
                    )
    return contracts, meta


def _mid_from_ticker(ticker) -> tuple[float | None, float | None, float | None]:
    bid = _clean(getattr(ticker, "bid", None), positive=True)
    ask = _clean(getattr(ticker, "ask", None), positive=True)
    last = _clean(getattr(ticker, "last", None), positive=True)
    close = _clean(getattr(ticker, "close", None), positive=True)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0, bid, ask
    return last or close, bid, ask


def _scan_symbol(ib: IB, symbol: str, args: argparse.Namespace) -> list[dict]:
    stock, spot = _stock_and_spot(ib, symbol, args.spot_wait_sec)
    expiries_raw, strikes_raw = _option_params(ib, stock, symbol)
    expiries = _choose_expiries(expiries_raw, args.dtes)
    strikes = _choose_strikes(strikes_raw, spot, args)
    contracts, meta = _qualify_options(ib, symbol, expiries, strikes, args.rights)
    print(
        f"{symbol} spot={spot:.2f} expiries={','.join(e for e, _ in expiries) or '-'} "
        f"strikes={len(strikes)} contracts={len(contracts)}"
    )

    tickers = []
    for start in range(0, len(contracts), args.chunk_size):
        chunk = contracts[start : start + args.chunk_size]
        tickers.extend([ib.reqMktData(contract, "100,101,106", False, False) for contract in chunk])
        ib.sleep(args.chunk_sleep)
    ib.sleep(args.option_wait_sec)

    ts = datetime.now(ET).isoformat()
    rows: list[dict] = []
    for ticker, item in zip(tickers, meta):
        price, bid, ask = _mid_from_ticker(ticker)
        greeks = getattr(ticker, "modelGreeks", None)
        oi_attr = "callOpenInterest" if item["right"] == "C" else "putOpenInterest"
        volume = _clean(getattr(ticker, "volume", None))
        rows.append(
            {
                "ts": ts,
                "trading_date": trading_date_str(),
                **item,
                "spot": spot,
                "moneyness": item["strike"] / spot,
                "bid": bid,
                "ask": ask,
                "mid": price,
                "last": _clean(getattr(ticker, "last", None), positive=True),
                "close": _clean(getattr(ticker, "close", None), positive=True),
                "volume": volume,
                "open_interest": _clean(getattr(ticker, oi_attr, None)),
                "iv": _clean(getattr(greeks, "impliedVol", None), positive=True) if greeks else None,
                "delta": _clean(getattr(greeks, "delta", None)) if greeks else None,
                "gamma": _clean(getattr(greeks, "gamma", None)) if greeks else None,
                "theta": _clean(getattr(greeks, "theta", None)) if greeks else None,
                "vega": _clean(getattr(greeks, "vega", None)) if greeks else None,
                "notional_volume": (volume or 0.0) * (price or 0.0) * 100.0,
            }
        )

    for ticker in tickers:
        try:
            ib.cancelMktData(ticker.contract)
        except Exception:
            pass
    return rows


def _daily_path(args: argparse.Namespace) -> Path:
    return args.data_dir / f"short_dated_option_flow_{trading_date_str()}.parquet"


def _snapshot_path(args: argparse.Namespace) -> Path:
    tag = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    return args.data_dir / "snapshots" / f"short_dated_option_flow_snapshot_{tag}.parquet"


def _add_volume_delta(current: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    out = current.copy()
    if prior.empty:
        out["prior_ts"] = None
        out["prior_volume"] = None
        out["volume_delta"] = out["volume"]
        out["notional_volume_delta"] = out["notional_volume"]
        return out

    keys = ["symbol", "expiry", "strike", "right"]
    hist = prior.copy()
    hist["_ts_dt"] = pd.to_datetime(hist["ts"], utc=True, errors="coerce")
    last = hist.sort_values("_ts_dt").groupby(keys, as_index=False).tail(1)
    last = last[keys + ["ts", "volume", "notional_volume"]].rename(
        columns={
            "ts": "prior_ts",
            "volume": "prior_volume",
            "notional_volume": "prior_notional_volume",
        }
    )
    out = out.merge(last, on=keys, how="left")
    out["volume_delta"] = out["volume"] - out["prior_volume"]
    out.loc[out["volume_delta"].isna(), "volume_delta"] = out["volume"]
    out.loc[out["volume_delta"] < 0, "volume_delta"] = out["volume"]
    out["notional_volume_delta"] = out["volume_delta"].fillna(0) * out["mid"].fillna(0) * 100.0
    return out


def _read_daily_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("Could not read daily flow history %s: %s", path, exc)
        return pd.DataFrame()


def _write_daily(path: Path, combined: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.drop_duplicates(
        subset=["ts", "symbol", "expiry", "strike", "right"],
        keep="last",
    )
    tmp = path.with_name(path.name + ".tmp.parquet")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)


def _print_top(df: pd.DataFrame, top: int) -> None:
    if df.empty:
        return
    active = df[df["volume_delta"].fillna(0) > 0].copy()
    if active.empty:
        print("no positive volume_delta")
        return
    active = active.sort_values(["notional_volume_delta", "volume_delta"], ascending=False)
    print("symbol expiry dte type K spot vol dvol mid IV delta notional_delta")
    for row in active.head(top).itertuples(index=False):
        print(
            f"{row.symbol:5s} {row.expiry} {int(row.dte):1d} {row.right} "
            f"{float(row.strike):8.2f} {_fmt(getattr(row, 'spot')):>8s} "
            f"{_fmt(getattr(row, 'volume'), 0):>8s} {_fmt(getattr(row, 'volume_delta'), 0):>8s} "
            f"{_fmt(getattr(row, 'mid')):>7s} {_fmt(getattr(row, 'iv') * 100 if getattr(row, 'iv') else None):>7s} "
            f"{_fmt(getattr(row, 'delta'), 3):>7s} {_fmt(getattr(row, 'notional_volume_delta'), 0):>12s}"
        )


def collect_once(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    ib = _connect(args)
    try:
        for symbol in args.symbols:
            try:
                rows.extend(_scan_symbol(ib, symbol, args))
            except Exception as exc:
                log.exception("Failed scanning %s", symbol)
                print(f"{symbol} error={exc}")
            ib.sleep(args.symbol_sleep)
    finally:
        ib.disconnect()

    current = pd.DataFrame(rows)
    if current.empty:
        print("no rows")
        return current

    daily_path = _daily_path(args)
    prior = _read_daily_history(daily_path)
    enriched = _add_volume_delta(current, prior)
    snapshot_path = _snapshot_path(args)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(snapshot_path, index=False)
    combined = pd.concat([prior, enriched], ignore_index=True) if not prior.empty else enriched
    _write_daily(daily_path, combined)
    print(f"snapshot_written={snapshot_path}")
    print(f"daily_written={daily_path} rows_added={len(enriched)} daily_rows={len(combined)}")
    _print_top(enriched, args.top)
    return enriched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Big 7 0DTE/1DTE option-flow snapshots")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=416)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--market-data-type", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--symbols", type=_parse_csv_strings, default=BIG7)
    parser.add_argument("--dtes", type=_parse_dtes, default=[0, 1])
    parser.add_argument("--rights", type=_parse_csv_strings, default=["C", "P"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--rth-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-moneyness", type=float, default=0.90)
    parser.add_argument("--max-moneyness", type=float, default=1.10)
    parser.add_argument("--max-strikes", type=int, default=36)
    parser.add_argument("--spot-wait-sec", type=float, default=1.0)
    parser.add_argument("--option-wait-sec", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--chunk-sleep", type=float, default=0.25)
    parser.add_argument("--symbol-sleep", type=float, default=0.25)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.data_dir.mkdir(parents=True, exist_ok=True)
    while True:
        if not args.rth_only or is_market_open():
            collect_once(args)
        else:
            print(f"{datetime.now(ET).isoformat()} market closed; skip")
        if not args.monitor:
            break
        time.sleep(args.interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
