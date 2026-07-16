"""IB option flow scanner.

Utilities for recurring discretionary option-flow checks:

* ``scan-volume``: snapshot option volume/OI/IV for selected expiries/strikes.
* ``event-calls``: find an underlying intraday event window and aggregate
  historical call trades around it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Option, Stock

from gex_monitor.email_notifier import EmailConfig, EmailNotifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "analysis"
ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpiryBucket:
    label: str
    expiry: str
    dte: int


@dataclass(frozen=True)
class KDJSnapshot:
    ts: datetime
    close: float
    k: float
    d: float
    j: float
    high: float
    low: float


@dataclass(frozen=True)
class HourlyBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    wap: float | None = None
    count: int | None = None


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


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _date_from_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _dte(expiry: str, asof: date | None = None) -> int:
    asof = asof or date.today()
    return (_date_from_yyyymmdd(expiry) - asof).days


def _mid_from_ticker(ticker) -> tuple[float | None, float | None, float | None]:
    bid = _clean(getattr(ticker, "bid", None), positive=True)
    ask = _clean(getattr(ticker, "ask", None), positive=True)
    last = _clean(getattr(ticker, "last", None), positive=True)
    close = _clean(getattr(ticker, "close", None), positive=True)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0, bid, ask
    return last or close, bid, ask


def _connect(args: argparse.Namespace) -> IB:
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True, timeout=args.timeout)
    ib.reqMarketDataType(args.market_data_type)
    return ib


def _stock_and_spot(ib: IB, symbol: str, wait_sec: float = 1.5) -> tuple[Stock, float]:
    stock = Stock(symbol.upper(), "SMART", "USD")
    ib.qualifyContracts(stock)
    ticker = ib.reqMktData(stock, "", False, False)
    ib.sleep(wait_sec)
    spot = (
        _clean(ticker.marketPrice(), positive=True)
        or _clean(ticker.last, positive=True)
        or _clean(ticker.bid, positive=True)
        or _clean(ticker.ask, positive=True)
        or _clean(ticker.close, positive=True)
    )
    ib.cancelMktData(stock)
    if spot is None:
        raise RuntimeError(f"Could not get valid spot for {symbol}")
    return stock, spot


def _option_params(ib: IB, stock: Stock, symbol: str) -> tuple[list[str], list[float]]:
    chains = ib.reqSecDefOptParams(symbol.upper(), "", stock.secType, stock.conId)
    expiries = sorted({e for chain in chains for e in chain.expirations})
    strikes = sorted({float(s) for chain in chains for s in chain.strikes})
    return expiries, strikes


def _choose_expiries(
    expiries: list[str],
    *,
    targets: list[int] | None,
    explicit: list[str] | None,
    include_dec26: bool,
) -> list[ExpiryBucket]:
    today = date.today()
    parsed: list[tuple[str, int]] = []
    for expiry in expiries:
        try:
            dte = _dte(expiry, today)
        except ValueError:
            continue
        if dte >= 0:
            parsed.append((expiry, dte))

    out: list[ExpiryBucket] = []
    used: set[str] = set()
    if explicit:
        for expiry in explicit:
            if expiry in used:
                continue
            out.append(ExpiryBucket(expiry, expiry, _dte(expiry, today)))
            used.add(expiry)
        return out

    for target in targets or [7, 30, 60, 180, 365]:
        candidates = [item for item in parsed if item[0] not in used]
        if not candidates:
            break
        expiry, dte = min(candidates, key=lambda item: abs(item[1] - target))
        out.append(ExpiryBucket(f"{target}d", expiry, dte))
        used.add(expiry)

    if include_dec26:
        dec = [item for item in parsed if item[0].startswith("202612") and item[0] not in used]
        if dec:
            expiry, dte = min(dec, key=lambda item: abs(item[1] - 220))
            out.append(ExpiryBucket("Dec26", expiry, dte))
    return out


def _default_strikes(symbol: str, all_strikes: list[float], spot: float) -> list[float]:
    symbol = symbol.upper()
    if symbol == "NOK":
        return [s for s in all_strikes if spot * 0.65 <= s <= spot * 1.35 and float(s).is_integer()]
    if symbol == "ORCL":
        out = []
        for strike in all_strikes:
            if spot * 0.7 <= strike <= spot * 1.75 and abs(strike / 5 - round(strike / 5)) < 1e-9:
                out.append(strike)
        return out[:80]
    return [s for s in all_strikes if spot * 0.7 <= s <= spot * 1.5][:80]


def _qualify_options(
    ib: IB,
    symbol: str,
    expiries: list[ExpiryBucket],
    strikes: list[float],
    rights: list[str],
) -> tuple[list[Option], list[dict]]:
    contracts: list[Option] = []
    meta: list[dict] = []
    for bucket in expiries:
        for strike in strikes:
            for right in rights:
                contract = Option(symbol.upper(), bucket.expiry, strike, right, "SMART", currency="USD")
                try:
                    matches = ib.qualifyContracts(contract)
                except Exception:
                    matches = []
                if matches:
                    contracts.append(matches[0])
                    meta.append(
                        {
                            "symbol": symbol.upper(),
                            "bucket": bucket.label,
                            "expiry": bucket.expiry,
                            "dte": bucket.dte,
                            "strike": float(strike),
                            "right": right,
                        }
                    )
    return contracts, meta


def _select_strikes_near_spot(
    all_strikes: list[float],
    spot: float,
    *,
    pct: float,
    max_count: int,
) -> list[float]:
    candidates = [float(s) for s in all_strikes if spot * (1 - pct) <= float(s) <= spot * (1 + pct)]
    candidates = sorted(candidates, key=lambda strike: abs(strike - spot))[:max_count]
    return sorted(candidates)


def _collect_option_volume_rows(
    ib: IB,
    *,
    symbol: str,
    target_dtes: list[int] | None,
    expiries: list[str] | None,
    include_dec26: bool,
    strikes: list[float] | None,
    rights: list[str],
    wait_sec: float,
    chunk_size: int,
    chunk_sleep: float,
    strike_pct: float | None = None,
    max_strikes: int | None = None,
) -> tuple[float, list[dict]]:
    stock, spot = _stock_and_spot(ib, symbol)
    expiries_raw, strikes_raw = _option_params(ib, stock, symbol)
    chosen_expiries = _choose_expiries(
        expiries_raw,
        targets=target_dtes,
        explicit=expiries,
        include_dec26=include_dec26,
    )
    if strikes:
        chosen_strikes = strikes
    elif strike_pct is not None and max_strikes:
        chosen_strikes = _select_strikes_near_spot(
            strikes_raw,
            spot,
            pct=strike_pct,
            max_count=max_strikes,
        )
    else:
        chosen_strikes = _default_strikes(symbol, strikes_raw, spot)

    contracts, meta = _qualify_options(ib, symbol, chosen_expiries, chosen_strikes, rights)
    print(
        f"{symbol.upper()} spot={spot:.2f} contracts={len(contracts)} "
        f"expiries={','.join(b.expiry for b in chosen_expiries)}"
    )
    tickers = []
    for start in range(0, len(contracts), chunk_size):
        chunk = contracts[start : start + chunk_size]
        tickers.extend([ib.reqMktData(contract, "100,101,106", False, False) for contract in chunk])
        ib.sleep(chunk_sleep)
    ib.sleep(wait_sec)

    rows: list[dict] = []
    for ticker, item in zip(tickers, meta):
        price, bid, ask = _mid_from_ticker(ticker)
        greeks = getattr(ticker, "modelGreeks", None)
        volume = _clean(getattr(ticker, "volume", None), positive=True)
        oi_attr = "callOpenInterest" if item["right"] == "C" else "putOpenInterest"
        oi = _clean(getattr(ticker, oi_attr, None), positive=True)
        rows.append(
            {
                **item,
                "spot": spot,
                "bid": bid,
                "ask": ask,
                "mid": price,
                "last": _clean(getattr(ticker, "last", None), positive=True),
                "close": _clean(getattr(ticker, "close", None), positive=True),
                "volume": volume,
                "open_interest": oi,
                "iv": _clean(getattr(greeks, "impliedVol", None), positive=True) if greeks else None,
                "delta": _clean(getattr(greeks, "delta", None)) if greeks else None,
                "moneyness": item["strike"] / spot,
                "notional_volume": (volume or 0.0) * (price or 0.0) * 100.0,
            }
        )
    for ticker in tickers:
        try:
            ib.cancelMktData(ticker.contract)
        except Exception:
            pass
    return spot, rows


def scan_volume(args: argparse.Namespace) -> int:
    ib = _connect(args)
    rows: list[dict] = []
    try:
        for symbol in args.symbols:
            _, symbol_rows = _collect_option_volume_rows(
                ib,
                symbol=symbol,
                target_dtes=args.target_dtes,
                expiries=args.expiries,
                include_dec26=args.include_dec26,
                strikes=args.strikes,
                rights=args.rights,
                wait_sec=args.wait_sec,
                chunk_size=args.chunk_size,
                chunk_sleep=args.chunk_sleep,
            )
            rows.extend(symbol_rows)
    finally:
        ib.disconnect()

    output = args.output or DEFAULT_OUT_DIR / "option_volume_scan_ib.csv"
    _write_csv(output, rows)
    print(f"csv_written={output}")
    _print_top(rows, args.top)
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_top(rows: list[dict], top: int) -> None:
    for symbol in sorted({row["symbol"] for row in rows}):
        sym_rows = [row for row in rows if row["symbol"] == symbol]
        print(f"\n=== {symbol} TOP VOLUME ===")
        print("bucket expiry dte type K vol OI mid bid/ask IV delta mny")
        top_rows = sorted(
            [row for row in sym_rows if row.get("volume")],
            key=lambda row: (row["volume"], row.get("notional_volume") or 0.0),
            reverse=True,
        )[:top]
        for row in top_rows:
            print(
                f"{row['bucket']:6s} {row['expiry']} {int(row['dte']):3d} {row['right']} "
                f"{row['strike']:7.2f} {_fmt(row.get('volume'), 0):>8s} "
                f"{_fmt(row.get('open_interest'), 0):>8s} {_fmt(row.get('mid')):>7s} "
                f"{_fmt(row.get('bid'))}/{_fmt(row.get('ask'))} "
                f"{_fmt_pct(row.get('iv')):>7s} {_fmt(row.get('delta'), 3):>7s} "
                f"{_fmt(row.get('moneyness'), 2):>5s}"
            )

        print(f"\n=== {symbol} TOP OI ===")
        top_oi = sorted(
            [row for row in sym_rows if row.get("open_interest")],
            key=lambda row: row["open_interest"],
            reverse=True,
        )[:top]
        for row in top_oi:
            print(
                f"{row['bucket']:6s} {row['expiry']} {int(row['dte']):3d} {row['right']} "
                f"{row['strike']:7.2f} OI={_fmt(row.get('open_interest'), 0)} "
                f"vol={_fmt(row.get('volume'), 0)} mid={_fmt(row.get('mid'))} "
                f"IV={_fmt_pct(row.get('iv'))}"
            )


def _historical_bars_for_date(ib: IB, stock: Stock, yyyymmdd: str):
    end = f"{yyyymmdd} 16:00:00 US/Eastern"
    return ib.reqHistoricalData(
        stock,
        endDateTime=end,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
    )


def _find_event_window(ib: IB, stock: Stock, yyyymmdd: str, threshold: float | None, window_min: int):
    bars = _historical_bars_for_date(ib, stock, yyyymmdd)
    if not bars:
        raise RuntimeError(f"No historical bars for {stock.symbol} {yyyymmdd}")
    if threshold is not None:
        touched = [bar for bar in bars if float(bar.low) <= threshold]
        event_bar = touched[0] if touched else min(bars, key=lambda bar: abs(float(bar.low) - threshold))
    else:
        event_bar = min(bars, key=lambda bar: float(bar.low))
    event_dt = event_bar.date
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=ET)
    else:
        event_dt = event_dt.astimezone(ET)
    start = event_dt - timedelta(minutes=window_min)
    end = event_dt + timedelta(minutes=window_min)
    return event_bar, start, end, bars


def _ib_time(dt: datetime) -> str:
    return dt.astimezone(ET).strftime("%Y%m%d %H:%M:%S US/Eastern")


def event_calls(args: argparse.Namespace) -> int:
    symbol = args.symbol.upper()
    ib = _connect(args)
    rows: list[dict] = []
    try:
        stock, spot = _stock_and_spot(ib, symbol)
        event_bar, start, end, bars = _find_event_window(
            ib, stock, args.date, args.threshold, args.window_min
        )
        low_bar = min(bars, key=lambda bar: float(bar.low))
        print(
            f"{symbol} date={args.date} current_spot={spot:.2f} "
            f"event_time={event_bar.date} event_low={float(event_bar.low):.2f} "
            f"day_low={float(low_bar.low):.2f} window={start.strftime('%H:%M')}-{end.strftime('%H:%M')} ET"
        )

        expiries_raw, strikes_raw = _option_params(ib, stock, symbol)
        expiries = _choose_expiries(
            expiries_raw,
            targets=args.target_dtes,
            explicit=args.expiries,
            include_dec26=args.include_dec26,
        )
        strikes = args.strikes or _default_strikes(symbol, strikes_raw, float(event_bar.close))
        contracts, meta = _qualify_options(ib, symbol, expiries, strikes, ["C"])
        print(f"qualified_call_contracts={len(contracts)}")
        for contract, item in zip(contracts, meta):
            try:
                ticks = ib.reqHistoricalTicks(
                    contract,
                    _ib_time(start),
                    _ib_time(end),
                    args.max_ticks,
                    "Trades",
                    True,
                )
            except Exception as exc:
                rows.append({**item, "error": str(exc)})
                continue
            total_size = 0.0
            total_notional = 0.0
            prices: list[float] = []
            for tick in ticks:
                price = _clean(getattr(tick, "price", None), positive=True)
                size = _clean(getattr(tick, "size", None), positive=True) or 0.0
                if price is None or size <= 0:
                    continue
                prices.append(price)
                total_size += size
                total_notional += price * size * 100.0
            vwap = total_notional / (total_size * 100.0) if total_size > 0 else None
            rows.append(
                {
                    **item,
                    "event_time": event_bar.date.isoformat() if hasattr(event_bar.date, "isoformat") else str(event_bar.date),
                    "event_low": float(event_bar.low),
                    "event_close": float(event_bar.close),
                    "window_start_et": start.isoformat(),
                    "window_end_et": end.isoformat(),
                    "trade_count": len(prices),
                    "trade_size": total_size if total_size > 0 else None,
                    "vwap": vwap,
                    "min_trade": min(prices) if prices else None,
                    "max_trade": max(prices) if prices else None,
                    "notional": total_notional if total_notional > 0 else None,
                }
            )
            ib.sleep(args.hist_sleep)
    finally:
        ib.disconnect()

    output = args.output or DEFAULT_OUT_DIR / f"{symbol.lower()}_{args.date}_event_call_trades.csv"
    _write_csv(output, rows)
    print(f"csv_written={output}")
    active = sorted([r for r in rows if r.get("trade_size")], key=lambda r: r["trade_size"], reverse=True)
    print("\n=== CALL TRADES AROUND EVENT ===")
    print("expiry dte K trades size vwap min max notional")
    for row in active[: args.top]:
        print(
            f"{row['expiry']} {int(row['dte']):3d} {row['strike']:6.2f} "
            f"{int(row['trade_count']):6d} {_fmt(row.get('trade_size'), 0):>8s} "
            f"{_fmt(row.get('vwap')):>7s} {_fmt(row.get('min_trade')):>7s} "
            f"{_fmt(row.get('max_trade')):>7s} {_fmt(row.get('notional'), 0):>10s}"
        )
    return 0


def _parse_recipients(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _bar_datetime(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.strptime(str(value), "%Y%m%d  %H:%M:%S") if "  " in str(value) else datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def _historical_hourly_bars(
    ib: IB,
    stock: Stock,
    *,
    duration: str,
    use_rth: bool,
):
    return ib.reqHistoricalData(
        stock,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 hour",
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False,
    )


def _hourly_bar_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"hourly_bars_{symbol.upper()}.parquet"


def _ib_bars_to_hourly(bars) -> list[HourlyBar]:
    out: list[HourlyBar] = []
    for bar in bars:
        out.append(
            HourlyBar(
                ts=_bar_datetime(bar.date),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=_clean(getattr(bar, "volume", None)),
                wap=_clean(getattr(bar, "wap", None)),
                count=int(getattr(bar, "barCount", 0) or 0) or None,
            )
        )
    return out


def _hourly_to_df(symbol: str, bars: list[HourlyBar]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol.upper(),
                "ts": bar.ts.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "wap": bar.wap,
                "count": bar.count,
            }
            for bar in bars
        ]
    )


def _df_to_hourly_bars(df: pd.DataFrame) -> list[HourlyBar]:
    out: list[HourlyBar] = []
    if df.empty:
        return out
    for row in df.sort_values("ts").itertuples(index=False):
        ts = _bar_datetime(getattr(row, "ts"))
        out.append(
            HourlyBar(
                ts=ts,
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=_clean(getattr(row, "volume", None)),
                wap=_clean(getattr(row, "wap", None)),
                count=int(getattr(row, "count")) if _clean(getattr(row, "count", None)) else None,
            )
        )
    return out


def _load_hourly_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("Failed to read hourly cache %s: %s", path, exc)
        return pd.DataFrame()


def _save_hourly_cache(path: Path, df: pd.DataFrame, *, keep_days: int) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["_ts_dt"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["_ts_dt"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=keep_days)
    out = out[out["_ts_dt"] >= cutoff]
    out = out.sort_values("_ts_dt").drop_duplicates(subset=["symbol", "ts"], keep="last")
    out = out.drop(columns=["_ts_dt"]).reset_index(drop=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def _load_or_refresh_hourly_bars(
    ib: IB,
    stock: Stock,
    symbol: str,
    args: argparse.Namespace,
) -> tuple[list[HourlyBar], Path | None, int]:
    if not args.cache_bars:
        bars = _ib_bars_to_hourly(
            _historical_hourly_bars(
                ib,
                stock,
                duration=args.kdj_duration,
                use_rth=args.use_rth,
            )
        )
        return bars, None, len(bars)

    cache_path = _hourly_bar_cache_path(args.bar_cache_dir, symbol)
    cached = _load_hourly_cache(cache_path)
    duration = args.bar_backfill_duration if cached.empty else args.bar_refresh_duration
    fresh_bars = _ib_bars_to_hourly(
        _historical_hourly_bars(
            ib,
            stock,
            duration=duration,
            use_rth=args.use_rth,
        )
    )
    fresh = _hourly_to_df(symbol, fresh_bars)
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    _save_hourly_cache(cache_path, combined, keep_days=args.bar_cache_days)
    saved = _load_hourly_cache(cache_path)
    return _df_to_hourly_bars(saved), cache_path, len(fresh_bars)


def _calc_kdj(
    bars: list[HourlyBar],
    *,
    period: int,
    k_smooth: int,
    d_smooth: int,
) -> list[KDJSnapshot]:
    if len(bars) < period:
        return []
    k = 50.0
    d = 50.0
    out: list[KDJSnapshot] = []
    for idx, bar in enumerate(bars):
        if idx + 1 < period:
            continue
        window = bars[idx + 1 - period : idx + 1]
        low = min(item.low for item in window)
        high = max(item.high for item in window)
        close = bar.close
        rsv = 50.0 if high <= low else (close - low) / (high - low) * 100.0
        k = (k * (k_smooth - 1) + rsv) / k_smooth
        d = (d * (d_smooth - 1) + k) / d_smooth
        j = 3 * k - 2 * d
        out.append(
            KDJSnapshot(
                ts=bar.ts,
                close=close,
                k=k,
                d=d,
                j=j,
                high=bar.high,
                low=bar.low,
            )
        )
    return out


def _kdj_level(j_value: float, near_j: float, watch_j: float, extreme_j: float) -> str | None:
    if j_value <= extreme_j:
        return "extreme"
    if j_value <= watch_j:
        return "watch"
    if j_value <= near_j:
        return "near"
    return None


def _load_monitor_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"sent": {}}
    except json.JSONDecodeError:
        return {"sent": {}}


def _save_monitor_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _top_rows(rows: list[dict], *, right: str | None, top: int) -> list[dict]:
    filtered = [
        row
        for row in rows
        if row.get("volume") and (right is None or row.get("right") == right)
    ]
    return sorted(
        filtered,
        key=lambda row: (row.get("volume") or 0.0, row.get("notional_volume") or 0.0),
        reverse=True,
    )[:top]


def _format_option_rows(rows: list[dict]) -> str:
    if not rows:
        return "  no active option volume returned by IB"
    lines = ["  expiry dte type K volume OI mid IV delta notional"]
    for row in rows:
        lines.append(
            f"  {row['expiry']} {int(row['dte']):3d} {row['right']} "
            f"{row['strike']:7.2f} {_fmt(row.get('volume'), 0):>8s} "
            f"{_fmt(row.get('open_interest'), 0):>8s} {_fmt(row.get('mid')):>7s} "
            f"{_fmt_pct(row.get('iv')):>7s} {_fmt(row.get('delta'), 3):>7s} "
            f"{_fmt(row.get('notional_volume'), 0):>10s}"
        )
    return "\n".join(lines)


def _build_kdj_email_body(
    *,
    symbol: str,
    snap: KDJSnapshot,
    level: str,
    csv_path: Path,
    cache_path: Path | None,
    rows: list[dict],
    top: int,
) -> str:
    top_all = _top_rows(rows, right=None, top=top)
    top_calls = _top_rows(rows, right="C", top=top)
    top_puts = _top_rows(rows, right="P", top=top)
    return (
        "Option Flow KDJ 观察提醒\n\n"
        f"标的:        {symbol}\n"
        f"小时K时间:   {snap.ts.strftime('%Y-%m-%d %H:%M ET')}\n"
        f"触发等级:    {level}\n"
        f"价格/低高:   close={snap.close:.2f} low={snap.low:.2f} high={snap.high:.2f}\n"
        f"K/D/J:       K={snap.k:.2f} D={snap.d:.2f} J={snap.j:.2f}\n"
        f"CSV:         {csv_path}\n"
        f"小时K缓存:   {cache_path or 'disabled'}\n"
        "\n"
        "成交量最大期权:\n"
        f"{_format_option_rows(top_all)}\n\n"
        "Call 成交量:\n"
        f"{_format_option_rows(top_calls)}\n\n"
        "Put 成交量:\n"
        f"{_format_option_rows(top_puts)}\n\n"
        "说明: 这是小时 KDJ 低位触发后的 option flow 观察，不是交易指令。"
    )


def _email_notifier_from_args(args: argparse.Namespace) -> EmailNotifier:
    return EmailNotifier(
        EmailConfig(
            enabled=args.email_enabled,
            sender=args.email_sender,
            password_env=args.email_password_env,
            recipients=_parse_recipients(args.email_recipients),
            only_strong=False,
            cooldown_sec=args.email_cooldown_sec,
            subject_prefix=args.email_subject_prefix,
        )
    )


def _kdj_monitor_once(args: argparse.Namespace, state: dict, notifier: EmailNotifier) -> int:
    ib = _connect(args)
    sent_count = 0
    try:
        for raw_symbol in args.symbols:
            symbol = "AAPL" if raw_symbol.upper() in {"APPLE", "APPL"} else raw_symbol.upper()
            try:
                stock, _ = _stock_and_spot(ib, symbol, wait_sec=args.spot_wait_sec)
                bars, cache_path, fresh_count = _load_or_refresh_hourly_bars(ib, stock, symbol, args)
                snapshots = _calc_kdj(
                    bars,
                    period=args.kdj_period,
                    k_smooth=args.k_smooth,
                    d_smooth=args.d_smooth,
                )
                if not snapshots:
                    print(f"{symbol} no enough hourly bars for KDJ")
                    continue
                snap = snapshots[-1]
                level = _kdj_level(snap.j, args.near_j, args.watch_j, args.extreme_j)
                print(
                    f"{symbol} hourly KDJ {snap.ts.strftime('%Y-%m-%d %H:%M ET')} "
                    f"close={snap.close:.2f} K={snap.k:.2f} D={snap.d:.2f} J={snap.j:.2f} "
                    f"level={level or '-'} bars={len(bars)} fresh={fresh_count} "
                    f"cache={cache_path or '-'}"
                )
                if not level:
                    continue

                dedupe_key = f"{symbol}:{snap.ts.isoformat()}:{level}"
                if state.get("sent", {}).get(dedupe_key) and not args.force:
                    print(f"{symbol} {level} already alerted for {snap.ts.isoformat()}")
                    continue

                _, rows = _collect_option_volume_rows(
                    ib,
                    symbol=symbol,
                    target_dtes=args.target_dtes,
                    expiries=args.expiries,
                    include_dec26=args.include_dec26,
                    strikes=args.strikes,
                    rights=args.rights,
                    wait_sec=args.wait_sec,
                    chunk_size=args.chunk_size,
                    chunk_sleep=args.chunk_sleep,
                    strike_pct=args.strike_pct,
                    max_strikes=args.max_strikes,
                )
                ts_tag = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
                csv_path = args.output_dir / f"option_flow_kdj_{symbol.lower()}_{ts_tag}.csv"
                for row in rows:
                    row["kdj_bar_et"] = snap.ts.isoformat()
                    row["kdj_k"] = snap.k
                    row["kdj_d"] = snap.d
                    row["kdj_j"] = snap.j
                    row["kdj_level"] = level
                _write_csv(csv_path, rows)

                subject = (
                    f"{symbol} hourly KDJ {level} J={snap.j:.1f} "
                    f"close={snap.close:.2f}"
                )
                body = _build_kdj_email_body(
                    symbol=symbol,
                    snap=snap,
                    level=level,
                    csv_path=csv_path,
                    cache_path=cache_path,
                    rows=rows,
                    top=args.email_top,
                )
                if notifier.send_alert(subject, body):
                    sent_count += 1
                    state.setdefault("sent", {})[dedupe_key] = {
                        "sent_at": datetime.now(ET).isoformat(),
                        "j": snap.j,
                        "csv": str(csv_path),
                    }
                    _save_monitor_state(args.state_file, state)
                else:
                    print(f"{symbol} alert not sent; email disabled/missing password or SMTP failed")
            except Exception as exc:
                log.exception("KDJ option-flow monitor failed for %s", symbol)
                print(f"{symbol} error={exc}")
    finally:
        ib.disconnect()
    return sent_count


def kdj_monitor(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = _load_monitor_state(args.state_file)
    notifier = _email_notifier_from_args(args)
    while True:
        sent = _kdj_monitor_once(args, state, notifier)
        print(f"kdj_monitor_cycle_done sent={sent}")
        if not args.monitor:
            break
        time.sleep(args.interval_sec)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan IB option flow and event-window option trades")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=391)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--market-data-type", type=int, default=2, choices=[1, 2, 3, 4])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan-volume", help="Snapshot option volume/OI/IV")
    p.add_argument("--symbols", type=_parse_csv_strings, required=True)
    p.add_argument("--target-dtes", type=lambda s: [int(x) for x in _parse_csv_strings(s)], default=[7, 30, 60, 180, 365])
    p.add_argument("--expiries", type=_parse_csv_strings)
    p.add_argument("--include-dec26", action="store_true")
    p.add_argument("--strikes", type=_parse_csv_floats)
    p.add_argument("--rights", type=_parse_csv_strings, default=["C", "P"])
    p.add_argument("--wait-sec", type=float, default=8.0)
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--chunk-sleep", type=float, default=0.4)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=scan_volume)

    p = sub.add_parser("event-calls", help="Aggregate historical call trades around an underlying event")
    p.add_argument("--symbol", required=True)
    p.add_argument("--date", required=True, help="YYYYMMDD trading date")
    p.add_argument("--threshold", type=float, help="Event is first 1-min bar with low <= threshold; default day low")
    p.add_argument("--window-min", type=int, default=15)
    p.add_argument("--target-dtes", type=lambda s: [int(x) for x in _parse_csv_strings(s)], default=[30, 60, 120, 220])
    p.add_argument("--expiries", type=_parse_csv_strings)
    p.add_argument("--include-dec26", action="store_true")
    p.add_argument("--strikes", type=_parse_csv_floats)
    p.add_argument("--max-ticks", type=int, default=1000)
    p.add_argument("--hist-sleep", type=float, default=0.15)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=event_calls)

    p = sub.add_parser("kdj-monitor", help="Monitor hourly KDJ lows and email option-flow snapshots")
    p.add_argument("--symbols", type=_parse_csv_strings, default=["ORCL", "NVDA", "AAPL", "NOK", "AMD", "LITE"])
    p.add_argument("--monitor", action="store_true", help="Run continuously instead of one cycle")
    p.add_argument("--interval-sec", type=int, default=1800)
    p.add_argument("--state-file", type=Path, default=DEFAULT_OUT_DIR / "option_flow_kdj_monitor_state.json")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--force", action="store_true", help="Ignore per-bar dedupe")
    p.add_argument("--kdj-duration", default="45 D")
    p.add_argument("--kdj-period", type=int, default=9)
    p.add_argument("--k-smooth", type=int, default=3)
    p.add_argument("--d-smooth", type=int, default=3)
    p.add_argument("--use-rth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache-bars", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bar-cache-dir", type=Path, default=ROOT_DIR / "data" / "option_flow_kdj")
    p.add_argument("--bar-backfill-duration", default="180 D")
    p.add_argument("--bar-refresh-duration", default="5 D")
    p.add_argument("--bar-cache-days", type=int, default=370)
    p.add_argument("--near-j", type=float, default=25.0)
    p.add_argument("--watch-j", type=float, default=20.0)
    p.add_argument("--extreme-j", type=float, default=5.0)
    p.add_argument("--target-dtes", type=lambda s: [int(x) for x in _parse_csv_strings(s)], default=[14, 30, 60, 120, 220])
    p.add_argument("--expiries", type=_parse_csv_strings)
    p.add_argument("--include-dec26", action="store_true")
    p.add_argument("--strikes", type=_parse_csv_floats)
    p.add_argument("--rights", type=_parse_csv_strings, default=["C", "P"])
    p.add_argument("--strike-pct", type=float, default=0.30)
    p.add_argument("--max-strikes", type=int, default=28)
    p.add_argument("--spot-wait-sec", type=float, default=1.0)
    p.add_argument("--wait-sec", type=float, default=8.0)
    p.add_argument("--chunk-size", type=int, default=70)
    p.add_argument("--chunk-sleep", type=float, default=0.35)
    p.add_argument("--email-enabled", action="store_true")
    p.add_argument("--email-sender", default=os.environ.get("EMAIL_SENDER", "fzhouxu615@gmail.com"))
    p.add_argument("--email-password-env", default=os.environ.get("EMAIL_PASSWORD_ENV", "GMAIL_APP_PASSWORD"))
    p.add_argument("--email-recipients", default=os.environ.get("EMAIL_RECIPIENTS", "wenyi.hann@gmail.com"))
    p.add_argument("--email-subject-prefix", default=os.environ.get("EMAIL_SUBJECT_PREFIX", "[OptionFlow KDJ]"))
    p.add_argument("--email-cooldown-sec", type=int, default=1800)
    p.add_argument("--email-top", type=int, default=8)
    p.set_defaults(func=kdj_monitor)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
