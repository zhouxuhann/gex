"""Daily MA5 pullback monitor backed by IB Gateway market data."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB, Option, Stock

from gex_monitor.email_notifier import EmailConfig, EmailNotifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE = ROOT_DIR / "config" / "ma5_pullback_universe.txt"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "ma5_pullback"
DEFAULT_STATE_FILE = DEFAULT_DATA_DIR / "ma5_pullback_monitor_state.json"
ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyBar:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


@dataclass(frozen=True)
class IntradayBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


@dataclass(frozen=True)
class PullbackSnapshot:
    symbol: str
    ts_et: datetime
    price: float
    price_source: str
    ma5: float
    previous_ma5: float | None
    prev_close: float
    distance_pct: float
    prev_close_distance_pct: float
    today_low: float | None
    today_low_distance_pct: float | None
    five_closes: list[float]
    completed_days: list[date]
    touched_by_price: bool
    touched_by_low: bool
    prior_above: bool
    rising_ma5: bool | None
    signal: bool


@dataclass(frozen=True)
class NokCallSetup:
    symbol: str
    ts_et: datetime
    setup: str
    signal: bool
    reason: str
    price: float
    daily_ema10: float
    daily_ema20: float
    daily_gap_pct: float
    price_to_daily_ema10_pct: float
    price_to_daily_ema20_pct: float
    atr14: float
    intraday_ema10: float
    intraday_ema20: float
    price_to_intraday_ema10_pct: float
    price_to_intraday_ema20_pct: float
    recent_low: float
    latest_5m: IntradayBar
    trend_ok: bool
    daily_ema10_rising: bool
    daily_ema20_rising: bool
    option: dict | None = None


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
    return "-" if value is None else f"{value:+.2f}%"


def _parse_csv_strings(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        items = value.split(",")
    return [item.strip() for item in items if item and item.strip()]


def _parse_recipients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_symbol(symbol: str) -> str:
    out = symbol.strip().upper()
    if out in {"APPLE", "APPL"}:
        return "AAPL"
    return out


def _load_universe(args: argparse.Namespace) -> list[str]:
    symbols = [_normalize_symbol(item) for item in args.symbols]
    if args.universe_file and args.universe_file.exists():
        for line in args.universe_file.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            symbols.append(_normalize_symbol(item.split(",")[0].strip()))

    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol and symbol not in seen:
            out.append(symbol)
            seen.add(symbol)
    return out


def _connect(args: argparse.Namespace) -> IB:
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True, timeout=args.timeout)
    ib.reqMarketDataType(args.market_data_type)
    return ib


def _stock(ib: IB, symbol: str) -> Stock:
    contract = Stock(symbol.upper(), "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"Could not qualify stock contract for {symbol}")
    return qualified[0]


def _latest_price(ib: IB, stock: Stock, wait_sec: float) -> tuple[float, str]:
    ticker = ib.reqMktData(stock, "", False, False)
    ib.sleep(wait_sec)
    candidates = [
        ("market", _clean(ticker.marketPrice(), positive=True)),
        ("last", _clean(getattr(ticker, "last", None), positive=True)),
        ("close", _clean(getattr(ticker, "close", None), positive=True)),
        ("bid", _clean(getattr(ticker, "bid", None), positive=True)),
        ("ask", _clean(getattr(ticker, "ask", None), positive=True)),
    ]
    ib.cancelMktData(stock)
    for source, price in candidates:
        if price is not None:
            return price, source
    raise RuntimeError(f"Could not get valid latest price for {stock.symbol}")


def _bar_day(value) -> date:
    if isinstance(value, datetime):
        return value.astimezone(ET).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text[:8], fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(ET).date()
    except ValueError as exc:
        raise ValueError(f"Could not parse IB bar date: {value!r}") from exc


def _bar_datetime(value) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ET)
        return value.astimezone(ET)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=ET)
    text = str(value).strip()
    for fmt in ("%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d"):
        try:
            parsed = datetime.strptime(text[:8] if fmt == "%Y%m%d" else text, fmt)
            return parsed.replace(tzinfo=ET)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Could not parse IB bar datetime: {value!r}") from exc
    return parsed.replace(tzinfo=ET) if parsed.tzinfo is None else parsed.astimezone(ET)


def _daily_bars(
    ib: IB,
    stock: Stock,
    args: argparse.Namespace,
    duration: str | None = None,
) -> list[DailyBar]:
    bars = ib.reqHistoricalData(
        stock,
        endDateTime="",
        durationStr=duration or args.duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=args.use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    out: list[DailyBar] = []
    for bar in bars:
        close = _clean(getattr(bar, "close", None), positive=True)
        if close is None:
            continue
        out.append(
            DailyBar(
                day=_bar_day(getattr(bar, "date")),
                open=float(getattr(bar, "open")),
                high=float(getattr(bar, "high")),
                low=float(getattr(bar, "low")),
                close=close,
                volume=_clean(getattr(bar, "volume", None)),
            )
        )
    return sorted(out, key=lambda item: item.day)


def _intraday_bars(ib: IB, stock: Stock, args: argparse.Namespace) -> list[IntradayBar]:
    bars = ib.reqHistoricalData(
        stock,
        endDateTime="",
        durationStr=args.nok_intraday_duration,
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=args.use_rth,
        formatDate=2,
        keepUpToDate=False,
    )
    out: list[IntradayBar] = []
    for bar in bars:
        close = _clean(getattr(bar, "close", None), positive=True)
        if close is None:
            continue
        out.append(
            IntradayBar(
                ts=_bar_datetime(getattr(bar, "date")),
                open=float(getattr(bar, "open")),
                high=float(getattr(bar, "high")),
                low=float(getattr(bar, "low")),
                close=close,
                volume=_clean(getattr(bar, "volume", None)),
            )
        )
    return sorted(out, key=lambda item: item.ts)


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    out: list[float] = []
    ema = values[0]
    for idx, value in enumerate(values):
        ema = value if idx == 0 else value * k + ema * (1.0 - k)
        out.append(ema)
    return out


def _atr(bars: list[DailyBar], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs: list[float] = []
    for idx, bar in enumerate(bars):
        if idx == 0:
            trs.append(bar.high - bar.low)
            continue
        prev_close = bars[idx - 1].close
        trs.append(max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close)))
    return sum(trs[-period:]) / period


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"sent": {}}
    except json.JSONDecodeError:
        return {"sent": {}}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _state_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=ET) if parsed.tzinfo is None else parsed.astimezone(ET)


def _trim_state(state: dict, keep_days: int = 90) -> None:
    sent = state.setdefault("sent", {})
    cutoff_ord = datetime.now(ET).date().toordinal() - keep_days
    for key in list(sent):
        item = sent.get(key) or {}
        day_text = str(item.get("session_date") or "")
        try:
            day_ord = datetime.strptime(day_text, "%Y-%m-%d").date().toordinal()
        except ValueError:
            continue
        if day_ord < cutoff_ord:
            del sent[key]


def _evaluate_symbol(ib: IB, symbol: str, args: argparse.Namespace) -> PullbackSnapshot:
    contract = _stock(ib, symbol)
    price, source = _latest_price(ib, contract, args.spot_wait_sec)
    bars = _daily_bars(ib, contract, args)
    if len(bars) < 6:
        raise RuntimeError(f"Need at least 6 daily bars, got {len(bars)}")

    today_et = datetime.now(ET).date()
    completed = [bar for bar in bars if bar.day < today_et]
    today_bar = next((bar for bar in reversed(bars) if bar.day == today_et), None)
    if len(completed) < 6:
        completed = bars[:-1] if len(bars) >= 7 else bars
        if len(completed) < 6:
            raise RuntimeError(f"Need at least 6 completed daily bars, got {len(completed)}")

    five = completed[-5:]
    prev_five = completed[-6:-1]
    ma5 = sum(bar.close for bar in five) / 5.0
    previous_ma5 = sum(bar.close for bar in prev_five) / 5.0 if len(prev_five) == 5 else None
    prev_close = completed[-1].close
    distance_pct = (price - ma5) / ma5 * 100.0
    prev_close_distance_pct = (prev_close - ma5) / ma5 * 100.0
    today_low = today_bar.low if today_bar else None
    today_low_distance_pct = (today_low - ma5) / ma5 * 100.0 if today_low is not None else None

    upper = args.tolerance_pct
    lower = -args.max_below_pct
    touched_by_price = lower <= distance_pct <= upper
    touched_by_low = (
        today_low is not None
        and today_low_distance_pct is not None
        and today_low_distance_pct <= upper
        and distance_pct >= lower
        and distance_pct <= args.post_touch_max_above_pct
    )
    prior_above = prev_close_distance_pct >= args.min_prior_above_pct
    rising_ma5 = None if previous_ma5 is None else ma5 > previous_ma5
    rising_ok = not args.require_rising_ma5 or rising_ma5 is True
    signal = prior_above and rising_ok and (touched_by_price or touched_by_low)

    return PullbackSnapshot(
        symbol=symbol,
        ts_et=datetime.now(ET),
        price=price,
        price_source=source,
        ma5=ma5,
        previous_ma5=previous_ma5,
        prev_close=prev_close,
        distance_pct=distance_pct,
        prev_close_distance_pct=prev_close_distance_pct,
        today_low=today_low,
        today_low_distance_pct=today_low_distance_pct,
        five_closes=[bar.close for bar in five],
        completed_days=[bar.day for bar in five],
        touched_by_price=touched_by_price,
        touched_by_low=touched_by_low,
        prior_above=prior_above,
        rising_ma5=rising_ma5,
        signal=signal,
    )


def _email_notifier(args: argparse.Namespace) -> EmailNotifier:
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


def _option_mid_quote(ticker) -> tuple[float | None, float | None, float | None]:
    bid = _clean(getattr(ticker, "bid", None), positive=True)
    ask = _clean(getattr(ticker, "ask", None), positive=True)
    last = _clean(getattr(ticker, "last", None), positive=True)
    close = _clean(getattr(ticker, "close", None), positive=True)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0, bid, ask
    return last or close, bid, ask


def _nok_call_candidate(ib: IB, stock: Stock, price: float, args: argparse.Namespace) -> dict | None:
    try:
        chains = ib.reqSecDefOptParams(args.nok_symbol, "", stock.secType, stock.conId)
    except Exception as exc:
        log.warning("NOK option chain failed: %s", exc)
        return None
    expiries: list[tuple[str, int]] = []
    today = datetime.now(ET).date()
    for expiry in sorted({e for chain in chains for e in chain.expirations}):
        try:
            dte = (datetime.strptime(expiry, "%Y%m%d").date() - today).days
        except ValueError:
            continue
        if args.nok_option_min_dte <= dte <= args.nok_option_max_dte:
            expiries.append((expiry, dte))
    if not expiries:
        return None
    expiry, dte = min(expiries, key=lambda item: abs(item[1] - args.nok_option_target_dte))
    all_strikes = sorted({float(strike) for chain in chains for strike in chain.strikes})
    strikes = [
        strike
        for strike in all_strikes
        if price - args.nok_option_strike_window <= strike <= price + args.nok_option_strike_window
    ]
    if not strikes:
        return None

    contracts: list[Option] = []
    for strike in strikes:
        matches = ib.qualifyContracts(Option(args.nok_symbol, expiry, strike, "C", "SMART", currency="USD"))
        if matches:
            contracts.append(matches[0])
    if not contracts:
        return None

    tickers = [ib.reqMktData(contract, "100,101,104,106", False, False) for contract in contracts]
    ib.sleep(args.nok_option_wait_sec)
    rows: list[dict] = []
    for contract, ticker in zip(contracts, tickers):
        mid, bid, ask = _option_mid_quote(ticker)
        if mid is None:
            ib.cancelMktData(contract)
            continue
        greeks = getattr(ticker, "modelGreeks", None) or getattr(ticker, "bidGreeks", None) or getattr(ticker, "askGreeks", None)
        delta = _clean(getattr(greeks, "delta", None)) if greeks else None
        iv = _clean(getattr(greeks, "impliedVol", None)) if greeks else None
        spread_pct = (ask - bid) / mid * 100.0 if bid is not None and ask is not None and mid > 0 else None
        intrinsic = max(0.0, price - float(contract.strike))
        rows.append(
            {
                "expiry": expiry,
                "dte": dte,
                "strike": float(contract.strike),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "delta": delta,
                "iv": iv,
                "intrinsic": intrinsic,
                "extrinsic": mid - intrinsic,
                "volume": _clean(getattr(ticker, "volume", None)),
                "open_interest": _clean(getattr(ticker, "callOpenInterest", None)),
            }
        )
        ib.cancelMktData(contract)
    if not rows:
        return None

    def score(row: dict) -> tuple[float, float, float]:
        delta = row.get("delta")
        delta_penalty = abs((delta if delta is not None else 0.55) - args.nok_option_target_delta)
        spread_penalty = (row.get("spread_pct") if row.get("spread_pct") is not None else 50.0) / 100.0
        atm_penalty = abs(row["strike"] - price) / max(price, 1.0)
        return delta_penalty, spread_penalty, atm_penalty

    liquid = [row for row in rows if row.get("spread_pct") is None or row["spread_pct"] <= args.nok_option_max_spread_pct]
    return sorted(liquid or rows, key=score)[0]


def _evaluate_nok_call_setup(ib: IB, args: argparse.Namespace) -> NokCallSetup:
    symbol = args.nok_symbol.upper()
    stock = _stock(ib, symbol)
    price, _ = _latest_price(ib, stock, args.spot_wait_sec)
    daily = _daily_bars(ib, stock, args, args.nok_daily_duration)
    today = datetime.now(ET).date()
    completed = [bar for bar in daily if bar.day < today]
    if len(completed) < max(args.nok_daily_ema_slow + 5, 30):
        completed = daily
    if len(completed) < max(args.nok_daily_ema_slow + 5, 30):
        raise RuntimeError(f"{symbol} needs more daily bars for EMA monitor")

    closes = [bar.close for bar in completed]
    daily_ema10_values = _ema_series(closes, args.nok_daily_ema_fast)
    daily_ema20_values = _ema_series(closes, args.nok_daily_ema_slow)
    daily_ema10 = daily_ema10_values[-1]
    daily_ema20 = daily_ema20_values[-1]
    daily_ema10_prev = daily_ema10_values[-2]
    daily_ema20_prev = daily_ema20_values[-2]
    atr14 = _atr(completed, 14)
    if atr14 is None:
        raise RuntimeError(f"{symbol} needs more daily bars for ATR")

    five_min = _intraday_bars(ib, stock, args)
    min_intraday = max(args.nok_intraday_ema_slow + args.nok_reclaim_lookback_bars, 30)
    if len(five_min) < min_intraday:
        raise RuntimeError(f"{symbol} needs more 5m bars, got {len(five_min)}")
    five_closes = [bar.close for bar in five_min]
    five_ema10_values = _ema_series(five_closes, args.nok_intraday_ema_fast)
    five_ema20_values = _ema_series(five_closes, args.nok_intraday_ema_slow)
    latest = five_min[-1]
    previous = five_min[-2]
    intraday_ema10 = five_ema10_values[-1]
    intraday_ema20 = five_ema20_values[-1]
    recent = five_min[-args.nok_reclaim_lookback_bars :]
    recent_low = min(bar.low for bar in recent)

    daily_gap_pct = (daily_ema10 / daily_ema20 - 1.0) * 100.0
    price_to_daily_ema10_pct = (price / daily_ema10 - 1.0) * 100.0
    price_to_daily_ema20_pct = (price / daily_ema20 - 1.0) * 100.0
    price_to_intraday_ema10_pct = (latest.close / intraday_ema10 - 1.0) * 100.0
    price_to_intraday_ema20_pct = (latest.close / intraday_ema20 - 1.0) * 100.0
    daily_ema10_rising = daily_ema10 > daily_ema10_prev
    daily_ema20_rising = daily_ema20 >= daily_ema20_prev
    trend_ok = (
        daily_ema10 > daily_ema20
        and daily_ema10_rising
        and daily_ema20_rising
        and price >= daily_ema20 - args.nok_ema20_fail_atr * atr14
    )

    setup = "none"
    reason = "daily trend not ready" if not trend_ok else "waiting for EMA pullback/reclaim"
    signal = False

    daily_ema10_upper = daily_ema10 + args.nok_daily_ema10_touch_atr * atr14
    daily_ema10_lower = daily_ema10 - args.nok_daily_ema10_below_atr * atr14
    daily_ema20_upper = daily_ema20 + args.nok_daily_ema20_touch_atr * atr14
    recent_touched_daily_ema10 = recent_low <= daily_ema10_upper
    latest_reclaimed_daily_ema10 = latest.close >= daily_ema10_lower and latest.close >= daily_ema10
    recent_touched_daily_ema20 = recent_low <= daily_ema20_upper
    latest_reclaimed_daily_ema20 = latest.close >= daily_ema20
    recent_touched_intraday_ema20 = recent_low <= intraday_ema20 * (1.0 + args.nok_intraday_touch_pct / 100.0)
    latest_reclaimed_intraday = latest.close >= intraday_ema10 and latest.close >= intraday_ema20 and latest.close >= previous.close
    extension_ok = price_to_daily_ema10_pct <= args.nok_max_extension_pct

    if trend_ok and recent_touched_daily_ema10 and latest_reclaimed_daily_ema10:
        setup = "daily_ema10_reclaim"
        reason = "5m low touched daily EMA10 zone and latest 5m close reclaimed daily EMA10"
        signal = True
    elif (
        trend_ok
        and latest.close < daily_ema10
        and latest.close >= daily_ema20
        and recent_touched_daily_ema20
        and latest_reclaimed_daily_ema20
        and latest_reclaimed_intraday
    ):
        setup = "daily_ema20_deep_reclaim"
        reason = "price pulled into daily EMA10/20 band, held EMA20, and reclaimed 5m EMA stack"
        signal = True
    elif trend_ok and extension_ok and recent_touched_intraday_ema20 and latest_reclaimed_intraday:
        setup = "intraday_ema20_reclaim"
        reason = "daily trend intact; 5m pullback touched intraday EMA20 and reclaimed intraday EMA10"
        signal = True

    option = _nok_call_candidate(ib, stock, price, args) if signal and args.nok_option_lookup else None
    return NokCallSetup(
        symbol=symbol,
        ts_et=datetime.now(ET),
        setup=setup,
        signal=signal,
        reason=reason,
        price=price,
        daily_ema10=daily_ema10,
        daily_ema20=daily_ema20,
        daily_gap_pct=daily_gap_pct,
        price_to_daily_ema10_pct=price_to_daily_ema10_pct,
        price_to_daily_ema20_pct=price_to_daily_ema20_pct,
        atr14=atr14,
        intraday_ema10=intraday_ema10,
        intraday_ema20=intraday_ema20,
        price_to_intraday_ema10_pct=price_to_intraday_ema10_pct,
        price_to_intraday_ema20_pct=price_to_intraday_ema20_pct,
        recent_low=recent_low,
        latest_5m=latest,
        trend_ok=trend_ok,
        daily_ema10_rising=daily_ema10_rising,
        daily_ema20_rising=daily_ema20_rising,
        option=option,
    )


def _build_nok_call_email_body(setup: NokCallSetup, args: argparse.Namespace) -> str:
    option = setup.option or {}
    option_lines = "  no option quote selected"
    if option:
        option_lines = (
            f"  {setup.symbol} {option['expiry']} {option['strike']:.1f}C "
            f"DTE={option['dte']} bid={_fmt(option.get('bid'))} ask={_fmt(option.get('ask'))} "
            f"mid={_fmt(option.get('mid'))} delta={_fmt(option.get('delta'), 2)} "
            f"IV={_fmt((option.get('iv') or 0.0) * 100.0 if option.get('iv') is not None else None, 1)}% "
            f"spread={_fmt(option.get('spread_pct'), 1)}%"
        )
    hard_fail = setup.daily_ema20 - args.nok_ema20_fail_atr * setup.atr14
    preferred_zone_low = setup.daily_ema10 - args.nok_daily_ema10_below_atr * setup.atr14
    preferred_zone_high = setup.daily_ema10 + args.nok_daily_ema10_touch_atr * setup.atr14
    return (
        "NOK Call Setup Monitor\n\n"
        f"Symbol:          {setup.symbol}\n"
        f"Time:            {setup.ts_et.strftime('%Y-%m-%d %H:%M:%S ET')}\n"
        f"Setup:           {setup.setup}\n"
        f"Reason:          {setup.reason}\n\n"
        "Daily trend filter:\n"
        f"  price={setup.price:.2f}\n"
        f"  EMA10={setup.daily_ema10:.2f} ({_fmt_pct(setup.price_to_daily_ema10_pct)})\n"
        f"  EMA20={setup.daily_ema20:.2f} ({_fmt_pct(setup.price_to_daily_ema20_pct)})\n"
        f"  EMA10/EMA20 gap={setup.daily_gap_pct:.2f}%\n"
        f"  ATR14={setup.atr14:.2f}\n"
        f"  EMA10 rising={setup.daily_ema10_rising}, EMA20 rising={setup.daily_ema20_rising}\n\n"
        "5m timing:\n"
        f"  latest 5m={setup.latest_5m.ts.strftime('%Y-%m-%d %H:%M ET')} "
        f"O={setup.latest_5m.open:.2f} H={setup.latest_5m.high:.2f} "
        f"L={setup.latest_5m.low:.2f} C={setup.latest_5m.close:.2f}\n"
        f"  5m EMA10={setup.intraday_ema10:.2f} ({_fmt_pct(setup.price_to_intraday_ema10_pct)})\n"
        f"  5m EMA20={setup.intraday_ema20:.2f} ({_fmt_pct(setup.price_to_intraday_ema20_pct)})\n"
        f"  recent low={setup.recent_low:.2f}\n\n"
        "Call candidate:\n"
        f"{option_lines}\n\n"
        "Execution idea:\n"
        f"  preferred pullback zone: {preferred_zone_low:.2f} - {preferred_zone_high:.2f}\n"
        f"  hard invalidation: daily close / clean break below about {hard_fail:.2f}\n"
        "  use limit orders near option mid; avoid market orders when spread widens.\n\n"
        "This is a trend-continuation call-buying setup alert, not a trading instruction."
    )


def _maybe_send_nok_call_email(
    notifier: EmailNotifier,
    setup: NokCallSetup,
    state: dict,
    args: argparse.Namespace,
) -> bool:
    if not setup.signal:
        return False
    group = state.setdefault("nok_call", {})
    sent = group.setdefault("sent", {})
    last_by_setup = group.setdefault("last_sent_by_setup", {})
    last_sent = _state_time(last_by_setup.get(setup.setup))
    if last_sent is not None and not args.force:
        elapsed = (setup.ts_et - last_sent).total_seconds()
        if elapsed < args.nok_alert_cooldown_sec:
            print(f"{setup.symbol} {setup.setup} cooldown {int(elapsed)}s/{args.nok_alert_cooldown_sec}s")
            return False
    dedupe_key = f"{setup.symbol}:{setup.setup}:{setup.latest_5m.ts.isoformat()}"
    if sent.get(dedupe_key) and not args.force:
        print(f"{setup.symbol} {setup.setup} already alerted for {setup.latest_5m.ts.isoformat()}")
        return False

    subject = (
        f"{setup.symbol} call setup {setup.setup} price={setup.price:.2f} "
        f"dailyEMA10={setup.daily_ema10:.2f} 5mEMA20={setup.intraday_ema20:.2f}"
    )
    if not notifier.send_alert(subject, _build_nok_call_email_body(setup, args)):
        print(f"{setup.symbol} call setup not sent; email disabled/missing password or SMTP failed")
        return False
    sent[dedupe_key] = {
        "sent_at": setup.ts_et.isoformat(),
        "session_date": setup.ts_et.date().isoformat(),
        "price": setup.price,
        "setup": setup.setup,
        "daily_ema10": setup.daily_ema10,
        "daily_ema20": setup.daily_ema20,
        "intraday_ema10": setup.intraday_ema10,
        "intraday_ema20": setup.intraday_ema20,
        "option": setup.option,
    }
    last_by_setup[setup.setup] = setup.ts_et.isoformat()
    _trim_state(state)
    _save_state(args.state_file, state)
    return True


def _build_email_body(snapshot: PullbackSnapshot, args: argparse.Namespace) -> str:
    close_lines = []
    for day, close in zip(snapshot.completed_days, snapshot.five_closes):
        close_lines.append(f"  {day.isoformat()} close={close:.2f}")
    rising = "-" if snapshot.rising_ma5 is None else ("yes" if snapshot.rising_ma5 else "no")
    return (
        "MA5 Pullback Monitor\n\n"
        f"Symbol:          {snapshot.symbol}\n"
        f"Time:            {snapshot.ts_et.strftime('%Y-%m-%d %H:%M:%S ET')}\n"
        f"Latest price:    {snapshot.price:.2f} ({snapshot.price_source})\n"
        f"MA5:             {snapshot.ma5:.2f}\n"
        f"Distance to MA5: {_fmt_pct(snapshot.distance_pct)}\n"
        f"Prev close:      {snapshot.prev_close:.2f} ({_fmt_pct(snapshot.prev_close_distance_pct)} vs MA5)\n"
        f"Today low:       {_fmt(snapshot.today_low)} ({_fmt_pct(snapshot.today_low_distance_pct)} vs MA5)\n"
        f"Previous MA5:    {_fmt(snapshot.previous_ma5)}\n"
        f"MA5 rising:      {rising}\n\n"
        "Trigger logic:\n"
        f"  prior close above MA5 by >= {args.min_prior_above_pct:.2f}%: {snapshot.prior_above}\n"
        f"  price inside MA5 zone [{-args.max_below_pct:.2f}%, +{args.tolerance_pct:.2f}%]: {snapshot.touched_by_price}\n"
        f"  intraday low touched MA5 and price still holds: {snapshot.touched_by_low}\n"
        f"  require rising MA5: {args.require_rising_ma5}\n\n"
        "Five completed closes used for MA5:\n"
        f"{chr(10).join(close_lines)}\n\n"
        "This is a pullback observation alert, not a trading instruction."
    )


def _maybe_send_email(
    notifier: EmailNotifier,
    snapshot: PullbackSnapshot,
    state: dict,
    args: argparse.Namespace,
) -> bool:
    session_date = snapshot.ts_et.date().isoformat()
    dedupe_key = f"{snapshot.symbol}:{session_date}:ma5_pullback"
    if state.get("sent", {}).get(dedupe_key) and not args.force:
        print(f"{snapshot.symbol} already alerted for {session_date}")
        return False

    subject = (
        f"{snapshot.symbol} pullback to MA5 price={snapshot.price:.2f} "
        f"MA5={snapshot.ma5:.2f} dist={snapshot.distance_pct:+.2f}%"
    )
    if not notifier.send_alert(subject, _build_email_body(snapshot, args)):
        print(f"{snapshot.symbol} alert not sent; email disabled/missing password or SMTP failed")
        return False

    state.setdefault("sent", {})[dedupe_key] = {
        "session_date": session_date,
        "sent_at": snapshot.ts_et.isoformat(),
        "price": snapshot.price,
        "ma5": snapshot.ma5,
        "distance_pct": snapshot.distance_pct,
    }
    _trim_state(state)
    _save_state(args.state_file, state)
    return True


def run_once(args: argparse.Namespace, state: dict | None = None, notifier: EmailNotifier | None = None) -> int:
    symbols = _load_universe(args)
    if not symbols:
        raise RuntimeError("No symbols configured")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    state = state if state is not None else _load_state(args.state_file)
    notifier = notifier or _email_notifier(args)

    sent_count = 0
    ib = _connect(args)
    try:
        if args.nok_call_monitor_enabled:
            try:
                nok_setup = _evaluate_nok_call_setup(ib, args)
                nok_status = "SIGNAL" if nok_setup.signal else "-"
                print(
                    f"{nok_setup.symbol} CALL {nok_status} setup={nok_setup.setup} "
                    f"price={nok_setup.price:.2f} dailyEMA10={nok_setup.daily_ema10:.2f} "
                    f"dailyEMA20={nok_setup.daily_ema20:.2f} "
                    f"pxE10={nok_setup.price_to_daily_ema10_pct:+.2f}% "
                    f"5mEMA10={nok_setup.intraday_ema10:.2f} 5mEMA20={nok_setup.intraday_ema20:.2f} "
                    f"recentLow={nok_setup.recent_low:.2f} reason={nok_setup.reason}"
                )
                if _maybe_send_nok_call_email(notifier, nok_setup, state, args):
                    sent_count += 1
            except Exception as exc:
                log.exception("NOK call setup monitor failed")
                print(f"{args.nok_symbol.upper()} call error={exc}")

        for symbol in symbols:
            try:
                snap = _evaluate_symbol(ib, symbol, args)
                status = "SIGNAL" if snap.signal else "-"
                print(
                    f"{snap.symbol} {status} price={snap.price:.2f} ma5={snap.ma5:.2f} "
                    f"dist={snap.distance_pct:+.2f}% prev_above={snap.prior_above} "
                    f"rising={snap.rising_ma5} low_dist={_fmt_pct(snap.today_low_distance_pct)}"
                )
                if snap.signal and _maybe_send_email(notifier, snap, state, args):
                    sent_count += 1
            except Exception as exc:
                log.exception("MA5 pullback scan failed for %s", symbol)
                print(f"{symbol} error={exc}")
            ib.sleep(args.symbol_sleep)
    finally:
        ib.disconnect()
    _save_state(args.state_file, state)
    return sent_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor stocks pulling back to their daily MA5 via IB Gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=397)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--market-data-type", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--symbols", type=_parse_csv_strings, default=[])
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=1800)
    parser.add_argument("--duration", default="30 D")
    parser.add_argument("--use-rth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spot-wait-sec", type=float, default=1.2)
    parser.add_argument("--symbol-sleep", type=float, default=0.4)
    parser.add_argument("--tolerance-pct", type=float, default=0.70)
    parser.add_argument("--max-below-pct", type=float, default=0.50)
    parser.add_argument("--post-touch-max-above-pct", type=float, default=2.00)
    parser.add_argument("--min-prior-above-pct", type=float, default=1.00)
    parser.add_argument("--require-rising-ma5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--email-enabled", action="store_true")
    parser.add_argument("--email-sender", default=os.environ.get("EMAIL_SENDER", "fzhouxu615@gmail.com"))
    parser.add_argument("--email-password-env", default=os.environ.get("EMAIL_PASSWORD_ENV", "GMAIL_APP_PASSWORD"))
    parser.add_argument("--email-recipients", default=os.environ.get("EMAIL_RECIPIENTS", "wenyi.hann@gmail.com"))
    parser.add_argument("--email-subject-prefix", default=os.environ.get("EMAIL_SUBJECT_PREFIX", "[MA5 Pullback]"))
    parser.add_argument("--email-cooldown-sec", type=int, default=1800)
    parser.add_argument("--nok-call-monitor-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nok-symbol", default="NOK")
    parser.add_argument("--nok-daily-duration", default="120 D")
    parser.add_argument("--nok-intraday-duration", default="2 D")
    parser.add_argument("--nok-daily-ema-fast", type=int, default=10)
    parser.add_argument("--nok-daily-ema-slow", type=int, default=20)
    parser.add_argument("--nok-intraday-ema-fast", type=int, default=10)
    parser.add_argument("--nok-intraday-ema-slow", type=int, default=20)
    parser.add_argument("--nok-reclaim-lookback-bars", type=int, default=6)
    parser.add_argument("--nok-daily-ema10-touch-atr", type=float, default=0.25)
    parser.add_argument("--nok-daily-ema10-below-atr", type=float, default=0.10)
    parser.add_argument("--nok-daily-ema20-touch-atr", type=float, default=0.25)
    parser.add_argument("--nok-ema20-fail-atr", type=float, default=0.10)
    parser.add_argument("--nok-intraday-touch-pct", type=float, default=0.35)
    parser.add_argument("--nok-max-extension-pct", type=float, default=5.00)
    parser.add_argument("--nok-alert-cooldown-sec", type=int, default=7200)
    parser.add_argument("--nok-option-lookup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nok-option-target-dte", type=int, default=15)
    parser.add_argument("--nok-option-min-dte", type=int, default=8)
    parser.add_argument("--nok-option-max-dte", type=int, default=24)
    parser.add_argument("--nok-option-target-delta", type=float, default=0.55)
    parser.add_argument("--nok-option-max-spread-pct", type=float, default=20.0)
    parser.add_argument("--nok-option-strike-window", type=float, default=3.0)
    parser.add_argument("--nok-option-wait-sec", type=float, default=4.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    state = _load_state(args.state_file)
    notifier = _email_notifier(args)
    while True:
        try:
            sent = run_once(args, state=state, notifier=notifier)
            print(f"ma5_pullback_cycle_done sent={sent}")
        except Exception:
            log.exception("MA5 pullback monitor cycle failed")
            if not args.monitor:
                return 1
        if not args.monitor:
            break
        time.sleep(args.interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
