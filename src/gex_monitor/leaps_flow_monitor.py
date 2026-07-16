"""LEAPS option-flow monitor for broad watchlists.

The goal is to catch slow institutional-style accumulation signals:

* LEAPS call open interest rising versus prior local snapshots.
* Longer-dated IV lifting.
* Unusual LEAPS volume/notional in a contract that is hard to hide.

IB's option open interest is not truly real-time; it often updates from the
prior close. This monitor stores local snapshots so the useful signal is the
change across runs/days, while same-day volume is treated as a faster hint.
"""

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

import pandas as pd
from ib_insync import IB, Option, Stock

from gex_monitor.email_notifier import EmailConfig, EmailNotifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "leaps_flow"
DEFAULT_UNIVERSE = ROOT_DIR / "config" / "leaps_flow_universe.txt"
ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractMeta:
    symbol: str
    expiry: str
    dte: int
    strike: float
    right: str


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
    return "-" if value is None or pd.isna(value) else f"{float(value):,.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value) * 100:.1f}%"


def _parse_csv_strings(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_dtes(value: str | None) -> list[int]:
    return [int(item) for item in _parse_csv_strings(value)]


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol in {"APPLE", "APPL"}:
        return "AAPL"
    return symbol


def _load_universe(args: argparse.Namespace) -> list[str]:
    symbols = [_normalize_symbol(s) for s in args.symbols]
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
    asof = asof or date.today()
    return (datetime.strptime(expiry, "%Y%m%d").date() - asof).days


def _choose_expiries(expiries: list[str], args: argparse.Namespace) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for expiry in expiries:
        try:
            dte = _dte(expiry)
        except ValueError:
            continue
        if args.min_dte <= dte <= args.max_dte:
            parsed.append((expiry, dte))
    if not parsed:
        return []

    if args.target_dtes:
        chosen: list[tuple[str, int]] = []
        used: set[str] = set()
        for target in args.target_dtes:
            candidates = [item for item in parsed if item[0] not in used]
            if not candidates:
                break
            item = min(candidates, key=lambda row: abs(row[1] - target))
            chosen.append(item)
            used.add(item[0])
        return chosen[: args.max_expiries]

    return parsed[: args.max_expiries]


def _choose_strikes(strikes: list[float], spot: float, args: argparse.Namespace) -> list[float]:
    filtered = [s for s in strikes if args.min_moneyness <= s / spot <= args.max_moneyness]
    if len(filtered) <= args.max_strikes:
        return filtered
    if args.strike_mode == "nearest":
        return sorted(sorted(filtered, key=lambda s: abs(s / spot - 1.0))[: args.max_strikes])

    # Evenly cover the moneyness range so far OTM LEAPS are not missed.
    step = (len(filtered) - 1) / max(args.max_strikes - 1, 1)
    idxs = sorted({round(i * step) for i in range(args.max_strikes)})
    selected = [filtered[i] for i in idxs if 0 <= i < len(filtered)]
    atm = min(filtered, key=lambda s: abs(s / spot - 1.0))
    if atm not in selected:
        selected.append(atm)
    return sorted(selected)[: args.max_strikes]


def _qualify_options(
    ib: IB,
    symbol: str,
    expiries: list[tuple[str, int]],
    strikes: list[float],
    rights: list[str],
) -> tuple[list[Option], list[ContractMeta]]:
    contracts: list[Option] = []
    meta: list[ContractMeta] = []
    dte_by_expiry = dict(expiries)
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
                    meta.append(ContractMeta(symbol, expiry, dte_by_expiry[expiry], float(strike), right))
    return contracts, meta


def _mid(ticker) -> tuple[float | None, float | None, float | None]:
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
    expiries = _choose_expiries(expiries_raw, args)
    strikes = _choose_strikes(strikes_raw, spot, args)
    contracts, meta = _qualify_options(ib, symbol, expiries, strikes, args.rights)
    print(
        f"{symbol} spot={spot:.2f} expiries={len(expiries)} strikes={len(strikes)} "
        f"contracts={len(contracts)}"
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
        price, bid, ask = _mid(ticker)
        greeks = getattr(ticker, "modelGreeks", None)
        oi_attr = "callOpenInterest" if item.right == "C" else "putOpenInterest"
        oi = _clean(getattr(ticker, oi_attr, None), positive=True)
        volume = _clean(getattr(ticker, "volume", None), positive=True)
        rows.append(
            {
                "ts": ts,
                "symbol": item.symbol,
                "spot": spot,
                "expiry": item.expiry,
                "dte": item.dte,
                "strike": item.strike,
                "right": item.right,
                "moneyness": item.strike / spot,
                "bid": bid,
                "ask": ask,
                "mid": price,
                "last": _clean(getattr(ticker, "last", None), positive=True),
                "volume": volume,
                "open_interest": oi,
                "iv": _clean(getattr(greeks, "impliedVol", None), positive=True) if greeks else None,
                "delta": _clean(getattr(greeks, "delta", None)) if greeks else None,
                "vega": _clean(getattr(greeks, "vega", None)) if greeks else None,
                "theta": _clean(getattr(greeks, "theta", None)) if greeks else None,
                "notional_volume": (volume or 0.0) * (price or 0.0) * 100.0,
            }
        )
    for ticker in tickers:
        try:
            ib.cancelMktData(ticker.contract)
        except Exception:
            pass
    return rows


def _read_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _select_batch(symbols: list[str], args: argparse.Namespace) -> list[str]:
    if args.scan_all or args.batch_size <= 0 or len(symbols) <= args.batch_size:
        return symbols
    state = _read_json(args.cursor_file, {"cursor": 0})
    cursor = int(state.get("cursor", 0)) % len(symbols)
    batch = [symbols[(cursor + i) % len(symbols)] for i in range(args.batch_size)]
    state["cursor"] = (cursor + args.batch_size) % len(symbols)
    state["updated_at"] = datetime.now(ET).isoformat()
    _write_json(args.cursor_file, state)
    return batch


def _history_path(args: argparse.Namespace) -> Path:
    return args.data_dir / "leaps_flow_history.parquet"


def _snapshot_path(args: argparse.Namespace) -> Path:
    tag = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    return args.data_dir / "snapshots" / f"leaps_flow_snapshot_{tag}.parquet"


def _append_history(args: argparse.Namespace, current: pd.DataFrame) -> None:
    path = _history_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            current = pd.concat([old, current], ignore_index=True)
        except Exception as exc:
            log.warning("Could not read existing history %s: %s", path, exc)
    current["_ts_dt"] = pd.to_datetime(current["ts"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.history_days)
    current = current[current["_ts_dt"] >= cutoff]
    current = current.sort_values("_ts_dt").drop_duplicates(
        subset=["ts", "symbol", "expiry", "strike", "right"],
        keep="last",
    )
    current = current.drop(columns=["_ts_dt"]).reset_index(drop=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    current.to_parquet(tmp, index=False)
    tmp.replace(path)


def _load_history_before(args: argparse.Namespace, current_ts: str) -> pd.DataFrame:
    path = _history_path(args)
    if not path.exists():
        return pd.DataFrame()
    hist = pd.read_parquet(path)
    if hist.empty:
        return hist
    hist["_ts_dt"] = pd.to_datetime(hist["ts"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(current_ts, utc=True)
    return hist[hist["_ts_dt"] < cutoff].copy()


def _with_prior_metrics(args: argparse.Namespace, current: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return current
    hist = _load_history_before(args, str(current["ts"].iloc[0]))
    if hist.empty:
        out = current.copy()
        out["prior_ts"] = None
        out["prior_oi"] = None
        out["prior_iv"] = None
        out["oi_change"] = None
        out["oi_change_pct"] = None
        out["iv_change"] = None
        return out

    keys = ["symbol", "expiry", "strike", "right"]
    prior = hist.sort_values("_ts_dt").groupby(keys, as_index=False).tail(1)
    prior = prior[keys + ["ts", "open_interest", "iv"]].rename(
        columns={"ts": "prior_ts", "open_interest": "prior_oi", "iv": "prior_iv"}
    )
    out = current.merge(prior, on=keys, how="left")
    out["oi_change"] = out["open_interest"] - out["prior_oi"]
    out["oi_change_pct"] = out["oi_change"] / out["prior_oi"].replace(0, pd.NA)
    out["iv_change"] = out["iv"] - out["prior_iv"]
    return out


def _find_alerts(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["vol_oi_ratio"] = out["volume"] / out["open_interest"].replace(0, pd.NA)
    out["oi_flag"] = (
        (out["right"] == "C")
        & (out["oi_change"].fillna(0) >= args.min_oi_change)
        & (out["oi_change_pct"].fillna(0) >= args.min_oi_change_pct)
    )
    out["iv_flag"] = (
        (out["right"] == "C")
        & (out["iv_change"].fillna(0) >= args.min_iv_change)
        & (out["volume"].fillna(0) >= args.min_volume_for_iv)
    )
    out["volume_flag"] = (
        (out["right"] == "C")
        & (out["volume"].fillna(0) >= args.min_volume)
        & (out["notional_volume"].fillna(0) >= args.min_notional)
    )
    out["score"] = (
        out["oi_flag"].astype(int) * 3
        + out["iv_flag"].astype(int) * 2
        + out["volume_flag"].astype(int)
        + out["oi_change_pct"].fillna(0).clip(lower=0, upper=2)
        + out["iv_change"].fillna(0).clip(lower=0, upper=0.25) * 4
    )
    alerts = out[out["oi_flag"] | out["iv_flag"] | out["volume_flag"]]
    return alerts.sort_values(["score", "notional_volume", "oi_change"], ascending=False)


def _format_alert_table(alerts: pd.DataFrame, top: int) -> str:
    if alerts.empty:
        return "no LEAPS alerts"
    lines = ["symbol expiry dte K vol OI dOI dOI% IV dIV mid notional flags"]
    for row in alerts.head(top).itertuples(index=False):
        flags = ",".join(
            name
            for name, enabled in [
                ("OI", getattr(row, "oi_flag")),
                ("IV", getattr(row, "iv_flag")),
                ("VOL", getattr(row, "volume_flag")),
            ]
            if enabled
        )
        lines.append(
            f"{row.symbol:5s} {row.expiry} {int(row.dte):3d} {float(row.strike):8.2f} "
            f"{_fmt(getattr(row, 'volume'), 0):>7s} {_fmt(getattr(row, 'open_interest'), 0):>8s} "
            f"{_fmt(getattr(row, 'oi_change'), 0):>7s} {_fmt_pct(getattr(row, 'oi_change_pct')):>7s} "
            f"{_fmt_pct(getattr(row, 'iv')):>7s} {_fmt_pct(getattr(row, 'iv_change')):>7s} "
            f"{_fmt(getattr(row, 'mid')):>7s} {_fmt(getattr(row, 'notional_volume'), 0):>10s} {flags}"
        )
    return "\n".join(lines)


def _email_notifier(args: argparse.Namespace) -> EmailNotifier:
    return EmailNotifier(
        EmailConfig(
            enabled=args.email_enabled,
            sender=args.email_sender,
            password_env=args.email_password_env,
            recipients=[x.strip() for x in args.email_recipients.split(",") if x.strip()],
            only_strong=False,
            cooldown_sec=args.email_cooldown_sec,
            subject_prefix=args.email_subject_prefix,
        )
    )


def _maybe_send_email(
    notifier: EmailNotifier,
    alerts: pd.DataFrame,
    symbols: list[str],
    snapshot_path: Path,
    args: argparse.Namespace,
) -> bool:
    if alerts.empty or not args.email_enabled:
        return False
    title = f"LEAPS flow alerts {len(alerts)} contracts / {len(set(alerts['symbol']))} symbols"
    body = (
        "LEAPS Flow Monitor\n\n"
        f"扫描标的: {', '.join(symbols)}\n"
        f"快照: {snapshot_path}\n"
        f"历史库: {_history_path(args)}\n\n"
        f"{_format_alert_table(alerts, args.email_top)}\n\n"
        "解读: OI 是慢变量，通常看隔日/多日变化；盘中 volume 和 IV 抬升是更快的提示。"
        " 这是观察提醒，不是交易指令。"
    )
    return notifier.send_alert(title, body)


def run_once(args: argparse.Namespace, notifier: EmailNotifier | None = None) -> pd.DataFrame:
    symbols = _load_universe(args)
    if not symbols:
        raise RuntimeError("No symbols configured")
    batch = _select_batch(symbols, args)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ib = _connect(args)
    try:
        for symbol in batch:
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

    snapshot = _snapshot_path(args)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    enriched = _with_prior_metrics(args, current)
    alerts = _find_alerts(enriched, args)
    enriched.to_parquet(snapshot, index=False)
    _append_history(args, current)
    print(f"snapshot_written={snapshot}")
    print(f"history_written={_history_path(args)} rows={len(current)} alerts={len(alerts)}")
    if not alerts.empty:
        print(_format_alert_table(alerts, args.top))
    if notifier:
        _maybe_send_email(notifier, alerts, batch, snapshot, args)
    return alerts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor LEAPS OI/IV/volume across many stocks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=406)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--market-data-type", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--symbols", type=_parse_csv_strings, default=[])
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cursor-file", type=Path, default=DEFAULT_DATA_DIR / "leaps_flow_cursor.json")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=1800)
    parser.add_argument("--scan-all", action="store_true", help="Scan every symbol each cycle")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--rights", type=_parse_csv_strings, default=["C"])
    parser.add_argument("--target-dtes", type=_parse_dtes, default=[180, 360, 540, 720])
    parser.add_argument("--min-dte", type=int, default=150)
    parser.add_argument("--max-dte", type=int, default=900)
    parser.add_argument("--max-expiries", type=int, default=5)
    parser.add_argument("--min-moneyness", type=float, default=0.75)
    parser.add_argument("--max-moneyness", type=float, default=2.0)
    parser.add_argument("--max-strikes", type=int, default=24)
    parser.add_argument("--strike-mode", choices=["even", "nearest"], default="even")
    parser.add_argument("--spot-wait-sec", type=float, default=1.2)
    parser.add_argument("--option-wait-sec", type=float, default=8.0)
    parser.add_argument("--chunk-size", type=int, default=70)
    parser.add_argument("--chunk-sleep", type=float, default=0.35)
    parser.add_argument("--symbol-sleep", type=float, default=0.5)
    parser.add_argument("--min-oi-change", type=float, default=1000)
    parser.add_argument("--min-oi-change-pct", type=float, default=0.25)
    parser.add_argument("--min-iv-change", type=float, default=0.05)
    parser.add_argument("--min-volume-for-iv", type=float, default=100)
    parser.add_argument("--min-volume", type=float, default=500)
    parser.add_argument("--min-notional", type=float, default=100_000)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--email-enabled", action="store_true")
    parser.add_argument("--email-sender", default=os.environ.get("EMAIL_SENDER", "fzhouxu615@gmail.com"))
    parser.add_argument("--email-password-env", default=os.environ.get("EMAIL_PASSWORD_ENV", "GMAIL_APP_PASSWORD"))
    parser.add_argument("--email-recipients", default=os.environ.get("EMAIL_RECIPIENTS", "wenyi.hann@gmail.com"))
    parser.add_argument("--email-subject-prefix", default=os.environ.get("EMAIL_SUBJECT_PREFIX", "[LEAPS Flow]"))
    parser.add_argument("--email-cooldown-sec", type=int, default=1800)
    parser.add_argument("--email-top", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    notifier = _email_notifier(args)
    while True:
        run_once(args, notifier)
        if not args.monitor:
            break
        time.sleep(args.interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
