"""Ad-hoc GEX snapshot helpers.

This module is intentionally separate from the long-running monitor workers:
it connects to IB, fetches one option-chain snapshot for a user-supplied stock
symbol, computes GEX with the existing calculator, then releases subscriptions.
"""
from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from ib_insync import IB, Option, Stock

from .config import AppConfig
from .gex_calc import GEXResult, calculate_gex, pick_expiry
from .ib_client import GENERIC_TICKS, select_option_chain
from .time_utils import et_now, trading_date_str


def ensure_event_loop() -> None:
    """Ensure ib_insync has an asyncio loop in Dash/Flask worker threads."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@dataclass
class AdhocQuality:
    requested_contracts: int
    qualified_contracts: int
    greeks_count: int
    oi_count: int
    bidask_count: int
    quality: str

    @property
    def greeks_coverage(self) -> float:
        return self.greeks_count / self.qualified_contracts if self.qualified_contracts else 0.0

    @property
    def oi_coverage(self) -> float:
        return self.oi_count / self.qualified_contracts if self.qualified_contracts else 0.0

    @property
    def bidask_coverage(self) -> float:
        return self.bidask_count / self.qualified_contracts if self.qualified_contracts else 0.0


@dataclass
class AdhocGEXSnapshot:
    symbol: str
    spot: float
    expiry: str
    is_true_0dte: bool
    ts: datetime
    result: GEXResult | None
    quality: AdhocQuality
    error: str | None = None


def _is_number(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def select_nearby_strikes(
    chain_strikes,
    spot: float,
    strikes_each_side: int = 10,
) -> list[float]:
    """Select real chain strikes around spot without assuming integer spacing."""
    if not chain_strikes or not spot or spot <= 0:
        return []
    strikes = sorted(float(s) for s in chain_strikes if s and float(s) > 0)
    below = [s for s in strikes if s <= spot][-strikes_each_side:]
    above = [s for s in strikes if s > spot][:strikes_each_side]
    return sorted(set(below + above))


def classify_quality(greeks_count: int, qualified_contracts: int) -> str:
    if qualified_contracts <= 0:
        return "bad"
    coverage = greeks_count / qualified_contracts
    if coverage >= 0.8:
        return "ok"
    if coverage >= 0.5:
        return "partial"
    return "bad"


def _quality_from_tickers(tickers, requested_contracts: int, qualified_contracts: int) -> AdhocQuality:
    greeks_count = sum(1 for t in tickers if t.modelGreeks and t.modelGreeks.gamma is not None)
    oi_count = 0
    bidask_count = 0
    for t in tickers:
        right = getattr(t.contract, "right", None)
        oi = t.callOpenInterest if right == "C" else t.putOpenInterest
        if _is_number(oi):
            oi_count += 1
        if _is_number(t.bid) or _is_number(t.ask):
            bidask_count += 1
    return AdhocQuality(
        requested_contracts=requested_contracts,
        qualified_contracts=qualified_contracts,
        greeks_count=greeks_count,
        oi_count=oi_count,
        bidask_count=bidask_count,
        quality=classify_quality(greeks_count, qualified_contracts),
    )


def fetch_adhoc_gex(
    symbol: str,
    *,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 90,
    strikes_each_side: int = 10,
    wait_sec: float = 8.0,
    expiry: str | None = None,
) -> AdhocGEXSnapshot:
    """Fetch one ad-hoc stock-option GEX snapshot from IB."""
    ensure_event_loop()
    symbol = symbol.strip().upper()
    if not symbol:
        return AdhocGEXSnapshot(
            symbol="",
            spot=0.0,
            expiry="",
            is_true_0dte=False,
            ts=et_now(),
            result=None,
            quality=AdhocQuality(0, 0, 0, 0, 0, "bad"),
            error="Symbol is required",
        )

    ib = IB()
    underlying = None
    tickers = []
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
        underlying = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(underlying)
        if not underlying.conId:
            raise RuntimeError(f"Unable to qualify stock contract for {symbol}")

        chains = ib.reqSecDefOptParams(
            underlying.symbol, "", underlying.secType, underlying.conId
        )
        chain = select_option_chain(chains, symbol)
        if chain is None:
            available = sorted({c.tradingClass for c in chains})
            raise RuntimeError(f"No option chain for {symbol}; available tradingClass={available}")

        chosen_expiry, is_true_0dte = (
            (expiry, expiry == trading_date_str()) if expiry else pick_expiry(chain, trading_date_str())
        )
        if not chosen_expiry:
            raise RuntimeError(f"No future expiry available for {symbol}")
        if chosen_expiry not in chain.expirations:
            raise RuntimeError(f"Expiry {chosen_expiry} is not available for {symbol}")

        underlying_ticker = ib.reqMktData(underlying, "", False, False)
        ib.sleep(2)
        spot = underlying_ticker.marketPrice()
        if not _is_number(spot) or spot <= 0:
            raise RuntimeError(f"No valid spot price for {symbol}")

        strikes = select_nearby_strikes(chain.strikes, float(spot), strikes_each_side)
        if not strikes:
            raise RuntimeError(f"No strikes selected for {symbol}")

        raw_contracts = [
            Option(symbol, chosen_expiry, strike, right, "SMART", tradingClass=symbol)
            for strike in strikes
            for right in ("C", "P")
        ]
        contracts = list(ib.qualifyContracts(*raw_contracts))
        tickers = [ib.reqMktData(c, GENERIC_TICKS, False, False) for c in contracts]
        ib.sleep(wait_sec)

        quality = _quality_from_tickers(tickers, len(raw_contracts), len(contracts))
        result = calculate_gex(tickers, float(spot), oi_ready_threshold=0.0)
        return AdhocGEXSnapshot(
            symbol=symbol,
            spot=float(spot),
            expiry=chosen_expiry,
            is_true_0dte=is_true_0dte,
            ts=et_now(),
            result=result,
            quality=quality,
        )
    except Exception as exc:
        return AdhocGEXSnapshot(
            symbol=symbol,
            spot=0.0,
            expiry=expiry or "",
            is_true_0dte=False,
            ts=et_now(),
            result=None,
            quality=AdhocQuality(0, 0, 0, 0, 0, "bad"),
            error=str(exc),
        )
    finally:
        for t in tickers:
            try:
                ib.cancelMktData(t.contract)
            except Exception:
                pass
        if underlying is not None:
            try:
                ib.cancelMktData(underlying)
            except Exception:
                pass
        if ib.isConnected():
            ib.disconnect()


def save_adhoc_snapshot(snapshot: AdhocGEXSnapshot, data_dir: str | Path) -> tuple[Path, Path | None]:
    """Save an ad-hoc snapshot to separate files so it cannot pollute monitor history."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    stamp = snapshot.ts.strftime("%Y%m%d_%H%M%S")
    base = f"{snapshot.symbol}_{stamp}"

    summary_path = data_path / f"adhoc_gex_{base}.parquet"
    summary = {
        "ts": snapshot.ts,
        "symbol": snapshot.symbol,
        "spot": snapshot.spot,
        "expiry": snapshot.expiry,
        "is_true_0dte": snapshot.is_true_0dte,
        "quality": snapshot.quality.quality,
        "requested_contracts": snapshot.quality.requested_contracts,
        "qualified_contracts": snapshot.quality.qualified_contracts,
        "greeks_count": snapshot.quality.greeks_count,
        "oi_count": snapshot.quality.oi_count,
        "bidask_count": snapshot.quality.bidask_count,
        "total_gex": snapshot.result.total_gex if snapshot.result else None,
        "call_gex": snapshot.result.call_gex if snapshot.result else None,
        "put_gex": snapshot.result.put_gex if snapshot.result else None,
        "gamma_flip": snapshot.result.gamma_flip if snapshot.result else None,
        "call_wall": snapshot.result.call_wall if snapshot.result else None,
        "put_wall": snapshot.result.put_wall if snapshot.result else None,
        "max_pain": snapshot.result.max_pain if snapshot.result else None,
        "atm_iv_pct": snapshot.result.atm_iv_pct if snapshot.result else None,
        "error": snapshot.error,
    }
    pd.DataFrame([summary]).to_parquet(summary_path, index=False)

    strikes_path = None
    if snapshot.result is not None and not snapshot.result.df.empty:
        strikes_path = data_path / f"adhoc_strikes_{base}.parquet"
        df = snapshot.result.df.copy()
        df.insert(0, "ts", snapshot.ts)
        df.insert(1, "symbol", snapshot.symbol)
        df.insert(2, "expiry", snapshot.expiry)
        df.to_parquet(strikes_path, index=False)
    return summary_path, strikes_path


def _fmt_big(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value / 1e6:,.1f}M"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ad-hoc GEX snapshot")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=90)
    parser.add_argument("--strikes-each-side", type=int, default=10)
    parser.add_argument("--wait-sec", type=float, default=8.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config) if args.config else AppConfig.default()
    snapshot = fetch_adhoc_gex(
        args.symbol,
        host=args.host or config.ib.host,
        port=args.port or config.ib.port,
        client_id=args.client_id,
        strikes_each_side=args.strikes_each_side,
        wait_sec=args.wait_sec,
    )
    if snapshot.error:
        print(f"ERROR {snapshot.symbol}: {snapshot.error}")
        return
    q = snapshot.quality
    print(f"{snapshot.symbol} spot={snapshot.spot:.2f} expiry={snapshot.expiry} quality={q.quality}")
    print(
        f"coverage greeks={q.greeks_count}/{q.qualified_contracts} "
        f"oi={q.oi_count}/{q.qualified_contracts} bidask={q.bidask_count}/{q.qualified_contracts}"
    )
    if snapshot.result:
        r = snapshot.result
        print(
            f"total={_fmt_big(r.total_gex)} call={_fmt_big(r.call_gex)} "
            f"put={_fmt_big(r.put_gex)} flip={r.gamma_flip:.2f}"
        )
    if args.save:
        paths = save_adhoc_snapshot(snapshot, config.storage.data_dir)
        print("saved:", ", ".join(str(p) for p in paths if p is not None))


if __name__ == "__main__":
    main()
