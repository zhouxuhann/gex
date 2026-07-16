"""Stock IV/HV decoupling monitor.

Scans a small watchlist, estimates ATM option IV from IB model greeks, computes
daily HV and intraday realized volatility from historical bars, and emails when
price rises while IV/HV fails to confirm.
"""
from __future__ import annotations

import argparse
import logging
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock, util

from .config import AppConfig
from .email_notifier import EmailConfig, EmailNotifier
from .time_utils import ET, et_now, is_market_open, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CONFIG = ROOT_DIR / "config" / "config.yaml"
LOG_DIR = ROOT_DIR / "logs"

log = logging.getLogger("iv_hv_monitor")


@dataclass
class IVHVRuntimeConfig:
    symbols: list[str]
    client_id: int = 86
    interval_sec: int = 900
    target_dte: int = 30
    min_dte: int = 7
    lookback_points: int = 4
    min_price_up_pct: float = 0.5
    min_corr: float = 0.0
    iv_down_alert_pct: float = 2.0
    hv_down_alert_pct: float = 5.0
    cooldown_sec: int = 3600


@dataclass
class IVHVSnapshot:
    ts: pd.Timestamp
    symbol: str
    price: float
    atm_iv: float | None
    hv20: float | None
    intraday_hv: float | None
    expiry: str | None = None
    strike: float | None = None


@dataclass
class DecouplingAlert:
    symbol: str
    ts: pd.Timestamp
    price_change_pct: float
    iv_change_pct: float | None
    hv_change_pct: float | None
    corr_price_iv: float | None
    corr_price_hv: float | None
    reasons: list[str]
    latest: IVHVSnapshot


def annualized_hv_from_closes(
    closes: pd.Series, periods_per_year: float, window: int | None = None
) -> float | None:
    """Annualized log-return volatility from close prices."""
    s = pd.Series(closes).dropna().astype(float)
    if len(s) < 3:
        return None
    rets = np.log(s / s.shift(1)).dropna()
    if window is not None:
        rets = rets.tail(window)
    if len(rets) < 2 or rets.std() == 0 or pd.isna(rets.std()):
        return None
    return float(rets.std() * np.sqrt(periods_per_year))


def _pct_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None:
        return None
    if not np.isfinite(first) or not np.isfinite(last) or first == 0:
        return None
    return float((last / first - 1.0) * 100.0)


def _corr(xs: list[float | None], ys: list[float | None]) -> float | None:
    df = pd.DataFrame({"x": xs, "y": ys}).dropna()
    if len(df) < 3 or df["x"].std() == 0 or df["y"].std() == 0:
        return None
    value = df["x"].corr(df["y"])
    return float(value) if pd.notna(value) else None


def evaluate_decoupling(
    history: list[IVHVSnapshot], cfg: IVHVRuntimeConfig
) -> DecouplingAlert | None:
    """Return alert when price rises but IV/HV fails to confirm."""
    if len(history) < cfg.lookback_points:
        return None

    recent = history[-cfg.lookback_points:]
    first, last = recent[0], recent[-1]
    price_change = _pct_change(first.price, last.price)
    if price_change is None or price_change < cfg.min_price_up_pct:
        return None

    iv_change = _pct_change(first.atm_iv, last.atm_iv)
    hv_change = _pct_change(first.intraday_hv, last.intraday_hv)
    corr_iv = _corr([r.price for r in recent], [r.atm_iv for r in recent])
    corr_hv = _corr([r.price for r in recent], [r.intraday_hv for r in recent])

    reasons = []
    if iv_change is not None and iv_change <= -cfg.iv_down_alert_pct:
        reasons.append(f"IV down {iv_change:.1f}% while price up {price_change:.1f}%")
    if hv_change is not None and hv_change <= -cfg.hv_down_alert_pct:
        reasons.append(f"intraday HV down {hv_change:.1f}% while price up {price_change:.1f}%")
    if corr_iv is not None and corr_iv < cfg.min_corr:
        reasons.append(f"price/IV corr {corr_iv:.2f} < {cfg.min_corr:.2f}")
    if corr_hv is not None and corr_hv < cfg.min_corr:
        reasons.append(f"price/HV corr {corr_hv:.2f} < {cfg.min_corr:.2f}")

    if not reasons:
        return None

    return DecouplingAlert(
        symbol=last.symbol,
        ts=last.ts,
        price_change_pct=price_change,
        iv_change_pct=iv_change,
        hv_change_pct=hv_change,
        corr_price_iv=corr_iv,
        corr_price_hv=corr_hv,
        reasons=reasons,
        latest=last,
    )


class IVHVMonitor:
    def __init__(
        self,
        ib: IB,
        cfg: IVHVRuntimeConfig,
        data_dir: Path,
        email_notifier: EmailNotifier | None = None,
    ):
        self.ib = ib
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.email_notifier = email_notifier
        self.history: dict[str, list[IVHVSnapshot]] = {s: [] for s in cfg.symbols}
        self._last_alert: dict[str, float] = {}
        self._running = True

    def stop(self) -> None:
        self._running = False

    def start(self) -> None:
        log.info(
            "Started IV/HV monitor: symbols=%s interval=%ss target_dte=%s",
            self.cfg.symbols, self.cfg.interval_sec, self.cfg.target_dte,
        )
        while self._running:
            now = et_now()
            if is_market_open(now):
                self.scan_once()
            else:
                log.info("Market closed; IV/HV monitor sleeping")
            self.ib.sleep(self.cfg.interval_sec)

    def scan_once(self) -> list[IVHVSnapshot]:
        snapshots = []
        for symbol in self.cfg.symbols:
            try:
                snap = self.collect_symbol(symbol)
            except Exception:
                log.exception("[%s] IV/HV collection failed", symbol)
                continue
            if snap is None:
                continue
            snapshots.append(snap)
            self.history.setdefault(symbol, []).append(snap)
            self.history[symbol] = self.history[symbol][-max(self.cfg.lookback_points * 4, 20):]
            self._persist_snapshot(snap)
            alert = evaluate_decoupling(self.history[symbol], self.cfg)
            if alert is not None:
                self._handle_alert(alert)
        return snapshots

    def collect_symbol(self, symbol: str) -> IVHVSnapshot | None:
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        price = self._snapshot_price(contract)
        if price is None:
            log.warning("[%s] no stock price", symbol)
            return None

        hv20 = self._daily_hv(contract)
        intraday_hv = self._intraday_hv(contract)
        atm_iv, expiry, strike = self._atm_option_iv(contract, price)

        snap = IVHVSnapshot(
            ts=pd.Timestamp(et_now()),
            symbol=symbol,
            price=price,
            atm_iv=atm_iv,
            hv20=hv20,
            intraday_hv=intraday_hv,
            expiry=expiry,
            strike=strike,
        )
        log.info(
            "[%s] px=%.2f atm_iv=%s hv20=%s intraday_hv=%s expiry=%s strike=%s",
            symbol, price, _fmt_pct(atm_iv), _fmt_pct(hv20), _fmt_pct(intraday_hv),
            expiry, strike,
        )
        return snap

    def _snapshot_price(self, contract: Stock) -> float | None:
        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(1.5)
        price = ticker.marketPrice()
        if price is None or not np.isfinite(price) or price <= 0:
            price = ticker.last or ticker.close
        self.ib.cancelMktData(contract)
        return float(price) if price and np.isfinite(price) and price > 0 else None

    def _daily_hv(self, contract: Stock) -> float | None:
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="45 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        df = util.df(bars)
        if df is None or df.empty or "close" not in df:
            return None
        return annualized_hv_from_closes(df["close"], periods_per_year=252, window=20)

    def _intraday_hv(self, contract: Stock) -> float | None:
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        df = util.df(bars)
        if df is None or df.empty or "close" not in df:
            return None
        return annualized_hv_from_closes(df["close"], periods_per_year=252 * 78)

    def _atm_option_iv(self, underlying: Stock, spot: float) -> tuple[float | None, str | None, float | None]:
        chains = self.ib.reqSecDefOptParams(
            underlying.symbol, "", underlying.secType, underlying.conId
        )
        chain = _pick_chain(chains, underlying.symbol)
        if chain is None:
            log.warning("[%s] no option chain", underlying.symbol)
            return None, None, None

        expiry = _pick_expiry(chain.expirations, self.cfg.min_dte, self.cfg.target_dte)
        if expiry is None:
            log.warning("[%s] no expiry >= %s DTE", underlying.symbol, self.cfg.min_dte)
            return None, None, None

        strikes = [float(s) for s in chain.strikes if s and s > 0]
        if not strikes:
            return None, expiry, None
        strike = min(strikes, key=lambda s: abs(s - spot))

        ivs = []
        option_contracts = [
            Option(underlying.symbol, expiry, strike, right, "SMART", tradingClass=chain.tradingClass)
            for right in ("C", "P")
        ]
        self.ib.qualifyContracts(*option_contracts)
        tickers = [self.ib.reqMktData(c, "106", False, False) for c in option_contracts]
        self.ib.sleep(2.0)
        for ticker in tickers:
            greeks = ticker.modelGreeks
            if greeks and greeks.impliedVol is not None and np.isfinite(greeks.impliedVol):
                ivs.append(float(greeks.impliedVol))
        for c in option_contracts:
            self.ib.cancelMktData(c)
        return (float(np.mean(ivs)) if ivs else None), expiry, strike

    def _persist_snapshot(self, snap: IVHVSnapshot) -> None:
        path = self.data_dir / f"iv_hv_{snap.symbol}_{trading_date_str()}.parquet"
        _append_parquet(path, asdict(snap), unique_cols=["ts"])

    def _handle_alert(self, alert: DecouplingAlert) -> None:
        now = time.time()
        last = self._last_alert.get(alert.symbol, 0)
        if now - last < self.cfg.cooldown_sec:
            log.info("[%s] IV/HV alert suppressed by cooldown", alert.symbol)
            return

        self._last_alert[alert.symbol] = now
        alert_path = self.data_dir / f"iv_hv_alerts_{trading_date_str()}.parquet"
        row = {
            "ts": alert.ts.isoformat(),
            "symbol": alert.symbol,
            "price_change_pct": alert.price_change_pct,
            "iv_change_pct": alert.iv_change_pct,
            "hv_change_pct": alert.hv_change_pct,
            "corr_price_iv": alert.corr_price_iv,
            "corr_price_hv": alert.corr_price_hv,
            "reasons": "; ".join(alert.reasons),
            "price": alert.latest.price,
            "atm_iv": alert.latest.atm_iv,
            "hv20": alert.latest.hv20,
            "intraday_hv": alert.latest.intraday_hv,
        }
        _append_parquet(alert_path, row, unique_cols=["ts", "symbol"])

        subject = f"IV/HV decoupling {alert.symbol} px +{alert.price_change_pct:.1f}%"
        body = _format_alert_body(alert)
        log.warning("[%s] %s: %s", alert.symbol, subject, "; ".join(alert.reasons))
        if self.email_notifier is not None:
            self.email_notifier.send_alert(subject, body)


def _pick_chain(chains, symbol: str):
    if not chains:
        return None
    return (
        next((c for c in chains if c.exchange == "SMART" and c.tradingClass == symbol), None)
        or next((c for c in chains if c.exchange == "SMART"), None)
        or chains[0]
    )


def _pick_expiry(expirations, min_dte: int, target_dte: int) -> str | None:
    today = et_now().date()
    candidates = []
    for raw in expirations:
        try:
            exp_date = datetime.strptime(str(raw), "%Y%m%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte >= min_dte:
            candidates.append((abs(dte - target_dte), dte, str(raw)))
    if not candidates:
        return None
    return sorted(candidates)[0][2]


def _append_parquet(path: Path, row: dict, unique_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        try:
            df = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df = df.drop_duplicates(subset=unique_cols, keep="last")
    df.to_parquet(path, index=False)


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _fmt_change(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1f}%"


def _fmt_corr(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}"


def _format_alert_body(alert: DecouplingAlert) -> str:
    latest = alert.latest
    return (
        "IV/HV 脱节提醒\n\n"
        f"标的:        {alert.symbol}\n"
        f"时间:        {alert.ts.strftime('%Y-%m-%d %H:%M:%S ET')}\n"
        f"价格:        {latest.price:.2f}  ({alert.price_change_pct:+.2f}% lookback)\n"
        f"ATM IV:      {_fmt_pct(latest.atm_iv)}  change={_fmt_change(alert.iv_change_pct)}\n"
        f"HV20:        {_fmt_pct(latest.hv20)}\n"
        f"Intra HV:    {_fmt_pct(latest.intraday_hv)}  change={_fmt_change(alert.hv_change_pct)}\n"
        f"Expiry/ATM:  {latest.expiry or 'N/A'} / {latest.strike or 'N/A'}\n"
        f"Corr px-IV:  {_fmt_corr(alert.corr_price_iv)}\n"
        f"Corr px-HV:  {_fmt_corr(alert.corr_price_hv)}\n"
        "\n"
        "触发原因:\n"
        + "\n".join(f"- {r}" for r in alert.reasons)
        + "\n\n"
        "解释: 价格上涨但 IV/HV 未确认，可能代表上涨缺少波动率跟随、期权市场不买账，"
        "或行情进入低波 grind。仅作观察提醒，不是交易指令。"
    )


def _runtime_config(app_cfg: AppConfig) -> IVHVRuntimeConfig | None:
    raw = app_cfg.signals.iv_hv
    if not raw.enabled:
        return None
    return IVHVRuntimeConfig(
        symbols=[s.upper() for s in raw.symbols],
        client_id=raw.client_id,
        interval_sec=raw.interval_sec,
        target_dte=raw.target_dte,
        min_dte=raw.min_dte,
        lookback_points=raw.lookback_points,
        min_price_up_pct=raw.min_price_up_pct,
        min_corr=raw.min_corr,
        iv_down_alert_pct=raw.iv_down_alert_pct,
        hv_down_alert_pct=raw.hv_down_alert_pct,
        cooldown_sec=raw.cooldown_sec,
    )


def _email_notifier(app_cfg: AppConfig) -> EmailNotifier | None:
    email = app_cfg.signals.iv_hv.email
    if not email.enabled:
        return None
    return EmailNotifier(EmailConfig(
        enabled=email.enabled,
        sender=email.sender,
        password_env=email.password_env,
        recipients=email.recipients,
        smtp_host=email.smtp_host,
        smtp_port=email.smtp_port,
        only_strong=email.only_strong,
        cooldown_sec=email.cooldown_sec,
        subject_prefix=email.subject_prefix,
    ))


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"iv_hv_{trading_date_str()}.log"
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stock IV/HV decoupling monitor")
    p.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--client-id", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--once", action="store_true", help="scan once and exit")
    p.add_argument("--ib-readonly", action="store_true", help="Connect IB API in readonly mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    app_cfg = AppConfig.from_yaml(args.config)
    runtime = _runtime_config(app_cfg)
    if runtime is None:
        log.info("IV/HV monitor disabled in config")
        return
    if args.client_id is not None:
        runtime.client_id = args.client_id

    host = args.host or app_cfg.ib.host
    port = args.port or app_cfg.ib.port
    notifier = _email_notifier(app_cfg)
    ib = IB()
    monitor = None

    def _stop(*_):
        if monitor is not None:
            monitor.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        log.info(
            "Connecting IB %s:%s clientId=%s readonly=%s",
            host, port, runtime.client_id, args.ib_readonly,
        )
        ib.connect(host, port, clientId=runtime.client_id, timeout=20, readonly=args.ib_readonly)
        monitor = IVHVMonitor(ib, runtime, args.data_dir, notifier)
        if args.once:
            monitor.scan_once()
        else:
            monitor.start()
    finally:
        if ib.isConnected():
            ib.disconnect()
        log.info("Stopped")


if __name__ == "__main__":
    main()
