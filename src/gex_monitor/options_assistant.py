"""Options buying assistant for existing stock holdings.

Produces trade plans only. It does not submit orders.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock, util

from .config import AppConfig
from .email_notifier import EmailConfig, EmailNotifier
from .iv_hv_monitor import annualized_hv_from_closes, _pick_chain, _pick_expiry
from .time_utils import et_now, is_market_open, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "config" / "config.yaml"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

log = logging.getLogger("options_assistant")


@dataclass
class AssistantConfig:
    symbols: list[str]
    target_dte: int = 45
    min_dte: int = 21
    cheap_iv_hv: float = 1.05
    fair_iv_hv: float = 1.35
    min_trend_score: int = 2
    max_debit_pct_stock: float = 3.0
    long_call_delta: float = 0.60
    spread_short_delta: float = 0.30
    protective_put_delta: float = 0.30
    earnings_blackout_days_before: int = 7
    earnings_blackout_days_after: int = 2
    earnings_dates: dict[str, str | None] | None = None
    signal_actions: tuple[str, ...] = ("BUY_CALL", "BUY_CALL_DEBIT_SPREAD")


@dataclass
class TrendSnapshot:
    symbol: str
    price: float
    ret_5d_pct: float | None
    ret_20d_pct: float | None
    sma20: float | None
    sma50: float | None
    hv20: float | None
    trend_score: int


@dataclass
class OptionQuote:
    symbol: str
    expiry: str
    strike: float
    right: str
    delta: float | None
    iv: float | None
    bid: float | None
    ask: float | None
    mid: float | None


@dataclass
class OptionPlan:
    symbol: str
    action: str
    reason: str
    trend: TrendSnapshot
    iv_hv_ratio: float | None
    long_leg: OptionQuote | None = None
    short_leg: OptionQuote | None = None
    max_debit: float | None = None
    notes: list[str] | None = None
    earnings_date: str | None = None
    earnings_days_to: int | None = None
    earnings_blackout: bool = False


@dataclass
class EarningsInfo:
    date: str | None
    days_to: int | None
    in_blackout: bool


def trend_score_from_closes(symbol: str, closes: pd.Series) -> TrendSnapshot | None:
    s = pd.Series(closes).dropna().astype(float)
    if len(s) < 25:
        return None
    price = float(s.iloc[-1])
    ret_5d = (price / s.iloc[-6] - 1.0) * 100 if len(s) >= 6 else None
    ret_20d = (price / s.iloc[-21] - 1.0) * 100 if len(s) >= 21 else None
    sma20 = float(s.tail(20).mean()) if len(s) >= 20 else None
    sma50 = float(s.tail(50).mean()) if len(s) >= 50 else None
    hv20 = annualized_hv_from_closes(s, periods_per_year=252, window=20)

    score = 0
    if ret_5d is not None and ret_5d > 0:
        score += 1
    if ret_20d is not None and ret_20d > 0:
        score += 1
    if sma20 is not None and price > sma20:
        score += 1
    if sma20 is not None and sma50 is not None and sma20 > sma50:
        score += 1

    return TrendSnapshot(
        symbol=symbol,
        price=price,
        ret_5d_pct=ret_5d,
        ret_20d_pct=ret_20d,
        sma20=sma20,
        sma50=sma50,
        hv20=hv20,
        trend_score=score,
    )


def choose_plan(
    trend: TrendSnapshot,
    atm_iv: float | None,
    calls: list[OptionQuote],
    puts: list[OptionQuote],
    cfg: AssistantConfig,
) -> OptionPlan:
    """Pure decision logic for turning trend + IV/HV + chain quotes into a plan."""
    hv = trend.hv20
    iv_hv_ratio = (atm_iv / hv) if atm_iv is not None and hv and hv > 0 else None
    earnings = earnings_info(trend.symbol, cfg)
    notes = [
        "只生成候选计划，不自动下单",
        "财报黑窗内不发买 call/call spread 邮件",
    ]

    if trend.trend_score < cfg.min_trend_score:
        put = _pick_by_delta(puts, cfg.protective_put_delta)
        return OptionPlan(
            symbol=trend.symbol,
            action="WATCH_OR_PROTECTIVE_PUT",
            reason="趋势分数不足，优先保护持仓而不是追买 call",
            trend=trend,
            iv_hv_ratio=iv_hv_ratio,
            long_leg=put,
            max_debit=_option_debit(put),
            notes=notes,
            earnings_date=earnings.date,
            earnings_days_to=earnings.days_to,
            earnings_blackout=earnings.in_blackout,
        )

    long_call = _pick_by_delta(calls, cfg.long_call_delta)
    short_call = _pick_short_call(calls, long_call, cfg.spread_short_delta)
    long_debit = _option_debit(long_call)
    spread_debit = _spread_debit(long_call, short_call)

    if earnings.in_blackout:
        return OptionPlan(
            symbol=trend.symbol,
            action="WAIT_EARNINGS_BLACKOUT",
            reason=(
                f"距离财报 {earnings.days_to} 天，处于财报黑窗；"
                "避免在 IV crush/event risk 前追买 call"
            ),
            trend=trend,
            iv_hv_ratio=iv_hv_ratio,
            long_leg=long_call,
            short_leg=short_call,
            max_debit=spread_debit or long_debit,
            notes=notes,
            earnings_date=earnings.date,
            earnings_days_to=earnings.days_to,
            earnings_blackout=True,
        )

    if iv_hv_ratio is None:
        return OptionPlan(
            symbol=trend.symbol,
            action="WATCH",
            reason="缺少 IV 或 HV，暂不判断期权贵便宜",
            trend=trend,
            iv_hv_ratio=None,
            long_leg=long_call,
            notes=notes,
            earnings_date=earnings.date,
            earnings_days_to=earnings.days_to,
            earnings_blackout=False,
        )

    if iv_hv_ratio <= cfg.cheap_iv_hv:
        return OptionPlan(
            symbol=trend.symbol,
            action="BUY_CALL",
            reason="趋势向上且 IV/HV 不贵，适合用 long call 买 convexity",
            trend=trend,
            iv_hv_ratio=iv_hv_ratio,
            long_leg=long_call,
            max_debit=long_debit,
            notes=notes,
            earnings_date=earnings.date,
            earnings_days_to=earnings.days_to,
            earnings_blackout=False,
        )

    if iv_hv_ratio <= cfg.fair_iv_hv:
        return OptionPlan(
            symbol=trend.symbol,
            action="BUY_CALL_DEBIT_SPREAD",
            reason="趋势向上但 IV/HV 中性偏贵，用 call spread 降低 IV 回落伤害",
            trend=trend,
            iv_hv_ratio=iv_hv_ratio,
            long_leg=long_call,
            short_leg=short_call,
            max_debit=spread_debit,
            notes=notes,
            earnings_date=earnings.date,
            earnings_days_to=earnings.days_to,
            earnings_blackout=False,
        )

    return OptionPlan(
        symbol=trend.symbol,
        action="WAIT_PREMIUM_EXPENSIVE",
        reason="趋势向上但 IV/HV 偏贵，买裸期权性价比差，等待或只看小仓位 spread",
        trend=trend,
        iv_hv_ratio=iv_hv_ratio,
        long_leg=long_call,
        short_leg=short_call,
        max_debit=spread_debit,
        notes=notes,
        earnings_date=earnings.date,
        earnings_days_to=earnings.days_to,
        earnings_blackout=False,
    )


def earnings_info(
    symbol: str, cfg: AssistantConfig, as_of: date | None = None
) -> EarningsInfo:
    """Return configured earnings proximity and blackout status."""
    raw = (cfg.earnings_dates or {}).get(symbol.upper())
    if not raw:
        return EarningsInfo(date=None, days_to=None, in_blackout=False)
    try:
        earnings_date = pd.Timestamp(raw).date()
    except Exception:
        return EarningsInfo(date=str(raw), days_to=None, in_blackout=False)
    today = as_of or et_now().date()
    days_to = (earnings_date - today).days
    in_blackout = (
        -cfg.earnings_blackout_days_after
        <= days_to
        <= cfg.earnings_blackout_days_before
    )
    return EarningsInfo(
        date=earnings_date.isoformat(),
        days_to=int(days_to),
        in_blackout=bool(in_blackout),
    )


def _pick_by_delta(quotes: list[OptionQuote], target: float) -> OptionQuote | None:
    valid = [q for q in quotes if q.delta is not None and q.mid is not None and q.mid > 0]
    if not valid:
        return None
    return min(valid, key=lambda q: abs(abs(q.delta or 0) - target))


def _pick_short_call(
    calls: list[OptionQuote], long_call: OptionQuote | None, target_delta: float
) -> OptionQuote | None:
    if long_call is None:
        return None
    valid = [
        q for q in calls
        if q.delta is not None and q.mid is not None and q.mid > 0
        and q.strike > long_call.strike
    ]
    if not valid:
        return None
    return min(valid, key=lambda q: abs(abs(q.delta or 0) - target_delta))


def _option_debit(quote: OptionQuote | None) -> float | None:
    if quote is None or quote.ask is None or quote.ask <= 0:
        return None
    return float(quote.ask * 100)


def _spread_debit(long_leg: OptionQuote | None, short_leg: OptionQuote | None) -> float | None:
    if long_leg is None or short_leg is None:
        return None
    if long_leg.ask is None or short_leg.bid is None:
        return None
    debit = long_leg.ask - short_leg.bid
    return float(debit * 100) if debit > 0 else None


class OptionsAssistant:
    def __init__(self, ib: IB, cfg: AssistantConfig):
        self.ib = ib
        self.cfg = cfg

    def plan_symbol(self, symbol: str) -> OptionPlan | None:
        underlying = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(underlying)
        trend = self._trend(symbol, underlying)
        if trend is None:
            log.warning("[%s] insufficient historical bars", symbol)
            return None
        calls, puts, atm_iv = self._option_quotes(symbol, underlying, trend.price)
        return choose_plan(trend, atm_iv, calls, puts, self.cfg)

    def _trend(self, symbol: str, contract: Stock) -> TrendSnapshot | None:
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="75 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        df = util.df(bars)
        if df is None or df.empty or "close" not in df:
            return None
        return trend_score_from_closes(symbol, df["close"])

    def _option_quotes(
        self, symbol: str, underlying: Stock, spot: float
    ) -> tuple[list[OptionQuote], list[OptionQuote], float | None]:
        chains = self.ib.reqSecDefOptParams(
            underlying.symbol, "", underlying.secType, underlying.conId
        )
        chain = _pick_chain(chains, symbol)
        if chain is None:
            return [], [], None
        expiry = _pick_expiry(chain.expirations, self.cfg.min_dte, self.cfg.target_dte)
        if expiry is None:
            return [], [], None

        strikes = sorted(float(s) for s in chain.strikes if s and s > 0)
        strikes = [s for s in strikes if spot * 0.85 <= s <= spot * 1.25]
        if len(strikes) > 28:
            strikes = sorted(strikes, key=lambda s: abs(s - spot))[:28]
            strikes = sorted(strikes)

        contracts = [
            Option(symbol, expiry, strike, right, "SMART", tradingClass=chain.tradingClass)
            for strike in strikes
            for right in ("C", "P")
        ]
        qualified = self.ib.qualifyContracts(*contracts)
        tickers = [self.ib.reqMktData(c, "106", False, False) for c in qualified]
        self.ib.sleep(2.5)

        calls, puts = [], []
        atm_ivs = []
        for contract, ticker in zip(qualified, tickers):
            quote = _quote_from_ticker(symbol, contract, ticker)
            if quote is None:
                continue
            if abs(contract.strike - spot) / spot < 0.015 and quote.iv is not None:
                atm_ivs.append(quote.iv)
            if quote.right == "C":
                calls.append(quote)
            else:
                puts.append(quote)

        for contract in qualified:
            self.ib.cancelMktData(contract)
        atm_iv = float(np.mean(atm_ivs)) if atm_ivs else None
        return calls, puts, atm_iv


class OptionsAssistantMonitor:
    """Periodic runner that persists plans and emails actionable signals."""

    def __init__(
        self,
        assistant: OptionsAssistant,
        data_dir: Path,
        email_notifier: EmailNotifier | None = None,
        interval_sec: int = 1800,
        cooldown_sec: int = 14400,
    ):
        self.assistant = assistant
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.email_notifier = email_notifier
        self.interval_sec = interval_sec
        self.cooldown_sec = cooldown_sec
        self._last_sent: dict[tuple[str, str], float] = {}
        self._running = True

    def stop(self) -> None:
        self._running = False

    def start(self) -> None:
        log.info(
            "Started options assistant monitor: symbols=%s interval=%ss",
            self.assistant.cfg.symbols, self.interval_sec,
        )
        while self._running:
            if is_market_open(et_now()):
                self.scan_once()
            else:
                log.info("Market closed; options assistant sleeping")
            self.assistant.ib.sleep(self.interval_sec)

    def scan_once(self) -> list[OptionPlan]:
        plans = []
        for symbol in self.assistant.cfg.symbols:
            try:
                plan = self.assistant.plan_symbol(symbol)
            except Exception:
                log.exception("[%s] options assistant planning failed", symbol)
                continue
            if plan is None:
                continue
            plans.append(plan)
            self._persist_plan(plan)
            if plan.action in self.assistant.cfg.signal_actions:
                self._maybe_email(plan)
            log.info("[%s] %s iv/hv=%s", symbol, plan.action, _fmt_ratio(plan.iv_hv_ratio))
        return plans

    def _persist_plan(self, plan: OptionPlan) -> None:
        path = self.data_dir / f"options_assistant_plans_{trading_date_str()}.parquet"
        row = _plan_to_row(plan)
        df_new = pd.DataFrame([row])
        if path.exists():
            try:
                df = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
            except Exception:
                df = df_new
        else:
            df = df_new
        df.to_parquet(path, index=False)

    def _maybe_email(self, plan: OptionPlan) -> bool:
        if self.email_notifier is None:
            return False
        key = (plan.symbol, plan.action)
        now = time.time()
        last = self._last_sent.get(key, 0)
        if now - last < self.cooldown_sec:
            log.info("[%s] %s email suppressed by cooldown", plan.symbol, plan.action)
            return False
        subject = f"Options signal {plan.symbol} {plan.action}"
        body = _format_email_body(plan)
        sent = self.email_notifier.send_alert(subject, body)
        if sent:
            self._last_sent[key] = now
        return sent


def _quote_from_ticker(symbol: str, contract: Option, ticker) -> OptionQuote | None:
    greeks = ticker.modelGreeks
    delta = float(greeks.delta) if greeks and greeks.delta is not None else None
    iv = float(greeks.impliedVol) if greeks and greeks.impliedVol is not None else None
    bid = float(ticker.bid) if ticker.bid is not None and np.isfinite(ticker.bid) and ticker.bid > 0 else None
    ask = float(ticker.ask) if ticker.ask is not None and np.isfinite(ticker.ask) and ticker.ask > 0 else None
    mid = None
    if bid is not None and ask is not None and ask >= bid:
        mid = (bid + ask) / 2
    elif ticker.last is not None and np.isfinite(ticker.last) and ticker.last > 0:
        mid = float(ticker.last)
    return OptionQuote(
        symbol=symbol,
        expiry=contract.lastTradeDateOrContractMonth,
        strike=float(contract.strike),
        right=contract.right,
        delta=delta,
        iv=iv,
        bid=bid,
        ask=ask,
        mid=mid,
    )


def format_plan(plan: OptionPlan) -> str:
    t = plan.trend
    lines = [
        f"{plan.symbol}: {plan.action}",
        f"  reason: {plan.reason}",
        f"  price={t.price:.2f} trend_score={t.trend_score}/4 "
        f"ret5d={_fmt_pct_value(t.ret_5d_pct)} ret20d={_fmt_pct_value(t.ret_20d_pct)}",
        f"  hv20={_fmt_vol(t.hv20)} iv/hv={_fmt_ratio(plan.iv_hv_ratio)}",
    ]
    if plan.earnings_date is not None:
        blackout = " BLACKOUT" if plan.earnings_blackout else ""
        days_txt = "N/A" if plan.earnings_days_to is None else f"{plan.earnings_days_to:+d}d"
        lines.append(
            f"  earnings={plan.earnings_date} "
            f"({days_txt}){blackout}"
        )
    if plan.long_leg is not None:
        lines.append("  long:  " + _format_leg(plan.long_leg))
    if plan.short_leg is not None:
        lines.append("  short: " + _format_leg(plan.short_leg))
    if plan.max_debit is not None:
        lines.append(f"  estimated max debit: ${plan.max_debit:.0f} per 1-lot")
    if plan.notes:
        lines.append("  notes: " + "; ".join(plan.notes))
    return "\n".join(lines)


def _plan_to_row(plan: OptionPlan) -> dict:
    row = {
        "ts": et_now().isoformat(),
        "symbol": plan.symbol,
        "action": plan.action,
        "reason": plan.reason,
        "price": plan.trend.price,
        "trend_score": plan.trend.trend_score,
        "ret_5d_pct": plan.trend.ret_5d_pct,
        "ret_20d_pct": plan.trend.ret_20d_pct,
        "hv20": plan.trend.hv20,
        "iv_hv_ratio": plan.iv_hv_ratio,
        "max_debit": plan.max_debit,
        "earnings_date": plan.earnings_date,
        "earnings_days_to": plan.earnings_days_to,
        "earnings_blackout": plan.earnings_blackout,
    }
    if plan.long_leg is not None:
        for key, value in asdict(plan.long_leg).items():
            row[f"long_{key}"] = value
    if plan.short_leg is not None:
        for key, value in asdict(plan.short_leg).items():
            row[f"short_{key}"] = value
    return row


def _format_email_body(plan: OptionPlan) -> str:
    return (
        "买期权助手信号\n\n"
        + format_plan(plan)
        + "\n\n"
        "执行前检查:\n"
        "- 确认财报/重大事件日期\n"
        "- 确认 bid/ask 没有异常变宽\n"
        "- 按计划限制单笔权利金风险\n"
        "- 这是候选计划，不是自动下单指令"
    )


def _format_leg(q: OptionQuote) -> str:
    return (
        f"{q.expiry} {q.strike:g}{q.right} "
        f"delta={_fmt_num(q.delta)} iv={_fmt_vol(q.iv)} "
        f"bid/ask={_fmt_num(q.bid)}/{_fmt_num(q.ask)}"
    )


def _fmt_num(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _fmt_vol(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _fmt_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _fmt_pct_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Options buying assistant; no orders are submitted")
    p.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--symbols", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--client-id", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--monitor", action="store_true", help="run continuously and email actionable plans")
    p.add_argument("--interval-sec", type=int, default=None)
    p.add_argument("--cooldown-sec", type=int, default=None)
    p.add_argument("--ib-readonly", action="store_true", help="Connect IB API in readonly mode")
    return p.parse_args()


def _assistant_config_from_app(app_cfg: AppConfig, symbols_arg: str | None) -> AssistantConfig:
    raw = app_cfg.signals.options_assistant
    symbols = (
        [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
        if symbols_arg
        else [s.upper() for s in (raw.symbols or ["AMD", "NOK", "ORCL"])]
    )
    return AssistantConfig(
        symbols=symbols,
        target_dte=raw.target_dte,
        min_dte=raw.min_dte,
        cheap_iv_hv=raw.cheap_iv_hv,
        fair_iv_hv=raw.fair_iv_hv,
        min_trend_score=raw.min_trend_score,
        long_call_delta=raw.long_call_delta,
        spread_short_delta=raw.spread_short_delta,
        protective_put_delta=raw.protective_put_delta,
        earnings_blackout_days_before=raw.earnings_blackout_days_before,
        earnings_blackout_days_after=raw.earnings_blackout_days_after,
        earnings_dates={k.upper(): v for k, v in raw.earnings_dates.items()},
        signal_actions=tuple(raw.signal_actions),
    )


def _email_notifier_from_app(app_cfg: AppConfig) -> EmailNotifier | None:
    email = app_cfg.signals.options_assistant.email
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app_cfg = AppConfig.from_yaml(args.config)
    raw_cfg = app_cfg.signals.options_assistant
    cfg = _assistant_config_from_app(app_cfg, args.symbols)
    ib = IB()
    try:
        host = args.host or app_cfg.ib.host
        port = args.port or app_cfg.ib.port
        client_id = args.client_id if args.client_id is not None else raw_cfg.client_id
        log.info(
            "Connecting IB %s:%s clientId=%s readonly=%s",
            host, port, client_id, args.ib_readonly,
        )
        ib.connect(host, port, clientId=client_id, timeout=20, readonly=args.ib_readonly)
        assistant = OptionsAssistant(ib, cfg)
        if args.monitor:
            interval = args.interval_sec or raw_cfg.interval_sec
            cooldown = args.cooldown_sec or raw_cfg.cooldown_sec
            monitor = OptionsAssistantMonitor(
                assistant=assistant,
                data_dir=args.data_dir,
                email_notifier=_email_notifier_from_app(app_cfg),
                interval_sec=interval,
                cooldown_sec=cooldown,
            )
            monitor.start()
        else:
            print(f"Options assistant run @ {et_now().strftime('%Y-%m-%d %H:%M:%S ET')}")
            print("This is a planning aid only; no orders are submitted.\n")
            for symbol in cfg.symbols:
                plan = assistant.plan_symbol(symbol)
                if plan is None:
                    print(f"{symbol}: no plan (missing data)\n")
                else:
                    print(format_plan(plan))
                    print()
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
