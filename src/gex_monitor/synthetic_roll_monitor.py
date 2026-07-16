"""Synthetic long stock roll/risk monitor.

This module observes configured synthetic long structures such as:

    long call(K) + short put(K) + optional long disaster put(H)

It never submits orders. It emits reminders when the structure has drifted far
enough that the trader may want to roll, reduce, or re-anchor protection.
"""
from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock

from .config import AppConfig
from .email_notifier import EmailConfig, EmailNotifier
from .iv_hv_monitor import _append_parquet
from .time_utils import et_now, is_market_open, trading_date_str

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "config" / "config.yaml"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOG_DIR = ROOT_DIR / "logs"

log = logging.getLogger("synthetic_roll_monitor")


@dataclass
class SyntheticPositionConfig:
    symbol: str
    expiry: str
    synthetic_strike: float
    hedge_put_strike: float | None = None
    quantity: int = 1
    entry_spot: float | None = None
    entry_debit: float | None = None


@dataclass
class SyntheticRollConfig:
    positions: list[SyntheticPositionConfig]
    client_id: int = 106
    interval_sec: int = 1800
    upside_roll_pct: float = 25.0
    min_hedge_strike_pct: float = 60.0
    short_put_warning_pct: float = 10.0
    hedge_warning_multiple: float = 1.30
    expiry_roll_dte: int = 90
    assignment_extrinsic_pct: float = 1.0
    cooldown_sec: int = 14400


@dataclass
class OptionLegSnapshot:
    symbol: str
    expiry: str
    strike: float
    right: str
    action: str
    bid: float | None
    ask: float | None
    mid: float | None
    mark: float | None
    mark_source: str
    delta: float | None
    iv: float | None
    open_interest: float | None


@dataclass
class SyntheticRollSnapshot:
    ts: pd.Timestamp
    symbol: str
    spot: float
    expiry: str
    dte: int
    synthetic_strike: float
    hedge_put_strike: float | None
    quantity: int
    entry_spot: float | None
    synthetic_debit: float | None
    implied_financing_rate: float | None
    implied_financing_source: str | None
    combo_mark: float | None
    net_delta: float | None
    short_put_extrinsic: float | None
    hedge_strike_pct: float | None
    call: OptionLegSnapshot | None
    short_put: OptionLegSnapshot | None
    hedge_put: OptionLegSnapshot | None


@dataclass
class RollAlert:
    symbol: str
    ts: pd.Timestamp
    action: str
    priority: str
    reasons: list[str]
    recommendation: str
    snapshot: SyntheticRollSnapshot


def evaluate_roll_need(
    snap: SyntheticRollSnapshot,
    cfg: SyntheticRollConfig,
) -> RollAlert | None:
    """Pure decision logic for synthetic roll/risk reminders."""
    reasons: list[str] = []
    actions: set[str] = set()

    if snap.entry_spot and snap.entry_spot > 0:
        gain_pct = (snap.spot / snap.entry_spot - 1.0) * 100.0
        if gain_pct >= cfg.upside_roll_pct:
            actions.add("ROLL_UP")
            reasons.append(
                f"spot is {gain_pct:.1f}% above entry_spot {snap.entry_spot:.2f}"
            )

    if snap.hedge_put_strike and snap.hedge_strike_pct is not None:
        if snap.hedge_strike_pct < cfg.min_hedge_strike_pct:
            actions.add("REANCHOR_HEDGE")
            reasons.append(
                f"hedge strike is only {snap.hedge_strike_pct:.1f}% of spot"
            )
        if snap.spot <= snap.hedge_put_strike * cfg.hedge_warning_multiple:
            actions.add("DOWNSIDE_RISK")
            reasons.append(
                f"spot is within {cfg.hedge_warning_multiple:.2f}x hedge strike"
            )

    short_put_trigger = snap.synthetic_strike * (1.0 + cfg.short_put_warning_pct / 100.0)
    if snap.spot <= short_put_trigger:
        actions.add("DOWNSIDE_RISK")
        reasons.append(
            f"spot {snap.spot:.2f} is close to short put strike {snap.synthetic_strike:.2f}"
        )

    if snap.dte <= cfg.expiry_roll_dte:
        actions.add("ROLL_EXPIRY")
        reasons.append(f"{snap.dte} DTE <= expiry roll threshold {cfg.expiry_roll_dte}")

    extrinsic_threshold = snap.synthetic_strike * cfg.assignment_extrinsic_pct / 100.0
    if (
        snap.spot < snap.synthetic_strike
        and snap.short_put_extrinsic is not None
        and snap.short_put_extrinsic <= extrinsic_threshold
    ):
        actions.add("ASSIGNMENT_RISK")
        reasons.append(
            "short put is ITM with low extrinsic "
            f"({snap.short_put_extrinsic:.2f} <= {extrinsic_threshold:.2f})"
        )

    if not reasons:
        return None

    if "DOWNSIDE_RISK" in actions or "ASSIGNMENT_RISK" in actions:
        priority = "high"
    elif "ROLL_EXPIRY" in actions:
        priority = "medium"
    else:
        priority = "watch"

    action = "+".join(sorted(actions))
    recommendation = _recommendation(actions, snap)
    return RollAlert(
        symbol=snap.symbol,
        ts=snap.ts,
        action=action,
        priority=priority,
        reasons=reasons,
        recommendation=recommendation,
        snapshot=snap,
    )


def _recommendation(actions: set[str], snap: SyntheticRollSnapshot) -> str:
    if "DOWNSIDE_RISK" in actions or "ASSIGNMENT_RISK" in actions:
        return (
            "Check excess liquidity first. Consider reducing size, rolling the "
            "synthetic strike down/out, or moving the protective put closer."
        )
    if "ROLL_UP" in actions or "REANCHOR_HEDGE" in actions:
        new_synth = _round_to_5(snap.spot * 0.90)
        new_hedge = _round_to_5(snap.spot * 0.70)
        return (
            "Consider taking partial profit or rolling up/re-anchoring. "
            f"A starting template is synthetic K~{new_synth:.0f} with hedge put "
            f"K~{new_hedge:.0f}, then choose by live liquidity and financing."
        )
    return "Consider rolling to a farther expiry while bid/ask liquidity is still reasonable."


def _round_to_5(value: float) -> float:
    return round(value / 5.0) * 5.0


class SyntheticRollMonitor:
    def __init__(
        self,
        ib: IB,
        cfg: SyntheticRollConfig,
        data_dir: Path,
        email_notifier: EmailNotifier | None = None,
    ):
        self.ib = ib
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.email_notifier = email_notifier
        self._last_alert: dict[tuple[str, str], float] = {}
        self._running = True

    def stop(self) -> None:
        self._running = False

    def start(self) -> None:
        log.info(
            "Started synthetic roll monitor: positions=%s interval=%ss",
            len(self.cfg.positions), self.cfg.interval_sec,
        )
        while self._running:
            now = et_now()
            if is_market_open(now):
                self.scan_once()
            else:
                log.info("Market closed; synthetic roll monitor sleeping")
            self.ib.sleep(self.cfg.interval_sec)

    def scan_once(self) -> list[SyntheticRollSnapshot]:
        snapshots = []
        for pos in self.cfg.positions:
            try:
                snap = self.collect_position(pos)
            except Exception:
                log.exception("[%s] synthetic roll collection failed", pos.symbol)
                continue
            if snap is None:
                continue
            snapshots.append(snap)
            self._persist_snapshot(snap)
            alert = evaluate_roll_need(snap, self.cfg)
            if alert is not None:
                self._handle_alert(alert)
            else:
                log.info(
                    "[%s] no roll alert: spot=%.2f dte=%s net_delta=%s combo=%s",
                    snap.symbol, snap.spot, snap.dte, _fmt_num(snap.net_delta),
                    _fmt_num(snap.combo_mark),
                )
        return snapshots

    def collect_position(self, pos: SyntheticPositionConfig) -> SyntheticRollSnapshot | None:
        symbol = pos.symbol.upper()
        stock = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(stock)
        spot = self._snapshot_price(stock)
        if spot is None:
            log.warning("[%s] no stock price", symbol)
            return None

        dte = _dte(pos.expiry)
        call_contract = Option(symbol, pos.expiry, pos.synthetic_strike, "C", "SMART")
        put_contract = Option(symbol, pos.expiry, pos.synthetic_strike, "P", "SMART")
        contracts = [call_contract, put_contract]
        hedge_contract = None
        if pos.hedge_put_strike is not None:
            hedge_contract = Option(symbol, pos.expiry, pos.hedge_put_strike, "P", "SMART")
            contracts.append(hedge_contract)
        self.ib.qualifyContracts(*contracts)

        call = self._leg_snapshot(call_contract, "BUY")
        short_put = self._leg_snapshot(put_contract, "SELL")
        hedge_put = self._leg_snapshot(hedge_contract, "BUY") if hedge_contract else None

        synthetic_debit = _sub_marks(call, short_put)
        implied_financing_rate = _implied_financing_rate(
            spot=spot,
            strike=pos.synthetic_strike,
            call=call,
            put=short_put,
            dte=dte,
        )
        implied_financing_source = _financing_source(call, short_put)
        combo_mark = synthetic_debit
        if combo_mark is not None and hedge_put and hedge_put.mark is not None:
            combo_mark += hedge_put.mark

        net_delta = _net_delta(call, short_put, hedge_put)
        short_put_extrinsic = _put_extrinsic(short_put, spot)
        hedge_strike_pct = (
            pos.hedge_put_strike / spot * 100.0
            if pos.hedge_put_strike is not None and spot > 0
            else None
        )

        snap = SyntheticRollSnapshot(
            ts=pd.Timestamp(et_now()),
            symbol=symbol,
            spot=spot,
            expiry=pos.expiry,
            dte=dte,
            synthetic_strike=pos.synthetic_strike,
            hedge_put_strike=pos.hedge_put_strike,
            quantity=pos.quantity,
            entry_spot=pos.entry_spot,
            synthetic_debit=synthetic_debit,
            implied_financing_rate=implied_financing_rate,
            implied_financing_source=implied_financing_source,
            combo_mark=combo_mark,
            net_delta=net_delta,
            short_put_extrinsic=short_put_extrinsic,
            hedge_strike_pct=hedge_strike_pct,
            call=call,
            short_put=short_put,
            hedge_put=hedge_put,
        )
        log.info(
            "[%s] spot=%.2f K=%.0f hedge=%s dte=%s combo=%s r_impl=%s src=%s delta=%s hedge%%=%s",
            symbol, spot, pos.synthetic_strike, _fmt_num(pos.hedge_put_strike), dte,
            _fmt_num(combo_mark), _fmt_pct(implied_financing_rate),
            implied_financing_source or "N/A", _fmt_num(net_delta), _fmt_num(hedge_strike_pct),
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

    def _leg_snapshot(self, contract: Option, action: str) -> OptionLegSnapshot:
        ticker = self.ib.reqMktData(contract, "100,101,106", False, False)
        self.ib.sleep(2.0)
        greeks = ticker.modelGreeks
        bid = _clean(ticker.bid)
        ask = _clean(ticker.ask)
        mid = (bid + ask) / 2.0 if bid and ask and bid > 0 and ask > 0 else None
        model_price = _clean(greeks.optPrice) if greeks else None
        close = _clean(ticker.close)
        if mid is not None:
            mark, source = mid, "mid"
        elif model_price is not None and model_price > 0:
            mark, source = model_price, "model"
        else:
            mark, source = close, "close"

        oi_attr = "callOpenInterest" if contract.right == "C" else "putOpenInterest"
        snap = OptionLegSnapshot(
            symbol=contract.symbol,
            expiry=contract.lastTradeDateOrContractMonth,
            strike=float(contract.strike),
            right=contract.right,
            action=action,
            bid=bid,
            ask=ask,
            mid=mid,
            mark=mark,
            mark_source=source,
            delta=_clean(greeks.delta) if greeks else None,
            iv=_clean(greeks.impliedVol) if greeks else None,
            open_interest=_clean(getattr(ticker, oi_attr, None)),
        )
        self.ib.cancelMktData(contract)
        return snap

    def _persist_snapshot(self, snap: SyntheticRollSnapshot) -> None:
        path = self.data_dir / f"synthetic_roll_{snap.symbol}_{trading_date_str()}.parquet"
        row = _snapshot_row(snap)
        _append_parquet(path, row, unique_cols=["ts", "symbol", "expiry", "synthetic_strike"])

    def _handle_alert(self, alert: RollAlert) -> None:
        key = (alert.symbol, alert.action)
        now = time.time()
        last = self._last_alert.get(key, 0)
        if now - last < self.cfg.cooldown_sec:
            log.info("[%s] synthetic roll alert suppressed by cooldown", alert.symbol)
            return
        self._last_alert[key] = now

        alert_path = self.data_dir / f"synthetic_roll_alerts_{trading_date_str()}.parquet"
        row = _snapshot_row(alert.snapshot)
        row.update({
            "action": alert.action,
            "priority": alert.priority,
            "reasons": "; ".join(alert.reasons),
            "recommendation": alert.recommendation,
        })
        _append_parquet(alert_path, row, unique_cols=["ts", "symbol", "action"])

        subject = _alert_subject(alert)
        body = _format_alert_body(alert)
        log.warning("[%s] %s: %s", alert.symbol, subject, "; ".join(alert.reasons))
        if self.email_notifier is not None:
            self.email_notifier.send_alert(subject, body)


def _snapshot_row(snap: SyntheticRollSnapshot) -> dict:
    row = asdict(snap)
    row["ts"] = snap.ts.isoformat()
    for leg_name in ("call", "short_put", "hedge_put"):
        leg = row.pop(leg_name)
        if leg is None:
            continue
        prefix = f"{leg_name}_"
        for key, value in leg.items():
            row[prefix + key] = value
    return row


def _net_delta(
    call: OptionLegSnapshot | None,
    short_put: OptionLegSnapshot | None,
    hedge_put: OptionLegSnapshot | None,
) -> float | None:
    deltas = []
    if call and call.delta is not None:
        deltas.append(call.delta)
    if short_put and short_put.delta is not None:
        deltas.append(-short_put.delta)
    if hedge_put and hedge_put.delta is not None:
        deltas.append(hedge_put.delta)
    return float(sum(deltas)) if deltas else None


def _sub_marks(left: OptionLegSnapshot | None, right: OptionLegSnapshot | None) -> float | None:
    if left is None or right is None or left.mark is None or right.mark is None:
        return None
    return float(left.mark - right.mark)


def _put_extrinsic(put: OptionLegSnapshot | None, spot: float) -> float | None:
    if put is None or put.mark is None:
        return None
    intrinsic = max(put.strike - spot, 0.0)
    return max(float(put.mark - intrinsic), 0.0)


def _implied_financing_rate(
    spot: float,
    strike: float,
    call: OptionLegSnapshot | None,
    put: OptionLegSnapshot | None,
    dte: int,
) -> float | None:
    """Annualized continuous implied financing from put-call parity.

    C - P = S - K * exp(-rT), so:
    r = -ln((S - C + P) / K) / T

    For dividend-paying stocks this is a net carry rate unless dividend PV is
    modeled separately.
    """
    if dte <= 0 or spot <= 0 or strike <= 0:
        return None
    if call is None or put is None or call.mark is None or put.mark is None:
        return None
    discount_numerator = spot - call.mark + put.mark
    if discount_numerator <= 0:
        return None
    t_years = dte / 365.0
    try:
        return float(-math.log(discount_numerator / strike) / t_years)
    except (ValueError, ZeroDivisionError):
        return None


def _financing_source(
    call: OptionLegSnapshot | None,
    put: OptionLegSnapshot | None,
) -> str | None:
    if call is None or put is None:
        return None
    return f"{call.mark_source}/{put.mark_source}"


def _dte(expiry: str) -> int:
    exp_date = datetime.strptime(str(expiry), "%Y%m%d").date()
    return (exp_date - et_now().date()).days


def _clean(value) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value < 0:
            return None
        return value
    except Exception:
        return None


def _fmt_num(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _alert_subject(alert: RollAlert) -> str:
    if "ROLL_UP" in alert.action:
        return f"{alert.symbol} 要 roll up / 重设保护"
    if "DOWNSIDE_RISK" in alert.action or "ASSIGNMENT_RISK" in alert.action:
        return f"{alert.symbol} synthetic 下跌风险检查"
    if "ROLL_EXPIRY" in alert.action:
        return f"{alert.symbol} synthetic 到期 roll 提醒"
    return f"{alert.symbol} synthetic roll 提醒"


def _format_alert_body(alert: RollAlert) -> str:
    s = alert.snapshot
    return (
        "合成多头 roll / 风险提醒\n\n"
        f"标的:        {s.symbol}\n"
        f"时间:        {alert.ts.strftime('%Y-%m-%d %H:%M:%S ET')}\n"
        f"提醒:        {_alert_subject(alert)}\n"
        f"Action code: {alert.action} ({alert.priority})\n"
        f"现价:        {s.spot:.2f}\n"
        f"结构:        +{s.synthetic_strike:.0f}C / -{s.synthetic_strike:.0f}P"
        + (f" / +{s.hedge_put_strike:.0f}P" if s.hedge_put_strike else "")
        + f"  x{s.quantity}\n"
        f"到期/DTE:    {s.expiry} / {s.dte}\n"
        f"入场现价:    {_fmt_num(s.entry_spot)}\n"
        f"组合 mark:   {_fmt_num(s.combo_mark)}\n"
        f"隐含融资:    {_fmt_pct(s.implied_financing_rate)}"
        + (f" ({s.implied_financing_source})" if s.implied_financing_source else "")
        + "\n"
        f"净 delta:    {_fmt_num(s.net_delta)}\n"
        f"保护/现价:   {_fmt_num(s.hedge_strike_pct)}%\n"
        f"short put extrinsic: {_fmt_num(s.short_put_extrinsic)}\n"
        "\n触发原因:\n"
        + "\n".join(f"- {r}" for r in alert.reasons)
        + "\n\n建议:\n"
        + alert.recommendation
        + "\n\n脚本没有提交任何订单；这是人工检查提醒。"
    )


def _runtime_config(app_cfg: AppConfig) -> SyntheticRollConfig | None:
    raw = app_cfg.signals.synthetic_roll
    if not raw.enabled:
        return None
    positions = [
        SyntheticPositionConfig(
            symbol=p.symbol.upper(),
            expiry=p.expiry,
            synthetic_strike=p.synthetic_strike,
            hedge_put_strike=p.hedge_put_strike,
            quantity=p.quantity,
            entry_spot=p.entry_spot,
            entry_debit=p.entry_debit,
        )
        for p in raw.positions
    ]
    return SyntheticRollConfig(
        positions=positions,
        client_id=raw.client_id,
        interval_sec=raw.interval_sec,
        upside_roll_pct=raw.upside_roll_pct,
        min_hedge_strike_pct=raw.min_hedge_strike_pct,
        short_put_warning_pct=raw.short_put_warning_pct,
        hedge_warning_multiple=raw.hedge_warning_multiple,
        expiry_roll_dte=raw.expiry_roll_dte,
        assignment_extrinsic_pct=raw.assignment_extrinsic_pct,
        cooldown_sec=raw.cooldown_sec,
    )


def _email_notifier(app_cfg: AppConfig) -> EmailNotifier | None:
    email = app_cfg.signals.synthetic_roll.email
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
    log_file = LOG_DIR / f"synthetic_roll_{trading_date_str()}.log"
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
    p = argparse.ArgumentParser(description="Synthetic long stock roll monitor")
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
    if runtime is None or not runtime.positions:
        log.info("Synthetic roll monitor disabled or has no positions")
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
        monitor = SyntheticRollMonitor(ib, runtime, args.data_dir, notifier)
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
