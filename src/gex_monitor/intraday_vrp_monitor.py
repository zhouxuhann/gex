"""0DTE ATM straddle 日内观测器。

本模块只测量，不包含任何下单路径。原始报价与结算结果分开保存，方便以后
修改结算公式后从原始数据重建 observations。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IntradayVRPConfig
from .intraday_vrp_audit import build_vrp_daily_audit, write_vrp_daily_audit
from .storage import StorageManager, read_parquet_et
from .time_utils import ET, et_now, trading_date_str
from .vrp_context import VRPEventCalendar, opex_context, path_features

log = logging.getLogger(__name__)


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _ticker_time(ticker) -> datetime | None:
    value = getattr(ticker, "time", None)
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=ET)
    return value.astimezone(ET)


def _sum_optional(*values) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) if finite else None


class IntradayVRPMonitor:
    """由 IBWorker 驱动的固定时点 0DTE straddle 观测器。"""

    def __init__(self, symbol: str, storage: StorageManager,
                 config: IntradayVRPConfig):
        if not config.observation_only:
            raise ValueError("IntradayVRPMonitor only supports observation_only=true")
        self.symbol = symbol
        self.storage = storage
        self.config = config
        self._schedule = tuple(config.schedule_et)
        self._recorded_date: str | None = None
        self._recorded: set[str] = set()
        self._event_calendar = VRPEventCalendar(storage.data_dir)
        self._mtm_date: str | None = None
        self._mtm_recorded: set[tuple[str, str]] = set()
        self._fly_mtm_recorded: set[tuple[str, float, str]] = set()
        self._mtm_quotes: list[dict] = []
        self._mtm_flies: list[dict] = []

    def _load_recorded(self, date_str: str) -> None:
        if self._recorded_date == date_str:
            return
        df = self.storage.load_vrp_quotes(self.symbol, date_str)
        self._recorded = set(df.get("scheduled_time", pd.Series(dtype=str)).astype(str))
        self._recorded_date = date_str

    def due_slot(self, now: datetime) -> tuple[str, datetime] | None:
        now = now.astimezone(ET)
        date_str = trading_date_str(now)
        self._load_recorded(date_str)
        for slot in self._schedule:
            if slot in self._recorded:
                continue
            hour, minute = (int(x) for x in slot.split(":"))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            delay = (now - target).total_seconds()
            if 0 <= delay <= self.config.sample_window_seconds:
                return slot, target
        return None

    def on_gex_update(self, ib, contracts: list, *, now: datetime, spot: float,
                      expiry: str, is_true_0dte: bool, gex_state: dict,
                      intraday_bars_provider=None) -> bool:
        self._capture_mtm_checkpoints(ib, contracts, now=now, expiry=expiry)
        due = self.due_slot(now)
        if due is None:
            return False
        slot, target = due
        date_str = trading_date_str(now)
        row = self._build_quote_row(
            ib, contracts, now, target, slot, spot, expiry, is_true_0dte, gex_state
        )
        bars = intraday_bars_provider() if intraday_bars_provider is not None else None
        row.update(path_features(bars, now, spot))
        self.storage.persist_vrp_quote(self.symbol, date_str, row)
        wing_rows, fly_rows = self._build_wings_and_flies(
            ib, contracts, now=now, quote_row=row
        )
        self.storage.persist_vrp_wing_quotes(self.symbol, date_str, wing_rows)
        self.storage.persist_vrp_iron_flies(self.symbol, date_str, fly_rows)
        self._ensure_mtm_state(date_str)
        self._mtm_quotes.append(dict(row))
        self._mtm_flies.extend(dict(item) for item in fly_rows)
        self._recorded.add(slot)
        log.info("[%s] VRP %s recorded status=%s strike=%s wings=%s flies=%s",
                 self.symbol, slot, row["status"], row.get("strike"),
                 len(wing_rows), len(fly_rows))
        return True

    def _ensure_mtm_state(self, date_str: str) -> None:
        if self._mtm_date == date_str:
            return
        quotes = self.storage.load_vrp_quotes(self.symbol, date_str)
        flies = self.storage.load_vrp_iron_flies(self.symbol, date_str)
        mtm = self.storage.load_vrp_mtm(self.symbol, date_str)
        fly_mtm = self.storage.load_vrp_iron_fly_mtm(self.symbol, date_str)
        self._mtm_quotes = quotes.to_dict("records")
        self._mtm_flies = flies.to_dict("records")
        self._mtm_recorded = set(zip(
            mtm.get("scheduled_time", pd.Series(dtype=str)).astype(str),
            mtm.get("checkpoint", pd.Series(dtype=str)).astype(str),
        ))
        self._fly_mtm_recorded = set(zip(
            fly_mtm.get("scheduled_time", pd.Series(dtype=str)).astype(str),
            pd.to_numeric(fly_mtm.get("target_wing_width", pd.Series(dtype=float)),
                          errors="coerce").fillna(-1.0),
            fly_mtm.get("checkpoint", pd.Series(dtype=str)).astype(str),
        ))
        self._mtm_date = date_str

    @staticmethod
    def _as_et(value) -> datetime | None:
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize(ET)
        else:
            ts = ts.tz_convert(ET)
        return ts.to_pydatetime()

    def _checkpoint_targets(self, row: dict) -> list[tuple[str, datetime]]:
        observed = self._as_et(row.get("observed_at"))
        if observed is None:
            return []
        targets = [(f"+{minutes}m", observed + timedelta(minutes=minutes))
                   for minutes in self.config.mtm_checkpoints_minutes]
        for clock in self.config.mtm_fixed_times_et:
            hour, minute = (int(value) for value in clock.split(":"))
            target = observed.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target > observed:
                targets.append((clock, target))
        return [(name, target) for name, target in targets
                if target.time() < datetime.strptime("15:59", "%H:%M").time()]

    def _capture_mtm_checkpoints(self, ib, contracts: list, *, now: datetime,
                                 expiry: str) -> int:
        """Mark original structures at elapsed/fixed checkpoints using executable NBBO."""
        now = now.astimezone(ET)
        date_str = trading_date_str(now)
        self._ensure_mtm_state(date_str)
        due: list[tuple[dict, str, datetime]] = []
        for row in self._mtm_quotes:
            if row.get("status") != "ok" or str(row.get("expiry")) != str(expiry):
                continue
            slot = str(row.get("scheduled_time"))
            for checkpoint, target in self._checkpoint_targets(row):
                delay = (now - target).total_seconds()
                quote_missing = (slot, checkpoint) not in self._mtm_recorded
                fly_missing = any(
                    (slot, _finite(fly.get("target_wing_width")) or -1.0, checkpoint)
                    not in self._fly_mtm_recorded
                    for fly in self._mtm_flies
                    if str(fly.get("scheduled_time")) == slot
                )
                if ((quote_missing or fly_missing)
                        and 0 <= delay <= self.config.sample_window_seconds):
                    due.append((row, checkpoint, target))
        if not due:
            return 0

        tickers: dict[tuple[float, str], object] = {}
        for contract in contracts:
            if str(getattr(contract, "lastTradeDateOrContractMonth", "")) != str(expiry):
                continue
            strike = _finite(getattr(contract, "strike", None))
            right = getattr(contract, "right", None)
            if strike is not None and right in ("C", "P"):
                ticker = ib.ticker(contract)
                if ticker is not None:
                    tickers[(strike, right)] = ticker

        def leg(strike: float, right: str) -> dict | None:
            ticker = tickers.get((strike, right))
            if ticker is None:
                return None
            bid = _finite(getattr(ticker, "bid", None))
            ask = _finite(getattr(ticker, "ask", None))
            ts = _ticker_time(ticker)
            age = max(0.0, (now - ts).total_seconds()) if ts else None
            if (bid is None or ask is None or bid < 0 or ask < bid or age is None
                    or age > self.config.max_quote_age_seconds):
                return None
            return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "age": age}

        rows, fly_rows = [], []
        for entry, checkpoint, target in due:
            slot = str(entry["scheduled_time"])
            strike = _finite(entry.get("strike"))
            if strike is None:
                continue
            call, put = leg(strike, "C"), leg(strike, "P")
            if call is None or put is None:
                continue
            entry_credit = _finite(entry.get("sell_credit_bid"))
            entry_mid = _finite(entry.get("straddle_mid"))
            close_ask = call["ask"] + put["ask"]
            close_mid = call["mid"] + put["mid"]
            fees = 2 * self.config.commission_per_straddle / 100.0
            if (slot, checkpoint) not in self._mtm_recorded:
                rows.append({
                    "schema_version": 1, "symbol": self.symbol,
                    "trading_date": date_str, "scheduled_time": slot,
                    "checkpoint": checkpoint, "checkpoint_at": target,
                    "observed_at": now, "delay_seconds": (now - target).total_seconds(),
                    "expiry": expiry, "strike": strike,
                    "entry_credit_bid": entry_credit, "entry_mid": entry_mid,
                    "close_cost_ask": close_ask, "close_mark_mid": close_mid,
                    "pnl_executable_roundtrip": entry_credit - close_ask - fees
                    if entry_credit is not None else None,
                    "pnl_mid_mark": entry_mid - close_mid if entry_mid is not None else None,
                    "estimated_roundtrip_fees_dollars": fees * 100,
                    "quote_age_seconds": max(call["age"], put["age"]), "status": "ok",
                })
                self._mtm_recorded.add((slot, checkpoint))

            matching = [fly for fly in self._mtm_flies
                        if str(fly.get("scheduled_time")) == slot]
            for fly in matching:
                width = _finite(fly.get("target_wing_width"))
                key = (slot, width if width is not None else -1.0, checkpoint)
                if width is None or key in self._fly_mtm_recorded:
                    continue
                lower = _finite(fly.get("lower_put_strike"))
                upper = _finite(fly.get("upper_call_strike"))
                lower_put = leg(lower, "P") if lower is not None else None
                upper_call = leg(upper, "C") if upper is not None else None
                if lower_put is None or upper_call is None:
                    continue
                close_debit = close_ask - lower_put["bid"] - upper_call["bid"]
                entry_net = _finite(fly.get("net_credit_after_fees"))
                exit_fees = self.config.commission_per_straddle * 2 / 100.0
                fly_rows.append({
                    "schema_version": 1, "symbol": self.symbol,
                    "trading_date": date_str, "scheduled_time": slot,
                    "target_wing_width": width, "checkpoint": checkpoint,
                    "checkpoint_at": target, "observed_at": now,
                    "delay_seconds": (now - target).total_seconds(), "expiry": expiry,
                    "atm_strike": strike, "entry_net_credit_after_fees": entry_net,
                    "close_debit_executable": close_debit,
                    "pnl_executable_roundtrip": entry_net - close_debit - exit_fees
                    if entry_net is not None else None,
                    "estimated_exit_fees_dollars": exit_fees * 100,
                    "quote_age_seconds": max(call["age"], put["age"],
                                             lower_put["age"], upper_call["age"]),
                    "status": "ok",
                })
                self._fly_mtm_recorded.add(key)
        self.storage.persist_vrp_mtm(self.symbol, date_str, rows)
        self.storage.persist_vrp_iron_fly_mtm(self.symbol, date_str, fly_rows)
        if rows:
            log.info("[%s] VRP MTM captured: straddles=%s iron_flies=%s",
                     self.symbol, len(rows), len(fly_rows))
        return len(rows)

    def _build_wings_and_flies(self, ib, contracts: list, *, now: datetime,
                               quote_row: dict) -> tuple[list[dict], list[dict]]:
        """构造标准化保护翼报价及有限风险 iron fly 候选。"""
        atm_strike = _finite(quote_row.get("strike"))
        expiry = str(quote_row.get("expiry") or "")
        if atm_strike is None or quote_row.get("status") in {
                "missing_pair", "not_true_0dte"}:
            return [], []

        tickers: dict[tuple[float, str], object] = {}
        for contract in contracts:
            if str(getattr(contract, "lastTradeDateOrContractMonth", "")) != expiry:
                continue
            strike = _finite(getattr(contract, "strike", None))
            right = getattr(contract, "right", None)
            if strike is None or right not in ("C", "P"):
                continue
            ticker = ib.ticker(contract)
            if ticker is not None:
                tickers[(strike, right)] = ticker

        available = sorted({strike for strike, _ in tickers})
        if atm_strike not in available:
            return [], []
        atm_index = available.index(atm_strike)
        n = max(0, self.config.wing_strikes_each_side)
        selected = available[max(0, atm_index - n): atm_index + n + 1]
        wing_rows = []
        for strike in selected:
            for right in ("C", "P"):
                ticker = tickers.get((strike, right))
                if ticker is None:
                    continue
                bid = _finite(getattr(ticker, "bid", None))
                ask = _finite(getattr(ticker, "ask", None))
                quote_ts = _ticker_time(ticker)
                age = max(0.0, (now - quote_ts).total_seconds()) if quote_ts else None
                greeks = getattr(ticker, "modelGreeks", None)
                valid = (bid is not None and ask is not None and bid >= 0
                         and ask >= bid and age is not None)
                wing_rows.append({
                    "schema_version": 2,
                    "symbol": self.symbol,
                    "trading_date": quote_row["trading_date"],
                    "scheduled_time": quote_row["scheduled_time"],
                    "observed_at": now,
                    "expiry": expiry,
                    "spot": quote_row["spot"],
                    "atm_strike": atm_strike,
                    "strike": strike,
                    "right": right,
                    "distance_from_atm": strike - atm_strike,
                    "bid": bid,
                    "ask": ask,
                    "bid_size": _finite(getattr(ticker, "bidSize", None)),
                    "ask_size": _finite(getattr(ticker, "askSize", None)),
                    "delta": _finite(getattr(greeks, "delta", None)),
                    "gamma": _finite(getattr(greeks, "gamma", None)),
                    "theta": _finite(getattr(greeks, "theta", None)),
                    "vega": _finite(getattr(greeks, "vega", None)),
                    "implied_vol": _finite(getattr(greeks, "impliedVol", None)),
                    "quote_ts": quote_ts,
                    "quote_age_seconds": age,
                    "status": "ok" if valid else "invalid_nbbo",
                })

        quote_map = {(r["strike"], r["right"]): r for r in wing_rows}
        atm_call = quote_map.get((atm_strike, "C"), {})
        atm_put = quote_map.get((atm_strike, "P"), {})
        short_credit = None
        if atm_call.get("status") == "ok" and atm_put.get("status") == "ok":
            short_credit = atm_call["bid"] + atm_put["bid"]

        fly_rows = []
        for target_width in self.config.iron_fly_widths:
            target_width = float(target_width)
            lower = atm_strike - target_width
            upper = atm_strike + target_width
            lower_put = quote_map.get((lower, "P"))
            upper_call = quote_map.get((upper, "C"))
            if (short_credit is None or lower_put is None or upper_call is None
                    or lower_put["status"] != "ok" or upper_call["status"] != "ok"):
                continue
            gross_credit = short_credit - lower_put["ask"] - upper_call["ask"]
            fees_dollars = self.config.commission_per_straddle * 2
            credit_after_fees = gross_credit - fees_dollars / 100.0
            downside_width = atm_strike - lower
            upside_width = upper - atm_strike
            max_loss_points = max(downside_width, upside_width) - credit_after_fees
            fly_rows.append({
                "schema_version": 1,
                "symbol": self.symbol,
                "trading_date": quote_row["trading_date"],
                "scheduled_time": quote_row["scheduled_time"],
                "observed_at": now,
                "expiry": expiry,
                "spot": quote_row["spot"],
                "atm_strike": atm_strike,
                "lower_put_strike": lower,
                "upper_call_strike": upper,
                "target_wing_width": target_width,
                "downside_wing_width": downside_width,
                "upside_wing_width": upside_width,
                "short_straddle_bid_credit": short_credit,
                "lower_put_ask": lower_put["ask"],
                "upper_call_ask": upper_call["ask"],
                "gross_net_credit": gross_credit,
                "estimated_fees_dollars": fees_dollars,
                "net_credit_after_fees": credit_after_fees,
                "lower_breakeven": atm_strike - credit_after_fees,
                "upper_breakeven": atm_strike + credit_after_fees,
                "max_loss_points": max_loss_points,
                "max_loss_dollars": max_loss_points * 100,
                "quote_age_seconds": max(
                    atm_call["quote_age_seconds"], atm_put["quote_age_seconds"],
                    lower_put["quote_age_seconds"], upper_call["quote_age_seconds"],
                ),
                "status": "ok" if credit_after_fees > 0 else "nonpositive_credit",
            })
        return wing_rows, fly_rows

    def _build_quote_row(self, ib, contracts: list, now: datetime, target: datetime,
                         slot: str, spot: float, expiry: str, is_true_0dte: bool,
                         gex: dict) -> dict:
        base = {
            "schema_version": 3,
            "symbol": self.symbol,
            "trading_date": trading_date_str(now),
            "scheduled_time": slot,
            "observed_at": now,
            "delay_seconds": (now - target).total_seconds(),
            "expiry": expiry,
            "is_true_0dte": bool(is_true_0dte),
            "spot": spot,
            "status": "missing_pair",
            "total_gex": gex.get("total_gex"),
            "gamma_flip": gex.get("gamma_flip"),
            "positive_gamma": gex.get("positive_gamma"),
            "call_wall": gex.get("call_wall"),
            "put_wall": gex.get("put_wall"),
            "max_pain": gex.get("max_pain"),
            "atm_iv_pct": gex.get("atm_iv_pct"),
            "regime_code": gex.get("regime_code"),
            # 复用 GEX 引擎已计算的 0DTE skew，不增加 IB API 请求。
            "rr_25": gex.get("rr_25"),
            "skew_slope": gex.get("skew_slope"),
            "rr_25_zscore": gex.get("rr_25_zscore"),
            "skew_signal": gex.get("skew_signal"),
            "drr_25": gex.get("drr_25"),
            "drr_25_zscore": gex.get("drr_25_zscore"),
            "skew_alert_level": gex.get("skew_alert_level"),
            "skew_alert_score": gex.get("skew_alert_score"),
            "regime_tags_json": json.dumps(gex.get("regime_tags"), ensure_ascii=False,
                                            sort_keys=True, default=str),
            "gex_partial": bool(gex.get("partial", False)),
            "gex_quality_reasons": ";".join(gex.get("quality_reasons") or []),
        }
        flip = _finite(gex.get("gamma_flip"))
        base["dist_to_flip_pct"] = ((spot - flip) / spot) if flip is not None else None
        base.update(opex_context(now))
        base.update(self._event_calendar.context(now))
        if not is_true_0dte:
            base["status"] = "not_true_0dte"
            return base

        pairs: dict[float, dict[str, tuple]] = {}
        for contract in contracts:
            if str(getattr(contract, "lastTradeDateOrContractMonth", "")) != expiry:
                continue
            strike = _finite(getattr(contract, "strike", None))
            right = getattr(contract, "right", None)
            if strike is None or right not in ("C", "P"):
                continue
            ticker = ib.ticker(contract)
            if ticker is not None:
                pairs.setdefault(strike, {})[right] = (contract, ticker)

        candidates = []
        for strike, legs in pairs.items():
            if "C" not in legs or "P" not in legs:
                continue
            distance = abs(strike - spot)
            candidates.append((distance, strike, legs))
        candidates.sort(key=lambda x: x[0])
        limit = 1 + 2 * max(0, self.config.candidate_strikes_each_side)
        candidates = candidates[:limit]

        best = None
        for distance, strike, legs in candidates:
            call_t = legs["C"][1]
            put_t = legs["P"][1]
            call_g = getattr(call_t, "modelGreeks", None)
            put_g = getattr(put_t, "modelGreeks", None)
            call_delta = _finite(getattr(call_g, "delta", None))
            put_delta = _finite(getattr(put_g, "delta", None))
            delta_score = (abs(call_delta + put_delta)
                           if call_delta is not None and put_delta is not None else None)
            score = (0 if delta_score is not None else 1,
                     delta_score if delta_score is not None else distance)
            if best is None or score < best[0]:
                best = (score, strike, legs, call_delta, put_delta,
                        "net_delta" if delta_score is not None else "nearest_strike")
        if best is None:
            return base

        _, strike, legs, call_delta, put_delta, method = best
        values = {}
        max_age = 0.0
        for right, prefix in (("C", "call"), ("P", "put")):
            ticker = legs[right][1]
            greeks = getattr(ticker, "modelGreeks", None)
            values[f"{prefix}_bid"] = _finite(getattr(ticker, "bid", None))
            values[f"{prefix}_ask"] = _finite(getattr(ticker, "ask", None))
            values[f"{prefix}_bid_size"] = _finite(getattr(ticker, "bidSize", None))
            values[f"{prefix}_ask_size"] = _finite(getattr(ticker, "askSize", None))
            values[f"{prefix}_gamma"] = _finite(getattr(greeks, "gamma", None))
            values[f"{prefix}_theta"] = _finite(getattr(greeks, "theta", None))
            values[f"{prefix}_vega"] = _finite(getattr(greeks, "vega", None))
            values[f"{prefix}_implied_vol"] = _finite(
                getattr(greeks, "impliedVol", None)
            )
            values[f"{prefix}_quote_ts"] = _ticker_time(ticker)
            quote_ts = values[f"{prefix}_quote_ts"]
            if quote_ts is not None:
                max_age = max(max_age, max(0.0, (now - quote_ts).total_seconds()))
            else:
                max_age = float("inf")
        values["call_delta"] = call_delta
        values["put_delta"] = put_delta
        base.update(values)
        base.update({
            "strike": strike,
            "strike_distance_pct": (strike - spot) / spot,
            "atm_method": method,
            "quote_age_seconds": max_age if np.isfinite(max_age) else None,
            "straddle_net_delta": call_delta + put_delta
            if call_delta is not None and put_delta is not None else None,
            "straddle_gamma": _sum_optional(values.get("call_gamma"),
                                              values.get("put_gamma")),
            "straddle_theta": _sum_optional(values.get("call_theta"),
                                              values.get("put_theta")),
            "straddle_vega": _sum_optional(values.get("call_vega"),
                                             values.get("put_vega")),
        })

        cb, ca, pb, pa = (values.get(k) for k in
                          ("call_bid", "call_ask", "put_bid", "put_ask"))
        if any(v is None or v < 0 for v in (cb, ca, pb, pa)) or ca < cb or pa < pb:
            base["status"] = "invalid_nbbo"
            return base
        call_mid, put_mid = (cb + ca) / 2, (pb + pa) / 2
        straddle_mid = call_mid + put_mid
        sell_credit = cb + pb
        combined_width = (ca - cb) + (pa - pb)
        spread_ratio = combined_width / straddle_mid if straddle_mid > 0 else None
        base.update({
            "call_mid": call_mid,
            "put_mid": put_mid,
            "straddle_mid": straddle_mid,
            "sell_credit_bid": sell_credit,
            "buy_cost_ask": ca + pa,
            "implied_move_mid_pct": straddle_mid / spot,
            "sell_credit_bid_pct": sell_credit / spot,
            "combined_spread_ratio": spread_ratio,
            "execution_haircut_pct": (1 - sell_credit / straddle_mid)
            if straddle_mid > 0 else None,
        })
        for field, level in (
            ("flip", flip), ("call_wall", _finite(gex.get("call_wall"))),
            ("put_wall", _finite(gex.get("put_wall"))),
            ("max_pain", _finite(gex.get("max_pain"))),
        ):
            base[f"dist_to_{field}_im"] = (
                (spot - level) / straddle_mid
                if level is not None and straddle_mid > 0 else None
            )
        if (base["quote_age_seconds"] is None
                or base["quote_age_seconds"] > self.config.max_quote_age_seconds):
            base["status"] = "stale_quote"
        elif spread_ratio is None or spread_ratio > self.config.max_combined_spread_ratio:
            base["status"] = "wide_spread"
        elif cb <= 0 or pb <= 0:
            base["status"] = "zero_bid"
        else:
            base["status"] = "ok"
        return base

    def settle_date(self, date_str: str) -> int:
        quotes = self.storage.load_vrp_quotes(self.symbol, date_str)
        if quotes.empty:
            return 0
        bars, source = self._load_clean_bars(date_str)
        if bars.empty:
            log.warning("[%s] VRP settlement %s: no clean OHLC", self.symbol, date_str)
            self.audit_date(date_str)
            return 0
        settle = _finite(bars.iloc[-1]["close"])
        if settle is None:
            return 0
        returns = np.log(bars["close"].astype(float)).diff()
        rows = []
        for raw in quotes.to_dict("records"):
            row = dict(raw)
            observed = pd.Timestamp(row["observed_at"])
            if observed.tzinfo is None:
                observed = observed.tz_localize(ET)
            else:
                observed = observed.tz_convert(ET)
            path_mask = bars["ts"] >= observed.floor("min")
            path = bars[path_mask].copy()
            rv = float(np.sqrt(np.nansum(np.square(returns[path_mask]))))
            strike = _finite(row.get("strike"))
            mid = _finite(row.get("straddle_mid"))
            credit = _finite(row.get("sell_credit_bid"))
            payoff = abs(settle - strike) if strike is not None else None
            highs = pd.to_numeric(path.get("high", path.get("close")), errors="coerce")
            lows = pd.to_numeric(path.get("low", path.get("close")), errors="coerce")
            closes = pd.to_numeric(path.get("close"), errors="coerce").dropna()
            path_high = _finite(highs.max())
            path_low = _finite(lows.min())
            net_credit = (credit - self.config.commission_per_straddle / 100.0
                          if credit is not None else None)
            lower_be = strike - net_credit if strike is not None and net_credit is not None else None
            upper_be = strike + net_credit if strike is not None and net_credit is not None else None
            breached = path.iloc[0:0]
            if lower_be is not None and upper_be is not None:
                breached = path[(highs > upper_be) | (lows < lower_be)]
            first_breach = breached.iloc[0]["ts"] if not breached.empty else None
            minutes_outside = None
            if lower_be is not None and upper_be is not None and not closes.empty:
                minutes_outside = int(((closes < lower_be) | (closes > upper_be)).sum())
            path_efficiency = None
            if len(closes) >= 2:
                travelled = float(closes.diff().abs().sum())
                path_efficiency = abs(float(closes.iloc[-1] - closes.iloc[0])) / travelled \
                    if travelled > 0 else 0.0
            close_location = None
            if path_high is not None and path_low is not None and path_high > path_low:
                close_location = (settle - path_low) / (path_high - path_low)
            max_distance = None
            if strike is not None and path_high is not None and path_low is not None:
                max_distance = max(abs(path_high - strike), abs(path_low - strike))
            row.update({
                "settled_at": et_now(),
                "settlement_price": settle,
                "settlement_source": source,
                "settlement_quality": "complete" if len(bars) >= 389 else "partial",
                "rth_bar_count": len(bars),
                "terminal_payoff": payoff,
                "terminal_payoff_pct": payoff / row["spot"] if payoff is not None else None,
                "spot_move_pct": abs(settle - row["spot"]) / row["spot"],
                "rv_1m": rv,
                "path_high": path_high,
                "path_low": path_low,
                "max_up_move_pct": (path_high - row["spot"]) / row["spot"]
                if path_high is not None else None,
                "max_down_move_pct": (path_low - row["spot"]) / row["spot"]
                if path_low is not None else None,
                "max_distance_to_strike": max_distance,
                "max_distance_to_strike_pct": max_distance / row["spot"]
                if max_distance is not None else None,
                "executable_lower_breakeven": lower_be,
                "executable_upper_breakeven": upper_be,
                "first_breakeven_breach_at": first_breach,
                "minutes_close_outside_breakeven": minutes_outside,
                "breakeven_breached": first_breach is not None,
                "worst_intrinsic_pnl_executable": net_credit - max_distance
                if net_credit is not None and max_distance is not None else None,
                "path_trend_efficiency": path_efficiency,
                "close_location_in_path_range": close_location,
                "pnl_mid": mid - payoff if mid is not None and payoff is not None else None,
                # 期权报价单位是每股；配置佣金是每套 straddle 的美元金额。
                "pnl_executable": credit - payoff - self.config.commission_per_straddle / 100.0
                if credit is not None and payoff is not None else None,
            })
            rows.append(row)
        self.storage.persist_vrp_observations(self.symbol, date_str, rows)
        fly_count = self._settle_iron_flies(
            date_str=date_str,
            settlement_price=settle,
            settlement_source=source,
            rth_bar_count=len(bars),
        )
        self.audit_date(date_str)
        log.info("[%s] VRP settled %s: %s straddles, %s iron flies (%s)",
                 self.symbol, date_str, len(rows), fly_count, source)
        return len(rows)

    def _settle_iron_flies(self, *, date_str: str, settlement_price: float,
                           settlement_source: str, rth_bar_count: int) -> int:
        """按有限风险到期 payoff 回填所有候选 Iron fly。"""
        flies = self.storage.load_vrp_iron_flies(self.symbol, date_str)
        if flies.empty:
            return 0
        rows = []
        for raw in flies.to_dict("records"):
            row = dict(raw)
            strike = _finite(row.get("atm_strike"))
            down_width = _finite(row.get("downside_wing_width"))
            up_width = _finite(row.get("upside_wing_width"))
            credit = _finite(row.get("net_credit_after_fees"))
            max_loss_dollars = _finite(row.get("max_loss_dollars"))
            if None in (strike, down_width, up_width, credit):
                continue
            distance = settlement_price - strike
            applicable_width = up_width if distance >= 0 else down_width
            terminal_payoff = min(abs(distance), applicable_width)
            pnl_points = credit - terminal_payoff
            pnl_dollars = pnl_points * 100
            lower_be = _finite(row.get("lower_breakeven"))
            upper_be = _finite(row.get("upper_breakeven"))
            row.update({
                "settled_at": et_now(),
                "settlement_price": settlement_price,
                "settlement_source": settlement_source,
                "settlement_quality": "complete" if rth_bar_count >= 389 else "partial",
                "rth_bar_count": rth_bar_count,
                "distance_to_atm": distance,
                "terminal_fly_payoff": terminal_payoff,
                "iron_fly_pnl_points": pnl_points,
                "iron_fly_pnl_dollars": pnl_dollars,
                "return_on_max_risk": pnl_dollars / max_loss_dollars
                if max_loss_dollars is not None and max_loss_dollars > 0 else None,
                "hit_max_loss": bool(abs(distance) >= applicable_width),
                "expired_inside_breakeven": bool(
                    lower_be is not None and upper_be is not None
                    and lower_be <= settlement_price <= upper_be
                ),
            })
            rows.append(row)
        self.storage.persist_vrp_iron_fly_observations(
            self.symbol, date_str, rows
        )
        return len(rows)

    def audit_date(self, date_str: str) -> dict:
        """审计指定日期并写入 vrp_quality_<symbol>_<date>.json。"""
        quotes = self.storage.load_vrp_quotes(self.symbol, date_str)
        observations = self.storage.load_vrp_observations(self.symbol, date_str)
        mtm = self.storage.load_vrp_mtm(self.symbol, date_str)
        iron_fly_mtm = self.storage.load_vrp_iron_fly_mtm(self.symbol, date_str)
        report = build_vrp_daily_audit(
            symbol=self.symbol,
            date_str=date_str,
            schedule=self._schedule,
            quotes=quotes,
            observations=observations,
            mtm=mtm,
            iron_fly_mtm=iron_fly_mtm,
        )
        path = write_vrp_daily_audit(report, self.storage.data_dir)
        level = logging.INFO if report["quality"] == "good" else logging.WARNING
        log.log(
            level,
            "[%s] VRP audit %s quality=%s coverage=%.1f%% settled=%.1f%% report=%s",
            self.symbol,
            date_str,
            report["quality"],
            report["coverage"] * 100,
            report["settled_coverage"] * 100,
            path,
        )
        return report

    def recover_unsettled(self) -> int:
        today = trading_date_str()
        total = 0
        for path in sorted(Path(self.storage.data_dir).glob(
                f"vrp_quotes_{self.symbol}_*.parquet")):
            date_str = path.stem.rsplit("_", 1)[-1]
            if date_str < today:
                quotes = self.storage.load_vrp_quotes(self.symbol, date_str)
                observations = self.storage.load_vrp_observations(self.symbol, date_str)
                quote_keys = set(quotes.get("scheduled_time", pd.Series(dtype=str)).astype(str))
                settled_keys = set(
                    observations.get("scheduled_time", pd.Series(dtype=str)).astype(str)
                )
                if not quote_keys.issubset(settled_keys):
                    total += self.settle_date(date_str)
        return total

    def _load_clean_bars(self, date_str: str) -> tuple[pd.DataFrame, str]:
        official = Path(self.storage.data_dir) / f"official_ohlc_{self.symbol}_{date_str}.parquet"
        fallback = Path(self.storage.data_dir) / f"ohlc_{self.symbol}_{date_str}.parquet"
        path = official if official.exists() else fallback
        if not path.exists():
            return pd.DataFrame(), "missing"
        df = read_parquet_et(path)
        if df.empty or "ts" not in df or "close" not in df:
            return pd.DataFrame(), "invalid"
        target = datetime.strptime(date_str, "%Y%m%d").date()
        df = df[df["ts"].dt.date == target].copy()
        clock = df["ts"].dt.time
        df = df[(clock >= datetime.strptime("09:30", "%H:%M").time()) &
                (clock <= datetime.strptime("16:00", "%H:%M").time())]
        df = df.drop_duplicates("ts", keep="last").sort_values("ts")
        return df, "official_ohlc_1m" if path == official else "derived_ohlc_1m"
