"""0DTE ATM straddle 日内观测器。

本模块只测量，不包含任何下单路径。原始报价与结算结果分开保存，方便以后
修改结算公式后从原始数据重建 observations。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IntradayVRPConfig
from .intraday_vrp_audit import build_vrp_daily_audit, write_vrp_daily_audit
from .storage import StorageManager, read_parquet_et
from .time_utils import ET, et_now, trading_date_str

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
                      expiry: str, is_true_0dte: bool, gex_state: dict) -> bool:
        due = self.due_slot(now)
        if due is None:
            return False
        slot, target = due
        date_str = trading_date_str(now)
        row = self._build_quote_row(
            ib, contracts, now, target, slot, spot, expiry, is_true_0dte, gex_state
        )
        self.storage.persist_vrp_quote(self.symbol, date_str, row)
        self._recorded.add(slot)
        log.info("[%s] VRP %s recorded status=%s strike=%s",
                 self.symbol, slot, row["status"], row.get("strike"))
        return True

    def _build_quote_row(self, ib, contracts: list, now: datetime, target: datetime,
                         slot: str, spot: float, expiry: str, is_true_0dte: bool,
                         gex: dict) -> dict:
        base = {
            "schema_version": 1,
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
            "regime_tags_json": json.dumps(gex.get("regime_tags"), ensure_ascii=False,
                                            sort_keys=True, default=str),
            "gex_partial": bool(gex.get("partial", False)),
            "gex_quality_reasons": ";".join(gex.get("quality_reasons") or []),
        }
        flip = _finite(gex.get("gamma_flip"))
        base["dist_to_flip_pct"] = ((spot - flip) / spot) if flip is not None else None
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
            values[f"{prefix}_bid"] = _finite(getattr(ticker, "bid", None))
            values[f"{prefix}_ask"] = _finite(getattr(ticker, "ask", None))
            values[f"{prefix}_bid_size"] = _finite(getattr(ticker, "bidSize", None))
            values[f"{prefix}_ask_size"] = _finite(getattr(ticker, "askSize", None))
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
            rv = float(np.sqrt(np.nansum(np.square(returns[path_mask]))))
            strike = _finite(row.get("strike"))
            mid = _finite(row.get("straddle_mid"))
            credit = _finite(row.get("sell_credit_bid"))
            payoff = abs(settle - strike) if strike is not None else None
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
                "pnl_mid": mid - payoff if mid is not None and payoff is not None else None,
                # 期权报价单位是每股；配置佣金是每套 straddle 的美元金额。
                "pnl_executable": credit - payoff - self.config.commission_per_straddle / 100.0
                if credit is not None and payoff is not None else None,
            })
            rows.append(row)
        self.storage.persist_vrp_observations(self.symbol, date_str, rows)
        self.audit_date(date_str)
        log.info("[%s] VRP settled %s: %s observations (%s)",
                 self.symbol, date_str, len(rows), source)
        return len(rows)

    def audit_date(self, date_str: str) -> dict:
        """审计指定日期并写入 vrp_quality_<symbol>_<date>.json。"""
        quotes = self.storage.load_vrp_quotes(self.symbol, date_str)
        observations = self.storage.load_vrp_observations(self.symbol, date_str)
        report = build_vrp_daily_audit(
            symbol=self.symbol,
            date_str=date_str,
            schedule=self._schedule,
            quotes=quotes,
            observations=observations,
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
