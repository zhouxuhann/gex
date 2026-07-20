"""Paper-only atomic short-straddle executor for the intraday VRP experiment."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime

import pandas as pd
from ib_insync import ComboLeg, Contract, LimitOrder

from .config import IntradayVRPConfig
from .intraday_vrp_paper import (
    CANCELLED_STATUSES,
    PENDING_STATUSES,
    VRPPaperIronFlyExecutor,
    _finite,
)
from .storage import StorageManager
from .time_utils import et_now, trading_date_str

log = logging.getLogger(__name__)


class VRPPaperStraddleExecutor(VRPPaperIronFlyExecutor):
    """Submit one-lot naked ATM straddles only after the paper-account guard."""

    def __init__(self, symbol: str, storage: StorageManager,
                 config: IntradayVRPConfig):
        super().__init__(symbol, storage, config)

    @staticmethod
    def _order_ref(symbol: str, date_str: str, slot: str) -> str:
        return f"VRPS_{symbol}_{date_str}_{slot.replace(':', '')}"

    def _load_date(self, date_str: str) -> None:
        if self._date == date_str:
            return
        frame = self.storage.load_vrp_paper_straddle_orders(self.symbol, date_str)
        self._orders = {
            str(row["order_ref"]): row
            for row in frame.to_dict("records") if row.get("order_ref")
        }
        self._active = {}
        self._date = date_str

    def _save(self, row: dict) -> None:
        self._orders[str(row["order_ref"])] = row
        self.storage.persist_vrp_paper_straddle_order(
            self.symbol, str(row["trading_date"]), row
        )

    def maybe_submit(self, ib, contracts: list, *, now: datetime,
                     ib_port: int | None, quote_row: dict) -> bool:
        if not self.config.paper_straddle_execution_enabled:
            return False
        slot = str(quote_row.get("scheduled_time"))
        if slot not in self.config.paper_entry_slots:
            return False
        date_str = trading_date_str(now)
        self._load_date(date_str)
        order_ref = self._order_ref(self.symbol, date_str, slot)
        if order_ref in self._orders:
            return False
        row = {
            "schema_version": 1, "strategy": "paper_short_straddle",
            "symbol": self.symbol, "trading_date": date_str,
            "scheduled_time": slot, "order_ref": order_ref,
            "created_at": now, "updated_at": now,
            "expiry": quote_row.get("expiry"), "spot": quote_row.get("spot"),
            "atm_strike": quote_row.get("strike"),
            "quantity": self.config.paper_quantity, "status": "INTENT_RECORDED",
            "account": None, "ib_port": ib_port,
            "entry_context_json": json.dumps(
                quote_row, ensure_ascii=False, sort_keys=True, default=str
            ),
        }
        for field in (
            "positive_gamma", "event_flag", "opex_type", "vix_ma20_ratio",
            "total_gex", "gamma_flip", "rr_25", "combined_spread_ratio",
            "surface_term_spread_iv", "gap_pct", "trend_efficiency_session",
        ):
            row[field] = quote_row.get(field)
        if quote_row.get("status") != "ok":
            row.update({"status": "SKIPPED_INVALID_QUOTE",
                        "failure_reason": f"quote_status={quote_row.get('status')}"})
            self._save(row)
            return False
        credit = _finite(quote_row.get("sell_credit_bid"))
        if credit is None or credit <= 0:
            row.update({"status": "SKIPPED_NONPOSITIVE_CREDIT",
                        "failure_reason": f"credit={credit}"})
            self._save(row)
            return False
        row["intended_gross_credit"] = credit
        row["intended_net_credit_after_fees"] = (
            credit - self.config.commission_per_straddle / 100.0
        )
        account, reason = self._paper_account(ib, ib_port)
        if reason:
            row.update({"status": "BLOCKED_SAFETY", "failure_reason": reason})
            self._save(row)
            log.error("[%s] VRP paper straddle blocked: %s", self.symbol, reason)
            return False

        expiry = str(row["expiry"])
        atm = _finite(row["atm_strike"])
        contract_map = self._contract_map(contracts, expiry)
        atm_call = contract_map.get((atm, "C"))
        atm_put = contract_map.get((atm, "P"))
        if atm_call is None or atm_put is None:
            row.update({"status": "SKIPPED_MISSING_CONID",
                        "failure_reason": f"missing_atm_contracts:{atm}"})
            self._save(row)
            return False
        bag = Contract(
            secType="BAG", symbol=self.symbol, currency="USD", exchange="SMART",
            comboLegs=[
                ComboLeg(conId=atm_call.conId, ratio=1, action="SELL", exchange="SMART"),
                ComboLeg(conId=atm_put.conId, ratio=1, action="SELL", exchange="SMART"),
            ],
        )
        limit_price = -math.floor(credit * 100 + 1e-9) / 100.0
        order = LimitOrder(
            "BUY", self.config.paper_quantity, limit_price,
            tif="DAY", account=account, orderRef=order_ref,
        )
        row.update({
            "account": account, "limit_combo_price": limit_price,
            "limit_credit": -limit_price,
            "legs_json": json.dumps([
                {"conId": atm_call.conId, "strike": atm, "right": "C", "side": "SELL"},
                {"conId": atm_put.conId, "strike": atm, "right": "P", "side": "SELL"},
            ], sort_keys=True),
        })
        self._save(row)
        try:
            trade = ib.placeOrder(bag, order)
        except Exception as exc:
            row.update({"status": "SUBMIT_ERROR", "updated_at": et_now(),
                        "failure_reason": str(exc)})
            self._save(row)
            return False
        row.update({
            "status": str(getattr(trade.orderStatus, "status", "PendingSubmit")),
            "submitted_at": now, "updated_at": now,
            "order_id": getattr(trade.order, "orderId", None),
            "perm_id": getattr(trade.order, "permId", None),
        })
        self._active[order_ref] = trade
        self._save(row)
        log.warning("[%s] PAPER short Straddle submitted %s limit_credit=%.2f",
                    self.symbol, order_ref, -limit_price)
        return True

    def poll(self, ib, *, now: datetime, ib_port: int | None) -> None:
        if not self.config.paper_straddle_execution_enabled:
            return
        date_str = trading_date_str(now)
        self._load_date(date_str)
        for order_ref, row in list(self._orders.items()):
            status = str(row.get("status", ""))
            if status not in PENDING_STATUSES | {"Filled", "CancelRequested"}:
                continue
            trade = self._active.get(order_ref) or self._find_trade(ib, row)
            if trade is None:
                if status in PENDING_STATUSES | {"CancelRequested"}:
                    row.update({"status": "RECONCILE_MISSING", "updated_at": now,
                                "failure_reason": "order_not_found_after_restart"})
                    self._save(row)
                continue
            self._active[order_ref] = trade
            ib_status = str(getattr(trade.orderStatus, "status", status))
            filled = _finite(getattr(trade.orderStatus, "filled", 0)) or 0.0
            avg_price = _finite(getattr(trade.orderStatus, "avgFillPrice", None))
            fills_json, commission, commission_reports = self._fills_json(trade)
            row.update({
                "status": ib_status, "updated_at": now, "filled_quantity": filled,
                "avg_combo_fill_price": avg_price,
                "actual_gross_credit": -avg_price if avg_price is not None else None,
                "actual_commission_dollars": commission,
                "commission_report_count": commission_reports,
                "fills_json": fills_json,
                "order_id": getattr(trade.order, "orderId", row.get("order_id")),
                "perm_id": getattr(trade.order, "permId", row.get("perm_id")),
            })
            actual = row.get("actual_gross_credit")
            intended = _finite(row.get("intended_gross_credit"))
            row["credit_slippage"] = (
                actual - intended if actual is not None and intended is not None else None
            )
            if ib_status == "Filled" and filled >= self.config.paper_quantity:
                row.setdefault("filled_at", now)
            submitted = pd.Timestamp(row.get("submitted_at")) if row.get("submitted_at") else None
            if submitted is not None:
                if submitted.tzinfo is None:
                    submitted = submitted.tz_localize(now.tzinfo)
                else:
                    submitted = submitted.tz_convert(now.tzinfo)
                age = (now - submitted.to_pydatetime()).total_seconds()
                if (ib_status in PENDING_STATUSES
                        and age >= self.config.paper_order_timeout_seconds):
                    ib.cancelOrder(trade.order)
                    row.update({"status": "CancelRequested", "cancel_requested_at": now})
            self._save(row)
            if ib_status in CANCELLED_STATUSES:
                self._active.pop(order_ref, None)
        self._persist_mtm(date_str)

    def _persist_mtm(self, date_str: str) -> None:
        marks = self.storage.load_vrp_mtm(self.symbol, date_str)
        if marks.empty:
            return
        existing = self.storage.load_vrp_paper_straddle_mtm(self.symbol, date_str)
        existing_keys = set(zip(
            existing.get("order_ref", pd.Series(dtype=str)).astype(str),
            existing.get("checkpoint", pd.Series(dtype=str)).astype(str),
        ))
        rows = []
        for order_ref, order in self._orders.items():
            if str(order.get("status")) not in {"Filled", "EXPIRED"}:
                continue
            credit = _finite(order.get("actual_gross_credit"))
            commission = _finite(order.get("actual_commission_dollars")) or 0.0
            if credit is None:
                continue
            matching = marks[
                (marks["scheduled_time"].astype(str) == str(order["scheduled_time"])) &
                (marks["status"] == "ok")
            ]
            for mark in matching.to_dict("records"):
                key = (order_ref, str(mark["checkpoint"]))
                if key in existing_keys:
                    continue
                close_cost = _finite(mark.get("close_cost_ask"))
                # The theoretical mark stores round-trip fees (entry + exit).
                # Paper entry commission is already known and deducted separately,
                # so only add one configured closing commission here.
                exit_fees = float(self.config.commission_per_straddle)
                rows.append({
                    "schema_version": 1, "symbol": self.symbol,
                    "trading_date": date_str, "order_ref": order_ref,
                    "scheduled_time": order["scheduled_time"],
                    "checkpoint": mark["checkpoint"],
                    "checkpoint_at": mark.get("checkpoint_at"),
                    "actual_entry_credit": credit,
                    "actual_entry_commission_dollars": commission,
                    "close_cost_executable": close_cost,
                    "estimated_exit_fees_dollars": exit_fees,
                    "paper_pnl_dollars": (credit - close_cost) * 100
                    - commission - exit_fees if close_cost is not None else None,
                    "status": "ok",
                })
                existing_keys.add(key)
        self.storage.persist_vrp_paper_straddle_mtm(self.symbol, date_str, rows)

    def settle(self, date_str: str, observations: pd.DataFrame) -> int:
        self._load_date(date_str)
        count = 0
        for order_ref, row in list(self._orders.items()):
            if str(row.get("status")) != "Filled":
                continue
            matching = observations[
                observations["scheduled_time"].astype(str) == str(row["scheduled_time"])
            ]
            if matching.empty:
                continue
            observation = matching.iloc[-1]
            payoff = _finite(observation.get("terminal_payoff"))
            credit = _finite(row.get("actual_gross_credit"))
            commission = _finite(row.get("actual_commission_dollars")) or 0.0
            if payoff is None or credit is None:
                continue
            row.update({
                "status": "EXPIRED", "updated_at": et_now(),
                "settled_at": observation.get("settled_at"),
                "settlement_price": observation.get("settlement_price"),
                "terminal_payoff": payoff,
                "paper_realized_pnl_dollars": (credit - payoff) * 100 - commission,
                "commission_complete": int(row.get("commission_report_count") or 0) >= 2,
            })
            self._save(row)
            count += 1
        return count
