"""Paper-only, atomic Iron Fly executor for the intraday VRP experiment."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime

import pandas as pd
from ib_insync import ComboLeg, Contract, ExecutionFilter, LimitOrder

from .config import IntradayVRPConfig
from .storage import StorageManager
from .time_utils import ET, et_now, trading_date_str

log = logging.getLogger(__name__)

PENDING_STATUSES = {
    "PendingSubmit", "PreSubmitted", "Submitted", "ApiPending", "PendingCancel",
}
CANCELLED_STATUSES = {"Cancelled", "ApiCancelled", "Inactive"}


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class VRPPaperIronFlyExecutor:
    """Submit one-lot credit flies only after a strict paper-account check."""

    def __init__(self, symbol: str, storage: StorageManager,
                 config: IntradayVRPConfig):
        self.symbol = symbol
        self.storage = storage
        self.config = config
        self._date: str | None = None
        self._orders: dict[str, dict] = {}
        self._active: dict[str, object] = {}
        self._attached_events: set[str] = set()

    @staticmethod
    def _order_ref(symbol: str, date_str: str, slot: str, width: float) -> str:
        return f"VRP_{symbol}_{date_str}_{slot.replace(':', '')}_W{width:g}"

    def _load_date(self, date_str: str) -> None:
        if self._date == date_str:
            return
        frame = self.storage.load_vrp_paper_orders(self.symbol, date_str)
        self._orders = {
            str(row["order_ref"]): row
            for row in frame.to_dict("records") if row.get("order_ref")
        }
        self._active = {}
        self._attached_events = set()
        self._date = date_str

    def _save(self, row: dict) -> None:
        self._orders[str(row["order_ref"])] = row
        self.storage.persist_vrp_paper_order(
            self.symbol, str(row["trading_date"]), row
        )

    def _paper_account(self, ib, ib_port: int | None) -> tuple[str | None, str | None]:
        if int(ib_port or -1) != self.config.paper_required_port:
            return None, f"port_{ib_port}_is_not_paper_{self.config.paper_required_port}"
        accounts = [str(account) for account in (ib.managedAccounts() or [])]
        if len(accounts) != 1 or not accounts[0].startswith("DU"):
            return None, f"paper_account_guard_failed:{accounts}"
        if self.config.paper_quantity != 1:
            return None, "quantity_must_equal_1"
        return accounts[0], None

    @staticmethod
    def _contract_map(contracts: list, expiry: str) -> dict[tuple[float, str], object]:
        result = {}
        for contract in contracts:
            if str(getattr(contract, "lastTradeDateOrContractMonth", "")) != expiry:
                continue
            strike = _finite(getattr(contract, "strike", None))
            right = getattr(contract, "right", None)
            con_id = int(getattr(contract, "conId", 0) or 0)
            if strike is not None and right in {"C", "P"} and con_id > 0:
                result[(strike, right)] = contract
        return result

    def _fresh_quote(self, ib, contract, *, now: datetime) -> dict:
        """Read the latest subscribed NBBO immediately before order submission."""
        ticker = ib.ticker(contract)
        bid = _finite(getattr(ticker, "bid", None)) if ticker is not None else None
        ask = _finite(getattr(ticker, "ask", None)) if ticker is not None else None
        timestamp = getattr(ticker, "time", None) if ticker is not None else None
        age = None
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=ET)
            age = max(0.0, (now - timestamp.astimezone(ET)).total_seconds())
        if bid is None or ask is None or bid < 0 or ask < bid:
            status = "invalid_nbbo"
        elif age is None or age > self.config.max_quote_age_seconds:
            status = "stale_quote"
        else:
            status = "ok"
        return {"bid": bid, "ask": ask, "timestamp": timestamp,
                "age_seconds": age, "status": status}

    def maybe_submit(self, ib, contracts: list, *, now: datetime, ib_port: int | None,
                     quote_row: dict, fly_rows: list[dict]) -> bool:
        if not self.config.paper_execution_enabled:
            return False
        slot = str(quote_row.get("scheduled_time"))
        if slot not in self.config.paper_entry_slots:
            return False
        date_str = trading_date_str(now)
        self._load_date(date_str)
        width = float(self.config.paper_iron_fly_width)
        order_ref = self._order_ref(self.symbol, date_str, slot, width)
        if order_ref in self._orders:
            return False
        fly = next((row for row in fly_rows
                    if _finite(row.get("target_wing_width")) == width
                    and row.get("status") == "ok"), None)
        row = {
            "schema_version": 1, "symbol": self.symbol,
            "trading_date": date_str, "scheduled_time": slot,
            "order_ref": order_ref, "created_at": now, "updated_at": now,
            "expiry": quote_row.get("expiry"), "spot": quote_row.get("spot"),
            "atm_strike": quote_row.get("strike"), "target_wing_width": width,
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
        if fly is None:
            row.update({"status": "SKIPPED_NO_VALID_FLY",
                        "failure_reason": "configured_width_candidate_unavailable"})
            self._save(row)
            return False
        row.update({
            "lower_put_strike": fly["lower_put_strike"],
            "upper_call_strike": fly["upper_call_strike"],
            "intended_gross_credit": fly["gross_net_credit"],
            "intended_net_credit_after_fees": fly["net_credit_after_fees"],
            "max_loss_dollars_at_intent": fly["max_loss_dollars"],
            "snapshot_gross_credit": fly["gross_net_credit"],
        })
        account, reason = self._paper_account(ib, ib_port)
        if reason:
            row.update({"status": "BLOCKED_SAFETY", "failure_reason": reason})
            self._save(row)
            log.error("[%s] VRP paper order blocked: %s", self.symbol, reason)
            return False

        expiry = str(row["expiry"])
        contract_map = self._contract_map(contracts, expiry)
        atm = _finite(row["atm_strike"])
        lower = _finite(row["lower_put_strike"])
        upper = _finite(row["upper_call_strike"])
        keys = [(atm, "C"), (atm, "P"), (lower, "P"), (upper, "C")]
        legs = [contract_map.get(key) for key in keys]
        if any(contract is None for contract in legs):
            row.update({"status": "SKIPPED_MISSING_CONID",
                        "failure_reason": f"missing_contracts:{keys}"})
            self._save(row)
            return False
        atm_call, atm_put, lower_put, upper_call = legs
        priced_at = et_now()
        fresh = {
            "atm_call": self._fresh_quote(ib, atm_call, now=priced_at),
            "atm_put": self._fresh_quote(ib, atm_put, now=priced_at),
            "lower_put": self._fresh_quote(ib, lower_put, now=priced_at),
            "upper_call": self._fresh_quote(ib, upper_call, now=priced_at),
        }
        bad = {name: value["status"] for name, value in fresh.items()
               if value["status"] != "ok"}
        if bad:
            row.update({"status": "SKIPPED_STALE_SUBMIT_QUOTE",
                        "failure_reason": json.dumps(bad, sort_keys=True),
                        "submit_quote_at": priced_at})
            self._save(row)
            return False
        credit = (
            fresh["atm_call"]["bid"] + fresh["atm_put"]["bid"]
            - fresh["lower_put"]["ask"] - fresh["upper_call"]["ask"]
        )
        row.update({
            "submit_quote_at": priced_at,
            "submit_quote_json": json.dumps(fresh, sort_keys=True, default=str),
            "intended_gross_credit": credit,
            "intended_net_credit_after_fees": (
                credit - self.config.commission_per_straddle * 2 / 100.0
            ),
            "max_loss_dollars_at_intent": (
                width - credit + self.config.commission_per_straddle * 2 / 100.0
            ) * 100,
        })
        bag = Contract(
            secType="BAG", symbol=self.symbol, currency="USD", exchange="SMART",
            comboLegs=[
                ComboLeg(conId=atm_call.conId, ratio=1, action="SELL", exchange="SMART"),
                ComboLeg(conId=atm_put.conId, ratio=1, action="SELL", exchange="SMART"),
                ComboLeg(conId=lower_put.conId, ratio=1, action="BUY", exchange="SMART"),
                ComboLeg(conId=upper_call.conId, ratio=1, action="BUY", exchange="SMART"),
            ],
        )
        if credit is None or credit <= 0:
            row.update({"status": "SKIPPED_NONPOSITIVE_CREDIT",
                        "failure_reason": f"credit={credit}"})
            self._save(row)
            return False
        # BUY executes ComboLeg actions; a credit combo is represented by a negative price.
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
                {"conId": lower_put.conId, "strike": lower, "right": "P", "side": "BUY"},
                {"conId": upper_call.conId, "strike": upper, "right": "C", "side": "BUY"},
            ], sort_keys=True),
        })
        self._save(row)
        submitted_at = et_now()
        try:
            trade = ib.placeOrder(bag, order)
        except Exception as exc:
            row.update({"status": "SUBMIT_ERROR", "updated_at": et_now(),
                        "failure_reason": str(exc)})
            self._save(row)
            return False
        row.update({
            "status": str(getattr(trade.orderStatus, "status", "PendingSubmit")),
            "submitted_at": submitted_at, "updated_at": submitted_at,
            "order_id": getattr(trade.order, "orderId", None),
            "perm_id": getattr(trade.order, "permId", None),
        })
        self._active[order_ref] = trade
        self._save(row)
        self._attach_trade_events(order_ref, trade)
        log.warning("[%s] PAPER Iron Fly submitted %s limit_credit=%.2f",
                    self.symbol, order_ref, -limit_price)
        return True

    @staticmethod
    def _fills_json(trade) -> tuple[str, float, int]:
        rows, commission, reports = [], 0.0, 0
        for fill in getattr(trade, "fills", []) or []:
            execution = getattr(fill, "execution", None)
            report = getattr(fill, "commissionReport", None)
            value = _finite(getattr(report, "commission", None))
            if value is not None:
                commission += value
                reports += 1
            rows.append({
                "time": str(getattr(fill, "time", None)),
                "conId": getattr(getattr(fill, "contract", None), "conId", None),
                "side": getattr(execution, "side", None),
                "shares": getattr(execution, "shares", None),
                "price": getattr(execution, "price", None),
                "execId": getattr(execution, "execId", None),
                "commission": value,
            })
        return json.dumps(rows, sort_keys=True), commission, reports

    def _find_trade(self, ib, row: dict):
        order_ref = str(row["order_ref"])
        for trade in list(ib.openTrades()) + list(ib.trades()):
            if str(getattr(trade.order, "orderRef", "")) == order_ref:
                return self._hydrate_completed_fill(ib, trade, order_ref)
        try:
            completed = ib.reqCompletedOrders(False)
        except Exception:
            completed = []
        trade = next((item for item in completed
                      if str(getattr(item.order, "orderRef", "")) == order_ref), None)
        if trade is None:
            return None
        return self._hydrate_completed_fill(ib, trade, order_ref)

    @staticmethod
    def _hydrate_completed_fill(ib, trade, order_ref: str):
        """Merge CompletedOrder metadata with Executions quantity and price."""
        status = str(getattr(trade.orderStatus, "status", ""))
        filled = _finite(getattr(trade.orderStatus, "filled", 0)) or 0.0
        existing_fills = list(getattr(trade, "fills", []) or [])
        if status != "Filled" or (filled > 0 and existing_fills):
            return trade
        try:
            fills = [
                fill for fill in ib.reqExecutions(ExecutionFilter())
                if str(getattr(fill.execution, "orderRef", "")) == order_ref
            ]
        except Exception:
            fills = []
        if fills:
            trade.fills = fills
            bag_fill = next((fill for fill in fills
                             if getattr(fill.contract, "secType", "") == "BAG"), None)
            if bag_fill is not None:
                trade.orderStatus.filled = _finite(bag_fill.execution.shares) or 0.0
                trade.orderStatus.avgFillPrice = _finite(bag_fill.execution.avgPrice)
        return trade

    def _sync_trade_state(self, order_ref: str, trade, *, now: datetime) -> None:
        """Persist asynchronous IB order/fill/commission callbacks immediately."""
        row = self._orders.get(order_ref)
        if row is None:
            return
        status = str(getattr(trade.orderStatus, "status", row.get("status", "")))
        filled = _finite(getattr(trade.orderStatus, "filled", 0)) or 0.0
        avg_price = _finite(getattr(trade.orderStatus, "avgFillPrice", None))
        actual_credit = -avg_price if filled > 0 and avg_price is not None else None
        fills_json, commission, commission_reports = self._fills_json(trade)
        row.update({
            "status": status, "updated_at": now, "filled_quantity": filled,
            "avg_combo_fill_price": avg_price,
            "actual_gross_credit": actual_credit,
            "actual_commission_dollars": commission,
            "commission_report_count": commission_reports,
            "fills_json": fills_json,
            "order_id": getattr(trade.order, "orderId", row.get("order_id")),
            "perm_id": getattr(trade.order, "permId", row.get("perm_id")),
        })
        actual = actual_credit
        intended = _finite(row.get("intended_gross_credit"))
        row["credit_slippage"] = (
            actual - intended if actual is not None and intended is not None else None
        )
        if status == "Filled" and filled >= self.config.paper_quantity:
            row.setdefault("filled_at", now)
        self._save(row)

    def _attach_trade_events(self, order_ref: str, trade) -> None:
        if order_ref in self._attached_events:
            return

        def persist(*_args) -> None:
            try:
                self._sync_trade_state(order_ref, trade, now=et_now())
            except Exception:
                log.exception("[%s] Failed to persist paper order event %s",
                              self.symbol, order_ref)

        attached = False
        for name in ("statusEvent", "fillEvent", "commissionReportEvent"):
            event = getattr(trade, name, None)
            if event is None:
                continue
            event += persist
            attached = True
        if attached:
            self._attached_events.add(order_ref)

    def poll(self, ib, *, now: datetime, ib_port: int | None) -> None:
        if not self.config.paper_execution_enabled:
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
            self._attach_trade_events(order_ref, trade)
            ib_status = str(getattr(trade.orderStatus, "status", status))
            filled = _finite(getattr(trade.orderStatus, "filled", 0)) or 0.0
            avg_price = _finite(getattr(trade.orderStatus, "avgFillPrice", None))
            actual_credit = -avg_price if filled > 0 and avg_price is not None else None
            fills_json, commission, commission_reports = self._fills_json(trade)
            row.update({
                "status": ib_status, "updated_at": now, "filled_quantity": filled,
                "avg_combo_fill_price": avg_price,
                "actual_gross_credit": actual_credit,
                "actual_commission_dollars": commission,
                "commission_report_count": commission_reports,
                "fills_json": fills_json,
                "order_id": getattr(trade.order, "orderId", row.get("order_id")),
                "perm_id": getattr(trade.order, "permId", row.get("perm_id")),
            })
            intended = _finite(row.get("intended_gross_credit"))
            row["credit_slippage"] = (actual_credit - intended
                                      if actual_credit is not None and intended is not None
                                      else None)
            if ib_status == "Filled" and filled >= self.config.paper_quantity:
                row.setdefault("filled_at", now)
            submitted = pd.Timestamp(row.get("submitted_at")) if row.get("submitted_at") else None
            if submitted is not None:
                if submitted.tzinfo is None:
                    submitted = submitted.tz_localize(ET)
                else:
                    submitted = submitted.tz_convert(ET)
                age = (now - submitted.to_pydatetime()).total_seconds()
                if ib_status in PENDING_STATUSES and age >= self.config.paper_order_timeout_seconds:
                    ib.cancelOrder(trade.order)
                    row.update({"status": "CancelRequested", "cancel_requested_at": now})
            self._save(row)
            if ib_status in CANCELLED_STATUSES:
                self._active.pop(order_ref, None)
        self._persist_mtm(date_str)

    def _persist_mtm(self, date_str: str) -> None:
        marks = self.storage.load_vrp_iron_fly_mtm(self.symbol, date_str)
        if marks.empty:
            return
        existing = self.storage.load_vrp_paper_mtm(self.symbol, date_str)
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
                (pd.to_numeric(marks["target_wing_width"], errors="coerce") ==
                 float(order["target_wing_width"])) &
                (marks["status"] == "ok")
            ]
            for mark in matching.to_dict("records"):
                key = (order_ref, str(mark["checkpoint"]))
                if key in existing_keys:
                    continue
                close_debit = _finite(mark.get("close_debit_executable"))
                exit_fees = _finite(mark.get("estimated_exit_fees_dollars")) or 0.0
                rows.append({
                    "schema_version": 1, "symbol": self.symbol,
                    "trading_date": date_str, "order_ref": order_ref,
                    "scheduled_time": order["scheduled_time"],
                    "target_wing_width": order["target_wing_width"],
                    "checkpoint": mark["checkpoint"],
                    "checkpoint_at": mark.get("checkpoint_at"),
                    "actual_entry_credit": credit,
                    "actual_entry_commission_dollars": commission,
                    "close_debit_executable": close_debit,
                    "estimated_exit_fees_dollars": exit_fees,
                    "paper_pnl_dollars": (credit - close_debit) * 100 - commission - exit_fees
                    if close_debit is not None else None,
                    "status": "ok",
                })
                existing_keys.add(key)
        self.storage.persist_vrp_paper_mtm(self.symbol, date_str, rows)

    def settle(self, date_str: str, fly_observations: pd.DataFrame) -> int:
        self._load_date(date_str)
        count = 0
        for order_ref, row in list(self._orders.items()):
            if str(row.get("status")) != "Filled":
                continue
            matching = fly_observations[
                (fly_observations["scheduled_time"].astype(str) ==
                 str(row["scheduled_time"])) &
                (pd.to_numeric(fly_observations["target_wing_width"], errors="coerce") ==
                 float(row["target_wing_width"]))
            ]
            if matching.empty:
                continue
            observation = matching.iloc[-1]
            payoff = _finite(observation.get("terminal_fly_payoff"))
            credit = _finite(row.get("actual_gross_credit"))
            commission = _finite(row.get("actual_commission_dollars")) or 0.0
            if payoff is None or credit is None:
                continue
            row.update({
                "status": "EXPIRED", "updated_at": et_now(),
                "settled_at": observation.get("settled_at"),
                "settlement_price": observation.get("settlement_price"),
                "terminal_fly_payoff": payoff,
                "paper_realized_pnl_dollars": (credit - payoff) * 100 - commission,
                "paper_max_loss_dollars": (
                    float(row["target_wing_width"]) - credit
                ) * 100 + commission,
                "commission_complete": int(row.get("commission_report_count") or 0) >= 4,
            })
            max_loss = _finite(row.get("paper_max_loss_dollars"))
            row["paper_return_on_max_risk"] = (
                row["paper_realized_pnl_dollars"] / max_loss
                if max_loss is not None and max_loss > 0 else None
            )
            self._save(row)
            count += 1
        return count
