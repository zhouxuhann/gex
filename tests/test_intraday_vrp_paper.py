from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from gex_monitor.config import IntradayVRPConfig
from gex_monitor.intraday_vrp_paper import VRPPaperIronFlyExecutor
from gex_monitor.intraday_vrp_report import generate_vrp_report
from gex_monitor.storage import StorageManager
from gex_monitor.time_utils import ET


def _contract(strike, right, con_id):
    return SimpleNamespace(
        strike=strike, right=right, conId=con_id,
        lastTradeDateOrContractMonth="20260720",
    )


class FakeIB:
    def __init__(self, accounts):
        self.accounts = accounts
        self.placed = []
        self.trade = None

    def managedAccounts(self):
        return self.accounts

    def placeOrder(self, bag, order):
        self.placed.append((bag, order))
        self.trade = SimpleNamespace(
            contract=bag, order=order,
            orderStatus=SimpleNamespace(
                status="Submitted", filled=0, avgFillPrice=0,
            ),
            fills=[],
        )
        return self.trade

    def openTrades(self):
        return [self.trade] if self.trade and self.trade.orderStatus.status == "Submitted" else []

    def trades(self):
        return [self.trade] if self.trade else []

    def cancelOrder(self, order):
        self.trade.orderStatus.status = "Cancelled"


def _inputs():
    quote = {
        "symbol": "QQQ", "trading_date": "20260720", "scheduled_time": "10:00",
        "observed_at": datetime(2026, 7, 20, 10, 0, tzinfo=ET),
        "expiry": "20260720", "spot": 725.0, "strike": 725.0, "status": "ok",
    }
    fly = {
        "scheduled_time": "10:00", "target_wing_width": 3.0, "status": "ok",
        "atm_strike": 725.0, "lower_put_strike": 722.0,
        "upper_call_strike": 728.0, "gross_net_credit": 0.674,
        "net_credit_after_fees": 0.648, "max_loss_dollars": 235.2,
    }
    contracts = [
        _contract(725, "C", 1), _contract(725, "P", 2),
        _contract(722, "P", 3), _contract(728, "C", 4),
    ]
    return quote, [fly], contracts


def test_hard_blocks_non_paper_account(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(
        enabled=True, paper_execution_enabled=True,
        paper_entry_slots=["10:00"], paper_iron_fly_width=3,
    )
    executor = VRPPaperIronFlyExecutor("QQQ", storage, config)
    quote, flies, contracts = _inputs()
    ib = FakeIB(["U12345"])
    assert not executor.maybe_submit(
        ib, contracts, now=quote["observed_at"], ib_port=4002,
        quote_row=quote, fly_rows=flies,
    )
    assert not ib.placed
    row = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert row["status"] == "BLOCKED_SAFETY"
    storage.shutdown()


def test_paper_combo_fill_mtm_and_expiry_are_recorded(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(
        enabled=True, paper_execution_enabled=True,
        paper_entry_slots=["10:00"], paper_iron_fly_width=3,
    )
    executor = VRPPaperIronFlyExecutor("QQQ", storage, config)
    quote, flies, contracts = _inputs()
    now = quote["observed_at"]
    ib = FakeIB(["DU12345"])
    assert executor.maybe_submit(
        ib, contracts, now=now, ib_port=4002, quote_row=quote, fly_rows=flies,
    )
    bag, order = ib.placed[0]
    assert order.action == "BUY"
    assert order.lmtPrice == -0.67
    assert [leg.action for leg in bag.comboLegs] == ["SELL", "SELL", "BUY", "BUY"]

    ib.trade.orderStatus.status = "Filled"
    ib.trade.orderStatus.filled = 1
    ib.trade.orderStatus.avgFillPrice = -0.66
    ib.trade.fills = [
        SimpleNamespace(
            time=now, contract=SimpleNamespace(conId=index),
            execution=SimpleNamespace(side="BOT", shares=1, price=1, execId=str(index)),
            commissionReport=SimpleNamespace(commission=0.65),
        ) for index in range(1, 5)
    ]
    executor.poll(ib, now=now + timedelta(seconds=3), ib_port=4002)
    order_row = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert order_row["status"] == "Filled"
    assert order_row["actual_gross_credit"] == 0.66
    assert order_row["actual_commission_dollars"] == 2.6

    storage.persist_vrp_iron_fly_mtm("QQQ", "20260720", [{
        "symbol": "QQQ", "trading_date": "20260720",
        "scheduled_time": "10:00", "target_wing_width": 3.0,
        "checkpoint": "+5m", "checkpoint_at": now + timedelta(minutes=5),
        "close_debit_executable": 0.40, "estimated_exit_fees_dollars": 2.6,
        "status": "ok",
    }])
    executor.poll(ib, now=now + timedelta(minutes=5), ib_port=4002)
    mark = storage.load_vrp_paper_mtm("QQQ", "20260720").iloc[0]
    assert abs(mark["paper_pnl_dollars"] - 20.8) < 1e-9

    observations = pd.DataFrame([{
        "scheduled_time": "10:00", "target_wing_width": 3.0,
        "terminal_fly_payoff": 0.20, "settled_at": now + timedelta(hours=6),
        "settlement_price": 725.2,
    }])
    assert executor.settle("20260720", observations) == 1
    settled = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert settled["status"] == "EXPIRED"
    assert abs(settled["paper_realized_pnl_dollars"] - 43.4) < 1e-9
    assert bool(settled["commission_complete"])
    report = generate_vrp_report(tmp_path, "QQQ")
    assert "paper_iron_fly" in set(report["strategy"])
    storage.shutdown()


def test_unfilled_order_is_cancelled_without_chasing(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(
        enabled=True, paper_execution_enabled=True,
        paper_entry_slots=["10:00"], paper_iron_fly_width=3,
        paper_order_timeout_seconds=30,
    )
    executor = VRPPaperIronFlyExecutor("QQQ", storage, config)
    quote, flies, contracts = _inputs()
    now = quote["observed_at"]
    ib = FakeIB(["DU12345"])
    assert executor.maybe_submit(
        ib, contracts, now=now, ib_port=4002, quote_row=quote, fly_rows=flies,
    )
    executor.poll(ib, now=now + timedelta(seconds=31), ib_port=4002)
    requested = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert requested["status"] == "CancelRequested"
    executor.poll(ib, now=now + timedelta(seconds=34), ib_port=4002)
    cancelled = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert cancelled["status"] == "Cancelled"
    assert cancelled["filled_quantity"] == 0
    storage.shutdown()
