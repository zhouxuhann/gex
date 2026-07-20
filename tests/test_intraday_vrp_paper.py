from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from gex_monitor.config import IntradayVRPConfig
from gex_monitor.intraday_vrp_paper import VRPPaperIronFlyExecutor
from gex_monitor.intraday_vrp_paper_straddle import VRPPaperStraddleExecutor
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

    def ticker(self, contract):
        quotes = {
            1: (1.172, 1.18), 2: (1.173, 1.18),
            3: (0.79, 0.80), 4: (0.86, 0.871),
        }
        bid, ask = quotes[contract.conId]
        return SimpleNamespace(bid=bid, ask=ask, time=datetime.now(ET))

    def placeOrder(self, bag, order):
        class FakeEvent:
            def __init__(self):
                self.handlers = []

            def __iadd__(self, handler):
                self.handlers.append(handler)
                return self

            def emit(self, *args):
                for handler in list(self.handlers):
                    handler(*args)

        self.placed.append((bag, order))
        self.trade = SimpleNamespace(
            contract=bag, order=order,
            orderStatus=SimpleNamespace(
                status="Submitted", filled=0, avgFillPrice=0,
            ),
            fills=[],
            statusEvent=FakeEvent(), fillEvent=FakeEvent(),
            commissionReportEvent=FakeEvent(),
        )
        return self.trade

    def openTrades(self):
        return [self.trade] if self.trade and self.trade.orderStatus.status == "Submitted" else []

    def trades(self):
        return [self.trade] if self.trade else []

    def cancelOrder(self, order):
        self.trade.orderStatus.status = "Cancelled"

    def reqCompletedOrders(self, api_only):
        return []

    def reqExecutions(self, execution_filter):
        return []


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
    ib.trade.statusEvent.emit(ib.trade)
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
    submitted = pd.Timestamp(
        storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]["submitted_at"]
    ).to_pydatetime()
    executor.poll(ib, now=submitted + timedelta(seconds=29), ib_port=4002)
    waiting = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert waiting["status"] == "Submitted"
    executor.poll(ib, now=submitted + timedelta(seconds=31), ib_port=4002)
    requested = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert requested["status"] == "CancelRequested"
    executor.poll(ib, now=submitted + timedelta(seconds=34), ib_port=4002)
    cancelled = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert cancelled["status"] == "Cancelled"
    assert cancelled["filled_quantity"] == 0
    storage.shutdown()


def test_paper_short_straddle_fill_mtm_and_expiry_are_separate(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(
        enabled=True, paper_straddle_execution_enabled=True,
        paper_entry_slots=["10:00"], paper_order_timeout_seconds=30,
    )
    executor = VRPPaperStraddleExecutor("QQQ", storage, config)
    quote, _, contracts = _inputs()
    # Persisted snapshot is deliberately stale; submit-time NBBO is 2.345.
    quote.update({"sell_credit_bid": 2.50, "status": "ok"})
    now = quote["observed_at"]
    ib = FakeIB(["DU12345"])
    assert executor.maybe_submit(
        ib, contracts, now=now, ib_port=4002, quote_row=quote,
    )
    bag, order = ib.placed[0]
    assert order.action == "BUY"
    assert order.lmtPrice == -2.34
    assert [leg.action for leg in bag.comboLegs] == ["SELL", "SELL"]
    intent = storage.load_vrp_paper_straddle_orders("QQQ", "20260720").iloc[0]
    assert intent["snapshot_gross_credit"] == 2.50
    assert abs(intent["intended_gross_credit"] - 2.345) < 1e-9

    ib.trade.orderStatus.status = "Filled"
    ib.trade.orderStatus.filled = 1
    ib.trade.orderStatus.avgFillPrice = -2.33
    ib.trade.fills = [
        SimpleNamespace(
            time=now, contract=SimpleNamespace(conId=index),
            execution=SimpleNamespace(side="SLD", shares=1, price=1, execId=str(index)),
            commissionReport=SimpleNamespace(commission=0.65),
        ) for index in range(1, 3)
    ]
    executor.poll(ib, now=now + timedelta(seconds=3), ib_port=4002)
    order_row = storage.load_vrp_paper_straddle_orders("QQQ", "20260720").iloc[0]
    assert order_row["status"] == "Filled"
    assert order_row["actual_gross_credit"] == 2.33
    assert order_row["actual_commission_dollars"] == 1.3
    assert storage.load_vrp_paper_orders("QQQ", "20260720").empty

    storage.persist_vrp_mtm("QQQ", "20260720", [{
        "symbol": "QQQ", "trading_date": "20260720",
        "scheduled_time": "10:00", "checkpoint": "+5m",
        "checkpoint_at": now + timedelta(minutes=5),
        # The source mark contains theoretical entry + exit fees.  The paper MTM
        # must not add that whole amount after deducting actual entry commission.
        "close_cost_ask": 2.00, "estimated_roundtrip_fees_dollars": 2.6,
        "status": "ok",
    }])
    executor.poll(ib, now=now + timedelta(minutes=5), ib_port=4002)
    mark = storage.load_vrp_paper_straddle_mtm("QQQ", "20260720").iloc[0]
    assert abs(mark["paper_pnl_dollars"] - 30.4) < 1e-9
    assert mark["estimated_exit_fees_dollars"] == 1.3

    observations = pd.DataFrame([{
        "scheduled_time": "10:00", "terminal_payoff": 1.50,
        "settled_at": now + timedelta(hours=6), "settlement_price": 726.5,
    }])
    assert executor.settle("20260720", observations) == 1
    settled = storage.load_vrp_paper_straddle_orders("QQQ", "20260720").iloc[0]
    assert settled["status"] == "EXPIRED"
    assert abs(settled["paper_realized_pnl_dollars"] - 81.7) < 1e-9
    assert bool(settled["commission_complete"])
    report = generate_vrp_report(tmp_path, "QQQ")
    assert "paper_short_straddle" in set(report["strategy"])
    overall = report[
        (report["strategy"] == "paper_short_straddle") &
        (report["dimension"] == "overall")
    ].iloc[0]
    assert overall["sizing_reason"] == "unbounded_strategy"
    storage.shutdown()


def test_paper_short_straddle_hard_blocks_wrong_port(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(
        enabled=True, paper_straddle_execution_enabled=True,
        paper_entry_slots=["10:00"],
    )
    executor = VRPPaperStraddleExecutor("QQQ", storage, config)
    quote, _, contracts = _inputs()
    quote.update({"sell_credit_bid": 2.30, "status": "ok"})
    ib = FakeIB(["DU12345"])
    assert not executor.maybe_submit(
        ib, contracts, now=quote["observed_at"], ib_port=4001, quote_row=quote,
    )
    row = storage.load_vrp_paper_straddle_orders("QQQ", "20260720").iloc[0]
    assert row["status"] == "BLOCKED_SAFETY"
    assert not ib.placed
    storage.shutdown()


def test_restart_recovers_completed_fill_from_ib(tmp_path):
    storage = StorageManager(tmp_path)
    config = IntradayVRPConfig(
        enabled=True, paper_execution_enabled=True,
        paper_entry_slots=["10:00"], paper_iron_fly_width=3,
    )
    quote, flies, contracts = _inputs()
    ib = FakeIB(["DU12345"])
    first = VRPPaperIronFlyExecutor("QQQ", storage, config)
    assert first.maybe_submit(
        ib, contracts, now=quote["observed_at"], ib_port=4002,
        quote_row=quote, fly_rows=flies,
    )
    row = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0].to_dict()
    row["status"] = "PendingCancel"
    storage.persist_vrp_paper_order("QQQ", "20260720", row)

    completed = SimpleNamespace(
        order=ib.trade.order,
        orderStatus=SimpleNamespace(status="Filled", filled=0, avgFillPrice=0),
        fills=[],
    )
    bag_fill = SimpleNamespace(
        time=quote["observed_at"], contract=SimpleNamespace(conId=0, secType="BAG"),
        execution=SimpleNamespace(
            orderRef=row["order_ref"], side="BOT", shares=1,
            avgPrice=-0.66, price=-0.66, execId="bag",
        ),
        commissionReport=SimpleNamespace(commission=0.0),
    )
    leg_fills = [
        SimpleNamespace(
            time=quote["observed_at"],
            contract=SimpleNamespace(conId=index, secType="OPT"),
            execution=SimpleNamespace(
                orderRef=row["order_ref"], side="BOT", shares=1,
                avgPrice=1.0, price=1.0, execId=str(index),
            ),
            commissionReport=SimpleNamespace(commission=0.65),
        ) for index in range(1, 5)
    ]

    class ReconnectedIB(FakeIB):
        def openTrades(self):
            return []

        def trades(self):
            return []

        def reqCompletedOrders(self, api_only):
            return [completed]

        def reqExecutions(self, execution_filter):
            return [bag_fill, *leg_fills]

    second = VRPPaperIronFlyExecutor("QQQ", storage, config)
    second.poll(
        ReconnectedIB(["DU12345"]),
        now=quote["observed_at"] + timedelta(minutes=1), ib_port=4002,
    )
    recovered = storage.load_vrp_paper_orders("QQQ", "20260720").iloc[0]
    assert recovered["status"] == "Filled"
    assert recovered["filled_quantity"] == 1
    assert recovered["actual_gross_credit"] == 0.66
    assert recovered["actual_commission_dollars"] == 2.6
    storage.shutdown()
