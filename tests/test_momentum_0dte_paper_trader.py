from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from gex_monitor.momentum_0dte_paper_trader import (
    Momentum0DTEPaperTrader,
    OptionLeg,
    SymbolPosition,
    arrow_direction,
    select_triad_strikes,
)


def _trader(tmp_path: Path, accounts=None, port: int = 4002) -> Momentum0DTEPaperTrader:
    ib = MagicMock()
    ib.managedAccounts.return_value = accounts or []
    return Momentum0DTEPaperTrader(
        ib,
        ib_port=port,
        dry_run=True,
        state_path=tmp_path / "state.json",
        trade_log_path=tmp_path / "trades.csv",
    )


def test_select_call_triad_uses_lower_atm_upper_roles():
    choices = select_triad_strikes([498, 499, 500, 501, 502], 500.2, "C")
    assert [(choice.role, choice.strike) for choice in choices] == [
        ("ITM", 499.0),
        ("ATM", 500.0),
        ("OTM", 501.0),
    ]


def test_select_put_triad_reverses_itm_and_otm_roles():
    choices = select_triad_strikes([498, 499, 500, 501, 502], 500.2, "P")
    assert [(choice.role, choice.strike) for choice in choices] == [
        ("ITM", 501.0),
        ("ATM", 500.0),
        ("OTM", 499.0),
    ]


def test_real_execution_guard_requires_single_du_account_and_paper_port(tmp_path):
    trader = _trader(tmp_path, ["DU12345"], 4002)
    assert trader.verify_paper_account()
    assert trader.account == "DU12345"

    assert not _trader(tmp_path, ["U12345"], 4002).verify_paper_account()
    assert not _trader(tmp_path, ["DU12345"], 4001).verify_paper_account()
    assert not _trader(tmp_path, ["DU12345", "DU99999"], 4002).verify_paper_account()


def test_only_full_vote_arrow_defines_direction():
    assert arrow_direction(pd.Series({"long_signal": True, "short_signal": False})) == "long"
    assert arrow_direction(pd.Series({"long_signal": False, "short_signal": True})) == "short"
    assert arrow_direction(pd.Series({"long_signal": False, "short_signal": False})) is None


def test_arrow_disappearance_closes_existing_group(tmp_path):
    trader = _trader(tmp_path)
    trader.positions["QQQ"] = SymbolPosition(
        symbol="QQQ",
        direction="long",
        entry_bar="2026-08-05T10:00:00-04:00",
        legs=[OptionLeg("ATM", 1, "QQQ TEST", "20260805", 500, "C", 1)],
    )
    trader._exit_group = MagicMock(return_value=True)
    trader._enter_group = MagicMock()
    row = pd.Series(
        {
            "close": 500.0,
            "score": 0,
            "long_signal": False,
            "short_signal": False,
            "long_entry": False,
            "short_entry": False,
        }
    )

    trader._handle_completed_bar("QQQ", pd.Timestamp("2026-08-05 14:01", tz="UTC"), row)

    trader._exit_group.assert_called_once()
    trader._enter_group.assert_not_called()


def test_persistent_arrow_does_not_reenter_without_new_edge(tmp_path):
    trader = _trader(tmp_path)
    trader._enter_group = MagicMock()
    row = pd.Series(
        {
            "close": 500.0,
            "score": 2,
            "long_signal": True,
            "short_signal": False,
            "long_entry": False,
            "short_entry": False,
        }
    )

    trader._handle_completed_bar("QQQ", pd.Timestamp("2026-08-05 14:01", tz="UTC"), row)

    trader._enter_group.assert_not_called()


def test_new_short_arrow_enters_put_direction(tmp_path):
    trader = _trader(tmp_path)
    trader._enter_group = MagicMock()
    row = pd.Series(
        {
            "close": 500.0,
            "score": -2,
            "long_signal": False,
            "short_signal": True,
            "long_entry": False,
            "short_entry": True,
        }
    )
    bar_ts = pd.Timestamp("2026-08-05 14:01", tz="UTC")

    trader._handle_completed_bar("SPY", bar_ts, row)

    trader._enter_group.assert_called_once_with("SPY", "short", bar_ts, 500.0, -2)
