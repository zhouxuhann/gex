"""hedge_executor 测试"""
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from gex_monitor.hedge_signal import HedgeSignal
from gex_monitor.hedge_executor import (
    HedgeExecutor, TradeRecord, format_trade_result,
    DEFAULT_TARGET_DELTA, SPREAD_SHORT_DELTA,
)


@pytest.fixture
def mock_ib():
    ib = MagicMock()
    ib.managedAccounts.return_value = ['DU12345']  # Paper account
    ib.isConnected.return_value = True
    return ib


@pytest.fixture
def signal_hedge_now():
    return HedgeSignal(
        ts=datetime(2026, 4, 13, 15, 30),
        symbol='QQQ',
        action='HEDGE_NOW',
        urgency=0.7,
        skew_cheapness=20.0,
        gex_regime='negative',
        term_structure='backwardation',
        recommended_structure='outright_put',
        recommended_tenor='45D (exp 20260528)',
        reasoning='test',
    )


@pytest.fixture
def signal_hedge_spread():
    return HedgeSignal(
        ts=datetime(2026, 4, 13, 15, 30),
        symbol='QQQ',
        action='HEDGE_SPREAD',
        urgency=0.5,
        skew_cheapness=50.0,
        gex_regime='negative',
        term_structure='flat',
        recommended_structure='put_spread',
        recommended_tenor='30D (exp 20260513)',
        reasoning='test',
    )


@pytest.fixture
def signal_skip():
    return HedgeSignal(
        ts=datetime(2026, 4, 13, 15, 30),
        symbol='QQQ',
        action='SKIP',
        urgency=0.1,
        skew_cheapness=80.0,
        gex_regime='positive',
        term_structure='contango',
        recommended_structure='none',
        recommended_tenor='30-45D',
        reasoning='test',
    )


class TestPaperAccountVerification:
    def test_paper_account(self, mock_ib):
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)
        assert executor.verify_paper_account()

    def test_live_account_blocked(self, mock_ib):
        mock_ib.managedAccounts.return_value = ['U12345']  # Live account
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)
        assert not executor.verify_paper_account()

    def test_no_accounts(self, mock_ib):
        mock_ib.managedAccounts.return_value = []
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)
        assert not executor.verify_paper_account()


class TestExecuteSkipSignal:
    def test_skip_returns_none(self, mock_ib, signal_skip):
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)
        result = executor.execute(signal_skip, 480.0)
        assert result is None


class TestParseExpiry:
    def test_parse_with_expiry(self, mock_ib, signal_hedge_now):
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)
        expiry = executor._parse_expiry(signal_hedge_now)
        assert expiry == '20260528'

    def test_parse_spread_expiry(self, mock_ib, signal_hedge_spread):
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)
        expiry = executor._parse_expiry(signal_hedge_spread)
        assert expiry == '20260513'

    def test_parse_fallback(self, mock_ib):
        signal = HedgeSignal(
            ts=datetime.now(), symbol='QQQ', action='HEDGE_NOW',
            urgency=0.5, skew_cheapness=30, gex_regime='negative',
            term_structure='flat', recommended_structure='outright_put',
            recommended_tenor='30-45D',  # no explicit expiry
            reasoning='test',
        )
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)
        assert executor._parse_expiry(signal) is None


class TestDryRun:
    def test_outright_put_dry_run(self, mock_ib, signal_hedge_now):
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)

        # Mock _find_contract_by_delta
        mock_contract = MagicMock()
        mock_contract.strike = 470
        with patch.object(executor, '_find_contract_by_delta',
                          return_value=(mock_contract, -0.25, 0.22)):
            record = executor.execute(signal_hedge_now, 480.0)

        assert record is not None
        assert record.status == 'DRY_RUN'
        assert len(record.legs) == 1
        assert record.legs[0]['strike'] == 470
        assert record.legs[0]['side'] == 'BUY'
        assert record.legs[0]['right'] == 'P'

    def test_put_spread_dry_run(self, mock_ib, signal_hedge_spread):
        executor = HedgeExecutor(ib=mock_ib, dry_run=True)

        long_contract = MagicMock()
        long_contract.strike = 475
        short_contract = MagicMock()
        short_contract.strike = 465

        returns = [(long_contract, -0.25, 0.22), (short_contract, -0.10, 0.20)]
        with patch.object(executor, '_find_contract_by_delta',
                          side_effect=returns):
            record = executor.execute(signal_hedge_spread, 480.0)

        assert record is not None
        assert record.status == 'DRY_RUN'
        assert len(record.legs) == 2
        assert record.legs[0]['side'] == 'BUY'
        assert record.legs[0]['strike'] == 475
        assert record.legs[1]['side'] == 'SELL'
        assert record.legs[1]['strike'] == 465


class TestTradeRecord:
    def test_to_db_dict(self, signal_hedge_now):
        record = TradeRecord(
            signal=signal_hedge_now,
            legs=[{'strike': 470, 'right': 'P', 'side': 'BUY',
                   'qty': 1, 'fill_price': 3.50}],
            entry_ts=datetime(2026, 4, 13, 15, 31),
            entry_spot=480.0,
            entry_cost=350.0,
            status='OPEN',
        )
        d = record.to_db_dict()
        assert d['symbol'] == 'QQQ'
        assert d['action'] == 'HEDGE_NOW'
        assert d['structure'] == 'outright_put'
        assert d['entry_cost'] == 350.0
        assert d['status'] == 'OPEN'
        assert d['gex_regime'] == 'negative'

    def test_format_trade_result(self, signal_hedge_now):
        record = TradeRecord(
            signal=signal_hedge_now,
            legs=[{'strike': 470, 'right': 'P', 'expiry': '20260528',
                   'side': 'BUY', 'qty': 1, 'fill_price': 3.50, 'delta': -0.25}],
            entry_ts=datetime(2026, 4, 13, 15, 31),
            entry_spot=480.0,
            entry_cost=350.0,
            status='OPEN',
        )
        text = format_trade_result(record)
        assert 'HEDGE_NOW' in text
        assert '470P' in text
        assert '3.50' in text


class TestMaxPositions:
    def test_position_limit(self, mock_ib, signal_hedge_now):
        executor = HedgeExecutor(ib=mock_ib, dry_run=True, max_positions=0)
        # Mock DB returning 0 open positions — but limit is 0
        with patch.object(executor, 'get_open_positions', return_value=0):
            # max_positions=0 means no positions allowed
            result = executor.execute(signal_hedge_now, 480.0)
        assert result is None
