"""Tests for state module."""
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from gex_monitor.state import StateManager, StateRegistry, GEXSnapshot, OHLCBar

ET = ZoneInfo('America/New_York')


class TestStateManager:
    """Tests for StateManager class."""

    @pytest.fixture
    def state_manager(self):
        """Create a StateManager instance for testing."""
        return StateManager(symbol='TEST', max_history=100, max_logs=10)

    def test_init(self, state_manager):
        """Test StateManager initialization."""
        assert state_manager.symbol == 'TEST'
        assert state_manager.max_history == 100
        assert state_manager.max_logs == 10
        assert state_manager._spot == 0
        assert state_manager._connected is False

    def test_update_basic(self, state_manager):
        """Test basic state update."""
        df = pd.DataFrame([
            {'strike': 500, 'right': 'C', 'gex': 1e6, 'gamma': 0.1, 'oi': 1000, 'iv': 0.25}
        ])

        state_manager.update(
            spot=500.0,
            total_gex=1e6,
            gamma_flip=498.0,
            call_gex=1.5e6,
            put_gex=-0.5e6,
            atm_iv_pct=25.0,
            expiry='20240115',
            is_true_0dte=True,
            df=df,
        )

        snapshot = state_manager.get_snapshot()
        assert snapshot['spot'] == 500.0
        assert snapshot['total_gex'] == 1e6
        assert snapshot['gamma_flip'] == 498.0
        assert snapshot['connected'] is True
        assert snapshot['market_open'] is True

    def test_update_appends_history(self, state_manager):
        """Test that update appends to history."""
        df = pd.DataFrame()

        for i in range(5):
            state_manager.update(
                spot=500.0 + i,
                total_gex=1e6,
                gamma_flip=498.0,
                call_gex=1.5e6,
                put_gex=-0.5e6,
                atm_iv_pct=25.0,
                expiry='20240115',
                is_true_0dte=True,
                df=df,
            )

        history, version = state_manager.get_history_for_resample()
        assert len(history) == 5
        assert version == 5

    def test_get_history_returns_copy(self, state_manager):
        """Test that get_history_for_resample returns a copy, not reference."""
        df = pd.DataFrame()
        state_manager.update(
            spot=500.0, total_gex=1e6, gamma_flip=498.0,
            call_gex=1.5e6, put_gex=-0.5e6, atm_iv_pct=25.0,
            expiry='20240115', is_true_0dte=True, df=df,
        )

        history1, _ = state_manager.get_history_for_resample()
        history2, _ = state_manager.get_history_for_resample()

        # Should be different list objects
        assert history1 is not history2

        # Modifying one should not affect the other
        history1.append({'test': 'data'})
        assert len(history1) != len(history2)

    def test_max_history_limit(self, state_manager):
        """Test that history respects max_history limit."""
        df = pd.DataFrame()

        # Add more than max_history items
        for i in range(150):
            state_manager.update(
                spot=500.0 + i,
                total_gex=1e6,
                gamma_flip=498.0,
                call_gex=1.5e6,
                put_gex=-0.5e6,
                atm_iv_pct=25.0,
                expiry='20240115',
                is_true_0dte=True,
                df=df,
            )

        history, _ = state_manager.get_history_for_resample()
        assert len(history) <= 100  # max_history

    def test_set_status(self, state_manager):
        """Test set_status method."""
        state_manager.set_status(market_open=False, connected=True, updated='test')

        snapshot = state_manager.get_snapshot()
        assert snapshot['market_open'] is False
        assert snapshot['connected'] is True
        assert snapshot['updated'] == 'test'

    def test_log(self, state_manager):
        """Test log method."""
        state_manager.log('info', 'Test message')
        state_manager.log('warning', 'Warning message')

        logs = state_manager.get_logs()
        assert len(logs) == 2
        assert logs[0][0] == 'info'
        assert logs[0][2] == 'Test message'

    def test_max_logs_limit(self, state_manager):
        """Test that logs respect max_logs limit."""
        for i in range(20):
            state_manager.log('info', f'Message {i}')

        logs = state_manager.get_logs()
        assert len(logs) <= 10  # max_logs

    def test_get_df_returns_copy(self, state_manager):
        """Test that get_df returns a copy."""
        original_df = pd.DataFrame([{
            'strike': 500, 'right': 'C', 'gex': 1e6,
            'gamma': 0.1, 'oi': 1000, 'iv': 0.25
        }])
        state_manager.update(
            spot=500.0, total_gex=1e6, gamma_flip=498.0,
            call_gex=1.5e6, put_gex=-0.5e6, atm_iv_pct=25.0,
            expiry='20240115', is_true_0dte=True, df=original_df,
        )

        df1 = state_manager.get_df()
        df2 = state_manager.get_df()

        # Should be different DataFrame objects
        assert df1 is not df2

    def test_get_persist_data(self, state_manager):
        """Test get_persist_data method."""
        df = pd.DataFrame([{'strike': 500, 'right': 'C', 'gex': 1e6}])
        state_manager.update(
            spot=500.0, total_gex=1e6, gamma_flip=498.0,
            call_gex=1.5e6, put_gex=-0.5e6, atm_iv_pct=25.0,
            expiry='20240115', is_true_0dte=True, df=df,
        )

        hist, ohlc, strikes = state_manager.get_persist_data()

        assert isinstance(hist, list)
        assert isinstance(ohlc, list)
        assert isinstance(strikes, list)
        assert len(hist) == 1

    def test_resample_history_empty(self, state_manager):
        """Test resample_history with empty history."""
        result = state_manager.resample_history('1min')
        assert result.empty

    def test_resample_history_insufficient_data(self, state_manager):
        """Test resample_history with only one data point."""
        df = pd.DataFrame()
        state_manager.update(
            spot=500.0, total_gex=1e6, gamma_flip=498.0,
            call_gex=1.5e6, put_gex=-0.5e6, atm_iv_pct=25.0,
            expiry='20240115', is_true_0dte=True, df=df,
        )

        result = state_manager.resample_history('1min')
        assert result.empty  # Need at least 2 points

    def test_thread_safety(self, state_manager):
        """Test that StateManager is thread-safe."""
        errors = []
        df = pd.DataFrame()

        def writer():
            try:
                for i in range(100):
                    state_manager.update(
                        spot=500.0 + i,
                        total_gex=1e6,
                        gamma_flip=498.0,
                        call_gex=1.5e6,
                        put_gex=-0.5e6,
                        atm_iv_pct=25.0,
                        expiry='20240115',
                        is_true_0dte=True,
                        df=df,
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    state_manager.get_snapshot()
                    state_manager.get_history_for_resample()
                    state_manager.get_logs()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestStateRegistry:
    """Tests for StateRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a StateRegistry instance for testing."""
        return StateRegistry()

    def test_register_new_symbol(self, registry):
        """Test registering a new symbol."""
        state = registry.register('QQQ', max_history=1000)

        assert state is not None
        assert isinstance(state, StateManager)
        assert state.symbol == 'QQQ'

    def test_register_existing_symbol(self, registry):
        """Test registering an existing symbol returns same instance."""
        state1 = registry.register('QQQ', max_history=1000)
        state2 = registry.register('QQQ', max_history=2000)  # Different max_history

        assert state1 is state2  # Same instance

    def test_get_registered_symbol(self, registry):
        """Test getting a registered symbol."""
        registry.register('QQQ')
        state = registry.get('QQQ')

        assert state is not None
        assert state.symbol == 'QQQ'

    def test_get_unregistered_symbol(self, registry):
        """Test getting an unregistered symbol returns None."""
        state = registry.get('UNKNOWN')
        assert state is None

    def test_list_symbols(self, registry):
        """Test listing all registered symbols."""
        registry.register('QQQ')
        registry.register('SPY')
        registry.register('SPX')

        symbols = registry.list_symbols()

        assert len(symbols) == 3
        assert 'QQQ' in symbols
        assert 'SPY' in symbols
        assert 'SPX' in symbols

    def test_get_all(self, registry):
        """Test getting all state managers."""
        registry.register('QQQ')
        registry.register('SPY')

        all_managers = registry.get_all()

        assert len(all_managers) == 2
        assert 'QQQ' in all_managers
        assert 'SPY' in all_managers
        assert isinstance(all_managers['QQQ'], StateManager)

    def test_thread_safety(self, registry):
        """Test that StateRegistry is thread-safe."""
        errors = []

        def registerer(name):
            try:
                for i in range(50):
                    registry.register(f'{name}_{i}')
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    registry.list_symbols()
                    registry.get_all()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=registerer, args=('A',)),
            threading.Thread(target=registerer, args=('B',)),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestDataclasses:
    """Tests for dataclass definitions."""

    def test_gex_snapshot(self):
        """Test GEXSnapshot dataclass."""
        now = datetime.now(ET)
        snapshot = GEXSnapshot(
            ts=now,
            spot=500.0,
            total_gex=1e6,
            flip=498.0,
            call_gex=1.5e6,
            put_gex=-0.5e6,
            atm_iv_pct=25.0,
        )

        assert snapshot.ts == now
        assert snapshot.spot == 500.0
        assert snapshot.total_gex == 1e6

    def test_ohlc_bar(self):
        """Test OHLCBar dataclass."""
        now = datetime.now(ET)
        bar = OHLCBar(
            ts=now,
            open=500.0,
            high=502.0,
            low=499.0,
            close=501.0,
        )

        assert bar.open == 500.0
        assert bar.high == 502.0
        assert bar.low == 499.0
        assert bar.close == 501.0
