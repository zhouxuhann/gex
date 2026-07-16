"""db_storage 模块测试（mock psycopg2，不需要真实 DB）"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gex_monitor.config import DatabaseConfig


# Mock psycopg2 before importing db_storage
@pytest.fixture(autouse=True)
def mock_psycopg2():
    """全局 mock psycopg2"""
    with patch.dict('sys.modules', {
        'psycopg2': MagicMock(),
        'psycopg2.extras': MagicMock(),
    }):
        # 需要重新设置 HAS_PSYCOPG2
        import gex_monitor.db_storage as mod
        mod.HAS_PSYCOPG2 = True
        mod.psycopg2 = MagicMock()
        mod.psycopg2.extras = MagicMock()
        yield mod


@pytest.fixture
def config():
    return DatabaseConfig(
        enabled=True,
        host='localhost',
        port=5433,
        dbname='test_db',
        user='test_user',
        password='test_pass',
    )


@pytest.fixture
def db(mock_psycopg2, config):
    """创建带 mock 连接的 GEXDBStorage"""
    mock_conn = MagicMock()
    mock_psycopg2.psycopg2.connect.return_value = mock_conn
    storage = mock_psycopg2.GEXDBStorage(config)
    storage._conn = mock_conn
    return storage


class TestGEXDBStorageInit:
    def test_disabled_config(self, mock_psycopg2):
        config = DatabaseConfig(enabled=False)
        storage = mock_psycopg2.GEXDBStorage(config)
        assert not storage._enabled

    def test_enabled_connects(self, mock_psycopg2, config):
        mock_conn = MagicMock()
        mock_psycopg2.psycopg2.connect.return_value = mock_conn
        storage = mock_psycopg2.GEXDBStorage(config)
        assert storage._enabled

    def test_connect_failure_degrades(self, mock_psycopg2, config):
        mock_psycopg2.psycopg2.connect.side_effect = Exception("refused")
        storage = mock_psycopg2.GEXDBStorage(config)
        assert storage._conn is None
        assert not storage.is_available


class TestBufferAndFlush:
    def test_buffer_snapshot(self, db):
        record = {
            'symbol': 'QQQ', 'ts': datetime(2026, 4, 12, 10, 0),
            'spot': 480.0, 'total_gex': 1e6, 'call_gex': 5e5,
            'put_gex': -5e5, 'flip': 479.0,
            'call_wall': 485.0, 'put_wall': 475.0, 'max_pain': 480.0,
            'atm_iv_pct': 25.0, 'positive_gamma': True,
            'regime_code': 'long_gamma/above_flip/diffuse',
            'rr_25': 0.02, 'skew_slope': 0.1,
            'rr_25_zscore': 0.5, 'skew_signal': None,
        }
        db.buffer_snapshot(record)
        assert db.pending_count() == 1

    def test_flush_calls_execute_batch(self, db, mock_psycopg2):
        record = {
            'symbol': 'QQQ', 'ts': datetime(2026, 4, 12, 10, 0),
            'spot': 480.0, 'total_gex': 1e6, 'call_gex': 5e5,
            'put_gex': -5e5, 'flip': 479.0,
            'call_wall': None, 'put_wall': None, 'max_pain': None,
            'atm_iv_pct': None, 'positive_gamma': False,
            'regime_code': None, 'rr_25': None, 'skew_slope': None,
            'rr_25_zscore': None, 'skew_signal': None,
        }
        db.buffer_snapshot(record)

        # mock cursor
        mock_cursor = MagicMock()
        db._conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        db._conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = db.flush()
        assert count == 1
        assert db.pending_count() == 0
        mock_psycopg2.psycopg2.extras.execute_batch.assert_called_once()

    def test_flush_empty_buffer(self, db):
        assert db.flush() == 0

    def test_flush_no_connection(self, db, mock_psycopg2):
        db.buffer_snapshot({'symbol': 'QQQ', 'ts': datetime.now(), 'spot': 480})
        db._conn = None
        mock_psycopg2.psycopg2.connect.side_effect = Exception("refused")
        count = db.flush()
        assert count == 0

    def test_buffer_key_mapping(self, db):
        """验证 flip → gamma_flip 的 key 映射"""
        db.buffer_snapshot({'symbol': 'QQQ', 'ts': datetime.now(),
                           'spot': 480.0, 'flip': 479.5})
        records = list(db._buffer)
        assert records[0]['gamma_flip'] == 479.5

    def test_disabled_buffer_noop(self, mock_psycopg2):
        config = DatabaseConfig(enabled=False)
        storage = mock_psycopg2.GEXDBStorage(config)
        storage.buffer_snapshot({'symbol': 'QQQ'})
        assert storage.pending_count() == 0


class TestOHLCRead:
    def test_load_ohlc_returns_dataframe(self, db):
        mock_df = pd.DataFrame({
            'ts': pd.to_datetime(['2026-04-12 09:31:00', '2026-04-12 09:32:00']),
            'open': [480.0, 480.5],
            'high': [481.0, 481.5],
            'low': [479.5, 480.0],
            'close': [480.5, 481.0],
            'volume': [1000, 1200],
        })

        with patch('pandas.read_sql_query', return_value=mock_df):
            result = db.load_ohlc('QQQ', '20260412')
        assert result is not None
        assert len(result) == 2

    def test_load_ohlc_empty(self, db):
        with patch('pandas.read_sql_query', return_value=pd.DataFrame()):
            result = db.load_ohlc('QQQ', '20260412')
        assert result is None


class TestMarketBarWrite:
    def test_upsert_market_data_bars(self, db, mock_psycopg2):
        mock_cursor = MagicMock()
        db._conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        db._conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        records = [{
            'symbol': 'QQQ', 'bar_size': '1 min',
            'datetime': datetime(2026, 4, 12, 9, 30),
            'open': 500.0, 'high': 501.0, 'low': 499.0, 'close': 500.5,
            'volume': 1000, 'bar_count': 10, 'average': 500.2,
            'has_gaps': False, 'source': 'ib_historical_trades',
            'created_at': datetime(2026, 4, 12, 16, 1),
        }]

        count = db.upsert_market_data_bars(records)

        assert count == 1
        mock_psycopg2.psycopg2.extras.execute_batch.assert_called_once()

    def test_load_ohlc_no_connection(self, mock_psycopg2):
        config = DatabaseConfig(enabled=True, password='x')
        mock_psycopg2.psycopg2.connect.side_effect = Exception("refused")
        storage = mock_psycopg2.GEXDBStorage(config)
        assert storage.load_ohlc('QQQ', '20260412') is None


class TestListDates:
    def test_list_available_dates(self, db):
        from datetime import date
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (date(2026, 4, 10),),
            (date(2026, 4, 11),),
            (date(2026, 4, 12),),
        ]
        db._conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        db._conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        dates = db.list_available_dates('QQQ')
        assert dates == ['20260410', '20260411', '20260412']

    def test_list_dates_no_connection(self, mock_psycopg2):
        config = DatabaseConfig(enabled=True, password='x')
        mock_psycopg2.psycopg2.connect.side_effect = Exception("refused")
        storage = mock_psycopg2.GEXDBStorage(config)
        assert storage.list_available_dates('QQQ') == []


class TestShutdown:
    def test_shutdown_flushes_and_closes(self, db, mock_psycopg2):
        db.buffer_snapshot({'symbol': 'QQQ', 'ts': datetime.now(), 'spot': 480.0})

        mock_cursor = MagicMock()
        db._conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        db._conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        db.shutdown()
        # flush 应该被调用
        mock_psycopg2.psycopg2.extras.execute_batch.assert_called_once()
        # 连接应该关闭
        assert db._conn is None
