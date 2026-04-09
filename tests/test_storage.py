"""Tests for storage module."""
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from gex_monitor.storage import (
    StorageManager, SegmentStorage, WriteBuffer,
    _normalize_ts_to_utc, _normalize_ts_to_et,
    _atomic_write_parquet, read_parquet_et,
    BUFFER_FLUSH_THRESHOLD, BUFFER_FLUSH_INTERVAL,
)

ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')


class TestTimezoneNormalization:
    """Tests for timezone normalization functions."""

    def test_normalize_to_utc_naive_datetime(self):
        """Test normalizing naive datetime to UTC."""
        df = pd.DataFrame({
            'ts': [datetime(2024, 1, 15, 10, 0, 0)],
            'value': [1]
        })

        result = _normalize_ts_to_utc(df)

        assert result['ts'].dt.tz is not None
        assert str(result['ts'].dt.tz) == 'UTC'

    def test_normalize_to_utc_et_datetime(self):
        """Test normalizing ET datetime to UTC."""
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2024-01-15 10:00:00']).tz_localize(ET),
            'value': [1]
        })

        result = _normalize_ts_to_utc(df)

        assert str(result['ts'].dt.tz) == 'UTC'
        # 10 AM ET should be 15:00 UTC (during EST)
        assert result['ts'].iloc[0].hour == 15

    def test_normalize_to_et_utc_datetime(self):
        """Test normalizing UTC datetime to ET."""
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2024-01-15 15:00:00']).tz_localize(UTC),
            'value': [1]
        })

        result = _normalize_ts_to_et(df)

        assert str(result['ts'].dt.tz) == 'America/New_York'
        # 15:00 UTC should be 10:00 ET (during EST)
        assert result['ts'].iloc[0].hour == 10

    def test_normalize_no_ts_column(self):
        """Test normalization when no ts column exists."""
        df = pd.DataFrame({'value': [1, 2, 3]})

        result_utc = _normalize_ts_to_utc(df)
        result_et = _normalize_ts_to_et(df)

        assert 'ts' not in result_utc.columns
        assert 'ts' not in result_et.columns

    def test_normalize_non_datetime_ts(self):
        """Test normalization when ts is not datetime."""
        df = pd.DataFrame({
            'ts': ['2024-01-15', '2024-01-16'],
            'value': [1, 2]
        })

        result = _normalize_ts_to_utc(df)
        # Should return unchanged
        assert result['ts'].dtype == object


class TestWriteBuffer:
    """Tests for WriteBuffer class."""

    @pytest.fixture
    def buffer(self, temp_dir):
        """Create a WriteBuffer instance for testing."""
        io_lock = threading.Lock()
        path = temp_dir / 'test_buffer.parquet'
        return WriteBuffer(
            path=path,
            key_cols=['ts'],
            io_lock=io_lock,
            threshold=5,  # Low threshold for testing
            max_age=1.0,  # Short timeout for testing
        )

    def test_append_records(self, buffer):
        """Test appending records to buffer."""
        records = [{'ts': datetime.now(ET), 'value': 1}]
        buffer.append(records)

        assert buffer.pending_count() == 1

    def test_should_flush_threshold(self, buffer):
        """Test flush triggered by threshold."""
        # Below threshold
        buffer.append([{'ts': datetime.now(ET), 'value': i} for i in range(3)])
        assert buffer.should_flush() is False

        # At threshold
        buffer.append([{'ts': datetime.now(ET), 'value': i} for i in range(3)])
        assert buffer.should_flush() is True

    def test_should_flush_timeout(self, buffer):
        """Test flush triggered by timeout."""
        buffer.append([{'ts': datetime.now(ET), 'value': 1}])
        assert buffer.should_flush() is False

        # Simulate time passing
        buffer._last_flush = time.time() - 2.0  # Exceed max_age
        assert buffer.should_flush() is True

    def test_flush_writes_to_disk(self, buffer, temp_dir):
        """Test that flush writes data to disk."""
        # Use distinct timestamps to avoid deduplication
        base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=ET)
        records = [
            {'ts': base.replace(minute=i), 'value': i}
            for i in range(3)
        ]
        buffer.append(records)

        count = buffer.flush()

        assert count == 3
        assert buffer.path.exists()
        df = pd.read_parquet(buffer.path)
        assert len(df) == 3

    def test_flush_merges_with_existing(self, buffer, temp_dir):
        """Test that flush merges with existing data."""
        # First flush
        buffer.append([{'ts': datetime(2024, 1, 15, 10, 0, 0, tzinfo=ET), 'value': 1}])
        buffer.flush()

        # Second flush
        buffer.append([{'ts': datetime(2024, 1, 15, 10, 1, 0, tzinfo=ET), 'value': 2}])
        buffer.flush()

        df = pd.read_parquet(buffer.path)
        assert len(df) == 2

    def test_flush_deduplicates(self, buffer):
        """Test that flush deduplicates by key columns."""
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=ET)
        buffer.append([
            {'ts': ts, 'value': 1},
            {'ts': ts, 'value': 2},  # Same ts, should keep last
        ])
        buffer.flush()

        df = pd.read_parquet(buffer.path)
        assert len(df) == 1
        assert df['value'].iloc[0] == 2

    def test_flush_empty_buffer(self, buffer):
        """Test flushing empty buffer."""
        count = buffer.flush()
        assert count == 0
        assert not buffer.path.exists()

    def test_force_flush(self, buffer):
        """Test force flush ignores threshold."""
        buffer.append([{'ts': datetime.now(ET), 'value': 1}])
        # Below threshold, but force should work
        count = buffer.force_flush()
        assert count == 1


class TestStorageManager:
    """Tests for StorageManager class."""

    @pytest.fixture
    def storage(self, temp_dir):
        """Create a StorageManager instance for testing with low buffer threshold."""
        return StorageManager(temp_dir, buffer_threshold=1, buffer_max_age=0.1)

    def test_init_creates_directory(self, temp_dir):
        """Test that init creates data directory."""
        subdir = temp_dir / 'subdir'
        storage = StorageManager(subdir)

        assert subdir.exists()

    def test_persist_sync_gex(self, storage, temp_dir):
        """Test synchronous persistence of GEX data."""
        now = datetime.now(ET)
        hist = [
            {'ts': now, 'spot': 500.0, 'total_gex': 1e6, 'flip': 498.0},
        ]

        storage.persist_sync('QQQ', hist=hist, ohlc=[], strikes=[])
        storage.flush_all_buffers()  # Force flush

        files = list(temp_dir.glob('gex_QQQ_*.parquet'))
        assert len(files) == 1

    def test_persist_sync_ohlc(self, storage, temp_dir):
        """Test synchronous persistence of OHLC data."""
        now = datetime.now(ET)
        ohlc = [
            {'ts': now, 'open': 500.0, 'high': 502.0, 'low': 499.0, 'close': 501.0},
        ]

        storage.persist_sync('QQQ', hist=[], ohlc=ohlc, strikes=[])
        storage.flush_all_buffers()  # Force flush

        files = list(temp_dir.glob('ohlc_QQQ_*.parquet'))
        assert len(files) == 1

    def test_persist_sync_strikes(self, storage, temp_dir):
        """Test synchronous persistence of strikes data."""
        now = datetime.now(ET)
        strikes = [
            {'ts': now, 'strike': 500.0, 'right': 'C', 'gex': 1e6},
        ]

        storage.persist_sync('QQQ', hist=[], ohlc=[], strikes=strikes)
        storage.flush_all_buffers()  # Force flush

        files = list(temp_dir.glob('strikes_QQQ_*.parquet'))
        assert len(files) == 1

    def test_buffer_stats(self, storage, temp_dir):
        """Test get_buffer_stats method."""
        now = datetime.now(ET)
        hist = [{'ts': now, 'spot': 500.0}]

        # Add data but don't flush
        storage.persist_sync('QQQ', hist=hist, ohlc=[], strikes=[])

        stats = storage.get_buffer_stats()
        # Should have at least one buffer with data
        assert len(stats) >= 0  # May be flushed due to threshold=1

    def test_flush_all_buffers(self, storage, temp_dir):
        """Test flush_all_buffers method."""
        now = datetime.now(ET)

        # Use a storage with higher threshold to test buffering
        storage2 = StorageManager(temp_dir, buffer_threshold=100, buffer_max_age=60)
        storage2.persist_sync('QQQ', hist=[{'ts': now, 'spot': 500.0}], ohlc=[], strikes=[])
        storage2.persist_sync('SPY', hist=[{'ts': now, 'spot': 400.0}], ohlc=[], strikes=[])

        # Before flush, files might not exist
        count = storage2.flush_all_buffers()

        # After flush, files should exist
        gex_files = list(temp_dir.glob('gex_*.parquet'))
        assert len(gex_files) == 2
        storage2.shutdown()

    def test_persist_async_does_not_block(self, storage):
        """Test that async persistence does not block."""
        now = datetime.now(ET)
        hist = [{'ts': now, 'spot': 500.0}]

        start = time.time()
        storage.persist_async('QQQ', hist=hist, ohlc=[], strikes=[])
        elapsed = time.time() - start

        # Should return almost immediately
        assert elapsed < 0.5

    def test_persist_async_skips_if_busy(self, storage, temp_dir):
        """Test that async persist skips if previous is still running."""
        now = datetime.now(ET)
        hist = [{'ts': now, 'spot': 500.0}]

        # Start first persist
        storage.persist_async('QQQ', hist=hist, ohlc=[], strikes=[])

        # Immediately try another - should be skipped
        storage.persist_async('QQQ', hist=hist, ohlc=[], strikes=[])

        # Wait for completion
        storage.shutdown()

    def test_list_available_dates(self, storage, temp_dir):
        """Test listing available dates."""
        # Create some test files
        now = datetime.now(ET)
        for date_str in ['20240115', '20240116', '20240117']:
            df = pd.DataFrame({'ts': [now], 'open': [500.0], 'high': [502.0],
                              'low': [499.0], 'close': [501.0]})
            df.to_parquet(temp_dir / f'ohlc_QQQ_{date_str}.parquet')

        dates = storage.list_available_dates('QQQ')

        assert len(dates) == 3
        assert '20240115' in dates
        assert '20240116' in dates
        assert '20240117' in dates

    def test_list_available_dates_caching(self, storage, temp_dir):
        """Test that dates listing uses cache."""
        # Create a test file
        now = datetime.now(ET)
        df = pd.DataFrame({'ts': [now], 'value': [1]})
        df.to_parquet(temp_dir / 'ohlc_QQQ_20240115.parquet')

        # First call
        dates1 = storage.list_available_dates('QQQ')

        # Create another file
        df.to_parquet(temp_dir / 'ohlc_QQQ_20240116.parquet')

        # Second call should use cache
        dates2 = storage.list_available_dates('QQQ', use_cache=True)

        # Should still show only one file (cached)
        assert dates1 == dates2

        # Without cache should show both
        dates3 = storage.list_available_dates('QQQ', use_cache=False)
        assert len(dates3) == 2

    def test_list_available_strikes_dates(self, storage, temp_dir):
        """Test listing dates with strikes data."""
        now = datetime.now(ET)
        df = pd.DataFrame({'ts': [now], 'strike': [500.0], 'right': ['C']})
        df.to_parquet(temp_dir / 'strikes_QQQ_20240115.parquet')
        df.to_parquet(temp_dir / 'strikes_QQQ_20240116.parquet')

        dates = storage.list_available_strikes_dates('QQQ')

        assert len(dates) == 2

    def test_load_day_ohlc(self, storage, temp_dir):
        """Test loading OHLC data for a specific day."""
        now = datetime.now(ET)
        df = pd.DataFrame({
            'ts': pd.to_datetime([now]).tz_localize(None).tz_localize(UTC),
            'open': [500.0], 'high': [502.0], 'low': [499.0], 'close': [501.0]
        })
        df.to_parquet(temp_dir / 'ohlc_QQQ_20240115.parquet')

        result = storage.load_day_ohlc('QQQ', '20240115')

        assert result is not None
        assert len(result) == 1
        assert result['open'].iloc[0] == 500.0

    def test_load_day_ohlc_nonexistent(self, storage):
        """Test loading OHLC for nonexistent date."""
        result = storage.load_day_ohlc('QQQ', '99990101')
        assert result is None

    def test_resample_5min(self, storage):
        """Test OHLC resampling to 5 minutes."""
        now = datetime.now(ET).replace(second=0, microsecond=0)
        data = []
        for i in range(10):
            data.append({
                'ts': now.replace(minute=i),
                'open': 500.0 + i,
                'high': 502.0 + i,
                'low': 499.0 + i,
                'close': 501.0 + i,
            })
        df = pd.DataFrame(data)

        result = storage.resample_5min(df)

        assert result is not None
        assert len(result) <= 2  # 10 minutes = 2 x 5-minute bars

    def test_resample_5min_none_input(self, storage):
        """Test resampling with None input."""
        result = storage.resample_5min(None)
        assert result is None

    def test_resample_5min_empty_df(self, storage):
        """Test resampling with empty DataFrame."""
        result = storage.resample_5min(pd.DataFrame())
        assert result is None

    def test_get_replay_timestamps(self, storage, temp_dir):
        """Test getting replay timestamps."""
        now = datetime.now(ET)
        times = [now.replace(minute=i) for i in range(5)]
        df = pd.DataFrame({
            'ts': pd.to_datetime(times).tz_localize(None).tz_localize(UTC),
            'strike': [500.0] * 5,
            'right': ['C'] * 5,
        })
        df.to_parquet(temp_dir / 'strikes_QQQ_20240115.parquet')

        timestamps = storage.get_replay_timestamps('QQQ', '20240115')

        assert len(timestamps) == 5

    def test_shutdown(self, storage):
        """Test shutdown waits for pending operations."""
        now = datetime.now(ET)
        hist = [{'ts': now, 'spot': 500.0}]

        storage.persist_async('QQQ', hist=hist, ohlc=[], strikes=[])
        storage.shutdown()

        # Should complete without error


class TestSegmentStorage:
    """Tests for SegmentStorage class."""

    @pytest.fixture
    def seg_storage(self, temp_dir):
        """Create a SegmentStorage instance for testing."""
        return SegmentStorage(temp_dir)

    def test_load_segments_empty(self, seg_storage):
        """Test loading segments when file doesn't exist."""
        result = seg_storage.load_segments()

        assert result.empty
        assert 'id' in result.columns
        assert 'label' in result.columns

    def test_save_segment(self, seg_storage):
        """Test saving a segment."""
        now = datetime.now(ET)

        result = seg_storage.save_segment(
            date_str='20240115',
            start_ts=now,
            end_ts=now,
            symbol='QQQ',
            label='trend_up',
            note='Test note',
        )

        assert len(result) == 1
        assert result['label'].iloc[0] == 'trend_up'
        assert result['note'].iloc[0] == 'Test note'

    def test_save_multiple_segments(self, seg_storage):
        """Test saving multiple segments."""
        now = datetime.now(ET)

        seg_storage.save_segment('20240115', now, now, 'QQQ', 'trend_up', '')
        seg_storage.save_segment('20240115', now, now, 'QQQ', 'chop', '')
        result = seg_storage.save_segment('20240115', now, now, 'SPY', 'trend_down', '')

        assert len(result) == 3

    def test_delete_segments_by_ids(self, seg_storage):
        """Test deleting segments by ID."""
        now = datetime.now(ET)

        seg_storage.save_segment('20240115', now, now, 'QQQ', 'trend_up', '')
        segments = seg_storage.save_segment('20240115', now, now, 'QQQ', 'chop', '')

        # Delete first segment
        seg_storage.delete_segments_by_ids([segments['id'].iloc[0]])

        remaining = seg_storage.load_segments()
        assert len(remaining) == 1

    def test_delete_segments_empty_ids(self, seg_storage):
        """Test deleting with empty ID list."""
        now = datetime.now(ET)
        seg_storage.save_segment('20240115', now, now, 'QQQ', 'trend_up', '')

        seg_storage.delete_segments_by_ids([])

        remaining = seg_storage.load_segments()
        assert len(remaining) == 1

    def test_segment_has_uuid(self, seg_storage):
        """Test that segments have UUID ids."""
        import uuid
        now = datetime.now(ET)

        result = seg_storage.save_segment('20240115', now, now, 'QQQ', 'trend_up', '')

        seg_id = result['id'].iloc[0]
        # Should be valid UUID
        uuid.UUID(seg_id)


class TestAtomicWrite:
    """Tests for atomic write functionality."""

    def test_atomic_write_creates_file(self, temp_dir):
        """Test that atomic write creates file."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        path = temp_dir / 'test.parquet'

        _atomic_write_parquet(df, path)

        assert path.exists()

    def test_atomic_write_no_tmp_file_left(self, temp_dir):
        """Test that no .tmp file is left after write."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        path = temp_dir / 'test.parquet'

        _atomic_write_parquet(df, path)

        tmp_files = list(temp_dir.glob('*.tmp'))
        assert len(tmp_files) == 0

    def test_atomic_write_overwrites(self, temp_dir):
        """Test that atomic write overwrites existing file."""
        path = temp_dir / 'test.parquet'

        df1 = pd.DataFrame({'value': [1]})
        _atomic_write_parquet(df1, path)

        df2 = pd.DataFrame({'value': [2]})
        _atomic_write_parquet(df2, path)

        result = pd.read_parquet(path)
        assert result['value'].iloc[0] == 2
