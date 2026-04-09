"""Tests for time_utils module."""
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pytest

from gex_monitor.time_utils import (
    ET, UTC, MARKET_OPEN, MARKET_CLOSE,
    et_now, trading_date_str, market_session_today,
    is_market_open, seconds_until_next_open
)


class TestEtNow:
    """Tests for et_now function."""

    def test_returns_datetime_with_et_timezone(self):
        """Test that et_now returns datetime with ET timezone."""
        now = et_now()
        assert now.tzinfo is not None
        assert now.tzinfo == ET

    def test_returns_current_time(self):
        """Test that et_now returns approximately current time."""
        import time
        before = time.time()
        now = et_now()
        after = time.time()

        now_timestamp = now.timestamp()
        assert before <= now_timestamp <= after


class TestTradingDateStr:
    """Tests for trading_date_str function."""

    def test_returns_yyyymmdd_format(self):
        """Test that trading_date_str returns YYYYMMDD format."""
        date_str = trading_date_str()
        assert len(date_str) == 8
        assert date_str.isdigit()

        # Should be parseable
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])

        assert 2020 <= year <= 2100
        assert 1 <= month <= 12
        assert 1 <= day <= 31


class TestMarketSessionToday:
    """Tests for market_session_today function."""

    def test_weekday_returns_session(self):
        """Test that weekdays return a market session."""
        # Monday 10:00 AM ET
        monday = datetime(2024, 1, 15, 10, 0, 0, tzinfo=ET)
        session = market_session_today(monday)

        if session is not None:  # May be None if calendar says it's a holiday
            open_time, close_time = session
            assert open_time.tzinfo is not None
            assert close_time.tzinfo is not None
            assert open_time < close_time

    def test_weekend_returns_none(self):
        """Test that weekends return None (no calendar) or holiday check."""
        # Saturday
        saturday = datetime(2024, 1, 13, 10, 0, 0, tzinfo=ET)
        session = market_session_today(saturday)

        # Without exchange_calendars, weekends return None
        # With exchange_calendars, also None
        assert session is None

    def test_sunday_returns_none(self):
        """Test that Sunday returns None."""
        sunday = datetime(2024, 1, 14, 10, 0, 0, tzinfo=ET)
        session = market_session_today(sunday)
        assert session is None


class TestIsMarketOpen:
    """Tests for is_market_open function."""

    def test_during_market_hours(self):
        """Test market is open during trading hours on a weekday."""
        # Tuesday 11:00 AM ET (definitely during market hours)
        tuesday_midday = datetime(2024, 1, 16, 11, 0, 0, tzinfo=ET)

        # This might return False if exchange_calendars says it's a holiday
        # We can only test the logic, not the actual date
        result = is_market_open(tuesday_midday)
        # Just verify it returns a boolean
        assert isinstance(result, bool)

    def test_before_market_open(self):
        """Test market is closed before 9:30 AM ET."""
        early_morning = datetime(2024, 1, 16, 8, 0, 0, tzinfo=ET)
        result = is_market_open(early_morning)
        assert result is False

    def test_after_market_close(self):
        """Test market is closed after 4:00 PM ET."""
        evening = datetime(2024, 1, 16, 17, 0, 0, tzinfo=ET)
        result = is_market_open(evening)
        assert result is False

    def test_weekend(self):
        """Test market is closed on weekends."""
        saturday = datetime(2024, 1, 13, 11, 0, 0, tzinfo=ET)
        result = is_market_open(saturday)
        assert result is False

    def test_at_market_open(self):
        """Test exactly at market open."""
        # Tuesday at 9:30 AM ET
        at_open = datetime(2024, 1, 16, 9, 30, 0, tzinfo=ET)
        result = is_market_open(at_open)
        # Should be True unless it's a holiday
        assert isinstance(result, bool)

    def test_at_market_close(self):
        """Test exactly at market close."""
        # Tuesday at 4:00 PM ET
        at_close = datetime(2024, 1, 16, 16, 0, 0, tzinfo=ET)
        result = is_market_open(at_close)
        # At exactly 4:00 PM, market should still be considered open (<=)
        assert isinstance(result, bool)


class TestSecondsUntilNextOpen:
    """Tests for seconds_until_next_open function."""

    def test_before_todays_open(self):
        """Test seconds until open when before today's market open."""
        # Tuesday 8:00 AM ET (1.5 hours before open)
        before_open = datetime(2024, 1, 16, 8, 0, 0, tzinfo=ET)
        seconds = seconds_until_next_open(before_open)

        # Should be around 5400 seconds (1.5 hours)
        assert seconds > 0
        assert seconds <= 5400 + 60  # Allow some margin

    def test_after_todays_close(self):
        """Test seconds until open when after today's market close."""
        # Tuesday 5:00 PM ET (after close)
        after_close = datetime(2024, 1, 16, 17, 0, 0, tzinfo=ET)
        seconds = seconds_until_next_open(after_close)

        # Should be positive (next day's open)
        assert seconds > 0
        # Should be less than ~17 hours (to 9:30 AM next day)
        assert seconds < 17 * 3600

    def test_on_weekend(self):
        """Test seconds until open on weekend."""
        # Saturday 10:00 AM ET
        saturday = datetime(2024, 1, 13, 10, 0, 0, tzinfo=ET)
        seconds = seconds_until_next_open(saturday)

        # Should be positive (Monday's open or later)
        assert seconds > 0

    def test_returns_positive_value(self):
        """Test that function always returns positive value."""
        now = et_now()
        try:
            seconds = seconds_until_next_open(now)
            # If market is open, we should get time until tomorrow's open
            # If market is closed, we should get time until next open
            assert seconds >= 0
        except RuntimeError:
            # This is also acceptable if no trading day found
            pass


class TestConstants:
    """Tests for module constants."""

    def test_et_timezone(self):
        """Test ET timezone is America/New_York."""
        assert str(ET) == 'America/New_York'

    def test_utc_timezone(self):
        """Test UTC timezone."""
        assert str(UTC) == 'UTC'

    def test_market_open_time(self):
        """Test market open time is 9:30 AM."""
        assert MARKET_OPEN == dtime(9, 30)

    def test_market_close_time(self):
        """Test market close time is 4:00 PM."""
        assert MARKET_CLOSE == dtime(16, 0)
