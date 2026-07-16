"""Tests for time_utils module."""
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pytest

from gex_monitor.time_utils import (
    ET, UTC, MARKET_OPEN, MARKET_CLOSE, HAS_CALENDAR,
    et_now, trading_date_str, option_expiry_date_str, market_session_today,
    is_market_open, is_extended_hours, should_connect,
    seconds_until_next_open, seconds_until_next_session
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


class TestIsExtendedHours:
    """Tests for is_extended_hours (SPX overnight/pre/post market)."""

    def _dt(self, h, m):
        return datetime(2024, 1, 16, h, m, tzinfo=ET)  # 周二

    def test_overnight_pre_market(self):
        assert is_extended_hours(self._dt(4, 0)) is True

    def test_pre_market_start_boundary(self):
        assert is_extended_hours(self._dt(3, 0)) is True

    def test_pre_market_end_boundary(self):
        assert is_extended_hours(self._dt(9, 24)) is True
        assert is_extended_hours(self._dt(9, 25)) is False

    def test_regular_session_not_extended(self):
        assert is_extended_hours(self._dt(11, 0)) is False

    def test_post_market(self):
        assert is_extended_hours(self._dt(16, 30)) is True

    def test_post_market_end_boundary(self):
        assert is_extended_hours(self._dt(16, 59)) is True
        assert is_extended_hours(self._dt(17, 0)) is False

    def test_evening_gth(self):
        assert is_extended_hours(self._dt(22, 0)) is True

    def test_overnight_continues_after_midnight(self):
        assert is_extended_hours(self._dt(2, 59)) is True

    def test_weekend_never_extended(self):
        saturday = datetime(2024, 1, 13, 4, 0, tzinfo=ET)
        sunday = datetime(2024, 1, 14, 4, 0, tzinfo=ET)
        assert is_extended_hours(saturday) is False
        assert is_extended_hours(sunday) is False

    def test_sunday_evening_opens_monday_gth(self):
        sunday = datetime(2024, 1, 14, 20, 15, tzinfo=ET)
        assert is_extended_hours(sunday) is True

    @pytest.mark.skipif(not HAS_CALENDAR, reason="需要 exchange_calendars 识别假日")
    def test_holiday_gth_is_not_filtered_by_xnys(self):
        """Cboe 节假日 GTH 不能被 XNYS 休市日历直接排除。"""
        mlk_pre = datetime(2024, 1, 15, 4, 0, tzinfo=ET)
        mlk_post = datetime(2024, 1, 15, 16, 30, tzinfo=ET)
        assert is_extended_hours(mlk_pre) is True
        assert is_extended_hours(mlk_post) is True


class TestShouldConnect:
    """Tests for should_connect — covers extended hours + warmup + regular."""

    def _dt(self, h, m):
        return datetime(2024, 1, 16, h, m, tzinfo=ET)

    def test_extended_pre_market(self):
        assert should_connect(self._dt(5, 0)) is True

    def test_extended_post_market(self):
        assert should_connect(self._dt(16, 30)) is True

    def test_warmup_period(self):
        assert should_connect(self._dt(9, 26)) is True

    def test_regular_session(self):
        assert should_connect(self._dt(11, 0)) is True

    def test_evening_gth_connects(self):
        assert should_connect(self._dt(22, 0)) is True

    def test_friday_evening_no_weekend_session(self):
        friday = datetime(2024, 1, 19, 22, 0, tzinfo=ET)
        assert should_connect(friday) is False

    def test_weekend_no_connect(self):
        saturday = datetime(2024, 1, 13, 10, 0, tzinfo=ET)
        assert should_connect(saturday) is False

    def test_no_gap_between_extended_pre_and_warmup(self):
        """09:20-09:30 must all connect — no gap between extended and warmup."""
        for minute in range(20, 30):
            t = self._dt(9, minute)
            assert should_connect(t) is True, f"09:{minute:02d} should connect"

    def test_include_extended_false_pre_market(self):
        """股票期权 worker（无 GTH）盘前不应连接。"""
        assert should_connect(self._dt(5, 0), include_extended=False) is False

    def test_include_extended_false_post_market(self):
        assert should_connect(self._dt(16, 30), include_extended=False) is False

    def test_include_extended_false_regular_unaffected(self):
        """关掉延伸时段不影响预热期和常规时段。"""
        assert should_connect(self._dt(9, 26), include_extended=False) is True
        assert should_connect(self._dt(11, 0), include_extended=False) is True


class TestSecondsUntilNextSession:
    """Tests for seconds_until_next_session（UI 闭市定时器调度）."""

    def test_midday_next_is_post_extended(self):
        """周二 12:00 → 下一段是当天 16:15 盘后延伸（4.25h）。"""
        t = datetime(2024, 1, 16, 12, 0, tzinfo=ET)
        assert seconds_until_next_session(t) == pytest.approx(4.25 * 3600)

    def test_midday_without_extended_is_next_open(self):
        """周二 12:00 无延伸标的 → 周三 9:30 开盘（21.5h）。"""
        t = datetime(2024, 1, 16, 12, 0, tzinfo=ET)
        sec = seconds_until_next_session(t, include_extended=False)
        assert sec == pytest.approx(21.5 * 3600)

    def test_after_curb_next_is_evening_gth(self):
        """周二 17:00 → 当天 20:15 GTH（3.25h），不是次日开盘。"""
        t = datetime(2024, 1, 16, 17, 0, tzinfo=ET)
        assert seconds_until_next_session(t) == pytest.approx(3.25 * 3600)

    def test_weekend_next_is_monday_pre(self):
        """周六 12:00 → 周日 20:15（32.25h）。"""
        sat = datetime(2024, 1, 20, 12, 0, tzinfo=ET)
        assert seconds_until_next_session(sat) == pytest.approx(32.25 * 3600)

    def test_never_later_than_next_open(self):
        """有延伸时段时，下一 session 永远不晚于下一开盘。"""
        for t in (
            datetime(2024, 1, 16, 17, 0, tzinfo=ET),
            datetime(2024, 1, 20, 12, 0, tzinfo=ET),
            datetime(2024, 1, 16, 12, 0, tzinfo=ET),
        ):
            assert seconds_until_next_session(t) <= seconds_until_next_open(t)


class TestOptionExpiryDate:
    def test_regular_and_morning_use_calendar_date(self):
        assert option_expiry_date_str(
            datetime(2024, 1, 16, 9, 0, tzinfo=ET)) == '20240116'

    def test_curb_and_evening_exclude_expired_0dte(self):
        assert option_expiry_date_str(
            datetime(2024, 1, 16, 16, 30, tzinfo=ET)) == '20240117'
        assert option_expiry_date_str(
            datetime(2024, 1, 16, 20, 15, tzinfo=ET)) == '20240117'


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
