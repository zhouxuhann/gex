"""时区和交易日历工具"""
import logging
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

# 时区常量
ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# 美股交易日历（可选依赖）
try:
    import exchange_calendars as xcals
    XNYS = xcals.get_calendar("XNYS")
    HAS_CALENDAR = True
except ImportError:
    XNYS = None
    HAS_CALENDAR = False
    log.warning(
        "exchange_calendars not installed — using simple weekend check only. "
        "Market holidays (MLK Day, Good Friday, etc.) will NOT be detected. "
        "Install with: pip install exchange-calendars"
    )


def et_now() -> datetime:
    """返回当前美东时间"""
    return datetime.now(ET)


def trading_date_str() -> str:
    """返回当前交易日期字符串 YYYYMMDD"""
    return et_now().strftime('%Y%m%d')


def market_session_today(now: datetime = None) -> tuple[datetime, datetime] | None:
    """
    返回今日交易时段 (open, close)，非交易日返回 None

    Args:
        now: 指定时间，默认当前美东时间

    Returns:
        (open_dt, close_dt) 或 None
    """
    now = now or et_now()
    today = now.date()

    if HAS_CALENDAR:
        ts = pd.Timestamp(today)
        if not XNYS.is_session(ts):
            return None
        o = XNYS.session_open(ts).tz_convert(ET).to_pydatetime()
        c = XNYS.session_close(ts).tz_convert(ET).to_pydatetime()
        return o, c

    # 无日历时简单判断周末
    if now.weekday() >= 5:
        return None
    o = datetime.combine(today, MARKET_OPEN, tzinfo=ET)
    c = datetime.combine(today, MARKET_CLOSE, tzinfo=ET)
    return o, c


def is_market_open(now: datetime = None) -> bool:
    """判断当前是否在交易时段内"""
    now = now or et_now()
    sess = market_session_today(now)
    if sess is None:
        return False
    o, c = sess
    return o <= now <= c


def should_connect(now: datetime = None, warmup_minutes: int = 5) -> bool:
    """
    判断是否应该建立 IB 连接

    在开盘前 warmup_minutes 分钟就连接，以获取盘前价格用于预热

    Args:
        now: 指定时间，默认当前美东时间
        warmup_minutes: 开盘前多少分钟连接

    Returns:
        True 如果应该连接（盘前预热期或交易时段）
    """
    now = now or et_now()
    sess = market_session_today(now)
    if sess is None:
        return False
    o, c = sess
    warmup_start = o - timedelta(minutes=warmup_minutes)
    return warmup_start <= now <= c


def seconds_until_next_open(now: datetime = None) -> float:
    """
    计算距离下一个交易日开盘的秒数

    Raises:
        RuntimeError: 10 天内找不到交易日
    """
    now = now or et_now()
    sess = market_session_today(now)
    if sess is not None and now < sess[0]:
        return (sess[0] - now).total_seconds()

    d = now.date() + timedelta(days=1)
    for _ in range(10):
        probe = datetime.combine(d, dtime(0, 1), tzinfo=ET)
        sess = market_session_today(probe)
        if sess is not None:
            return (sess[0] - now).total_seconds()
        d += timedelta(days=1)
    raise RuntimeError("10 天内找不到下一个交易日，日历可能损坏")
