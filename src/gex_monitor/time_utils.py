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

# Cboe 指数期权延伸交易时段（美东时间）。
# GTH:  前一日 20:15 - 交易日 09:25
# Curb: 交易日 16:15 - 17:00
# 不用 XNYS 日历过滤 GTH：Cboe 在部分美股节假日仍开放 GTH。
EXTENDED_OVERNIGHT_START = dtime(20, 15)
EXTENDED_PRE_END = dtime(9, 25)
EXTENDED_POST_START = dtime(16, 15)
EXTENDED_POST_END = dtime(17, 0)

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


def trading_date_str(now: datetime = None) -> str:
    """返回当前交易日期字符串 YYYYMMDD"""
    return (now or et_now()).strftime('%Y%m%d')


def option_expiry_date_str(now: datetime = None) -> str:
    """返回当前时段可选的最早期权到期日。

    GTH 夜盘属于下一业务日；Curb 时当日 0DTE 已到期，
    两种情况都应排除当日 expiry。pick_expiry 会再跳到 chain 中
    下一个实际存在的到期日。
    """
    now = now or et_now()
    if now.time() >= EXTENDED_POST_START:
        return (now.date() + timedelta(days=1)).strftime('%Y%m%d')
    return now.strftime('%Y%m%d')


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
    """判断当前是否在正常交易时段内（9:30-16:00 ET）"""
    now = now or et_now()
    sess = market_session_today(now)
    if sess is None:
        return False
    o, c = sess
    return o <= now <= c


def is_extended_hours(now: datetime = None) -> bool:
    """判断当前是否在 SPX 期权延伸时段（夜盘/盘前/盘后）

    覆盖: 前一日 20:15-09:25 ET 和 16:15-17:00 ET。
    周日 20:15 是周一 GTH 的开始；周五 20:15 之后不开周六时段。
    注意这是全局时间谓词，只对支持 GTH 的指数期权（SPX/XSP）有意义，
    是否对某个标的生效由 SymbolConfig.extended_hours 决定。
    """
    now = now or et_now()
    t = now.time()

    weekday = now.weekday()  # Monday=0, Sunday=6
    if t >= EXTENDED_OVERNIGHT_START:
        return weekday in (6, 0, 1, 2, 3)
    if t < EXTENDED_PRE_END:
        return weekday in (0, 1, 2, 3, 4)
    if EXTENDED_POST_START <= t < EXTENDED_POST_END:
        return weekday in (0, 1, 2, 3, 4)
    return False


def should_connect(now: datetime = None, warmup_minutes: int = 5,
                   include_extended: bool = True) -> bool:
    """
    判断是否应该建立 IB 连接

    在开盘前 warmup_minutes 分钟就连接，以获取盘前价格用于预热

    Args:
        now: 指定时间，默认当前美东时间
        warmup_minutes: 开盘前多少分钟连接
        include_extended: 延伸时段是否算连接时段
                          （股票期权无 GTH，对应 worker 应传 False）

    Returns:
        True 如果应该连接（盘前预热期或交易时段）
    """
    now = now or et_now()

    # 延伸时段直接连接
    if include_extended and is_extended_hours(now):
        return True

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


def seconds_until_next_session(now: datetime = None,
                               include_extended: bool = True) -> float:
    """
    计算距下一个数据时段开始的秒数

    数据时段 = 常规开盘 (9:30)，以及 include_extended 时的 GTH
    (20:15 夜盘) / Curb (16:15) 起点。

    Raises:
        RuntimeError: 10 天内找不到交易日（来自 seconds_until_next_open）
    """
    now = now or et_now()
    candidates = [seconds_until_next_open(now)]
    if not include_extended:
        return candidates[0]

    for d in range(10):
        day = now.date() + timedelta(days=d)
        for start in (EXTENDED_POST_START, EXTENDED_OVERNIGHT_START):
            dt = datetime.combine(day, start, tzinfo=ET)
            # 用 is_extended_hours 验证该起点确实落在周常规延伸时段。
            if dt > now and is_extended_hours(dt):
                candidates.append((dt - now).total_seconds())
        if len(candidates) > 1:
            break  # 后面天数只会更晚，找到当天的就够了
    return min(candidates)
