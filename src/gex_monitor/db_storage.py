"""
PostgreSQL 存储模块

职责:
  1. GEX snapshots 写入 gex_snapshots 表（batch upsert）
  2. OHLC 价格数据从 market_data_bars 读取（替代 parquet）

设计:
  - 连接失败不影响主循环（静默降级，日志警告）
  - 批量写入，减少 DB 压力
  - 复用 ibkr-data-store 的 PostgreSQL 实例 (localhost:5433)
"""
import logging
import threading
import warnings
from collections import deque
from datetime import datetime

import pandas as pd

from .config import DatabaseConfig

warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')

log = logging.getLogger(__name__)

# 尝试导入 psycopg2，不可用时降级
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    log.warning(
        "psycopg2 not installed — DB storage disabled. "
        "Install with: pip install psycopg2-binary"
    )

# batch upsert SQL
_UPSERT_SQL = """
INSERT INTO gex_snapshots (
    symbol, datetime, spot,
    total_gex, call_gex, put_gex, gamma_flip,
    call_wall, put_wall, max_pain,
    atm_iv_pct, positive_gamma, regime_code,
    rr_25, skew_slope, rr_25_zscore, skew_signal,
    partial
) VALUES (
    %(symbol)s, %(datetime)s, %(spot)s,
    %(total_gex)s, %(call_gex)s, %(put_gex)s, %(gamma_flip)s,
    %(call_wall)s, %(put_wall)s, %(max_pain)s,
    %(atm_iv_pct)s, %(positive_gamma)s, %(regime_code)s,
    %(rr_25)s, %(skew_slope)s, %(rr_25_zscore)s, %(skew_signal)s,
    %(partial)s
)
ON CONFLICT (symbol, datetime) DO UPDATE SET
    spot = EXCLUDED.spot,
    total_gex = EXCLUDED.total_gex,
    call_gex = EXCLUDED.call_gex,
    put_gex = EXCLUDED.put_gex,
    gamma_flip = EXCLUDED.gamma_flip,
    call_wall = EXCLUDED.call_wall,
    put_wall = EXCLUDED.put_wall,
    max_pain = EXCLUDED.max_pain,
    atm_iv_pct = EXCLUDED.atm_iv_pct,
    positive_gamma = EXCLUDED.positive_gamma,
    regime_code = EXCLUDED.regime_code,
    rr_25 = EXCLUDED.rr_25,
    skew_slope = EXCLUDED.skew_slope,
    rr_25_zscore = EXCLUDED.rr_25_zscore,
    skew_signal = EXCLUDED.skew_signal,
    partial = EXCLUDED.partial
"""

# OHLC 查询 SQL
_OHLC_QUERY = """
SELECT datetime as ts, open, high, low, close, volume,
       bar_count, average, has_gaps, source
FROM market_data_bars
WHERE symbol = %s AND bar_size = %s
  AND datetime::date = %s::date
ORDER BY datetime
"""

_OHLC_RANGE_QUERY = """
SELECT datetime as ts, open, high, low, close, volume,
       bar_count, average, has_gaps, source
FROM market_data_bars
WHERE symbol = %s AND bar_size = %s
  AND datetime BETWEEN %s AND %s
ORDER BY datetime
"""

_MARKET_BAR_UPSERT_SQL = """
INSERT INTO market_data_bars (
    symbol, bar_size, datetime, open, high, low, close, volume,
    bar_count, average, has_gaps, source, created_at
) VALUES (
    %(symbol)s, %(bar_size)s, %(datetime)s, %(open)s, %(high)s, %(low)s,
    %(close)s, %(volume)s, %(bar_count)s, %(average)s, %(has_gaps)s,
    %(source)s, %(created_at)s
)
ON CONFLICT (symbol, bar_size, datetime) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    bar_count = EXCLUDED.bar_count,
    average = EXCLUDED.average,
    has_gaps = EXCLUDED.has_gaps,
    source = EXCLUDED.source,
    created_at = EXCLUDED.created_at
"""


class GEXDBStorage:
    """
    PostgreSQL 存储层

    - 写: GEX snapshots batch upsert
    - 读: OHLC from market_data_bars
    - 连接失败静默降级
    """

    def __init__(self, config: DatabaseConfig):
        self._config = config
        self._conn = None
        self._lock = threading.Lock()
        self._buffer: deque[dict] = deque(maxlen=5000)
        self._enabled = config.enabled and HAS_PSYCOPG2

        if self._enabled:
            self._try_connect()

    def _try_connect(self) -> bool:
        """尝试连接数据库"""
        if not self._enabled:
            return False
        try:
            self._conn = psycopg2.connect(
                host=self._config.host,
                port=self._config.port,
                dbname=self._config.dbname,
                user=self._config.user,
                password=self._config.password,
                connect_timeout=5,
            )
            self._conn.autocommit = False
            log.info(
                f"DB connected: {self._config.host}:{self._config.port}"
                f"/{self._config.dbname}"
            )
            return True
        except Exception as e:
            log.warning(f"DB connect failed (will retry): {e}")
            self._conn = None
            return False

    def _ensure_connection(self) -> bool:
        """确保连接可用"""
        if not self._enabled:
            return False
        if self._conn is not None:
            try:
                # 轻量心跳检测
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return True
            except Exception:
                self._conn = None
        return self._try_connect()

    @property
    def is_available(self) -> bool:
        return self._enabled and self._conn is not None

    # ==================== 写: GEX Snapshots ====================

    def buffer_snapshot(self, record: dict) -> None:
        """
        缓冲一条 GEX snapshot

        record 需要的 key:
          symbol, ts, spot, total_gex, call_gex, put_gex, gamma_flip,
          call_wall, put_wall, max_pain, atm_iv_pct, positive_gamma,
          regime_code, rr_25, skew_slope, rr_25_zscore, skew_signal
        """
        if not self._enabled:
            return

        # 转换 key 名 (state 用 ts/flip, DB 用 datetime/gamma_flip)
        # numpy 类型转 Python 原生类型，防止 psycopg2 报错
        def _py(v):
            if v is None:
                return None
            if hasattr(v, 'item'):  # numpy scalar
                return v.item()
            return v

        db_record = {
            'symbol': record.get('symbol'),
            'datetime': record.get('ts'),
            'spot': _py(record.get('spot')),
            'total_gex': _py(record.get('total_gex')),
            'call_gex': _py(record.get('call_gex')),
            'put_gex': _py(record.get('put_gex')),
            'gamma_flip': _py(record.get('flip') or record.get('gamma_flip')),
            'call_wall': _py(record.get('call_wall')),
            'put_wall': _py(record.get('put_wall')),
            'max_pain': _py(record.get('max_pain')),
            'atm_iv_pct': _py(record.get('atm_iv_pct')),
            'positive_gamma': bool(record.get('positive_gamma', False)),
            'regime_code': record.get('regime_code'),
            'rr_25': _py(record.get('rr_25')),
            'skew_slope': _py(record.get('skew_slope')),
            'rr_25_zscore': _py(record.get('rr_25_zscore')),
            'skew_signal': record.get('skew_signal'),
            'partial': bool(record.get('partial', False)),
        }
        self._buffer.append(db_record)

    def flush(self) -> int:
        """
        将缓冲区写入数据库

        Returns:
            写入的行数，失败返回 0
        """
        if not self._buffer:
            return 0
        if not self._enabled:
            self._buffer.clear()
            return 0

        with self._lock:
            records = list(self._buffer)
            self._buffer.clear()

        if not self._ensure_connection():
            log.warning(f"DB flush skipped: no connection ({len(records)} records lost)")
            return 0

        try:
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, _UPSERT_SQL, records, page_size=100)
            self._conn.commit()
            log.debug(f"DB flush: {len(records)} gex snapshots")
            return len(records)
        except Exception as e:
            log.error(f"DB flush failed: {e}")
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            return 0

    def pending_count(self) -> int:
        return len(self._buffer)

    def upsert_market_data_bars(self, records: list[dict], page_size: int = 500) -> int:
        """批量写入官方 OHLC Bar。``datetime`` 使用美东 naive 时间。"""
        if not records or not self._ensure_connection():
            return 0
        try:
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur, _MARKET_BAR_UPSERT_SQL, records, page_size=page_size
                )
            self._conn.commit()
            return len(records)
        except Exception as e:
            log.error(f"Market bar upsert failed: {e}")
            try:
                self._conn.rollback()
            except Exception:
                self._conn = None
            return 0

    # ==================== 读: OHLC from market_data_bars ====================

    def load_ohlc(
        self, symbol: str, date_str: str, bar_size: str = '1 min'
    ) -> pd.DataFrame | None:
        """
        从 market_data_bars 读取指定日期的 OHLC

        Args:
            symbol: 标的代码
            date_str: 日期 YYYYMMDD
            bar_size: K 线周期 (默认 '1 min')

        Returns:
            DataFrame with columns [ts, open, high, low, close, volume,
            bar_count, average, has_gaps, source]
            或 None
        """
        if not self._ensure_connection():
            return None

        try:
            date = datetime.strptime(date_str, '%Y%m%d').date()
            df = pd.read_sql_query(
                _OHLC_QUERY, self._conn,
                params=[symbol, bar_size, date]
            )
            if df.empty:
                return None
            # 时区处理：DB 存的是 naive timestamp，加上 ET
            from .time_utils import ET
            if df['ts'].dt.tz is None:
                df['ts'] = df['ts'].dt.tz_localize(ET)
            return df
        except Exception as e:
            log.warning(f"DB OHLC read failed for {symbol}/{date_str}: {e}")
            return None

    def load_ohlc_range(
        self, symbol: str, start: str, end: str, bar_size: str = '1 min'
    ) -> pd.DataFrame | None:
        """
        从 market_data_bars 读取日期范围的 OHLC

        Args:
            symbol: 标的代码
            start/end: YYYYMMDD
            bar_size: K 线周期

        Returns:
            DataFrame 或 None
        """
        if not self._ensure_connection():
            return None

        try:
            start_dt = datetime.strptime(start, '%Y%m%d')
            end_dt = datetime.strptime(end, '%Y%m%d').replace(
                hour=23, minute=59, second=59
            )
            df = pd.read_sql_query(
                _OHLC_RANGE_QUERY, self._conn,
                params=[symbol, bar_size, start_dt, end_dt]
            )
            if df.empty:
                return None
            from .time_utils import ET
            if df['ts'].dt.tz is None:
                df['ts'] = df['ts'].dt.tz_localize(ET)
            return df
        except Exception as e:
            log.warning(f"DB OHLC range read failed: {e}")
            return None

    def load_gex_history(
        self, symbol: str, date_str: str
    ) -> pd.DataFrame | None:
        """
        从 gex_snapshots 读取指定日期的 GEX 历史

        用于回放和特征计算
        """
        if not self._ensure_connection():
            return None

        try:
            date = datetime.strptime(date_str, '%Y%m%d').date()
            df = pd.read_sql_query(
                """
                SELECT datetime as ts, spot, total_gex,
                       gamma_flip as flip, call_gex, put_gex,
                       atm_iv_pct, call_wall, put_wall,
                       positive_gamma, max_pain,
                       rr_25, skew_slope, rr_25_zscore
                FROM gex_snapshots
                WHERE symbol = %s AND datetime::date = %s::date
                ORDER BY datetime
                """,
                self._conn, params=[symbol, date]
            )
            if df.empty:
                return None
            from .time_utils import ET
            if df['ts'].dt.tz is None:
                df['ts'] = df['ts'].dt.tz_localize(ET)
            return df
        except Exception as e:
            log.warning(f"DB GEX history read failed: {e}")
            return None

    def list_available_dates(self, symbol: str) -> list[str]:
        """列出有 OHLC 数据的日期"""
        if not self._ensure_connection():
            return []
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT datetime::date as d
                    FROM market_data_bars
                    WHERE symbol = %s AND bar_size = '1 min'
                    ORDER BY d
                    """,
                    [symbol]
                )
                return [row[0].strftime('%Y%m%d') for row in cur.fetchall()]
        except Exception as e:
            log.warning(f"DB list dates failed: {e}")
            return []

    # ==================== 生命周期 ====================

    def shutdown(self) -> None:
        """关闭连接前 flush 缓冲"""
        if self._buffer:
            self.flush()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            log.info("DB connection closed")
