"""Parquet 文件存储模块

优化策略:
1. 内存缓冲: 数据先写入内存 buffer，达到阈值或定时 flush 时才写磁盘
2. PyArrow 优化: snappy 压缩 + 多线程写入
3. 延迟读取: 只在首次 flush 时读取旧文件，之后保持内存状态
"""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .time_utils import ET, UTC, et_now

log = logging.getLogger(__name__)

# 写入优化参数
BUFFER_FLUSH_THRESHOLD = 100  # 缓冲记录数阈值
BUFFER_FLUSH_INTERVAL = 60.0  # 缓冲最大保留时间（秒）
PARQUET_ROW_GROUP_SIZE = 10000  # Parquet row group 大小
PARQUET_COMPRESSION = 'snappy'  # 压缩算法（snappy: 速度快，gzip: 压缩率高）


def _normalize_ts_to_utc(df: "pd.DataFrame") -> "pd.DataFrame":
    """将 df['ts'] 转换为 UTC 时区（用于存储前）"""
    if 'ts' not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        return df
    df = df.copy()
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize(ET).dt.tz_convert(UTC)
    else:
        df['ts'] = df['ts'].dt.tz_convert(UTC)
    return df


def _normalize_ts_to_et(df: "pd.DataFrame") -> "pd.DataFrame":
    """将 df['ts'] 转换为 ET 时区（用于读取后）"""
    if 'ts' not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        return df
    df = df.copy()
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize(UTC).dt.tz_convert(ET)
    else:
        df['ts'] = df['ts'].dt.tz_convert(ET)
    return df


def _atomic_write_parquet(df: "pd.DataFrame", path: Path) -> None:
    """原子写入 parquet 文件（先写临时文件再 rename）

    使用 PyArrow 优化:
    - snappy 压缩（速度/压缩率平衡）
    - 合理的 row group 大小
    - 多线程写入
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table, tmp,
        compression=PARQUET_COMPRESSION,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        use_dictionary=True,  # 对重复值使用字典编码
        write_statistics=True,  # 写入统计信息，加速查询
    )
    tmp.replace(path)


def _merge_and_write(path: Path, new_df: "pd.DataFrame", key_cols: list[str],
                     io_lock: threading.Lock) -> None:
    """合并新旧数据并写入文件"""
    if new_df.empty:
        return
    with io_lock:
        if path.exists():
            try:
                old = pd.read_parquet(path)
                combined = pd.concat([old, new_df], ignore_index=True)
            except Exception as e:
                log.warning(f"读取旧 parquet 失败，覆盖: {e}")
                combined = new_df
        else:
            combined = new_df
        combined = combined.drop_duplicates(subset=key_cols, keep='last')
        combined = _normalize_ts_to_utc(combined)
        if 'ts' in combined.columns:
            combined = combined.sort_values('ts').reset_index(drop=True)
        _atomic_write_parquet(combined, path)


def _read_parquet_with_lock(path: Path, io_lock: threading.Lock) -> "pd.DataFrame":
    """读取 parquet 文件（带锁）"""
    with io_lock:
        return pd.read_parquet(path)


def read_parquet_et(path: Path, io_lock: Optional[threading.Lock] = None) -> "pd.DataFrame":
    """读取 parquet 文件并转换时区为 ET"""
    if io_lock is not None:
        df = _read_parquet_with_lock(path, io_lock)
    else:
        df = pd.read_parquet(path)
    return _normalize_ts_to_et(df)


@dataclass
class WriteBuffer:
    """内存写缓冲区

    累积数据直到达到阈值或超时，然后批量写入磁盘。
    避免每次 persist 都读取旧文件。
    """
    path: Path
    key_cols: list[str]
    io_lock: threading.Lock
    threshold: int = BUFFER_FLUSH_THRESHOLD
    max_age: float = BUFFER_FLUSH_INTERVAL

    # 内部状态
    _records: list = field(default_factory=list)
    _last_flush: float = field(default_factory=time.time)
    _disk_loaded: bool = False
    _disk_data: Optional[pd.DataFrame] = None

    def append(self, records: list[dict]) -> None:
        """添加记录到缓冲区"""
        if not records:
            return
        self._records.extend(records)

    def should_flush(self) -> bool:
        """检查是否需要 flush"""
        if not self._records:
            return False
        if len(self._records) >= self.threshold:
            return True
        if time.time() - self._last_flush > self.max_age:
            return True
        return False

    def flush(self) -> int:
        """将缓冲区数据写入磁盘，返回写入记录数"""
        if not self._records:
            return 0

        records_to_write = self._records
        self._records = []
        count = len(records_to_write)

        new_df = pd.DataFrame(records_to_write)

        with self.io_lock:
            # 首次 flush 时加载磁盘数据
            if not self._disk_loaded and self.path.exists():
                try:
                    self._disk_data = pd.read_parquet(self.path)
                except Exception as e:
                    log.warning(f"读取旧 parquet 失败，忽略: {e}")
                    self._disk_data = None
                self._disk_loaded = True

            # 合并数据
            if self._disk_data is not None:
                combined = pd.concat([self._disk_data, new_df], ignore_index=True)
            else:
                combined = new_df

            # 去重、排序
            combined = combined.drop_duplicates(subset=self.key_cols, keep='last')
            combined = _normalize_ts_to_utc(combined)
            if 'ts' in combined.columns:
                combined = combined.sort_values('ts').reset_index(drop=True)

            # 写入磁盘
            _atomic_write_parquet(combined, self.path)

            # 更新内存缓存（保留最近数据用于下次合并）
            self._disk_data = combined

        self._last_flush = time.time()
        return count

    def force_flush(self) -> int:
        """强制 flush，不管阈值"""
        return self.flush()

    def pending_count(self) -> int:
        """返回待写入的记录数"""
        return len(self._records)


class StorageManager:
    """数据存储管理器

    优化:
    - 内存缓冲: 数据先写入 buffer，达到阈值或超时才写磁盘
    - 延迟读取: 只在首次 flush 时读取旧文件
    - PyArrow 压缩: snappy + 字典编码
    """

    def __init__(self, data_dir: Path | str,
                 buffer_threshold: int = BUFFER_FLUSH_THRESHOLD,
                 buffer_max_age: float = BUFFER_FLUSH_INTERVAL):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 缓冲配置
        self._buffer_threshold = buffer_threshold
        self._buffer_max_age = buffer_max_age

        # 实例级别的锁和线程池，避免全局状态
        self._io_lock = threading.Lock()
        self._buffer_lock = threading.Lock()  # 保护 buffers
        self._persist_lock = threading.Lock()  # 保护 _persist_futures
        self._persist_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='persist')
        self._persist_futures: dict[str, Future] = {}  # symbol -> future

        # 写缓冲区: (symbol, date, type) -> WriteBuffer
        self._buffers: dict[tuple[str, str, str], WriteBuffer] = {}

        # 可用日期缓存
        self._dates_cache: dict[str, tuple[float, list[str]]] = {}  # symbol -> (mtime, dates)
        self._dates_cache_ttl = 30.0  # 缓存 30 秒

    def _get_buffer(self, symbol: str, date_str: str, data_type: str,
                    key_cols: list[str]) -> WriteBuffer:
        """获取或创建指定的写缓冲区"""
        key = (symbol, date_str, data_type)
        with self._buffer_lock:
            if key not in self._buffers:
                path = self.data_dir / f'{data_type}_{symbol}_{date_str}.parquet'
                self._buffers[key] = WriteBuffer(
                    path=path,
                    key_cols=key_cols,
                    io_lock=self._io_lock,
                    threshold=self._buffer_threshold,
                    max_age=self._buffer_max_age,
                )
            return self._buffers[key]

    def persist_sync(self, symbol: str, hist: list[dict], ohlc: list[dict],
                     strikes: list[dict] | None = None) -> None:
        """同步落盘（使用缓冲区，达到阈值才写磁盘）"""
        date_str = et_now().strftime('%Y%m%d')
        try:
            # 添加到缓冲区
            if hist:
                buf = self._get_buffer(symbol, date_str, 'gex', ['ts'])
                buf.append(hist)
                if buf.should_flush():
                    count = buf.flush()
                    log.debug(f"[{symbol}] Flushed {count} gex records")

            if ohlc:
                buf = self._get_buffer(symbol, date_str, 'ohlc', ['ts'])
                buf.append(ohlc)
                if buf.should_flush():
                    count = buf.flush()
                    log.debug(f"[{symbol}] Flushed {count} ohlc records")

            if strikes:
                buf = self._get_buffer(symbol, date_str, 'strikes', ['ts', 'strike', 'right'])
                buf.append(strikes)
                if buf.should_flush():
                    count = buf.flush()
                    log.debug(f"[{symbol}] Flushed {count} strikes records")

        except Exception as e:
            log.error(f"persist failed for {symbol}: {e}")

    def persist_async(self, symbol: str, hist: list[dict], ohlc: list[dict],
                      strikes: list[dict] | None = None) -> None:
        """异步落盘，不阻塞调用方。每个 symbol 独立跟踪。"""
        with self._persist_lock:
            # 检查该 symbol 上一次是否还在写
            future = self._persist_futures.get(symbol)
            if future is not None and not future.done():
                return  # 跳过，上次还没完成

            # 提交新任务
            self._persist_futures[symbol] = self._persist_executor.submit(
                self.persist_sync, symbol, hist, ohlc, strikes
            )

    def list_available_dates(self, symbol: str, use_cache: bool = True) -> list[str]:
        """列出指定标的的所有可用日期（带缓存）"""
        import time
        now = time.time()

        if use_cache and symbol in self._dates_cache:
            cached_time, cached_dates = self._dates_cache[symbol]
            if now - cached_time < self._dates_cache_ttl:
                return cached_dates

        files = sorted(self.data_dir.glob(f'ohlc_{symbol}_*.parquet'))
        dates = [f.stem.split('_')[-1] for f in files]
        self._dates_cache[symbol] = (now, dates)
        return dates

    def list_available_strikes_dates(self, symbol: str) -> list[str]:
        """列出有 strikes 数据的日期（用于回放）"""
        files = sorted(self.data_dir.glob(f'strikes_{symbol}_*.parquet'))
        return [f.stem.split('_')[-1] for f in files]

    def load_day_ohlc(self, symbol: str, date_str: str) -> pd.DataFrame | None:
        """加载指定日期的 OHLC 数据"""
        p = self.data_dir / f'ohlc_{symbol}_{date_str}.parquet'
        if not p.exists():
            return None
        return read_parquet_et(p, self._io_lock)

    def resample_5min(self, ohlc_df: pd.DataFrame) -> pd.DataFrame | None:
        """将 OHLC 重采样为 5 分钟"""
        if ohlc_df is None or ohlc_df.empty:
            return None
        df = ohlc_df.set_index('ts')
        agg = df.resample('5min').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        return agg

    def persist_strikes_sync(self, symbol: str, strikes_data: list[dict]) -> None:
        """
        存储 strike-level GEX 数据（用于回放，使用缓冲区）

        strikes_data 格式: [{'ts': datetime, 'strike': float, 'right': str,
                            'gex': float, 'gamma': float, 'oi': int, 'iv': float}, ...]
        """
        if not strikes_data:
            return
        date_str = et_now().strftime('%Y%m%d')
        try:
            buf = self._get_buffer(symbol, date_str, 'strikes', ['ts', 'strike', 'right'])
            buf.append(strikes_data)
            if buf.should_flush():
                count = buf.flush()
                log.debug(f"[{symbol}] Flushed {count} strikes records (direct call)")
        except Exception as e:
            log.error(f"persist strikes failed for {symbol}: {e}")

    def load_day_strikes(self, symbol: str, date_str: str) -> pd.DataFrame | None:
        """加载指定日期的 strike-level GEX 数据"""
        p = self.data_dir / f'strikes_{symbol}_{date_str}.parquet'
        if not p.exists():
            return None
        return read_parquet_et(p, self._io_lock)

    def load_day_gex(self, symbol: str, date_str: str) -> pd.DataFrame | None:
        """加载指定日期的聚合 GEX 数据"""
        p = self.data_dir / f'gex_{symbol}_{date_str}.parquet'
        if not p.exists():
            return None
        return read_parquet_et(p, self._io_lock)

    def get_strikes_at_time(self, symbol: str, date_str: str,
                            target_ts: pd.Timestamp) -> pd.DataFrame | None:
        """获取指定时间点的 strike-level 数据"""
        strikes_df = self.load_day_strikes(symbol, date_str)
        if strikes_df is None or strikes_df.empty:
            return None

        # 找到最接近 target_ts 的时间点
        unique_ts = strikes_df['ts'].unique()
        if len(unique_ts) == 0:
            return None

        closest_ts = min(unique_ts, key=lambda t: abs((t - target_ts).total_seconds()))
        return strikes_df[strikes_df['ts'] == closest_ts]

    def get_replay_timestamps(self, symbol: str, date_str: str) -> list:
        """获取可回放的时间戳列表"""
        strikes_df = self.load_day_strikes(symbol, date_str)
        if strikes_df is None or strikes_df.empty:
            return []
        return sorted(strikes_df['ts'].unique())

    # ==================== OI 快照存储 ====================

    def save_oi_snapshot(self, symbol: str, date_str: str, oi_data: dict[float, dict]) -> None:
        """
        保存当日收盘 OI 快照

        Args:
            symbol: 标的代码
            date_str: 日期 YYYYMMDD
            oi_data: {strike: {'call_oi': int, 'put_oi': int}, ...}
        """
        if not oi_data:
            return
        rows = []
        for strike, data in oi_data.items():
            rows.append({
                'strike': strike,
                'call_oi': data.get('call_oi', 0),
                'put_oi': data.get('put_oi', 0),
            })
        df = pd.DataFrame(rows)
        path = self.data_dir / f'oi_snapshot_{symbol}_{date_str}.parquet'
        with self._io_lock:
            _atomic_write_parquet(df, path)
        log.info(f"[{symbol}] Saved OI snapshot: {len(rows)} strikes")

    def load_oi_snapshot(self, symbol: str, date_str: str) -> dict[float, dict] | None:
        """
        加载指定日期的 OI 快照

        Returns:
            {strike: {'call_oi': int, 'put_oi': int}, ...} 或 None
        """
        path = self.data_dir / f'oi_snapshot_{symbol}_{date_str}.parquet'
        if not path.exists():
            return None
        with self._io_lock:
            df = pd.read_parquet(path)
        result = {}
        for _, row in df.iterrows():
            result[row['strike']] = {
                'call_oi': int(row['call_oi']),
                'put_oi': int(row['put_oi']),
            }
        return result

    def get_previous_trading_day(self, date_str: str) -> str | None:
        """获取上一个有 OI 快照的交易日"""
        files = sorted(self.data_dir.glob('oi_snapshot_*_*.parquet'), reverse=True)
        for f in files:
            parts = f.stem.split('_')
            if len(parts) >= 3:
                file_date = parts[-1]
                if file_date < date_str:
                    return file_date
        return None

    def flush_all_buffers(self) -> int:
        """强制 flush 所有缓冲区"""
        with self._buffer_lock:
            buffers = list(self._buffers.values())

        total_flushed = 0
        for buf in buffers:
            try:
                count = buf.force_flush()
                total_flushed += count
            except Exception as e:
                log.error(f"Failed to flush buffer {buf.path}: {e}")

        if total_flushed > 0:
            log.info(f"Flushed {total_flushed} records from {len(buffers)} buffers")
        return total_flushed

    def get_buffer_stats(self) -> dict:
        """获取缓冲区统计信息"""
        with self._buffer_lock:
            stats = {}
            for key, buf in self._buffers.items():
                symbol, date_str, data_type = key
                stats[f"{symbol}/{date_str}/{data_type}"] = {
                    'pending': buf.pending_count(),
                    'last_flush': buf._last_flush,
                    'disk_loaded': buf._disk_loaded,
                }
            return stats

    def shutdown(self) -> None:
        """关闭存储管理器，flush 缓冲区并等待落盘完成"""
        # 先 flush 所有缓冲区
        log.info("Flushing all buffers before shutdown...")
        self.flush_all_buffers()

        # 等待异步任务完成
        with self._persist_lock:
            for symbol, future in self._persist_futures.items():
                if future is not None and not future.done():
                    try:
                        future.result(timeout=5)
                    except Exception as e:
                        log.warning(f"Persist for {symbol} did not complete: {e}")
            self._persist_futures.clear()
        self._persist_executor.shutdown(wait=True, cancel_futures=False)

        # 清理缓冲区
        with self._buffer_lock:
            self._buffers.clear()


# ==================== 分段标注存储 ====================
SEGMENT_COLUMNS = ['id', 'date', 'start_ts', 'end_ts', 'symbol',
                   'label', 'note', 'labeled_at']


class SegmentStorage:
    """分段标注存储"""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.segments_file = self.data_dir / 'segments.parquet'
        self._io_lock = threading.Lock()

    def load_segments(self) -> pd.DataFrame:
        """加载所有分段标注"""
        if self.segments_file.exists():
            df = read_parquet_et(self.segments_file, self._io_lock)
            if 'id' not in df.columns:
                df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
                df = df[SEGMENT_COLUMNS]
                with self._io_lock:
                    _atomic_write_parquet(df, self.segments_file)
            return df
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    def save_segment(self, date_str: str, start_ts, end_ts,
                     symbol: str, label: str, note: str) -> pd.DataFrame:
        """保存一个分段标注"""
        df = self.load_segments()
        new_row = pd.DataFrame([{
            'id': str(uuid.uuid4()),
            'date': date_str,
            'start_ts': pd.Timestamp(start_ts),
            'end_ts': pd.Timestamp(end_ts),
            'symbol': symbol,
            'label': label,
            'note': note or '',
            'labeled_at': et_now().isoformat(timespec='seconds'),
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        with self._io_lock:
            _atomic_write_parquet(df, self.segments_file)
        return df

    def delete_segments_by_ids(self, ids: list[str]) -> None:
        """删除指定 ID 的分段"""
        if not ids:
            return
        df = self.load_segments()
        df = df[~df['id'].isin(ids)]
        with self._io_lock:
            _atomic_write_parquet(df, self.segments_file)
