"""IB 官方股票 1 分钟 Bar 回补与实时采集（QQQ/SPY）。

盘中通过 ``reqHistoricalData(..., keepUpToDate=True)`` 接收 IB 更新中的 1m
TRADES Bar；收盘后再次按交易日请求完整历史 Bar，修正盘中最后一根和断线缺口。
数据库保存标准 ``market_data_bars``，Parquet 保存 ``official_ohlc_*`` 备份。
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from threading import Event
from typing import Iterable

import pandas as pd
from ib_insync import IB, Stock

from .config import AppConfig
from .db_storage import GEXDBStorage
from .storage import _atomic_write_parquet
from .time_utils import (
    ET,
    HAS_CALENDAR,
    UTC,
    XNYS,
    et_now,
    market_session_today,
    seconds_until_next_open,
)

log = logging.getLogger(__name__)

BAR_SIZE = '1 min'
WHAT_TO_SHOW = 'TRADES'
SOURCE = 'ib_historical_trades'
FINALIZE_RETRY_SEC = 60.0


def _session_bounds(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(datetime.strptime(date_str, '%Y%m%d').date())
    if HAS_CALENDAR and XNYS.is_session(day):
        return XNYS.session_open(day).tz_convert(ET), XNYS.session_close(day).tz_convert(ET)
    return (
        pd.Timestamp(datetime.combine(day.date(), dt_time(9, 30), tzinfo=ET)),
        pd.Timestamp(datetime.combine(day.date(), dt_time(16, 0), tzinfo=ET)),
    )


def _to_et(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    # formatDate=2 is UTC; naive values in tests/legacy mocks are treated as UTC.
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(ET)


def bar_to_record(bar, symbol: str = 'QQQ') -> dict:
    """把 ib_insync BarData 转换成 DB/Parquet 共用记录。"""
    ts = _to_et(getattr(bar, 'date'))
    naive_et = ts.to_pydatetime().replace(tzinfo=None)

    def number(name, default=None):
        value = getattr(bar, name, default)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)

    volume = getattr(bar, 'volume', 0)
    bar_count = getattr(bar, 'barCount', None)
    return {
        'symbol': symbol,
        'ts': ts.to_pydatetime(),
        'datetime': naive_et,
        'open': number('open', 0.0),
        'high': number('high', 0.0),
        'low': number('low', 0.0),
        'close': number('close', 0.0),
        'volume': int(volume or 0),
        'bar_count': int(bar_count) if bar_count is not None else None,
        'average': number('average'),
    }


def normalize_bars(
    bars: Iterable,
    symbol: str = 'QQQ',
    date_str: str | None = None,
) -> list[dict]:
    """过滤 RTH、按分钟去重并按时间排序。"""
    records: dict[datetime, dict] = {}
    target_date = date_str or ''
    start, end = _session_bounds(target_date) if target_date else (None, None)
    for bar in bars or []:
        try:
            record = bar_to_record(bar, symbol)
        except (TypeError, ValueError, AttributeError):
            continue
        ts = pd.Timestamp(record['ts'])
        if target_date and ts.strftime('%Y%m%d') != target_date:
            continue
        if start is not None and not (start <= ts < end):
            continue
        # IB 的 1m Bar 应该只有一个时间点；同一分钟更新时保留最后版本。
        minute_ts = ts.floor('min')
        record['ts'] = minute_ts.to_pydatetime()
        record['datetime'] = minute_ts.to_pydatetime().replace(tzinfo=None)
        records[record['datetime']] = record
    return [records[key] for key in sorted(records)]


def add_gap_flags(records: list[dict], expected_minutes: int) -> list[dict]:
    """给当天所有 Bar 写入统一的 has_gaps 标志。"""
    if not records:
        return records
    timestamps = pd.DatetimeIndex([r['ts'] for r in records]).sort_values()
    gaps = timestamps.to_series().diff().dt.total_seconds().dropna()
    has_gaps = len(records) != expected_minutes or bool((gaps > 60.5).any())
    for record in records:
        record['has_gaps'] = has_gaps
    return records


def is_complete_day(records: list[dict], date_str: str) -> bool:
    """Return whether a historical response contains the full RTH session."""
    if not records:
        return False
    start, end = _session_bounds(date_str)
    expected = int((end - start).total_seconds() // 60)
    return len(records) == expected and not records[0].get('has_gaps', True)


class OfficialQQQBarCollector:
    """采集并保存 QQQ 官方 1 分钟 TRADES Bar。"""

    def __init__(
        self,
        ib: IB,
        db_storage: GEXDBStorage | None,
        data_dir: str | Path,
        contract=None,
        symbol: str = 'QQQ',
        request_pause_sec: float = 1.0,
    ):
        self.ib = ib
        self.db_storage = db_storage
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.contract = contract or Stock(symbol, 'SMART', 'USD')
        self.request_pause_sec = max(0.2, request_pause_sec)

    def fetch_day(self, date_str: str) -> list[dict]:
        """从 IB 请求指定交易日的完整 RTH 1 分钟 Bar。"""
        _, close = _session_bounds(date_str)
        end_dt = close.to_pydatetime().replace(tzinfo=ET)
        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime=end_dt,
            durationStr='1 D',
            barSizeSetting=BAR_SIZE,
            whatToShow=WHAT_TO_SHOW,
            useRTH=True,
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )
        records = normalize_bars(bars, self.symbol, date_str)
        start, end = _session_bounds(date_str)
        expected = int((end - start).total_seconds() // 60)
        return add_gap_flags(records, expected)

    def _db_records(self, records: list[dict]) -> list[dict]:
        now = et_now().replace(tzinfo=None)
        return [
            {
                'symbol': self.symbol,
                'bar_size': BAR_SIZE,
                'datetime': r['datetime'],
                'open': r['open'],
                'high': r['high'],
                'low': r['low'],
                'close': r['close'],
                'volume': r['volume'],
                'bar_count': r['bar_count'],
                'average': r['average'],
                'has_gaps': r.get('has_gaps', False),
                'source': SOURCE,
                'created_at': now,
            }
            for r in records
        ]

    def _write_parquet(self, records: list[dict], date_str: str) -> Path | None:
        if not records:
            return None
        path = self.data_dir / f'official_ohlc_{self.symbol}_{date_str}.parquet'
        frame = pd.DataFrame(records).drop(columns=['datetime', 'symbol'], errors='ignore')
        _atomic_write_parquet(frame, path)
        return path

    def persist_day(self, date_str: str, records: list[dict]) -> int:
        """同时写数据库和本地官方 Parquet。"""
        if not records:
            log.warning('[%s] %s 没有收到官方 1m Bar', self.symbol, date_str)
            return 0
        db_count = 0
        if self.db_storage is not None:
            db_count = self.db_storage.upsert_market_data_bars(self._db_records(records))
        path = self._write_parquet(records, date_str)
        log.info(
            '[%s] official 1m %s: %d bars, db=%d, parquet=%s, gaps=%s',
            self.symbol,
            date_str,
            len(records),
            db_count,
            path,
            records[0].get('has_gaps', False),
        )
        return len(records)

    def backfill(self, start_date: str, end_date: str) -> dict[str, int]:
        """按交易日回补指定日期范围。"""
        start = pd.Timestamp(datetime.strptime(start_date, '%Y%m%d').date())
        end = pd.Timestamp(datetime.strptime(end_date, '%Y%m%d').date())
        if HAS_CALENDAR:
            sessions = XNYS.sessions_in_range(start, end)
            dates = [session.strftime('%Y%m%d') for session in sessions]
        else:
            dates = [
                day.strftime('%Y%m%d')
                for day in pd.date_range(start, end, freq='D')
                if day.weekday() < 5
            ]
        result: dict[str, int] = {}
        for date_str in dates:
            try:
                result[date_str] = self.persist_day(date_str, self.fetch_day(date_str))
            except Exception:
                log.exception('[%s] 回补 %s 失败', self.symbol, date_str)
                result[date_str] = 0
            time.sleep(self.request_pause_sec)
        return result

    def run_live(self, poll_sec: float = 5.0, stop_event: Event | None = None) -> None:
        """盘中订阅官方更新中的 1m Bar，收盘后回补最终版本。"""
        stop_event = stop_event or Event()
        bars = None
        current_date = None
        finalized_date = None
        last_signatures: dict[datetime, tuple] = {}
        last_parquet_write = 0.0
        latest_records: list[dict] = []
        try:
            while not stop_event.is_set():
                now = et_now()
                session = market_session_today(now)
                if session is None:
                    time.sleep(30)
                    continue
                open_dt, close_dt = session
                if now < open_dt:
                    time.sleep(min(30, max(1, (open_dt - now).total_seconds())))
                    continue
                if now >= close_dt:
                    date_str = now.strftime('%Y%m%d')
                    if finalized_date != date_str:
                        if bars is not None:
                            self.ib.cancelHistoricalData(bars)
                            bars = None
                            current_date = None
                            last_signatures.clear()
                        try:
                            latest_records = self.fetch_day(date_str)
                            self.persist_day(date_str, latest_records)
                        except Exception:
                            log.exception(
                                '[%s] %s 收盘回补请求失败，%.0fs 后重试',
                                self.symbol, date_str, FINALIZE_RETRY_SEC,
                            )
                            time.sleep(FINALIZE_RETRY_SEC)
                            continue
                        if not is_complete_day(latest_records, date_str):
                            log.warning(
                                '[%s] %s 收盘回补不完整 (%d bars)，%.0fs 后重试',
                                self.symbol, date_str, len(latest_records),
                                FINALIZE_RETRY_SEC,
                            )
                            time.sleep(FINALIZE_RETRY_SEC)
                            continue
                        finalized_date = date_str
                        log.info('[%s] %s 收盘回补完成，等待下一个交易日', self.symbol, date_str)
                    if bars is not None:
                        self.ib.cancelHistoricalData(bars)
                    bars = None
                    current_date = None
                    last_signatures.clear()
                    try:
                        sleep_sec = max(seconds_until_next_open(now) - 60, 30)
                    except RuntimeError:
                        sleep_sec = 900
                    time.sleep(min(sleep_sec, 1800))
                    continue

                date_str = now.strftime('%Y%m%d')
                if bars is None or current_date != date_str:
                    if bars is not None:
                        self.ib.cancelHistoricalData(bars)
                    bars = self.ib.reqHistoricalData(
                        self.contract,
                        endDateTime='',
                        durationStr='1 D',
                        barSizeSetting=BAR_SIZE,
                        whatToShow=WHAT_TO_SHOW,
                        useRTH=True,
                        formatDate=2,
                        keepUpToDate=True,
                        chartOptions=[],
                    )
                    current_date = date_str
                    last_signatures.clear()

                self.ib.waitOnUpdate(timeout=poll_sec)
                latest_records = normalize_bars(bars, self.symbol, date_str)
                if not latest_records:
                    continue
                start, end = _session_bounds(date_str)
                latest_records = add_gap_flags(
                    latest_records, int((end - start).total_seconds() // 60)
                )
                changed = []
                for record in latest_records:
                    signature = tuple(record.get(k) for k in (
                        'open', 'high', 'low', 'close', 'volume', 'average', 'bar_count'
                    ))
                    if last_signatures.get(record['datetime']) != signature:
                        last_signatures[record['datetime']] = signature
                        changed.append(record)
                if changed and self.db_storage is not None:
                    self.db_storage.upsert_market_data_bars(self._db_records(changed))
                if changed and time.time() - last_parquet_write >= 60:
                    self._write_parquet(latest_records, date_str)
                    last_parquet_write = time.time()
        finally:
            if bars is not None:
                try:
                    self.ib.cancelHistoricalData(bars)
                except Exception:
                    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Official IB stock 1-minute bars')
    parser.add_argument('--config', '-c', default='config/config.yaml')
    parser.add_argument('--host', default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--client-id', type=int, default=210)
    parser.add_argument('--symbol', default='QQQ')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--start', help='回补起始日期 YYYYMMDD')
    parser.add_argument('--end', help='回补结束日期 YYYYMMDD')
    parser.add_argument('--date', help='只回补一个交易日 YYYYMMDD')
    parser.add_argument('--live', action='store_true', help='盘中实时采集并收盘回补')
    parser.add_argument('--pause', type=float, default=1.0)
    parser.add_argument('--connect-retries', type=int, default=12)
    parser.add_argument('--connect-retry-sec', type=float, default=10.0)
    args = parser.parse_args(argv)
    if not args.live and not (args.date or (args.start and args.end)):
        parser.error('请指定 --date、--start/--end 或 --live')

    config = AppConfig.from_yaml(args.config)
    host = args.host or config.ib.host
    port = args.port or config.ib.port
    data_dir = args.data_dir or config.storage.data_dir
    db = GEXDBStorage(config.database) if config.database.enabled else None
    ib = IB()
    try:
        retries = max(1, args.connect_retries)
        for attempt in range(1, retries + 1):
            try:
                ib.connect(host, port, clientId=args.client_id, timeout=config.ib.connect_timeout)
                break
            except Exception:
                if attempt >= retries:
                    raise
                log.warning(
                    'IB 尚未就绪（%d/%d），%ss 后重试: %s',
                    attempt,
                    retries,
                    args.connect_retry_sec,
                    host,
                )
                if ib.isConnected():
                    ib.disconnect()
                time.sleep(max(1.0, args.connect_retry_sec))
        symbol = args.symbol.upper()
        contract = Stock(symbol, 'SMART', 'USD')
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f'{symbol} 合约验证失败')
        collector = OfficialQQQBarCollector(
            ib, db, data_dir, contract=qualified[0], symbol=symbol,
            request_pause_sec=args.pause
        )
        if args.date:
            collector.backfill(args.date, args.date)
        elif args.start:
            collector.backfill(args.start, args.end)
        if args.live:
            collector.run_live()
        return 0
    finally:
        if ib.isConnected():
            ib.disconnect()
        if db is not None:
            db.shutdown()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
