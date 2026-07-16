"""GEX 实时与日终数据质量检查。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from .time_utils import ET, HAS_CALENDAR, XNYS

CORE_GEX_FIELDS = ('spot', 'total_gex', 'call_gex', 'put_gex', 'max_pain')
DERIVED_GEX_FIELDS = ('flip', 'call_wall', 'put_wall', 'rr_25', 'rr_25_zscore')


@dataclass
class DailyQualityReport:
    symbol: str
    date: str
    status: str
    score: int
    expected_rth_minutes: int
    gex_rth_minutes: int = 0
    ohlc_rth_minutes: int = 0
    strikes_rth_minutes: int = 0
    gex_coverage: float = 0.0
    ohlc_coverage: float = 0.0
    strikes_coverage: float = 0.0
    max_gex_gap_seconds: float | None = None
    duplicate_gex_rows: int = 0
    duplicate_ohlc_rows: int = 0
    duplicate_strike_rows: int = 0
    core_null_ratio: float = 0.0
    derived_null_ratios: dict[str, float] = field(default_factory=dict)
    partial_snapshot_ratio: float = 0.0
    median_contracts_per_minute: float = 0.0
    oi_strike_count: int = 0
    missing_files: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(ET).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_tick_quality(
    result: Any,
    subscribed_contracts: int,
    min_contracts: int = 20,
    max_missing_ratio: float = 0.25,
) -> list[str]:
    """返回当前 GEX 快照的结构性质量问题。"""
    reasons: list[str] = []
    valid_contracts = len(result.df) if getattr(result, 'df', None) is not None else 0
    if valid_contracts < min_contracts:
        reasons.append(f'valid_contracts={valid_contracts}<{min_contracts}')

    missing = int(getattr(result, 'missing_greeks', 0)) + int(
        getattr(result, 'missing_oi', 0)
    )
    denominator = max(int(subscribed_contracts), 1)
    missing_ratio = min(1.0, missing / denominator)
    if missing_ratio > max_missing_ratio:
        reasons.append(f'missing_ratio={missing_ratio:.1%}>{max_missing_ratio:.1%}')

    for name in ('total_gex', 'call_gex', 'put_gex'):
        value = getattr(result, name, None)
        if value is None or not math.isfinite(float(value)):
            reasons.append(f'{name}=invalid')

    if getattr(result, 'max_pain', None) is None:
        reasons.append('max_pain=missing')
    if getattr(result, 'call_wall', None) is None:
        reasons.append('call_wall=missing')
    if getattr(result, 'put_wall', None) is None:
        reasons.append('put_wall=missing')
    return reasons


def _session_bounds(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    session = pd.Timestamp(datetime.strptime(date_str, '%Y%m%d').date())
    if HAS_CALENDAR and XNYS.is_session(session):
        return XNYS.session_open(session).tz_convert(ET), XNYS.session_close(session).tz_convert(ET)
    day = session.date()
    return (
        pd.Timestamp(datetime.combine(day, time(9, 30), tzinfo=ET)),
        pd.Timestamp(datetime.combine(day, time(16, 0), tzinfo=ET)),
    )


def _timestamps_et(df: pd.DataFrame) -> pd.Series:
    if 'ts' not in df or df.empty:
        return pd.Series(dtype=f'datetime64[ns, {ET}]')
    return pd.to_datetime(df['ts'], utc=True, errors='coerce').dt.tz_convert(ET)


def _rth_frame(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = _timestamps_et(df)
    valid = ts.notna() & (ts >= start) & (ts < end)
    result = df.loc[valid].copy()
    result['_quality_ts'] = ts.loc[valid]
    return result


def _coverage(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, float | None]:
    rth = _rth_frame(df, start, end)
    if rth.empty:
        return 0, None
    ts = rth['_quality_ts'].sort_values()
    minutes = int(ts.dt.floor('min').nunique())
    gaps = ts.diff().dt.total_seconds().dropna()
    return minutes, (float(gaps.max()) if not gaps.empty else None)


def audit_daily_data(
    data_dir: str | Path,
    symbol: str,
    date_str: str,
    *,
    min_rth_coverage: float = 0.95,
    max_gap_seconds: int = 60,
    max_derived_null_ratio: float = 0.10,
    min_contracts: int = 20,
) -> DailyQualityReport:
    """审计一个交易日的 GEX/OHLC/strikes/OI 文件。"""
    data_path = Path(data_dir)
    start, end = _session_bounds(date_str)
    expected = max(1, int((end - start).total_seconds() // 60))
    frames: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []

    for kind in ('gex', 'ohlc', 'strikes'):
        path = data_path / f'{kind}_{symbol}_{date_str}.parquet'
        if not path.is_file():
            missing_files.append(kind)
            frames[kind] = pd.DataFrame()
            continue
        try:
            frames[kind] = pd.read_parquet(path)
        except Exception:
            missing_files.append(f'{kind}:unreadable')
            frames[kind] = pd.DataFrame()

    gex_minutes, max_gap = _coverage(frames['gex'], start, end)
    ohlc_minutes, _ = _coverage(frames['ohlc'], start, end)
    strikes_minutes, _ = _coverage(frames['strikes'], start, end)

    gex = frames['gex']
    ohlc = frames['ohlc']
    strikes = frames['strikes']
    if len(gex):
        core_nulls = sum(
            int(gex[c].isna().sum()) if c in gex else len(gex)
            for c in CORE_GEX_FIELDS
        )
        core_null_ratio = float(core_nulls / (len(gex) * len(CORE_GEX_FIELDS)))
        derived_nulls = {
            c: (float(gex[c].isna().mean()) if c in gex else 1.0)
            for c in DERIVED_GEX_FIELDS
        }
    else:
        core_null_ratio = 1.0
        derived_nulls = {c: 1.0 for c in DERIVED_GEX_FIELDS}
    partial_ratio = (
        float(gex['partial'].fillna(False).astype(bool).mean())
        if 'partial' in gex
        else 0.0
    )

    strike_rth = _rth_frame(strikes, start, end)
    if strike_rth.empty:
        median_contracts = 0.0
    else:
        per_minute = strike_rth.groupby(strike_rth['_quality_ts'].dt.floor('min')).size()
        median_contracts = float(per_minute.median())

    oi_path = data_path / f'oi_snapshot_{symbol}_{date_str}.parquet'
    oi_count = 0
    if oi_path.is_file():
        try:
            oi_count = len(pd.read_parquet(oi_path))
        except Exception:
            missing_files.append('oi:unreadable')
    else:
        missing_files.append('oi')

    duplicate_gex = int(gex.duplicated(['ts']).sum()) if 'ts' in gex else 0
    duplicate_ohlc = int(ohlc.duplicated(['ts']).sum()) if 'ts' in ohlc else 0
    strike_keys = [c for c in ('ts', 'strike', 'right') if c in strikes]
    duplicate_strikes = (
        int(strikes.duplicated(strike_keys).sum()) if len(strike_keys) == 3 else 0
    )

    coverage = {
        'gex': gex_minutes / expected,
        'ohlc': ohlc_minutes / expected,
        'strikes': strikes_minutes / expected,
    }
    reasons: list[str] = []
    if missing_files:
        reasons.append(f'missing_files={",".join(missing_files)}')
    for kind, ratio in coverage.items():
        if ratio < min_rth_coverage:
            reasons.append(f'{kind}_coverage={ratio:.1%}')
    if max_gap is not None and max_gap > max_gap_seconds:
        reasons.append(f'max_gex_gap={max_gap:.0f}s')
    if core_null_ratio > 0:
        reasons.append(f'core_null_ratio={core_null_ratio:.1%}')
    bad_derived = {k: v for k, v in derived_nulls.items() if v > max_derived_null_ratio}
    if bad_derived:
        reasons.append('derived_nulls=' + ','.join(f'{k}:{v:.1%}' for k, v in bad_derived.items()))
    if partial_ratio > 0:
        reasons.append(f'partial_snapshots={partial_ratio:.1%}')
    if median_contracts < min_contracts:
        reasons.append(f'median_contracts={median_contracts:.0f}<{min_contracts}')
    if oi_count < min_contracts:
        reasons.append(f'oi_strikes={oi_count}<{min_contracts}')
    duplicate_total = duplicate_gex + duplicate_ohlc + duplicate_strikes
    if duplicate_total:
        reasons.append(f'duplicate_rows={duplicate_total}')

    missing_required = any(
        item.split(':')[0] in {'gex', 'ohlc', 'strikes'} for item in missing_files
    )
    invalid = bool(missing_files and missing_required)
    invalid = invalid or coverage['gex'] < 0.50 or core_null_ratio > 0.05 or median_contracts < 10
    status = 'invalid' if invalid else ('partial' if reasons else 'good')

    score = 100
    score -= 20 * sum(1 for x in missing_files if x.split(':')[0] in {'gex', 'ohlc', 'strikes'})
    score -= round((1 - min(coverage.values())) * 40)
    score -= 15 if max_gap is not None and max_gap > max_gap_seconds else 0
    score -= min(20, round(core_null_ratio * 100))
    score -= 10 if bad_derived else 0
    score -= 15 if median_contracts < min_contracts else 0
    score -= 5 if oi_count < min_contracts else 0
    score -= 5 if duplicate_total else 0

    return DailyQualityReport(
        symbol=symbol,
        date=date_str,
        status=status,
        score=max(0, min(100, score)),
        expected_rth_minutes=expected,
        gex_rth_minutes=gex_minutes,
        ohlc_rth_minutes=ohlc_minutes,
        strikes_rth_minutes=strikes_minutes,
        gex_coverage=coverage['gex'],
        ohlc_coverage=coverage['ohlc'],
        strikes_coverage=coverage['strikes'],
        max_gex_gap_seconds=max_gap,
        duplicate_gex_rows=duplicate_gex,
        duplicate_ohlc_rows=duplicate_ohlc,
        duplicate_strike_rows=duplicate_strikes,
        core_null_ratio=core_null_ratio,
        derived_null_ratios=derived_nulls,
        partial_snapshot_ratio=partial_ratio,
        median_contracts_per_minute=median_contracts,
        oi_strike_count=oi_count,
        missing_files=missing_files,
        reasons=reasons,
    )


def write_quality_report(report: DailyQualityReport, data_dir: str | Path) -> Path:
    """原子写入日终质量 JSON。"""
    path = Path(data_dir) / f'quality_{report.symbol}_{report.date}.json'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
    return path
