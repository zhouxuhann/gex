from types import SimpleNamespace

import pandas as pd

from gex_monitor.data_quality import (
    audit_daily_data,
    evaluate_tick_quality,
    write_quality_report,
)


def _write_complete_day(data_dir, symbol='QQQ', date='20260115'):
    ts = pd.date_range(f'{date[:4]}-{date[4:6]}-{date[6:]} 09:30', periods=390,
                       freq='min', tz='America/New_York')
    gex = pd.DataFrame({
        'ts': ts,
        'spot': 500.0,
        'total_gex': 1e9,
        'call_gex': 2e9,
        'put_gex': -1e9,
        'flip': 495.0,
        'call_wall': 505.0,
        'put_wall': 490.0,
        'max_pain': 500.0,
        'rr_25': -0.02,
        'rr_25_zscore': 0.1,
        'partial': False,
    })
    ohlc = pd.DataFrame({
        'ts': ts, 'open': 500.0, 'high': 501.0, 'low': 499.0, 'close': 500.0,
    })
    strikes = pd.DataFrame({
        'ts': ts.repeat(20),
        'strike': list(range(490, 510)) * len(ts),
        'right': ['C', 'P'] * 10 * len(ts),
        'gex': 1.0,
    })
    oi = pd.DataFrame({'strike': range(490, 510), 'call_oi': 100, 'put_oi': 100})
    gex.to_parquet(data_dir / f'gex_{symbol}_{date}.parquet')
    ohlc.to_parquet(data_dir / f'ohlc_{symbol}_{date}.parquet')
    strikes.to_parquet(data_dir / f'strikes_{symbol}_{date}.parquet')
    oi.to_parquet(data_dir / f'oi_snapshot_{symbol}_{date}.parquet')


def test_complete_day_is_good(temp_dir):
    _write_complete_day(temp_dir)

    report = audit_daily_data(temp_dir, 'QQQ', '20260115')

    assert report.status == 'good'
    assert report.score == 100
    assert report.gex_rth_minutes == 390
    assert report.median_contracts_per_minute == 20
    path = write_quality_report(report, temp_dir)
    assert path.is_file()


def test_short_narrow_day_is_invalid(temp_dir):
    ts = pd.date_range('2026-01-15 12:00', periods=100, freq='min',
                       tz='America/New_York')
    pd.DataFrame({
        'ts': ts, 'spot': 500.0, 'total_gex': 1.0, 'call_gex': 2.0,
        'put_gex': -1.0, 'max_pain': 500.0,
    }).to_parquet(temp_dir / 'gex_QQQ_20260115.parquet')
    pd.DataFrame({'ts': ts, 'open': 1, 'high': 1, 'low': 1, 'close': 1}).to_parquet(
        temp_dir / 'ohlc_QQQ_20260115.parquet'
    )
    pd.DataFrame({
        'ts': ts.repeat(8), 'strike': list(range(8)) * len(ts),
        'right': ['C', 'P'] * 4 * len(ts),
    }).to_parquet(temp_dir / 'strikes_QQQ_20260115.parquet')

    report = audit_daily_data(temp_dir, 'QQQ', '20260115')

    assert report.status == 'invalid'
    assert report.gex_coverage < 0.50
    assert any(reason.startswith('median_contracts=') for reason in report.reasons)


def test_tick_quality_marks_missing_contracts_and_fields():
    result = SimpleNamespace(
        df=pd.DataFrame({'strike': range(8)}),
        missing_greeks=10,
        missing_oi=5,
        total_gex=1.0,
        call_gex=2.0,
        put_gex=-1.0,
        max_pain=None,
        call_wall=505.0,
        put_wall=None,
    )

    reasons = evaluate_tick_quality(result, subscribed_contracts=40)

    assert any(reason.startswith('valid_contracts=') for reason in reasons)
    assert any(reason.startswith('missing_ratio=') for reason in reasons)
    assert 'max_pain=missing' in reasons
    assert 'put_wall=missing' in reasons
