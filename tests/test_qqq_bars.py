from types import SimpleNamespace

import pandas as pd

from gex_monitor.qqq_bars import add_gap_flags, bar_to_record, normalize_bars


def _bar(ts, close=500.0, volume=1000):
    return SimpleNamespace(
        date=ts, open=499.0, high=501.0, low=498.0, close=close,
        volume=volume, barCount=12, average=500.0,
    )


def test_bar_to_record_normalizes_utc_to_et():
    record = bar_to_record(_bar(pd.Timestamp('2026-01-15 14:30:00', tz='UTC')))

    assert record['datetime'].strftime('%Y-%m-%d %H:%M') == '2026-01-15 09:30'
    assert record['volume'] == 1000
    assert record['average'] == 500.0


def test_normalize_bars_filters_rth_and_deduplicates():
    bars = [
        _bar(pd.Timestamp('2026-01-15 14:29:00', tz='UTC')),
        _bar(pd.Timestamp('2026-01-15 14:30:00', tz='UTC'), close=500.0),
        _bar(pd.Timestamp('2026-01-15 14:30:30', tz='UTC'), close=501.0),
        _bar(pd.Timestamp('2026-01-15 21:00:00', tz='UTC')),
    ]

    records = normalize_bars(bars, date_str='20260115')

    assert len(records) == 1
    assert records[0]['close'] == 501.0


def test_add_gap_flags_detects_missing_minutes():
    bars = normalize_bars([
        _bar(pd.Timestamp('2026-01-15 14:30:00', tz='UTC')),
        _bar(pd.Timestamp('2026-01-15 14:32:00', tz='UTC')),
    ], date_str='20260115')

    result = add_gap_flags(bars, expected_minutes=390)

    assert all(row['has_gaps'] for row in result)
