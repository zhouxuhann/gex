from datetime import datetime, timezone

from gex_monitor.gex_regime_reader import GEXRegimeReader, GEXSnapshot


def test_partial_snapshot_fails_open():
    reader = GEXRegimeReader(symbol='QQQ')
    reader.get_latest = lambda: GEXSnapshot(
        ts=datetime.now(timezone.utc),
        spot=500.0,
        total_gex=-1e9,
        positive_gamma=False,
        partial=True,
        age_sec=0.0,
    )

    allowed, reason = reader.allows('A_LONG')

    assert allowed is True
    assert reason == 'gex_partial'


def test_complete_snapshot_still_gates_by_regime():
    reader = GEXRegimeReader(symbol='QQQ')
    reader.get_latest = lambda: GEXSnapshot(
        ts=datetime.now(timezone.utc),
        spot=500.0,
        total_gex=-1e9,
        positive_gamma=False,
        partial=False,
        age_sec=0.0,
    )

    allowed, reason = reader.allows('A_LONG')

    assert allowed is False
    assert reason == 'neg_γ_blocks_AB'
