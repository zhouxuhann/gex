import numpy as np
import pandas as pd

from gex_monitor.features import compute_snapshot_features, classify_regime


def test_snapshot_flip_uses_main_crossing_logic_and_gex_flip_column():
    df = pd.DataFrame([
        {'strike': 100.0, 'right': 'C', 'gex': 100.0, 'gex_flip': -5.0},
        {'strike': 101.0, 'right': 'C', 'gex': 100.0, 'gex_flip': 10.0},
    ])

    feat = compute_snapshot_features(df, spot=100.4)

    assert feat['flip'] == 100.5
    assert feat['spot_to_flip_pct'] == (100.4 - 100.5) / 100.4


def test_no_flip_is_not_classified_as_below_flip():
    df = pd.DataFrame([
        {'strike': 100.0, 'right': 'C', 'gex': 10.0},
        {'strike': 101.0, 'right': 'C', 'gex': 20.0},
    ])

    feat = compute_snapshot_features(df, spot=100.5)
    _, tags = classify_regime(feat)

    assert np.isnan(feat['flip'])
    assert np.isnan(feat['spot_to_flip_pct'])
    assert tags['position'] == 'no_flip'
