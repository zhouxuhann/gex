from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gex_monitor.tech_theme_rotation_monitor import (
    DEFAULT_THEMES,
    MonitorParams,
    avg_ret_like_pine,
    basket_index_like_pine,
    compute_theme_rotation,
    dvol_like_pine,
)


def test_basket_index_starts_at_100_like_pine_var_initialization():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    basket_ret = pd.Series([0.25, 0.10, -0.05, 0.0], index=idx)

    out = basket_index_like_pine(basket_ret)

    assert out.iloc[0] == 100.0
    assert out.iloc[1] == pytest.approx(110.0)
    assert out.iloc[2] == pytest.approx(104.5)
    assert out.iloc[3] == pytest.approx(104.5)


def test_avg_ret_returns_zero_when_no_member_has_valid_daily_return():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    close_df = pd.DataFrame(
        {
            "A": [np.nan, np.nan, np.nan],
            "B": [10.0, np.nan, np.nan],
        },
        index=idx,
    )

    out = avg_ret_like_pine(close_df, ("A", "B"))

    assert out.tolist() == [0.0, 0.0, 0.0]


def test_dvol_uses_nz_semantics_for_missing_close_or_volume():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    close = pd.Series([10.0, np.nan, 12.0], index=idx)
    volume = pd.Series([100.0, 200.0, np.nan], index=idx)

    out = dvol_like_pine(close, volume)

    assert out.tolist() == [1000.0, 0.0, 0.0]


def test_compute_keeps_original_valid_semantics_before_ma_windows_exist():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    close_df = pd.DataFrame(
        {
            "QQQ": [100, 101, 102, 103, 104],
            "SMH": [50, 51, 52, 53, 54],
        },
        index=idx,
        dtype=float,
    )
    volume_df = pd.DataFrame(
        {
            "QQQ": [1000, 1000, 1000, 1000, 1000],
            "SMH": [100, 100, 100, 100, 100],
        },
        index=idx,
        dtype=float,
    )

    rows, histories = compute_theme_rotation(
        close_df,
        volume_df,
        themes=(DEFAULT_THEMES[0],),
        params=MonitorParams(trend_len=20, slow_len=50, vol_len=20),
    )

    assert rows.loc[0, "主题"] == "半导体"
    assert histories["半导体"]["score"].iloc[0] == 10.0
    assert rows.loc[0, "状态"] == "中性观察"
