"""Tests for gex_calc module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from gex_monitor.gex_calc import (
    calculate_gex, pick_expiry, GEXResult,
    _calculate_gamma_flip, _calculate_repriced_gamma_flip,
    _calculate_atm_iv, _calculate_max_pain,
    ATM_MAX_DEVIATION_PCT
)


class TestCalculateGex:
    """Tests for calculate_gex function."""

    def test_calculate_gex_basic(self, mock_ib_ticker):
        """Test basic GEX calculation with valid tickers."""
        tickers = [
            mock_ib_ticker(500, 'C', gamma=0.10, oi=1000),
            mock_ib_ticker(500, 'P', gamma=0.10, oi=800),
            mock_ib_ticker(505, 'C', gamma=0.08, oi=1200),
            mock_ib_ticker(505, 'P', gamma=0.08, oi=900),
        ]
        spot = 502.0

        result = calculate_gex(tickers, spot)

        assert result is not None
        assert isinstance(result, GEXResult)
        assert result.total_gex != 0
        assert result.call_gex > 0  # Calls have positive GEX
        assert result.put_gex < 0   # Puts have negative GEX
        assert result.gross_gex == pytest.approx(
            abs(result.call_gex) + abs(result.put_gex)
        )
        assert result.net_gex_ratio == pytest.approx(
            result.total_gex / result.gross_gex
        )
        assert result.gross_volume_gamma >= abs(result.volume_gamma)
        assert len(result.df) == 4

    def test_calculate_gex_empty_tickers(self):
        """Test with empty ticker list."""
        result = calculate_gex([], spot=500.0)
        assert result is None

    def test_calculate_gex_none_tickers(self):
        """Test with None tickers in list."""
        result = calculate_gex([None, None], spot=500.0)
        assert result is None

    def test_calculate_gex_missing_greeks(self, mock_ib_ticker):
        """Test handling of tickers with missing Greeks."""
        ticker_good = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000)
        ticker_no_greeks = MagicMock()
        ticker_no_greeks.modelGreeks = None

        tickers = [ticker_good, ticker_no_greeks]
        result = calculate_gex(tickers, spot=500.0)

        assert result is not None
        assert result.missing_greeks == 1
        assert len(result.df) == 1

    def test_calculate_gex_missing_oi(self, mock_ib_ticker):
        """Test handling of tickers with missing OI."""
        ticker_good = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000)
        ticker_no_oi = mock_ib_ticker(505, 'C', gamma=0.10, oi=0)
        ticker_no_oi.callOpenInterest = None

        tickers = [ticker_good, ticker_no_oi]
        # 传入低阈值以测试边缘情况
        result = calculate_gex(tickers, spot=500.0, oi_ready_threshold=0)

        assert result is not None
        assert result.missing_oi == 1

    def test_calculate_gex_nan_oi(self, mock_ib_ticker):
        """Test handling of NaN OI values."""
        ticker_good = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000)
        ticker_nan_oi = mock_ib_ticker(505, 'C', gamma=0.10, oi=1000)
        ticker_nan_oi.callOpenInterest = float('nan')

        tickers = [ticker_good, ticker_nan_oi]
        # 传入低阈值以测试边缘情况
        result = calculate_gex(tickers, spot=500.0, oi_ready_threshold=0)

        assert result is not None
        assert result.missing_oi == 1

    def test_partial_flag_default_false(self, mock_ib_ticker):
        """Test that partial flag defaults to False."""
        tickers = [
            mock_ib_ticker(500, 'C', gamma=0.10, oi=1000),
            mock_ib_ticker(500, 'P', gamma=0.10, oi=800),
        ]
        result = calculate_gex(tickers, spot=502.0)
        assert result is not None
        assert result.partial is False

    def test_partial_flag_settable(self, mock_ib_ticker):
        """Test that partial flag can be set to True (used by degraded mode)."""
        tickers = [
            mock_ib_ticker(500, 'C', gamma=0.10, oi=1000),
            mock_ib_ticker(500, 'P', gamma=0.10, oi=800),
        ]
        result = calculate_gex(tickers, spot=502.0)
        assert result is not None
        result.partial = True
        assert result.partial is True

    def test_zero_oi_threshold_accepts_all(self, mock_ib_ticker):
        """Test oi_ready_threshold=0.0 accepts data even with all OI missing (degraded mode)."""
        # All tickers have missing OI
        t1 = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000)
        t1.callOpenInterest = None
        t2 = mock_ib_ticker(500, 'P', gamma=0.10, oi=800)
        t2.putOpenInterest = None
        tickers = [t1, t2]
        # Default threshold would reject this
        result_default = calculate_gex(tickers, spot=502.0, oi_ready_threshold=0.8)
        assert result_default is None
        # Zero threshold accepts it
        result_degraded = calculate_gex(tickers, spot=502.0, oi_ready_threshold=0.0)
        assert result_degraded is not None
        assert result_degraded.missing_oi == 2

    def test_missing_oi_with_volume_does_not_fake_position_gex(self, mock_ib_ticker):
        """Volume is activity, not an OI substitute for position GEX."""
        t1 = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000)
        t1.callOpenInterest = float('nan')
        t1.volume = 1200
        t2 = mock_ib_ticker(500, 'P', gamma=0.10, oi=800)
        t2.putOpenInterest = float('nan')
        t2.volume = 900

        result = calculate_gex([t1, t2], spot=502.0)

        assert result is None

        degraded = calculate_gex([t1, t2], spot=502.0, oi_ready_threshold=0.0)
        assert degraded is not None
        assert degraded.missing_oi == 2
        assert degraded.total_gex == 0
        assert degraded.volume_gamma != 0

    def test_gamma_flip_single_strike(self, mock_ib_ticker):
        """Test gamma flip calculation with single strike."""
        tickers = [
            mock_ib_ticker(500, 'C', gamma=0.10, oi=1000),
        ]
        result = calculate_gex(tickers, spot=500.0)

        assert result is not None
        assert result.gamma_flip is None
        assert result.gamma_flip_method == 'unavailable'

    def test_gamma_flip_multiple_strikes(self, mock_ib_ticker):
        """Test gamma flip with multiple strikes."""
        # Create a scenario where flip should be around 500
        tickers = [
            mock_ib_ticker(495, 'C', gamma=0.10, oi=500),
            mock_ib_ticker(495, 'P', gamma=0.10, oi=1500),  # More puts = negative
            mock_ib_ticker(500, 'C', gamma=0.10, oi=1000),
            mock_ib_ticker(500, 'P', gamma=0.10, oi=1000),
            mock_ib_ticker(505, 'C', gamma=0.10, oi=1500),  # More calls = positive
            mock_ib_ticker(505, 'P', gamma=0.10, oi=500),
        ]
        result = calculate_gex(tickers, spot=500.0)

        assert result is not None
        # Mock contracts have no real expiry, so the repricing solver must not
        # fabricate a strike-cumsum value.
        assert result.gamma_flip is None

    def test_atm_iv_calculation(self, mock_ib_ticker):
        """Test ATM IV calculation."""
        tickers = [
            mock_ib_ticker(495, 'C', gamma=0.10, oi=1000, iv=0.30),
            mock_ib_ticker(495, 'P', gamma=0.10, oi=1000, iv=0.31),
            mock_ib_ticker(500, 'C', gamma=0.10, oi=1000, iv=0.25),
            mock_ib_ticker(500, 'P', gamma=0.10, oi=1000, iv=0.26),
            mock_ib_ticker(505, 'C', gamma=0.10, oi=1000, iv=0.28),
            mock_ib_ticker(505, 'P', gamma=0.10, oi=1000, iv=0.29),
        ]
        spot = 500.5  # Closest to 500 strike

        result = calculate_gex(tickers, spot)

        assert result is not None
        assert result.atm_iv_pct is not None
        # ATM IV should be average of 500 strike call (25%) and put (26%)
        assert 25 <= result.atm_iv_pct <= 26

    def test_atm_iv_none_when_no_iv(self, mock_ib_ticker):
        """Test ATM IV is None when IV data missing."""
        ticker = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000, iv=None)
        ticker.modelGreeks.impliedVol = None

        result = calculate_gex([ticker], spot=500.0)

        assert result is not None
        assert result.atm_iv_pct is None

    def test_gex_formula_correctness(self, mock_ib_ticker):
        """Test that GEX formula is applied correctly."""
        # gex = sign * gamma * OI * multiplier * spot^2 * 0.01
        gamma = 0.10
        oi = 1000
        spot = 500.0
        multiplier = 100

        # For a call: sign = +1
        expected_call_gex = 1 * gamma * oi * multiplier * spot**2 * 0.01

        ticker = mock_ib_ticker(500, 'C', gamma=gamma, oi=oi)
        result = calculate_gex([ticker], spot)

        assert result is not None
        actual_gex = result.df[result.df.right == 'C']['gex'].iloc[0]
        assert abs(actual_gex - expected_call_gex) < 0.01

    def test_daily_volume_never_replaces_open_interest(self, mock_ib_ticker):
        ticker = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000)
        ticker.volume = 50_000

        result = calculate_gex([ticker], spot=500.0)

        expected_oi_gex = 0.10 * 1000 * 100 * 500**2 * 0.01
        expected_volume_gamma = 0.10 * 50_000 * 100 * 500**2 * 0.01
        assert result.total_gex == pytest.approx(expected_oi_gex)
        assert result.volume_gamma == pytest.approx(expected_volume_gamma)
        assert result.df.iloc[0]['gex'] == pytest.approx(expected_oi_gex)

    def test_delta_oi_requires_same_expiry(self, mock_ib_ticker):
        ticker = mock_ib_ticker(500, 'C', gamma=0.10, oi=1200)
        ticker.contract.lastTradeDateOrContractMonth = '20260724'

        wrong_expiry = {
            500: {'call_oi': 1000, 'put_oi': 0, 'expiry': '20260723'},
        }
        result = calculate_gex(
            [ticker], spot=500.0, prev_oi=wrong_expiry,
        )
        assert result.delta_oi_df['call_delta_oi'].isna().all()
        assert result.max_call_delta_oi_strike is None

        same_expiry = {
            500: {'call_oi': 1000, 'put_oi': 0, 'expiry': '20260724'},
        }
        result = calculate_gex(
            [ticker], spot=500.0, prev_oi=same_expiry,
        )
        assert result.delta_oi_df.iloc[0]['call_delta_oi'] == 200
        assert result.max_call_delta_oi_strike == 500

    def test_walls_use_call_and_put_sides_without_netting(self, mock_ib_ticker):
        tickers = [
            mock_ib_ticker(99, 'P', gamma=0.10, oi=5000),
            mock_ib_ticker(99, 'C', gamma=0.10, oi=6000),
            mock_ib_ticker(101, 'P', gamma=0.10, oi=7000),
            mock_ib_ticker(101, 'C', gamma=0.10, oi=8000),
        ]

        result = calculate_gex(tickers, spot=100.0)

        assert result.put_wall == 99.0
        assert result.call_wall == 101.0

    def test_invalid_contract_right(self, mock_ib_ticker):
        """Test handling of invalid contract right."""
        ticker_good = mock_ib_ticker(500, 'C', gamma=0.10, oi=1000)
        ticker_invalid = mock_ib_ticker(505, 'X', gamma=0.10, oi=1000)  # Invalid right
        ticker_invalid.contract.right = 'X'

        tickers = [ticker_good, ticker_invalid]
        result = calculate_gex(tickers, spot=500.0)

        assert result is not None
        assert result.invalid_contracts == 1
        assert len(result.df) == 1  # Only the valid one

    def test_zero_oi_is_ready_but_contributes_no_gex(self, mock_ib_ticker):
        """A finite zero OI is valid IB data and must pass the ready threshold."""
        tickers = [
            mock_ib_ticker(500, 'C', gamma=0.10, oi=0),
            mock_ib_ticker(500, 'P', gamma=0.10, oi=0),
            mock_ib_ticker(505, 'C', gamma=0.10, oi=0),
            mock_ib_ticker(505, 'P', gamma=0.10, oi=0),
        ]

        result = calculate_gex(tickers, spot=500.0, oi_ready_threshold=0.8)

        assert result is not None
        assert result.missing_oi == 0
        assert result.total_gex == 0
        assert len(result.df) == 4


class TestPickExpiry:
    """Tests for pick_expiry function."""

    def test_pick_expiry_0dte_available(self):
        """Test when today's expiry is available."""
        chain = MagicMock()
        chain.expirations = ['20240115', '20240117', '20240119', '20240122']
        today_str = '20240115'

        expiry, is_true_0dte = pick_expiry(chain, today_str)

        assert expiry == '20240115'
        assert is_true_0dte is True

    def test_pick_expiry_no_0dte(self):
        """Test when today's expiry is not available."""
        chain = MagicMock()
        chain.expirations = ['20240117', '20240119', '20240122']
        today_str = '20240115'

        expiry, is_true_0dte = pick_expiry(chain, today_str)

        assert expiry == '20240117'
        assert is_true_0dte is False

    def test_pick_expiry_no_future_expiries(self):
        """Test when no future expiries available."""
        chain = MagicMock()
        chain.expirations = ['20240110', '20240112']
        today_str = '20240115'

        expiry, is_true_0dte = pick_expiry(chain, today_str)

        assert expiry is None
        assert is_true_0dte is False

    def test_pick_expiry_empty_chain(self):
        """Test with empty expiration list."""
        chain = MagicMock()
        chain.expirations = []
        today_str = '20240115'

        expiry, is_true_0dte = pick_expiry(chain, today_str)

        assert expiry is None
        assert is_true_0dte is False

    def test_pick_expiry_unsorted_chain(self):
        """Test with unsorted expiration list."""
        chain = MagicMock()
        chain.expirations = ['20240122', '20240115', '20240119', '20240117']
        today_str = '20240115'

        expiry, is_true_0dte = pick_expiry(chain, today_str)

        assert expiry == '20240115'
        assert is_true_0dte is True


class TestCalculateGammaFlip:
    """Tests for _calculate_gamma_flip function."""

    def test_empty_series(self):
        """Test with empty Series."""
        by_strike = pd.Series([], dtype=float)
        result = _calculate_gamma_flip(by_strike, spot=500.0)
        assert result is None

    def test_single_strike(self):
        """Test with single strike."""
        by_strike = pd.Series([1e6], index=[500.0])
        result = _calculate_gamma_flip(by_strike, spot=502.0)
        assert result is None

    def test_no_zero_crossing(self):
        """Test when cumsum never crosses zero (all positive)."""
        by_strike = pd.Series([1e6, 2e6, 1.5e6], index=[495.0, 500.0, 505.0])
        result = _calculate_gamma_flip(by_strike, spot=500.0)
        assert result is None

    def test_zero_crossing_interpolation(self):
        """Test linear interpolation when cumsum crosses zero."""
        # cumsum: -1e6 at 495, +1e6 at 505
        # Zero crossing should be exactly at 500
        by_strike = pd.Series([-1e6, 2e6], index=[495.0, 505.0])
        result = _calculate_gamma_flip(by_strike, spot=500.0)
        # cumsum: [-1e6, 1e6], crossing happens between them
        # Linear interpolation: 495 + (505-495) * (1e6 / 2e6) = 495 + 5 = 500
        assert abs(result - 500.0) < 0.01

    def test_zero_crossing_asymmetric(self):
        """Test interpolation with asymmetric values."""
        # cumsum: -2e6 at 490, +1e6 at 500
        # Zero at: 490 + 10 * (2e6 / 3e6) = 490 + 6.67 = 496.67
        by_strike = pd.Series([-2e6, 3e6], index=[490.0, 500.0])
        result = _calculate_gamma_flip(by_strike, spot=495.0)
        expected = 490.0 + 10.0 * (2e6 / 3e6)
        assert abs(result - expected) < 0.01

    def test_multiple_zero_crossings(self):
        """Test with multiple zero crossings - should return nearest to spot."""
        # cumsum: -1, 1, -1, 1
        by_strike = pd.Series([-1e6, 2e6, -2e6, 2e6], index=[490.0, 495.0, 500.0, 505.0])
        result = _calculate_gamma_flip(by_strike, spot=497.0)
        assert result == pytest.approx(497.5)


class TestRepricedGammaFlip:
    def test_reprices_gamma_and_finds_spot_root(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        df = pd.DataFrame([
            {'strike': 80.0, 'right': 'P', 'oi': 1, 'iv': 0.25,
             'multiplier': 100, 'expiry': '20270115'},
            {'strike': 90.0, 'right': 'P', 'oi': 1000, 'iv': 0.25,
             'multiplier': 100, 'expiry': '20270115'},
            {'strike': 110.0, 'right': 'C', 'oi': 1000, 'iv': 0.25,
             'multiplier': 100, 'expiry': '20270115'},
            {'strike': 120.0, 'right': 'C', 'oi': 1, 'iv': 0.25,
             'multiplier': 100, 'expiry': '20270115'},
        ])
        flip = _calculate_repriced_gamma_flip(
            df, 100.0,
            as_of=datetime(2026, 7, 15, 12, tzinfo=ZoneInfo('America/New_York')),
        )
        assert flip is not None
        assert 90.0 < flip < 110.0

    def test_no_crossing_returns_none(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        df = pd.DataFrame([
            {'strike': 95.0, 'right': 'C', 'oi': 1000, 'iv': 0.25,
             'multiplier': 100, 'expiry': '20270115'},
            {'strike': 105.0, 'right': 'C', 'oi': 1000, 'iv': 0.25,
             'multiplier': 100, 'expiry': '20270115'},
        ])
        assert _calculate_repriced_gamma_flip(
            df, 100.0,
            as_of=datetime(2026, 7, 15, 12, tzinfo=ZoneInfo('America/New_York')),
        ) is None


class TestCalculateAtmIv:
    """Tests for _calculate_atm_iv function."""

    def test_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = _calculate_atm_iv(df, spot=500.0)
        assert result is None

    def test_normal_atm_iv(self):
        """Test normal ATM IV calculation."""
        df = pd.DataFrame([
            {'strike': 500, 'right': 'C', 'iv': 0.25},
            {'strike': 500, 'right': 'P', 'iv': 0.27},
        ])
        result = _calculate_atm_iv(df, spot=500.0)
        # Average of 25% and 27% = 26%
        assert result is not None
        assert abs(result - 26.0) < 0.01

    def test_atm_iv_call_only(self):
        """Test ATM IV with only call IV available."""
        df = pd.DataFrame([
            {'strike': 500, 'right': 'C', 'iv': 0.25},
            {'strike': 500, 'right': 'P', 'iv': None},
        ])
        result = _calculate_atm_iv(df, spot=500.0)
        assert result is not None
        assert abs(result - 25.0) < 0.01

    def test_atm_strike_too_far(self):
        """Test when closest strike is too far from spot."""
        df = pd.DataFrame([
            {'strike': 480, 'right': 'C', 'iv': 0.25},
            {'strike': 480, 'right': 'P', 'iv': 0.26},
        ])
        # 480 is 4% away from 500, exceeds ATM_MAX_DEVIATION_PCT (2%)
        result = _calculate_atm_iv(df, spot=500.0)
        assert result is None

    def test_atm_iv_with_nan(self):
        """Test ATM IV skips NaN values."""
        df = pd.DataFrame([
            {'strike': 500, 'right': 'C', 'iv': 0.25},
            {'strike': 500, 'right': 'P', 'iv': float('nan')},
        ])
        result = _calculate_atm_iv(df, spot=500.0)
        assert result is not None
        assert abs(result - 25.0) < 0.01

    def test_closest_strike_selection(self):
        """Test that closest strike to spot is selected."""
        df = pd.DataFrame([
            {'strike': 495, 'right': 'C', 'iv': 0.30},
            {'strike': 495, 'right': 'P', 'iv': 0.31},
            {'strike': 500, 'right': 'C', 'iv': 0.25},
            {'strike': 500, 'right': 'P', 'iv': 0.26},
            {'strike': 505, 'right': 'C', 'iv': 0.28},
            {'strike': 505, 'right': 'P', 'iv': 0.29},
        ])
        # spot=501 is closest to 500
        result = _calculate_atm_iv(df, spot=501.0)
        assert result is not None
        # Should use 500 strike: average of 25% and 26%
        assert abs(result - 25.5) < 0.1


class TestCalculateMaxPain:
    """Tests for _calculate_max_pain function."""

    def test_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = _calculate_max_pain(df)
        assert result is None

    def test_no_oi_column(self):
        """Test when OI column is missing."""
        df = pd.DataFrame([
            {'strike': 500, 'right': 'C', 'gamma': 0.1},
        ])
        result = _calculate_max_pain(df)
        assert result is None

    def test_simple_max_pain(self):
        """Test basic Max Pain calculation."""
        # Scenario: Most OI at 500 strike
        # Call at 490: OI=100, Put at 510: OI=100
        # Max Pain should be around where both have minimal intrinsic value
        df = pd.DataFrame([
            {'strike': 490, 'right': 'C', 'oi': 100},
            {'strike': 500, 'right': 'C', 'oi': 1000},
            {'strike': 500, 'right': 'P', 'oi': 1000},
            {'strike': 510, 'right': 'P', 'oi': 100},
        ])
        result = _calculate_max_pain(df)
        assert result is not None
        # Max Pain should be 500 where the OI is concentrated
        assert result == 500

    def test_max_pain_puts_only(self):
        """Test Max Pain with only puts."""
        df = pd.DataFrame([
            {'strike': 495, 'right': 'P', 'oi': 100},
            {'strike': 500, 'right': 'P', 'oi': 500},
            {'strike': 505, 'right': 'P', 'oi': 200},
        ])
        result = _calculate_max_pain(df)
        assert result is not None
        # With only puts, Max Pain should be highest strike (505)
        # because at 505, all puts expire worthless
        assert result == 505

    def test_max_pain_calls_only(self):
        """Test Max Pain with only calls."""
        df = pd.DataFrame([
            {'strike': 495, 'right': 'C', 'oi': 100},
            {'strike': 500, 'right': 'C', 'oi': 500},
            {'strike': 505, 'right': 'C', 'oi': 200},
        ])
        result = _calculate_max_pain(df)
        assert result is not None
        # With only calls, Max Pain should be lowest strike (495)
        # because at 495, all calls expire worthless
        assert result == 495

    def test_max_pain_symmetric(self):
        """Test Max Pain with symmetric OI distribution."""
        df = pd.DataFrame([
            {'strike': 490, 'right': 'C', 'oi': 500},
            {'strike': 490, 'right': 'P', 'oi': 500},
            {'strike': 500, 'right': 'C', 'oi': 500},
            {'strike': 500, 'right': 'P', 'oi': 500},
            {'strike': 510, 'right': 'C', 'oi': 500},
            {'strike': 510, 'right': 'P', 'oi': 500},
        ])
        result = _calculate_max_pain(df)
        assert result is not None
        # With symmetric distribution, Max Pain should be middle strike
        assert result == 500

    def test_max_pain_in_gex_result(self, mock_ib_ticker):
        """Test that max_pain is included in GEXResult."""
        tickers = [
            mock_ib_ticker(495, 'C', gamma=0.10, oi=500),
            mock_ib_ticker(495, 'P', gamma=0.10, oi=500),
            mock_ib_ticker(500, 'C', gamma=0.10, oi=1000),
            mock_ib_ticker(500, 'P', gamma=0.10, oi=1000),
            mock_ib_ticker(505, 'C', gamma=0.10, oi=500),
            mock_ib_ticker(505, 'P', gamma=0.10, oi=500),
        ]
        result = calculate_gex(tickers, spot=500.0)

        assert result is not None
        assert result.max_pain is not None
        # Max Pain should be 500 where OI is highest
        assert result.max_pain == 500
