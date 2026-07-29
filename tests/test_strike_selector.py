"""Unit tests for gex_monitor.strike_selector.select_strikes.

Validates the fix for the strike_range dead-code bug in ib_client.py
(see data/analysis/flip_bug_report.md).

Run: PYTHONPATH=src pytest tests/test_strike_selector.py -v
"""

from __future__ import annotations

import pytest

from gex_monitor.strike_selector import select_strikes


def _qqq_int_chain(lo: int = 580, hi: int = 700) -> list[float]:
    """Realistic integer-only QQQ chain."""
    return [float(s) for s in range(lo, hi + 1)]


def _qqq_half_chain(lo: int = 580, hi: int = 700) -> list[float]:
    """Chain with $0.5 resolution (like near-dated QQQ weeklies)."""
    out = []
    for s in range(lo, hi + 1):
        out.append(float(s))
        out.append(s + 0.5)
    return out


class TestRangeHonored:
    def test_integer_chain_4pct_around_635(self):
        """spot=635, range=0.04 → ±$25.4 → strikes 610-660 (integers)."""
        chain = _qqq_int_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=0.04)
        assert min(strikes) == 610.0
        assert max(strikes) == 660.0
        # 25 below + 25 above + spot == 635 (inclusive) = 51
        assert len(strikes) == 51

    def test_integer_chain_1pct_around_635(self):
        """spot=635, range=0.01 → ±$6.35 → strikes 629-641."""
        chain = _qqq_int_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=0.01)
        assert min(strikes) == 629.0
        assert max(strikes) == 641.0

    def test_range_scales_with_spot(self):
        """Same 4% range should give roughly ±4% coverage regardless of spot."""
        chain = _qqq_int_chain(lo=300, hi=900)
        for spot in [400.0, 500.0, 635.0, 800.0]:
            strikes = select_strikes(chain, spot=spot, strike_range=0.04)
            coverage_down = (spot - min(strikes)) / spot
            coverage_up = (max(strikes) - spot) / spot
            assert 0.035 <= coverage_down <= 0.045, \
                f"spot={spot}: down={coverage_down:.3f}"
            assert 0.035 <= coverage_up <= 0.045, \
                f"spot={spot}: up={coverage_up:.3f}"


class TestHalfDollarGranularity:
    def test_default_integer_only(self):
        """Default (include_half_dollar=False) drops $0.5 strikes."""
        chain = _qqq_half_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=0.02)
        assert all(s == int(s) for s in strikes)

    def test_include_half_dollar_true(self):
        """include_half_dollar=True keeps $0.5 strikes."""
        chain = _qqq_half_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=0.02,
                                 include_half_dollar=True)
        assert 635.5 in strikes
        assert 634.5 in strikes
        # Count: ~2x density vs integer-only (half-dollar mode can pick up a
        # couple extra strikes at the window boundaries, so we allow ±2).
        int_strikes = select_strikes(chain, spot=635.0, strike_range=0.02,
                                     include_half_dollar=False)
        expected = 2 * len(int_strikes)
        assert abs(len(strikes) - expected) <= 2, \
            f"expected ~{expected} half-dollar strikes, got {len(strikes)}"

    def test_quarter_dollar_rejected(self):
        """$0.25 strikes (rare) should be dropped by both modes."""
        chain = [635.0, 635.25, 635.5, 635.75, 636.0]
        strikes_int = select_strikes(chain, spot=635.5, strike_range=0.01)
        strikes_half = select_strikes(chain, spot=635.5, strike_range=0.01,
                                      include_half_dollar=True)
        assert 635.25 not in strikes_int
        assert 635.25 not in strikes_half


class TestMinStrikesFloor:
    def test_floor_kicks_in_when_range_too_tight(self):
        """spot=635, range=0.0001 (too tight) → fall back to floor."""
        chain = _qqq_int_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=0.0001,
                                 min_strikes_each_side=5)
        # Should have at least 5 below and 5 above
        below = [s for s in strikes if s <= 635]
        above = [s for s in strikes if s > 635]
        assert len(below) >= 5
        assert len(above) >= 5

    def test_floor_does_not_apply_when_range_is_plenty(self):
        """When range captures enough strikes, floor is not used."""
        chain = _qqq_int_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=0.04,
                                 min_strikes_each_side=5)
        # Range-based selection produces ~25 each side, floor irrelevant
        assert len(strikes) == 51


class TestSubscriptionCap:
    def test_cap_keeps_closest_strikes(self):
        chain = _qqq_int_chain()
        strikes = select_strikes(
            chain,
            spot=635.0,
            strike_range=0.04,
            max_strikes=20,
        )

        assert len(strikes) == 20
        assert strikes == [float(s) for s in range(626, 646)]

    def test_cap_preserves_sorted_unique_output(self):
        chain = [634.0, 635.0, 635.0, 636.0, 637.0]
        strikes = select_strikes(
            chain,
            spot=635.0,
            strike_range=0.04,
            min_strikes_each_side=1,
            max_strikes=3,
        )

        assert strikes == [634.0, 635.0, 636.0]


class TestEdgeCases:
    def test_empty_chain_returns_empty(self):
        assert select_strikes([], spot=635.0, strike_range=0.04) == []

    def test_none_spot_returns_empty(self):
        assert select_strikes(_qqq_int_chain(), spot=None,
                              strike_range=0.04) == []

    def test_zero_spot_returns_empty(self):
        assert select_strikes(_qqq_int_chain(), spot=0.0,
                              strike_range=0.04) == []

    def test_negative_spot_returns_empty(self):
        assert select_strikes(_qqq_int_chain(), spot=-10.0,
                              strike_range=0.04) == []

    def test_zero_strike_range_returns_empty(self):
        assert select_strikes(_qqq_int_chain(), spot=635.0,
                              strike_range=0.0) == []

    def test_chain_fully_below_spot(self):
        """Chain only has strikes below spot — above[] should still have floor."""
        chain = [float(s) for s in range(580, 636)]  # 580..635
        strikes = select_strikes(chain, spot=635.0, strike_range=0.04,
                                 min_strikes_each_side=5)
        # Below is plentiful; above is empty in range, fallback gets [] too
        # (no strikes strictly above spot exist in chain). That's fine.
        above = [s for s in strikes if s > 635]
        assert above == []

    def test_chain_fully_above_spot(self):
        chain = [float(s) for s in range(636, 700)]  # 636..699
        strikes = select_strikes(chain, spot=635.0, strike_range=0.04,
                                 min_strikes_each_side=5)
        below = [s for s in strikes if s <= 635]
        assert below == []


class TestLegacyCompatibility:
    def test_result_sorted_unique(self):
        """Output must be sorted and deduplicated."""
        chain = _qqq_int_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=0.04)
        assert strikes == sorted(set(strikes))

    def test_legacy_20_strike_behavior_via_small_range_and_floor(self):
        """Sanity check that a tiny strike_range + floor of 10 reproduces
        the legacy 'below[-10:] + above[:10]' behaviour, to ease migration."""
        chain = _qqq_int_chain()
        strikes = select_strikes(chain, spot=635.0, strike_range=1e-6,
                                 min_strikes_each_side=10)
        # Below 635: should be 626..635 (10 strikes ≤ spot, last 10)
        # Above 635: should be 636..645 (first 10 > spot)
        below = sorted(s for s in strikes if s <= 635)
        above = sorted(s for s in strikes if s > 635)
        assert below == [float(s) for s in range(626, 636)]
        assert above == [float(s) for s in range(636, 646)]
