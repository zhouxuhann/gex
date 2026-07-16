"""Unit tests for the gamma flip fix proposal in gex_calc_flip_patch.

These tests exercise the specific bugs documented in
data/analysis/flip_bug_report.md and lock the expected behaviour of the
replacement implementation.

Run with:  pytest tests/test_flip_patch.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gex_monitor.gex_calc_flip_patch import _calculate_gamma_flip_v2 as flip_v2


def _series(pairs):
    """Helper: [(strike, gex), ...] → sorted pd.Series indexed by strike."""
    pairs = sorted(pairs, key=lambda p: p[0])
    idx = [p[0] for p in pairs]
    val = [p[1] for p in pairs]
    return pd.Series(val, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------------------------

class TestGuards:
    def test_empty_series_returns_none(self):
        assert flip_v2(pd.Series([], dtype=float), spot=100.0) is None

    def test_none_series_returns_none(self):
        assert flip_v2(None, spot=100.0) is None

    def test_none_spot_returns_none(self):
        s = _series([(100, -1), (101, 2)])
        assert flip_v2(s, spot=None) is None

    def test_nan_spot_returns_none(self):
        s = _series([(100, -1), (101, 2)])
        assert flip_v2(s, spot=float("nan")) is None

    def test_single_strike_returns_that_strike(self):
        s = _series([(100, 5.0)])
        assert flip_v2(s, spot=100.0) == 100.0


# ---------------------------------------------------------------------------
# Core interpolation behaviour
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_simple_single_crossing_midpoint(self):
        """cumsum: 100→-1, 101→-1+2=+1. Zero at t = -(-1)/(1-(-1)) = 0.5."""
        s = _series([(100, -1.0), (101, 2.0)])
        result = flip_v2(s, spot=100.5)
        assert math.isclose(result, 100.5, rel_tol=1e-9)

    def test_single_crossing_asymmetric(self):
        """cumsum: 100→-3, 101→-3+4=+1. t = 3/4 = 0.75, flip = 100.75."""
        s = _series([(100, -3.0), (101, 4.0)])
        result = flip_v2(s, spot=100.0)
        assert math.isclose(result, 100.75, rel_tol=1e-9)

    def test_crossing_exactly_at_strike_high(self):
        """cumsum: 100→-2, 101→0. denom=2, t=1 → flip at strike 101."""
        s = _series([(100, -2.0), (101, 2.0)])
        # Note: np.sign(0) == 0, so this doesn't count as a strict sign change.
        # Bit of an edge case — verify behaviour either way:
        result = flip_v2(s, spot=100.5)
        # Our impl uses signs[:-1]*signs[1:] < 0, which is strict; the 0 boundary
        # is not a crossing. So we expect None.
        assert result is None


# ---------------------------------------------------------------------------
# Bug B (取样错误): multi-crossing → pick closest to spot
# ---------------------------------------------------------------------------

class TestPickNearestCrossing:
    def test_two_crossings_picks_nearest(self):
        """Construct a cumsum profile with two zero crossings; verify the
        chosen one is the one closest to spot, not `sign_changes[0]`.

        Strikes:        90   91   92   93   94   95   96
        per-strike GEX: -5  +10   -5  +10  -10  +10   -5
        cumsum:         -5  +5    0   10   0   10    5

        Strict sign changes (product < 0) happen between (90→91) cumsum -5→+5.
        The second crossing is at 92→93 where cumsum goes 0→10 and between
        93→94 where 10→0. Let's reshape so we get two clear crossings.
        """
        # Deliberate construction:
        # cumsum should be: -10, +5, -3, +2 at strikes 90, 91, 92, 93.
        # So per-strike GEX = [-10, 15, -8, 5].
        s = _series([(90, -10.0), (91, 15.0), (92, -8.0), (93, 5.0)])
        cumsum = s.cumsum().to_numpy()
        assert list(cumsum) == [-10.0, 5.0, -3.0, 2.0]

        # Sign changes at (90→91) and (91→92) and (92→93) — actually 3!
        # Let's compute crossings:
        # -10→5:  t = 10/15 ≈ 0.667,  candidate ≈ 90.667
        # 5→-3:   t = 5/(-8) = -0.625 → wait, denom = -3 - 5 = -8, t = -5/-8 = 0.625, candidate ≈ 91.625
        # -3→2:   t = 3/5 = 0.6,  candidate ≈ 92.6
        # Three candidates: 90.667, 91.625, 92.6

        # If spot is 92.5, nearest should be 92.6, not 90.667 (which is what the
        # buggy `sign_changes[0]` would pick).
        result = flip_v2(s, spot=92.5)
        assert math.isclose(result, 92.6, abs_tol=0.01), \
            f"expected ~92.6 (nearest to spot 92.5), got {result}"

        # If spot is 90.5, nearest should be 90.667.
        result_low = flip_v2(s, spot=90.5)
        assert math.isclose(result_low, 90.667, abs_tol=0.01)

        # Old buggy code would return 90.667 regardless of spot.

    def test_multiple_crossings_picks_nearest_not_first(self):
        """Three sign changes in the strike grid; verify we pick the one
        closest to spot, not `sign_changes[0]` (which is the lowest-strike
        crossing)."""
        # Construct cumsum that crosses zero 3 times:
        # strikes: 630  631  632  633  634  635
        # GEX:     -5   -3   +10  -5   +5   +3
        # cumsum:  -5   -8   +2   -3   +2   +5
        # Crossings: 631→632 (-8→+2), 632→633 (+2→-3), 633→634 (-3→+2)
        s = _series([(630, -5), (631, -3), (632, 10),
                     (633, -5), (634, 5), (635, 3)])
        cumsum = s.cumsum().to_numpy()
        signs = np.sign(cumsum)
        sc = np.where(signs[:-1] * signs[1:] < 0)[0]
        assert len(sc) == 3, f"expected 3 crossings, got {len(sc)}"

        # Interpolated crossings:
        #   631→632: t = 8/10 = 0.8  → flip = 631.8
        #   632→633: t = 2/5  = 0.4  → flip = 632.4
        #   633→634: t = 3/5  = 0.6  → flip = 633.6
        # Spot = 632.5 → nearest is 632.4 (distance 0.1), not 631.8 (buggy).
        result = flip_v2(s, spot=632.5)
        assert math.isclose(result, 632.4, abs_tol=0.01), \
            f"expected ~632.4 (nearest to spot 632.5), got {result}"

        # Now put spot near the highest crossing → should pick 633.6.
        result_high = flip_v2(s, spot=633.5)
        assert math.isclose(result_high, 633.6, abs_tol=0.01)

        # And spot near the lowest crossing → should pick 631.8.
        result_low = flip_v2(s, spot=631.9)
        assert math.isclose(result_low, 631.8, abs_tol=0.01)

    def test_deep_otm_put_crossing_not_chosen_when_spot_far_above(self):
        """Realistic 0DTE scenario: put OI piles up far below spot, call OI
        clusters near ATM. cumsum has TWO crossings — one in the put-wall
        region and one near ATM. The buggy impl would pin flip to the
        put-wall region (far from spot)."""
        # Designed cumsum: negative through 620-631, crosses to positive at
        # 632, then back negative at 633, then back positive at 634.
        # We expect the implementation to pick the crossing near spot=635,
        # not the earliest one.
        rows = [
            (620, -5), (621, -5), (622, -3), (623, -2),
            (624, 1),  # cumsum: -5, -10, -13, -15, -14
            (625, 1), (626, 1), (627, 1), (628, 1),  # -13, -12, -11, -10
            (629, 2), (630, 3), (631, 5),  # -8, -5, 0  (touches zero, no strict crossing)
            (632, 10),  # cumsum = 10 (crossing 631→632: from 0 to 10 is NOT strict)
            # Actually cumsum at 631 was 0, so 631→632 transition has
            # signs[:-1]*signs[1:] = 0*1 = 0 which is NOT < 0. So the first
            # *strict* crossing happens elsewhere.
            (633, -18),  # cumsum = -8 (strict crossing 632→633: +10 → -8)
            (634, 20),   # cumsum = 12 (strict crossing 633→634: -8 → +12)
            (635, 5), (636, 3), (637, 2), (638, 1),
        ]
        s = _series(rows)
        cumsum = s.cumsum().to_numpy()
        signs = np.sign(cumsum)
        sc = np.where(signs[:-1] * signs[1:] < 0)[0]
        assert len(sc) >= 2, \
            f"test setup: expected >=2 strict crossings, got {len(sc)}: " \
            f"{cumsum.tolist()}"

        # With spot=635, the nearest crossing is the 633→634 one (near ATM).
        # A buggy `sign_changes[0]` impl would pick the earliest crossing.
        result = flip_v2(s, spot=635.0)
        assert result is not None
        assert abs(result - 635.0) < 5.0, \
            f"flip={result} is too far from spot=635; bug B would pick the " \
            f"earliest crossing"


# ---------------------------------------------------------------------------
# Bug C (fallback): no crossing → None, not a boundary strike
# ---------------------------------------------------------------------------

class TestNoCrossing:
    def test_all_positive_cumsum_returns_none(self):
        """dealer is long gamma across the whole strike grid → flip undefined."""
        s = _series([(100, 5), (101, 3), (102, 2), (103, 1)])
        assert (s.cumsum() > 0).all()
        assert flip_v2(s, spot=101.0) is None

    def test_all_negative_cumsum_returns_none(self):
        """dealer is short gamma across the whole strike grid → flip undefined."""
        s = _series([(100, -5), (101, -3), (102, -2), (103, -1)])
        assert (s.cumsum() < 0).all()
        assert flip_v2(s, spot=101.0) is None

    def test_cumsum_touches_zero_without_crossing_is_none(self):
        """cumsum goes +5, +0, +3 — touches zero but does not strictly cross.
        Old impl would pick the min-abs strike (= the one where cumsum=0)."""
        s = _series([(100, 5), (101, -5), (102, 3)])
        cumsum = s.cumsum().tolist()
        assert cumsum == [5.0, 0.0, 3.0]
        assert flip_v2(s, spot=101.0) is None


# ---------------------------------------------------------------------------
# Stability: tiny perturbation in an ATM strike shouldn't jump flip across
# the grid (smoke test for the "19-dollar jumps" observed in live data).
# ---------------------------------------------------------------------------

class TestStability:
    def test_small_atm_perturbation_keeps_flip_near_spot(self):
        """A tiny change in a single ATM strike's GEX should change flip
        by <$1 when spot is stable. The buggy impl can jump $10+ when a
        secondary sign change moves across the grid."""
        # cumsum: -5, -8, -9, -5, -1, +2  → single crossing 634→635
        base = [(630, -5), (631, -3), (632, -1), (633, 4), (634, 4), (635, 3)]
        spot = 632.5

        s1 = _series(base)
        f1 = flip_v2(s1, spot=spot)
        assert f1 is not None

        # Perturb strike 633 slightly
        perturbed = [(k, v + 0.001 if k == 633 else v) for k, v in base]
        s2 = _series(perturbed)
        f2 = flip_v2(s2, spot=spot)
        assert f2 is not None

        assert abs(f1 - f2) < 0.1, \
            f"flip jumped too much under tiny perturbation: {f1} -> {f2}"


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------

def test_return_type_is_float_or_none():
    s = _series([(100, -1), (101, 2)])
    out = flip_v2(s, spot=100.5)
    assert isinstance(out, float)
    assert flip_v2(pd.Series([], dtype=float), spot=100) is None
