"""Proposed fix for gamma flip calculation bugs.

See data/analysis/flip_bug_report.md for full analysis.

This module is a PROPOSAL and does not modify gex_calc.py. To integrate:

    # in gex_calc.py, replace the existing _calculate_gamma_flip with:
    from .gex_calc_flip_patch import _calculate_gamma_flip_v2 as _calculate_gamma_flip

And update the GEXResult dataclass annotation:
    gamma_flip: float | None   # was: float

Downstream consumers that currently do things like
    f"Flip: {s['gamma_flip']:.0f}"
or
    h['flip'].std()
must be made None/NaN-safe. See callers in:
    - src/gex_monitor/ib_client.py (flip buffer + median)
    - src/gex_monitor/ui/callbacks.py (Dash rendering)
    - src/gex_monitor/features.py (flip derived features)
    - src/gex_monitor/market_narrator.py (narration)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _calculate_gamma_flip_v2(
    by_strike: pd.Series, spot: float | None
) -> float | None:
    """Compute the gamma flip price (cumsum-based), picking the crossing
    closest to spot and returning None when no crossing exists.

    Fixes vs. the original `_calculate_gamma_flip`:
      * Fix 1 (取样)   — pick the sign change closest to `spot`, not `sign_changes[0]`,
                        which otherwise pins flip to the lowest-strike crossing
                        (often deep-OTM when put/call OI is distributed on both
                        sides of ATM).
      * Fix 2 (fallback) — return None when there is no sign crossing at all,
                          instead of `cumsum.abs().idxmin()` which yields a
                          boundary strike unrelated to spot.

    Not addressed here (see report for details):
      * Bug A (cumsum 近似 vs perturbed-spot solver) — methodological,
        would require re-pricing GEX(s') for candidate s'.
      * Bug D (OI_DECAY_FACTOR = 0.3 hardcoded) — lives in calculate_gex,
        not in this helper.

    Args:
        by_strike: series of GEX values summed per strike, indexed by strike
            ascending. Signs follow dealer convention (+call, -put).
        spot: current spot price. If None, we cannot pick the nearest crossing
            and return None.

    Returns:
        Gamma flip price as float, or None when ill-defined.
    """
    # Guard clauses
    if by_strike is None or len(by_strike) == 0:
        return None
    if spot is None or not np.isfinite(spot):
        return None

    if len(by_strike) == 1:
        return float(by_strike.index[0])

    strikes = by_strike.index.to_numpy(dtype=float)
    cumsum_vals = by_strike.cumsum().to_numpy(dtype=float)

    # Find all sign changes (strict — exclude zeros on the boundary)
    signs = np.sign(cumsum_vals)
    sign_changes = np.where(signs[:-1] * signs[1:] < 0)[0]

    if len(sign_changes) == 0:
        # cumsum never crosses zero — dealer is consistently long or short
        # gamma across all strikes. Flip is ill-defined here; prefer None
        # over a misleading boundary strike.
        return None

    best_candidate = None
    best_dist = float("inf")
    for idx in sign_changes:
        s_low = strikes[idx]
        s_high = strikes[idx + 1]
        c_low = cumsum_vals[idx]
        c_high = cumsum_vals[idx + 1]

        denom = c_high - c_low
        if denom != 0:
            t = -c_low / denom
            # Clamp t to [0,1] for numerical safety (should already be in range)
            t = max(0.0, min(1.0, t))
            candidate = s_low + (s_high - s_low) * t
        else:
            candidate = (s_low + s_high) / 2.0

        d = abs(candidate - spot)
        if d < best_dist:
            best_dist = d
            best_candidate = candidate

    return float(best_candidate) if best_candidate is not None else None
