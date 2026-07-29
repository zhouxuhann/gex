"""Strike selection helper for IB option subscriptions.

Fixes the issue where `self.strike_range` (config `strike_range: 0.04`) was
dead code in ib_client.py — all three strike-selection sites hardcoded
`below[-10:] + above[:10]`, yielding only ±$10 (~±1.5% on QQQ at $635)
regardless of config.

See data/analysis/flip_bug_report.md §"NaN 根因" and the follow-up on
采集端 strike 窗口不足.

Integration (three 1-line replacements in ib_client.py):

    # top of ib_client.py
    from .strike_selector import select_strikes

    # inside _warmup (around line 227-230):
    strikes = select_strikes(self.chain.strikes, spot, self.strike_range,
                             include_half_dollar=True)

    # inside _update_warmup (around line 281-284):
    new_strikes = select_strikes(self.chain.strikes, spot, self.strike_range,
                                 include_half_dollar=True)

    # inside main loop (around line 388-396):
    strikes = select_strikes(self.chain.strikes, spot, self.strike_range,
                             include_half_dollar=True)

Also recommended (related fixes, NOT provided here):
  * Tighten the main-loop hysteresis threshold from $1 to `spot * 0.002`
    or scale with strike_range, so spot drift of a couple dollars doesn't
    leave the window edge unsampled.
  * Consider a subscription count guard — expanding from 20 to ~50 strikes
    per symbol can bump into IB's concurrent market-data line limits when
    multiple symbols are enabled.
"""

from __future__ import annotations

from collections.abc import Iterable


def select_strikes(
    chain_strikes: Iterable[float],
    spot: float,
    strike_range: float,
    *,
    include_half_dollar: bool = False,
    min_strikes_each_side: int = 5,
    max_strikes: int | None = None,
) -> list[float]:
    """Select the strikes to subscribe to around spot.

    Args:
        chain_strikes: all strikes available on the option chain (any order,
            any granularity — we filter here).
        spot: current spot price. Must be > 0.
        strike_range: fractional range around spot to cover on each side
            (e.g. 0.04 means ±4%). Must be > 0.
        include_half_dollar: if True, keep $0.5 strikes (e.g. 635.5) in
            addition to integer strikes. QQQ listed options have $0.5
            intervals near ATM on daily/weekly expiries; dropping them
            halves resolution near spot. Default False preserves legacy
            behaviour (integer-only).
        min_strikes_each_side: absolute floor on strikes each side. If the
            computed range captures fewer than this, fall back to the
            nearest `min_strikes_each_side` strikes on that side. Protects
            against degenerate cases where `chain_strikes` is sparse.
        max_strikes: optional hard cap after range selection. The closest
            strikes to spot are retained. This protects the shared IB
            streaming market-data allowance when multiple symbols run.

    Returns:
        Sorted list of strikes to subscribe to. Always includes spot-adjacent
        strikes on both sides; empty list only if `chain_strikes` is empty
        or `spot` is non-positive.
    """
    if not chain_strikes or spot is None or spot <= 0 or strike_range <= 0:
        return []

    # Granularity filter
    if include_half_dollar:
        def _keep(s: float) -> bool:
            return s == int(s) or (s * 2) == int(s * 2)
    else:
        def _keep(s: float) -> bool:
            return s == int(s)

    filtered = sorted(s for s in chain_strikes if _keep(s))
    if not filtered:
        return []

    half_width = spot * strike_range
    lo_bound = spot - half_width
    hi_bound = spot + half_width

    below = [s for s in filtered if lo_bound <= s <= spot]
    above = [s for s in filtered if spot < s <= hi_bound]

    # Floor: if range-based selection is too thin, fall back to N nearest on that side
    if len(below) < min_strikes_each_side:
        below = [s for s in filtered if s <= spot][-min_strikes_each_side:]
    if len(above) < min_strikes_each_side:
        above = [s for s in filtered if s > spot][:min_strikes_each_side]

    selected = sorted(set(below + above))
    if max_strikes is not None:
        limit = max(1, int(max_strikes))
        if len(selected) > limit:
            below_limit = (limit + 1) // 2
            above_limit = limit - below_limit
            capped = (
                [s for s in selected if s <= spot][-below_limit:]
                + [s for s in selected if s > spot][:above_limit]
            )
            if len(capped) < limit:
                chosen = set(capped)
                remainder = sorted(
                    (s for s in selected if s not in chosen),
                    key=lambda s: (abs(s - spot), s),
                )
                capped.extend(remainder[:limit - len(capped)])
            selected = sorted(capped)
    return selected
