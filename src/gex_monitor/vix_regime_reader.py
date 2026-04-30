"""VIX-based regime gate for KDJ live trader and backtest.

Reads VIX daily OHLC from PostgreSQL `market_data_bars` and classifies
each trading day into:

    low   (VIX close < low_thresh)   → complacent, mean-revert bias
                                        → allow A/B, suppress T
    mid   (low_thresh ≤ VIX ≤ high_thresh)
                                        → neutral, allow all
    high  (VIX close > high_thresh)  → stress, trend bias
                                        → suppress A/B, allow T

Defaults: low=15, high=25 (stable 30-year QQQ heuristics).

In backtest mode: call `load_historical(start, end)` once, then
`allows(signal, date)` for each bar. Regime is determined by **previous
trading day's VIX close** (avoids lookahead).

In live mode: call `get_today_regime()` which queries the DB for latest
VIX daily bar. Falls back to fail-open if data missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd


@dataclass
class VIXRegime:
    regime: str        # 'low' | 'mid' | 'high' | 'unknown'
    close: float
    ref_date: date     # the trading day whose close determines regime
    age_days: int = 0  # days since ref_date


class VIXRegimeReader:

    def __init__(
        self,
        low_thresh: float = 15.0,
        high_thresh: float = 25.0,
        max_age_days: int = 3,
        db_dsn: Optional[dict] = None,
    ):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.max_age_days = max_age_days
        self.db_dsn = db_dsn or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5433)),
            'database': os.getenv('DB_NAME', 'ibkr_market_data'),
            'user': os.getenv('DB_USER', 'ibkr_user'),
            'password': os.getenv('DB_PASSWORD', 'ibkr_secure_password_2026'),
        }
        self._daily_df: Optional[pd.DataFrame] = None

    def _classify(self, close: float) -> str:
        if close < self.low_thresh:
            return 'low'
        if close > self.high_thresh:
            return 'high'
        return 'mid'

    # ─── Backtest mode ───
    def load_historical(self, start: str, end: str) -> int:
        """Load VIX daily (YYYY-MM-DD start/end) from DB."""
        import psycopg2
        try:
            conn = psycopg2.connect(**self.db_dsn)
            # Pull a bit earlier to warm up (30d buffer for rolling options)
            start_buf = (pd.Timestamp(start) - pd.Timedelta('45 days')).strftime('%Y-%m-%d')
            q = """
                SELECT datetime::date AS trade_date, close
                FROM market_data_bars
                WHERE symbol='VIX' AND bar_size='1 day'
                  AND datetime >= %s AND datetime <= %s
                ORDER BY datetime ASC
            """
            df = pd.read_sql_query(q, conn, params=[start_buf, end + ' 23:59:59'])
            conn.close()
        except Exception:
            df = pd.DataFrame()
        if len(df) == 0:
            self._daily_df = None
            return 0
        df['close'] = df['close'].astype(float)
        df = df.drop_duplicates(subset=['trade_date']).sort_values('trade_date')
        df['regime'] = df['close'].apply(self._classify)
        df = df.set_index('trade_date')
        self._daily_df = df
        return len(df)

    def get_regime_for(self, trade_date) -> Optional[VIXRegime]:
        """Return VIX regime for a given trading date, based on **prior
        trading day's** close (avoids intraday lookahead)."""
        if self._daily_df is None or len(self._daily_df) == 0:
            return None
        trade_date = pd.Timestamp(trade_date).date() if not isinstance(trade_date, date) else trade_date
        # Find most recent date in df strictly before trade_date
        prev_dates = self._daily_df.index[self._daily_df.index < trade_date]
        if len(prev_dates) == 0:
            return None
        ref = prev_dates[-1]
        row = self._daily_df.loc[ref]
        age = (trade_date - ref).days
        if age > self.max_age_days:
            # Data stale (e.g. long holiday gap or missing data) — report as unknown
            return VIXRegime(regime='unknown', close=float(row['close']),
                             ref_date=ref, age_days=age)
        return VIXRegime(
            regime=str(row['regime']),
            close=float(row['close']),
            ref_date=ref,
            age_days=age,
        )

    # ─── Live mode ───
    def get_today_regime(self) -> Optional[VIXRegime]:
        """Query DB for most recent VIX daily; return regime based on yesterday's close."""
        import psycopg2
        try:
            conn = psycopg2.connect(**self.db_dsn)
            cur = conn.cursor()
            cur.execute("""
                SELECT datetime::date, close FROM market_data_bars
                WHERE symbol='VIX' AND bar_size='1 day'
                ORDER BY datetime DESC LIMIT 1
            """)
            row = cur.fetchone()
            conn.close()
            if row is None:
                return None
            ref, close = row[0], float(row[1])
            age = (date.today() - ref).days
            return VIXRegime(
                regime=self._classify(close) if age <= self.max_age_days else 'unknown',
                close=close, ref_date=ref, age_days=age,
            )
        except Exception:
            return None

    # ─── Decision helper ───
    def allows(self, signal_key: str, trade_date=None) -> tuple[bool, str]:
        """
        Return (allowed, reason) based on VIX regime.

        Policy:
          - A/B signals (reversion)  → allowed when regime in {low, mid},
                                        blocked on high VIX
          - T signals (trend)        → allowed when regime in {mid, high},
                                        blocked on low VIX
          - PURE_C → neutral, always allowed
          - unknown/missing → fail-open (allow)
        """
        regime = self.get_regime_for(trade_date) if trade_date else self.get_today_regime()
        if regime is None or regime.regime == 'unknown':
            age = getattr(regime, 'age_days', None) if regime else None
            return True, f'vix_unknown' + (f'({age}d)' if age else '')

        is_ab = signal_key in ('A_LONG', 'A_SHORT', 'B_LONG', 'B_SHORT')
        is_t = signal_key in ('T_LONG', 'T_SHORT')

        r = regime.regime
        if is_ab:
            if r == 'high':
                return False, f'vix_high({regime.close:.1f})_blocks_AB'
            return True, f'vix_{r}({regime.close:.1f})'
        if is_t:
            if r == 'low':
                return False, f'vix_low({regime.close:.1f})_blocks_T'
            return True, f'vix_{r}({regime.close:.1f})'
        return True, 'vix_no_gate_for_signal'
