"""GEX regime state reader for KDJ live trader and backtest.

Reads the latest GEX snapshot for a symbol and exposes a simple regime
classification used to gate signals:

  positive_gamma == True  → dealers long γ → mean-revert bias
                              → allow A/B (reversal) signals
                              → suppress T (trend) signals
  positive_gamma == False → dealers short γ → trend-amplify bias
                              → suppress A/B signals
                              → allow T signals

Two backends:
  - PostgreSQL `gex_snapshots` table (primary, only source for backtest)
  - `src/data/gex_{SYMBOL}_{YYYYMMDD}.parquet` file (fallback for live if DB down)

If the latest snapshot is older than `stale_sec_max` seconds, regime is
reported as 'unknown' and the gate defaults to allowing all signals
(fail-open, so the strategy still trades when GEX feed dies).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class GEXSnapshot:
    ts: datetime           # UTC timestamp of snapshot
    spot: float
    total_gex: float
    positive_gamma: bool
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    max_pain: Optional[float] = None
    atm_iv_pct: Optional[float] = None
    regime_code: Optional[str] = None
    partial: bool = False
    age_sec: float = 0.0   # seconds since snapshot

    @property
    def is_stale(self) -> bool:
        return self.age_sec > 60

    @property
    def regime(self) -> str:
        """'positive' | 'negative' | 'unknown'."""
        if self.positive_gamma is None:
            return 'unknown'
        return 'positive' if self.positive_gamma else 'negative'


class GEXRegimeReader:
    """Thin wrapper around GEX snapshot sources.

    In live mode: queries Postgres every call (~1ms localhost).
    In backtest mode: pass `historical_df` to seed from parquet/DB and
    use `get_at(ts)` with an asof lookup — no DB/file I/O per bar.
    """

    def __init__(
        self,
        symbol: str = 'QQQ',
        stale_sec_max: float = 120,
        db_dsn: Optional[dict] = None,
        parquet_dir: Optional[str] = None,
        historical_df: Optional[pd.DataFrame] = None,
    ):
        self.symbol = symbol
        self.stale_sec_max = stale_sec_max
        self.db_dsn = db_dsn or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5433)),
            'database': os.getenv('DB_NAME', 'ibkr_market_data'),
            'user': os.getenv('DB_USER', 'ibkr_user'),
            'password': os.getenv('DB_PASSWORD', 'ibkr_secure_password_2026'),
        }
        self.parquet_dir = Path(parquet_dir) if parquet_dir else \
            Path(__file__).parent.parent / 'data'
        self._conn = None
        self._hist_df = historical_df  # DataFrame indexed by ts (tz-aware UTC) for backtest

    # ─── Live mode ───
    def get_latest(self) -> Optional[GEXSnapshot]:
        """Returns latest GEX snapshot, or None if no data."""
        snap = self._query_db_latest()
        if snap is None:
            snap = self._read_parquet_latest()
        return snap

    def _query_db_latest(self) -> Optional[GEXSnapshot]:
        try:
            import psycopg2
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(**self.db_dsn)
            cur = self._conn.cursor()
            cur.execute("""
                SELECT datetime, spot, total_gex, positive_gamma,
                       call_wall, put_wall, max_pain, atm_iv_pct, regime_code,
                       partial
                FROM gex_snapshots WHERE symbol=%s
                ORDER BY datetime DESC LIMIT 1
            """, (self.symbol,))
            row = cur.fetchone()
            cur.close()
            if row is None:
                return None
            ts = row[0]
            # gex_snapshots.datetime is 'timestamp without time zone', treat as UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return GEXSnapshot(
                ts=ts,
                spot=float(row[1]) if row[1] is not None else 0.0,
                total_gex=float(row[2]) if row[2] is not None else 0.0,
                positive_gamma=bool(row[3]) if row[3] is not None else False,
                call_wall=float(row[4]) if row[4] is not None else None,
                put_wall=float(row[5]) if row[5] is not None else None,
                max_pain=float(row[6]) if row[6] is not None else None,
                atm_iv_pct=float(row[7]) if row[7] is not None else None,
                regime_code=row[8],
                partial=bool(row[9]) if row[9] is not None else False,
                age_sec=age,
            )
        except Exception as e:
            # Silent fallback — caller will try parquet
            self._conn = None
            return None

    def _read_parquet_latest(self) -> Optional[GEXSnapshot]:
        date_str = datetime.now(timezone.utc).astimezone().strftime('%Y%m%d')
        path = self.parquet_dir / f'gex_{self.symbol}_{date_str}.parquet'
        if not path.exists():
            # Try yesterday (handle midnight / pre-market)
            from datetime import timedelta
            y = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y%m%d')
            path = self.parquet_dir / f'gex_{self.symbol}_{y}.parquet'
            if not path.exists():
                return None
        try:
            df = pd.read_parquet(path)
            if len(df) == 0:
                return None
            row = df.iloc[-1]
            ts = row['ts']
            if not hasattr(ts, 'tzinfo') or ts.tzinfo is None:
                ts = pd.Timestamp(ts).tz_localize('UTC')
            age = (pd.Timestamp.now(tz='UTC') - ts).total_seconds()
            return GEXSnapshot(
                ts=ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                spot=float(row['spot']),
                total_gex=float(row['total_gex']),
                positive_gamma=bool(row.get('positive_gamma', False)),
                call_wall=float(row['call_wall']) if pd.notna(row.get('call_wall')) else None,
                put_wall=float(row['put_wall']) if pd.notna(row.get('put_wall')) else None,
                max_pain=float(row['max_pain']) if pd.notna(row.get('max_pain')) else None,
                atm_iv_pct=float(row['atm_iv_pct']) if pd.notna(row.get('atm_iv_pct')) else None,
                regime_code=None,
                partial=bool(row.get('partial', False)),
                age_sec=age,
            )
        except Exception:
            return None

    # ─── Backtest mode ───
    def load_historical(self, start: str, end: str) -> int:
        """Load historical GEX snapshots from DB or parquet. Return n rows."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_dsn)
            q = """
                SELECT datetime as ts, spot, total_gex, positive_gamma,
                       call_wall, put_wall, max_pain, atm_iv_pct, regime_code,
                       partial
                FROM gex_snapshots
                WHERE symbol=%s AND datetime >= %s AND datetime <= %s
                ORDER BY datetime ASC
            """
            df = pd.read_sql_query(q, conn, params=[self.symbol, start, end + ' 23:59:59'])
            conn.close()
        except Exception:
            df = self._load_historical_parquet(start, end)

        if len(df) == 0:
            self._hist_df = None
            return 0
        df['ts'] = pd.to_datetime(df['ts'])
        if df['ts'].dt.tz is None:
            df['ts'] = df['ts'].dt.tz_localize('UTC')
        else:
            df['ts'] = df['ts'].dt.tz_convert('UTC')
        df = df.sort_values('ts').reset_index(drop=True)
        df = df.set_index('ts')
        self._hist_df = df
        return len(df)

    def _load_historical_parquet(self, start: str, end: str) -> pd.DataFrame:
        start_d = pd.Timestamp(start).date()
        end_d = pd.Timestamp(end).date()
        frames = []
        d = start_d
        from datetime import timedelta as _td
        while d <= end_d:
            p = self.parquet_dir / f'gex_{self.symbol}_{d.strftime("%Y%m%d")}.parquet'
            if p.exists():
                frames.append(pd.read_parquet(p))
            d += _td(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_at(self, ts) -> Optional[GEXSnapshot]:
        """Asof lookup for backtest. ts must be tz-aware UTC-compatible."""
        if self._hist_df is None or len(self._hist_df) == 0:
            return None
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        idx = self._hist_df.index.asof(ts)
        if pd.isna(idx):
            return None
        row = self._hist_df.loc[idx]
        age = (ts - idx).total_seconds()
        return GEXSnapshot(
            ts=idx.to_pydatetime(),
            spot=float(row['spot']) if pd.notna(row['spot']) else 0.0,
            total_gex=float(row['total_gex']) if pd.notna(row['total_gex']) else 0.0,
            positive_gamma=bool(row.get('positive_gamma', False)),
            call_wall=float(row['call_wall']) if pd.notna(row.get('call_wall')) else None,
            put_wall=float(row['put_wall']) if pd.notna(row.get('put_wall')) else None,
            max_pain=float(row['max_pain']) if pd.notna(row.get('max_pain')) else None,
            atm_iv_pct=float(row['atm_iv_pct']) if pd.notna(row.get('atm_iv_pct')) else None,
            regime_code=row.get('regime_code'),
            partial=bool(row.get('partial', False)),
            age_sec=age,
        )

    # ─── Decision helper ───
    def allows(self, signal_key: str, ts=None) -> tuple[bool, str]:
        """
        Return (allowed, reason) based on current/asof GEX regime.

        Policy:
          - A_LONG/A_SHORT/B_LONG/B_SHORT → require positive γ
          - T_LONG/T_SHORT → require negative γ
          - PURE_C_LONG/PURE_C_SHORT → allow either (context signal)
          - If snapshot stale/missing → fail-open (allow), log reason
        """
        snap = self.get_at(ts) if ts is not None else self.get_latest()
        if snap is None:
            return True, 'gex_missing'
        if snap.age_sec > self.stale_sec_max:
            return True, f'gex_stale({int(snap.age_sec)}s)'
        if snap.partial:
            return True, 'gex_partial'

        is_ab = signal_key in ('A_LONG', 'A_SHORT', 'B_LONG', 'B_SHORT')
        is_t = signal_key in ('T_LONG', 'T_SHORT')

        if is_ab:
            if snap.positive_gamma:
                return True, f'pos_γ'
            return False, f'neg_γ_blocks_AB'
        if is_t:
            if not snap.positive_gamma:
                return True, f'neg_γ'
            return False, f'pos_γ_blocks_T'
        return True, 'no_gate_for_this_signal'
