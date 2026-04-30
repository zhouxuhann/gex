#!/usr/bin/env python3
"""Per-snapshot upside / downside gamma band observer.

把 strikes_*.parquet 的逐 strike GEX 按 spot 相对 band 重切片：
  上方 band: [spot, spot×(1+X)]   — 净 GEX < 0 = upside short γ → squeeze fuel
  下方 band: [spot×(1-X), spot]   — 净 GEX > 0 = downside long γ → 阻尼下跌

输出: data/gamma_bands/gamma_bands_{symbol}_{YYYYMMDD}.parquet

支持模式:
  --backfill 20260414 20260424   批量历史
  --today                          今天 (ET)
  --date 20260424                  指定日期

注意当前 strikes parquet 只覆盖一个 picked expiry (通常 0DTE)。
要看全本周多个 expiry, 需要先扩 ib_client._subscribe_options 拉多 expiry.
"""
import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

ET = pytz.timezone('America/New_York')
DATA_DIR = Path('/Users/fanzhouxu/Downloads/gex/src/data')
OUT_DIR = Path('/Users/fanzhouxu/Downloads/gex/data/gamma_bands')

BANDS = [
    (0.005, '50bps'),    # 0.5%
    (0.01,  '100bps'),   # 1%
    (0.02,  '200bps'),   # 2%
    (0.03,  '300bps'),   # 3%
]


def _safe_mean(s: pd.Series) -> float:
    """Drop garbage IV (≤0 or >2.0 = 200%, far-OTM 0DTE 噪声) 后取均值."""
    if 'iv' not in s.index.names and not isinstance(s, pd.Series):
        return float('nan')
    clean = s[(s > 0) & (s < 2.0)]
    return float(clean.mean()) if len(clean) else float('nan')


def compute_one_snapshot(snap_df: pd.DataFrame, spot: float) -> dict:
    """对一个 ts 的 strike snapshot 计算各 band 的 net GEX、strike 数、IV 统计."""
    out: dict = {}
    for pct, label in BANDS:
        upper = spot * (1 + pct)
        lower = spot * (1 - pct)
        above = snap_df[(snap_df.strike > spot) & (snap_df.strike <= upper)]
        below = snap_df[(snap_df.strike < spot) & (snap_df.strike >= lower)]
        out[f'up_{label}'] = float(above['gex'].sum())
        out[f'dn_{label}'] = float(below['gex'].sum())
        out[f'up_n_{label}'] = int(len(above))
        out[f'dn_n_{label}'] = int(len(below))
        # IV 均值 (剔除 ≤0 / >200% 的噪声值)
        out[f'up_iv_{label}'] = _safe_mean(above['iv']) if 'iv' in above.columns else float('nan')
        out[f'dn_iv_{label}'] = _safe_mean(below['iv']) if 'iv' in below.columns else float('nan')
    return out


def process_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    strikes_file = DATA_DIR / f'strikes_{symbol}_{date_str}.parquet'
    gex_file = DATA_DIR / f'gex_{symbol}_{date_str}.parquet'
    if not strikes_file.exists() or not gex_file.exists():
        return None

    strikes = pd.read_parquet(strikes_file)
    gex = pd.read_parquet(gex_file)

    # gex parquet 提供 spot + 总览字段
    keep_cols = ['ts', 'spot', 'total_gex']
    for c in ('call_wall', 'put_wall', 'max_pain', 'positive_gamma'):
        if c in gex.columns:
            keep_cols.append(c)
    gex_summary = gex[keep_cols].sort_values('ts').reset_index(drop=True)

    # strikes ts 是整分钟, gex ts 是 ~3 秒, 用 merge_asof 找最近的 gex 帧
    strikes_ts_uniq = strikes[['ts']].drop_duplicates().sort_values('ts').reset_index(drop=True)
    # 统一 dtype 防 merge_asof 报 ns vs us 错
    strikes_ts_uniq['ts'] = strikes_ts_uniq['ts'].astype(gex_summary['ts'].dtype)
    matched = pd.merge_asof(
        strikes_ts_uniq,
        gex_summary,
        on='ts',
        direction='nearest',
        tolerance=pd.Timedelta('30s'),
    )
    matched = matched.dropna(subset=['spot'])

    rows = []
    strikes_grouped = dict(tuple(strikes.groupby('ts')))
    for _, m in matched.iterrows():
        ts = m['ts']
        snap = strikes_grouped.get(ts)
        if snap is None or snap.empty:
            continue
        spot = float(m['spot']) if pd.notna(m['spot']) else None
        if spot is None or spot <= 0:
            continue
        row = compute_one_snapshot(snap, spot)
        row['ts'] = ts
        row['symbol'] = symbol
        row['spot'] = spot
        row['total_gex'] = float(m['total_gex']) if pd.notna(m['total_gex']) else None
        for c in ('call_wall', 'put_wall', 'max_pain'):
            if c in matched.columns and pd.notna(m.get(c)):
                row[c] = float(m[c])
        if 'positive_gamma' in matched.columns and pd.notna(m.get('positive_gamma')):
            row['positive_gamma'] = bool(m['positive_gamma'])
        rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='QQQ', choices=['QQQ', 'SPX', 'SPY'])
    ap.add_argument('--date', help='YYYYMMDD')
    ap.add_argument('--backfill', nargs=2, metavar=('START', 'END'), help='YYYYMMDD YYYYMMDD')
    ap.add_argument('--today', action='store_true', help='use today ET')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.today:
        dates = [datetime.now(ET).strftime('%Y%m%d')]
    elif args.backfill:
        start = datetime.strptime(args.backfill[0], '%Y%m%d').date()
        end = datetime.strptime(args.backfill[1], '%Y%m%d').date()
        dates = []
        d = start
        while d <= end:
            dates.append(d.strftime('%Y%m%d'))
            d += timedelta(days=1)
    elif args.date:
        dates = [args.date]
    else:
        print('Usage: --today | --date YYYYMMDD | --backfill YYYYMMDD YYYYMMDD')
        return

    print(f'Processing {len(dates)} day(s) for {args.symbol}, output → {OUT_DIR}')
    print(f'{"date":10}  {"n_rows":>7}  {"sample upside band 1%":<24}')
    print('-' * 60)
    for date_str in dates:
        df = process_day(args.symbol, date_str)
        if df is None:
            print(f'{date_str:10}  {"--":>7}  no data')
            continue
        out_file = OUT_DIR / f'gamma_bands_{args.symbol}_{date_str}.parquet'
        df.to_parquet(out_file, index=False)
        last = df.iloc[-1]
        print(f'{date_str:10}  {len(df):>7}  '
              f'last spot={last.spot:.2f}  up_100bps={last["up_100bps"]:+.2e}')


if __name__ == '__main__':
    main()
