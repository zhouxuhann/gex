"""
GEX 特征工程模块

设计目标：
  - 不做未来预测，只描述"此刻市场处于什么状态"
  - 每个特征都可解释、可监控
  - 既能实时调用（ib_worker 主循环），也能批量处理历史 parquet

用法：
  # 实时：在 ib_worker 里
  from features import compute_all_features, classify_regime
  feat = compute_all_features(df_tick, history, historical_rank_data)
  regime, tags = classify_regime(feat)

  # 批量：命令行
  python features.py --date 20260409           # 处理某天
  python features.py --all                      # 处理所有可用日期
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

DATA_DIR = Path('data')
SYMBOL = 'QQQ'

# ============================================================
# 原子层：12 个特征（从单个 snapshot 算）
# ============================================================
def compute_snapshot_features(df: pd.DataFrame, spot: float) -> dict:
    """
    从单个 tick 的 by-contract DataFrame 算原子特征。

    df 需要的列：strike, right ('C'/'P'), gex
    spot: 当前现货价
    """
    if df.empty or spot <= 0:
        return _empty_snapshot_features()

    # 按 strike 聚合
    by_strike = df.groupby('strike')['gex'].sum().sort_index()
    strikes = by_strike.index.values.astype(float)
    gex_vals = by_strike.values.astype(float)
    abs_gex = np.abs(gex_vals)
    total_abs = abs_gex.sum()

    if total_abs < 1e-6:
        return _empty_snapshot_features()

    # ---- Level ----
    total_gex = float(gex_vals.sum())
    call_gex = float(df[df['right'] == 'C']['gex'].sum())
    put_gex = float(df[df['right'] == 'P']['gex'].sum())
    call_abs = abs(call_gex)
    put_abs = abs(put_gex)
    call_gex_ratio = call_abs / (call_abs + put_abs) if (call_abs + put_abs) > 0 else 0.5

    # Flip：累积 GEX 绝对值最小的 strike
    cum = by_strike.cumsum()
    flip = float(strikes[np.argmin(np.abs(cum.values))])
    spot_to_flip_pct = (spot - flip) / spot

    # ---- Shape ----
    # Herfindahl 集中度
    weights = abs_gex / total_abs
    gex_concentration = float((weights ** 2).sum())

    # Top-3 占比
    top3_strike_share = float(np.sort(weights)[-3:].sum()) if len(weights) >= 3 else float(weights.sum())

    # 加权重心（按 |GEX| 加权，衡量"质量中心"在哪）
    com = float((strikes * abs_gex).sum() / total_abs)
    com_vs_spot = (com - spot) / spot

    # 分布宽度（加权 std）
    variance = ((strikes - com) ** 2 * abs_gex).sum() / total_abs
    spread = float(np.sqrt(variance))
    spread_pct = spread / spot

    # ---- Walls ----
    # Call wall：最大正 GEX 的 strike（不一定是 call，而是 net positive）
    pos_mask = gex_vals > 0
    neg_mask = gex_vals < 0

    if pos_mask.any():
        call_wall_idx = np.argmax(gex_vals)
        call_wall_strike = float(strikes[call_wall_idx])
        call_wall_strength = float(gex_vals[call_wall_idx] / total_abs)
        call_wall_distance_pct = (call_wall_strike - spot) / spot
    else:
        call_wall_strike = np.nan
        call_wall_strength = 0.0
        call_wall_distance_pct = np.nan

    if neg_mask.any():
        put_wall_idx = np.argmin(gex_vals)
        put_wall_strike = float(strikes[put_wall_idx])
        put_wall_strength = float(abs(gex_vals[put_wall_idx]) / total_abs)
        put_wall_distance_pct = (put_wall_strike - spot) / spot
    else:
        put_wall_strike = np.nan
        put_wall_strength = 0.0
        put_wall_distance_pct = np.nan

    # Wall 不对称性：call 强度 / put 强度
    wall_asymmetry = (call_wall_strength / put_wall_strength
                      if put_wall_strength > 1e-9 else np.inf)

    return {
        # Level
        'total_gex': total_gex,
        'call_gex_ratio': call_gex_ratio,
        'spot_to_flip_pct': spot_to_flip_pct,
        'flip': flip,
        # Shape
        'gex_concentration': gex_concentration,
        'top3_strike_share': top3_strike_share,
        'com_vs_spot': com_vs_spot,
        'spread_pct': spread_pct,
        # Walls
        'call_wall_strike': call_wall_strike,
        'put_wall_strike': put_wall_strike,
        'call_wall_distance_pct': call_wall_distance_pct,
        'put_wall_distance_pct': put_wall_distance_pct,
        'wall_asymmetry': wall_asymmetry,
    }


def _empty_snapshot_features():
    return {
        'total_gex': 0.0, 'call_gex_ratio': 0.5, 'spot_to_flip_pct': 0.0,
        'flip': np.nan, 'gex_concentration': 0.0, 'top3_strike_share': 0.0,
        'com_vs_spot': 0.0, 'spread_pct': 0.0,
        'call_wall_strike': np.nan, 'put_wall_strike': np.nan,
        'call_wall_distance_pct': np.nan, 'put_wall_distance_pct': np.nan,
        'wall_asymmetry': 1.0,
    }


# ============================================================
# 日内层：4 个特征（需要今日 history DataFrame）
# ============================================================
def _empty_intraday_features() -> dict:
    """日内特征的空值/默认值"""
    return {
        'gex_pct_of_day_range': 0.5,
        'flip_stability': 0.0,
        'regime_duration_min': 0.0,
        'spot_flip_cross_count': 0,
    }


def compute_intraday_features(
    history_df: pd.DataFrame,
    current_ts: pd.Timestamp,
    current_total_gex: float,
    current_flip: float,
    current_spot: float,
    strict: bool = True
) -> dict:
    """
    计算日内特征，严格防止数据泄露

    Args:
        history_df: 今日的 history，需要列 ts, total_gex, flip, spot
        current_ts: 当前时间戳（特征计算时刻）
        current_total_gex: 当前 total_gex
        current_flip: 当前 flip
        current_spot: 当前 spot
        strict: True=自动过滤未来数据并警告，False=信任调用方

    Returns:
        日内特征字典
    """
    import logging

    if history_df is None or len(history_df) < 2:
        return _empty_intraday_features()

    if 'ts' not in history_df.columns:
        logging.warning("history_df 缺少 'ts' 列，无法进行时间过滤")
        return _empty_intraday_features()

    # ========== 防泄露核心逻辑 ==========
    if strict:
        # 严格模式：只使用 current_ts 之前的数据
        past_mask = history_df['ts'] < current_ts
        n_future = (~past_mask).sum()
        if n_future > 0:
            logging.debug(f"防泄露：过滤掉 {n_future} 条 >= current_ts 的数据")
        h = history_df[past_mask].copy()
    else:
        h = history_df.copy()
    # =====================================

    if len(h) < 2:
        return _empty_intraday_features()

    # 按时间排序
    h = h.sort_values('ts').reset_index(drop=True)

    # 1. 今日 total_gex 在历史 (min, max) 中的位置
    gmin, gmax = h['total_gex'].min(), h['total_gex'].max()
    if gmax - gmin > 1e-6:
        pct_of_range = (current_total_gex - gmin) / (gmax - gmin)
    else:
        pct_of_range = 0.5
    pct_of_range = float(np.clip(pct_of_range, 0, 1))

    # 2. Flip 稳定性：flip 的 std / 当前 spot
    flip_stability = float(h['flip'].std() / current_spot) if current_spot > 0 else 0.0

    # 3. 当前 gamma regime（正/负）已持续多少分钟
    signs = np.sign(h['total_gex'].values)
    current_sign = np.sign(current_total_gex)

    # 从末尾往前找 sign 变化点
    regime_start_idx = len(signs) - 1
    for i in range(len(signs) - 1, -1, -1):
        if signs[i] != current_sign:
            regime_start_idx = i + 1
            break
        regime_start_idx = i

    if regime_start_idx < len(h):
        start_ts = h['ts'].iloc[regime_start_idx]
        end_ts = h['ts'].iloc[-1]
        regime_duration_min = (end_ts - start_ts).total_seconds() / 60
    else:
        regime_duration_min = 0.0

    # 4. 今日 spot 穿越 flip 的次数
    above = (h['spot'] > h['flip']).astype(int)
    crosses = int((above.diff().abs() == 1).sum())

    return {
        'gex_pct_of_day_range': pct_of_range,
        'flip_stability': flip_stability,
        'regime_duration_min': float(regime_duration_min),
        'spot_flip_cross_count': crosses,
    }


# ============================================================
# 跨日层：3 个特征（需要历史 N 天汇总数据）
# ============================================================
def _empty_cross_day_features() -> dict:
    """跨日特征的空值/默认值"""
    return {
        'total_gex_pct_rank_20d': 0.5,
        'gex_concentration_pct_rank_20d': 0.5,
        'flip_atr_20d': 0.0,
    }


def compute_cross_day_features(
    current_feat: dict,
    current_date: str,
    historical_df: Optional[pd.DataFrame],
    strict: bool = True
) -> dict:
    """
    计算跨日特征，严格防止数据泄露

    Args:
        current_feat: 当前 snapshot 的特征 dict
        current_date: 当前日期 YYYYMMDD（用于排除当天数据）
        historical_df: 过去 N 天每个 tick 的 features，需要列
            total_gex, gex_concentration, flip, spot, date
        strict: True=自动过滤当天及未来数据，False=信任调用方

    Returns:
        跨日特征字典，分位数 0~1。数据不足时返回 0.5（中性）。
    """
    import logging

    if historical_df is None or len(historical_df) < 100:
        return _empty_cross_day_features()

    # ========== 防泄露核心逻辑 ==========
    if strict and 'date' in historical_df.columns:
        # 严格模式：只使用 current_date 之前的数据
        past_mask = historical_df['date'] < current_date
        n_future = (~past_mask).sum()
        if n_future > 0:
            logging.debug(f"防泄露：过滤掉 {n_future} 条 >= current_date 的数据")
        h = historical_df[past_mask]
    else:
        h = historical_df
    # =====================================

    if len(h) < 100:
        return _empty_cross_day_features()

    # 1. total_gex 分位数
    total_rank = float((h['total_gex'] < current_feat['total_gex']).mean())

    # 2. 集中度分位数（如果有该列）
    if 'gex_concentration' in h.columns and not np.isnan(current_feat.get('gex_concentration', np.nan)):
        valid_conc = h['gex_concentration'].dropna()
        if len(valid_conc) > 0:
            conc_rank = float((valid_conc < current_feat['gex_concentration']).mean())
        else:
            conc_rank = 0.5
    else:
        conc_rank = 0.5

    # 3. Flip 20 日移动范围（简化 ATR：每日 flip max-min 的均值）
    if 'date' in h.columns:
        daily_range = h.groupby('date')['flip'].agg(lambda x: x.max() - x.min())
        flip_atr = float(daily_range.mean()) if len(daily_range) > 0 else 0.0
    else:
        flip_atr = float(h['flip'].max() - h['flip'].min())

    return {
        'total_gex_pct_rank_20d': total_rank,
        'gex_concentration_pct_rank_20d': conc_rank,
        'flip_atr_20d': flip_atr,
    }


# ============================================================
# 统一入口：把三层拼起来
# ============================================================
def compute_all_features(
    df_tick: pd.DataFrame,
    spot: float,
    current_ts: pd.Timestamp,
    current_date: str,
    history_df: Optional[pd.DataFrame] = None,
    historical_df: Optional[pd.DataFrame] = None,
    strict: bool = True
) -> dict:
    """
    统一入口，计算所有特征（严格防泄露）

    Args:
        df_tick: 当前 tick 的 by-contract DataFrame (strike, right, gex)
        spot: 当前现货价格
        current_ts: 当前时间戳（用于日内防泄露）
        current_date: 当前日期 YYYYMMDD（用于跨日防泄露）
        history_df: 今日 history (可选，没有就跳过日内层)
        historical_df: 跨日 history (可选，没有就跳过跨日层)
        strict: True=严格防泄露模式，False=信任调用方

    Returns:
        完整特征字典
    """
    # Layer 1: Snapshot 特征（无泄露风险）
    feat = compute_snapshot_features(df_tick, spot)

    # Layer 2: 日内特征（需要时间过滤）
    if history_df is not None and len(history_df) > 0:
        feat.update(compute_intraday_features(
            history_df=history_df,
            current_ts=current_ts,
            current_total_gex=feat['total_gex'],
            current_flip=feat['flip'],
            current_spot=spot,
            strict=strict
        ))
    else:
        feat.update(_empty_intraday_features())

    # Layer 3: 跨日特征（需要日期过滤）
    feat.update(compute_cross_day_features(
        current_feat=feat,
        current_date=current_date,
        historical_df=historical_df,
        strict=strict
    ))

    return feat


# ============================================================
# 规则分类：给出人类可读的 regime 标签
# ============================================================
REGIME_THRESHOLDS = {
    'concentration_high': 0.15,
    'at_flip_tol': 0.002,        # |spot_to_flip| 小于 0.2% 视为"at_flip"
    'near_wall_tol': 0.003,      # 距 wall < 0.3% 视为"接近"
    'rank_long_gamma': 0.7,
    'rank_short_gamma': 0.3,
    'rank_extreme_high': 0.9,
    'rank_extreme_low': 0.1,
    'rank_weak_low': 0.4,
    'rank_weak_high': 0.6,
}


def classify_regime(feat: dict, thresholds: dict = None) -> tuple:
    """
    返回 (regime_code, tags_dict)。
    regime_code 是三段式字符串，方便做 groupby 统计。
    tags_dict 包含所有单维标签，UI 可以分别显示。
    """
    t = thresholds or REGIME_THRESHOLDS
    tags = {}

    # ---- 1. Gamma 符号（基于跨日分位数）----
    rank = feat.get('total_gex_pct_rank_20d', 0.5)
    if rank > t['rank_long_gamma']:
        tags['gamma_sign'] = 'long_gamma'
    elif rank < t['rank_short_gamma']:
        tags['gamma_sign'] = 'short_gamma'
    else:
        tags['gamma_sign'] = 'neutral'

    # ---- 2. 集中度 ----
    tags['concentration'] = ('concentrated'
                             if feat['gex_concentration'] > t['concentration_high']
                             else 'diffuse')

    # ---- 3. Spot 相对 flip 位置 ----
    d = feat['spot_to_flip_pct']
    if abs(d) < t['at_flip_tol']:
        tags['position'] = 'at_flip'
    elif d > 0:
        tags['position'] = 'above_flip'
    else:
        tags['position'] = 'below_flip'

    # ---- 4. 接近哪个 wall ----
    call_d = feat.get('call_wall_distance_pct', np.nan)
    put_d = feat.get('put_wall_distance_pct', np.nan)
    call_near = (not np.isnan(call_d)) and abs(call_d) < t['near_wall_tol']
    put_near = (not np.isnan(put_d)) and abs(put_d) < t['near_wall_tol']
    if call_near and put_near:
        tags['wall_proximity'] = 'pinned'   # 被两个 wall 夹住
    elif call_near:
        tags['wall_proximity'] = 'near_call_wall'
    elif put_near:
        tags['wall_proximity'] = 'near_put_wall'
    else:
        tags['wall_proximity'] = 'mid_range'

    # ---- 5. 强度 ----
    if rank > t['rank_extreme_high'] or rank < t['rank_extreme_low']:
        tags['regime_strength'] = 'extreme'
    elif t['rank_weak_low'] < rank < t['rank_weak_high']:
        tags['regime_strength'] = 'weak'
    else:
        tags['regime_strength'] = 'normal'

    # 三段式 code，方便 groupby
    regime_code = f"{tags['gamma_sign']}/{tags['position']}/{tags['concentration']}"

    return regime_code, tags


# ============================================================
# 人类可读的 regime 描述
# ============================================================
REGIME_DESCRIPTIONS = {
    'long_gamma': '🟢 强正 gamma（波动压制）',
    'neutral':    '⚪ 中性 gamma',
    'short_gamma': '🔴 负 gamma（波动放大）',
    'above_flip': '📈 Spot 在稳定区',
    'below_flip': '📉 Spot 在不稳定区',
    'at_flip':    '⚡ 临近 flip（翻转风险）',
    'concentrated': '🧲 集中（磁铁效应强）',
    'diffuse':    '🌫️  分散（无明显磁铁）',
    'near_call_wall': '🧱 接近 call wall（上方阻力）',
    'near_put_wall':  '🧱 接近 put wall（下方支撑）',
    'pinned':     '📌 被双 wall 夹住',
    'mid_range':  '🆓 Walls 之间',
    'extreme':    '⚠️ 极端',
    'normal':     '✓ 正常',
    'weak':       '～ 弱信号',
}


def describe_regime(tags: dict) -> str:
    """把 tags 转成多行人类可读描述"""
    lines = []
    for key in ['gamma_sign', 'position', 'concentration',
                'wall_proximity', 'regime_strength']:
        label = tags.get(key)
        if label and label in REGIME_DESCRIPTIONS:
            lines.append(f"  {REGIME_DESCRIPTIONS[label]}")
    return '\n'.join(lines)


# ============================================================
# 批量处理：对某天的 parquet 算 features 并存盘
# ============================================================
def load_historical_features(symbol: str, n_days: int = 20) -> Optional[pd.DataFrame]:
    """加载过去 N 天的 features，用于算跨日分位数"""
    files = sorted(DATA_DIR.glob(f'features_{symbol}_*.parquet'))
    if not files:
        return None
    dfs = []
    for f in files[-n_days:]:
        try:
            df = pd.read_parquet(f)
            date_str = f.stem.split('_')[-1]
            df['date'] = date_str
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def process_date(
    date_str: str,
    symbol: str = SYMBOL,
    historical_df: Optional[pd.DataFrame] = None,
    strict: bool = True
) -> Optional[pd.DataFrame]:
    """
    处理某一天的 gex parquet，产出 features parquet（严格防泄露）

    Args:
        date_str: 日期 YYYYMMDD
        symbol: 标的代码
        historical_df: 过去 N 天的特征数据（用于跨日特征）
        strict: True=严格防泄露模式

    注意：gex_*.parquet 里存的是已聚合的 total_gex，没有 by-strike 的原始分布。
    要算 snapshot 特征需要每 tick 的 by-strike 数据——这个版本先用 history 里已有的字段
    （total_gex, flip, spot, call_gex, put_gex），shape 特征要做实时改造。
    """
    gex_file = DATA_DIR / f'gex_{symbol}_{date_str}.parquet'
    if not gex_file.exists():
        print(f"  {date_str}: 找不到 {gex_file}")
        return None

    df = pd.read_parquet(gex_file).sort_values('ts').reset_index(drop=True)
    if df.empty:
        return None

    print(f"  {date_str}: {len(df)} ticks (strict={strict})")

    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        current_ts = row['ts']
        spot = row['spot']

        # 从 history 字段重建一个最小 feature dict
        # （因为当前 history 里没存 by-strike 原始数据，shape 特征需要扩展 history
        #  我们用可得的字段填充）
        call_abs = abs(row.get('call_gex', 0))
        put_abs = abs(row.get('put_gex', 0))
        feat = {
            'total_gex': row['total_gex'],
            'call_gex_ratio': call_abs / (call_abs + put_abs) if (call_abs + put_abs) > 0 else 0.5,
            'spot_to_flip_pct': (spot - row['flip']) / spot if spot > 0 else 0,
            'flip': row['flip'],
            # Shape 特征留空（需要 by-strike 数据，见下方说明）
            'gex_concentration': np.nan,
            'top3_strike_share': np.nan,
            'com_vs_spot': np.nan,
            'spread_pct': np.nan,
            'call_wall_strike': np.nan,
            'put_wall_strike': np.nan,
            'call_wall_distance_pct': np.nan,
            'put_wall_distance_pct': np.nan,
            'wall_asymmetry': np.nan,
        }

        # 日内特征（传入 current_ts 进行防泄露过滤）
        history_so_far = df.iloc[:i + 1]  # 可以多传，函数内部会过滤
        feat.update(compute_intraday_features(
            history_df=history_so_far,
            current_ts=current_ts,
            current_total_gex=feat['total_gex'],
            current_flip=feat['flip'],
            current_spot=spot,
            strict=strict
        ))

        # 跨日特征（传入 current_date 进行防泄露过滤）
        feat.update(compute_cross_day_features(
            current_feat=feat,
            current_date=date_str,
            historical_df=historical_df,
            strict=strict
        ))

        # 分类
        regime_code, tags = classify_regime(feat)
        feat['regime_code'] = regime_code
        feat.update({f'tag_{k}': v for k, v in tags.items()})

        # 保留时间戳和 spot
        feat['ts'] = row['ts']
        feat['spot'] = spot

        rows.append(feat)

    out = pd.DataFrame(rows)
    out_file = DATA_DIR / f'features_{symbol}_{date_str}.parquet'
    out.to_parquet(out_file)
    print(f"  → {out_file}  ({len(out)} rows)")

    # 简要统计
    if 'regime_code' in out.columns:
        print(f"  Regime 分布 Top 5:")
        for code, cnt in out['regime_code'].value_counts().head(5).items():
            pct = cnt / len(out) * 100
            print(f"    {code:50s}  {cnt:5d} ({pct:4.1f}%)")

    return out


def batch_process_all(strict: bool = True):
    """
    处理所有可用日期（严格防泄露）

    Args:
        strict: True=严格防泄露模式
    """
    gex_files = sorted(DATA_DIR.glob(f'gex_{SYMBOL}_*.parquet'))
    if not gex_files:
        print("data/ 里没有 gex parquet 文件")
        return

    dates = [f.stem.split('_')[-1] for f in gex_files]
    print(f"发现 {len(dates)} 天数据: {dates[0]} → {dates[-1]}")
    print(f"防泄露模式: {'开启' if strict else '关闭'}")

    # 滚动处理：每天用之前所有天作为历史（严格排除当天及之后）
    processed = []
    for i, date_str in enumerate(dates):
        if i == 0:
            hist = None
        else:
            # 只用之前处理过的天数据
            hist = pd.concat(processed, ignore_index=True) if processed else None

        result = process_date(date_str, SYMBOL, historical_df=hist, strict=strict)
        if result is not None:
            # 保留用于后续跨日计算的字段
            processed.append(result[['total_gex', 'gex_concentration',
                                      'flip', 'spot']].assign(date=date_str))


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GEX 特征工程（严格防泄露版本）'
    )
    parser.add_argument('--date', help='处理指定日期 YYYYMMDD')
    parser.add_argument('--all', action='store_true', help='处理所有可用日期')
    parser.add_argument('--no-strict', action='store_true',
                        help='关闭严格防泄露模式（仅用于调试）')
    args = parser.parse_args()

    strict = not args.no_strict

    if args.all:
        batch_process_all(strict=strict)
    elif args.date:
        hist = load_historical_features(SYMBOL, n_days=20)
        process_date(args.date, SYMBOL, historical_df=hist, strict=strict)
    else:
        parser.print_help()
