"""
GEX 计算模块

GEX 约定
--------
本程序计算的是 *dealer* gamma exposure，符号约定:
    gex = sign * gamma * OI * multiplier * spot^2 * 0.01
其中 sign = +1 (call), -1 (puts)。
隐含假设: "dealers are short puts and long calls" (经典 dealer positioning 假设)。
单位: 美元 per 1% spot 变动。

注意: IB 返回的 OI 是前一交易日收盘数字，盘中不会变。
因此盘中 GEX 的波动 100% 来自 gamma 和 spot^2 ——
本实现严格说是 "基于前日 OI 的理论 dealer GEX"。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ATM IV 计算时，strike 偏离 spot 的最大允许比例
ATM_MAX_DEVIATION_PCT = 0.02


@dataclass
class GEXResult:
    """GEX 计算结果"""
    df: pd.DataFrame           # 每个期权的详细数据
    total_gex: float           # 总 GEX
    call_gex: float            # Call GEX
    put_gex: float             # Put GEX
    gamma_flip: float          # Gamma Flip 行权价
    atm_iv_pct: float | None   # ATM IV (百分比)
    missing_greeks: int        # 缺少 Greeks 的合约数
    missing_oi: int            # 缺少 OI 的合约数
    invalid_contracts: int     # 无效合约数（right 不是 C/P）
    # 新增字段
    call_wall: float | None    # Call Wall（spot 上方正 GEX 最大的 strike）
    put_wall: float | None     # Put Wall（spot 下方负 GEX 绝对值最大的 strike）
    positive_gamma: bool       # 是否正 Gamma 环境
    # ΔOI 相关
    delta_oi_df: pd.DataFrame | None = None  # ΔOI 数据 (strike, call_delta_oi, put_delta_oi)
    max_call_delta_oi_strike: float | None = None  # Call ΔOI 最大的 strike
    max_put_delta_oi_strike: float | None = None   # Put ΔOI 最大的 strike


def calculate_gex(
    tickers,
    spot: float,
    oi_ready_threshold: float = 0.8,
    prev_oi: dict[float, dict] | None = None,
) -> GEXResult | None:
    """
    从 IB tickers 计算 GEX

    Args:
        tickers: IB ticker 列表
        spot: 当前现货价格
        oi_ready_threshold: OI 就绪比例阈值，低于此值返回 None
        prev_oi: 前一交易日 OI 快照 {strike: {'call_oi': int, 'put_oi': int}}

    Returns:
        GEXResult 或 None（无有效数据或 OI 未就绪时）
    """
    rows = []
    missing_oi = 0
    missing_greeks = 0
    invalid_contracts = 0
    total_with_greeks = 0  # 有 Greeks 的合约数

    for t in tickers:
        if t is None:
            continue
        g = t.modelGreeks
        if not g or g.gamma is None:
            missing_greeks += 1
            continue
        c = t.contract

        # 显式验证 right 字段
        if c.right == 'C':
            sign = 1
            oi = t.callOpenInterest
        elif c.right == 'P':
            sign = -1
            oi = t.putOpenInterest
        else:
            invalid_contracts += 1
            continue

        total_with_greeks += 1

        # OI: 用于计算 Flip（稳定的 dealer 仓位）
        if oi is None or (isinstance(oi, float) and np.isnan(oi)) or oi <= 0:
            missing_oi += 1
            oi_qty = 0
        else:
            oi_qty = oi

        # Volume: 用于计算 GEX（盘中实时活动）
        vol = getattr(t, 'volume', None)
        if vol is None or not isinstance(vol, (int, float)) or (isinstance(vol, float) and np.isnan(vol)) or vol <= 0:
            vol_qty = oi_qty  # fallback 到 OI
        else:
            vol_qty = vol

        # dealer 约定: +1 for calls, -1 for puts
        multiplier = int(c.multiplier) if c.multiplier else 100
        gex_oi = sign * g.gamma * oi_qty * multiplier * spot ** 2 * 0.01    # 用于 Flip
        gex_vol = sign * g.gamma * vol_qty * multiplier * spot ** 2 * 0.01  # 用于 GEX 总量

        rows.append({
            'strike': c.strike,
            'right': c.right,
            'gamma': g.gamma,
            'oi': oi_qty,
            'volume': vol_qty,
            'gex_oi': gex_oi,    # OI-based GEX (for flip)
            'gex': gex_vol,      # Volume-based GEX (for total)
            'iv': g.impliedVol,
        })

    if not rows:
        return None

    # 检查 OI 就绪比例
    if total_with_greeks > 0:
        oi_ready_ratio = (total_with_greeks - missing_oi) / total_with_greeks
        if oi_ready_ratio < oi_ready_threshold:
            return None  # OI 数据未就绪，等待

    df = pd.DataFrame(rows)

    # 按行权价汇总（OI-based 用于 Flip）
    by_strike_oi = df.groupby('strike')['gex_oi'].sum().sort_index()

    # Gamma Flip: 用 OI-based GEX 计算（稳定）
    gamma_flip = _calculate_gamma_flip(by_strike_oi, spot)

    # GEX 总量: 用 Volume-based GEX 计算（实时）
    total_gex = df['gex'].sum()
    call_gex = df[df.right == 'C']['gex'].sum()
    put_gex = df[df.right == 'P']['gex'].sum()

    # ATM IV
    atm_iv_pct = _calculate_atm_iv(df, spot)

    # Call Wall / Put Wall 计算（用 OI-based GEX，与 Flip 一致）
    call_wall, put_wall = _calculate_walls(by_strike_oi, spot)

    # 是否正 Gamma 环境
    positive_gamma = total_gex > 0

    # 计算 ΔOI（与前一交易日对比）
    delta_oi_df = None
    max_call_delta_oi_strike = None
    max_put_delta_oi_strike = None

    if prev_oi:
        delta_oi_rows = []
        # 按 strike 聚合当前 OI
        call_oi_today = df[df.right == 'C'].groupby('strike')['oi'].sum()
        put_oi_today = df[df.right == 'P'].groupby('strike')['oi'].sum()

        all_strikes = set(call_oi_today.index) | set(put_oi_today.index) | set(prev_oi.keys())
        for strike in sorted(all_strikes):
            call_today = call_oi_today.get(strike, 0)
            put_today = put_oi_today.get(strike, 0)
            prev = prev_oi.get(strike, {'call_oi': 0, 'put_oi': 0})
            call_prev = prev.get('call_oi', 0)
            put_prev = prev.get('put_oi', 0)

            delta_oi_rows.append({
                'strike': strike,
                'call_oi_today': call_today,
                'put_oi_today': put_today,
                'call_oi_prev': call_prev,
                'put_oi_prev': put_prev,
                'call_delta_oi': call_today - call_prev,
                'put_delta_oi': put_today - put_prev,
            })

        if delta_oi_rows:
            delta_oi_df = pd.DataFrame(delta_oi_rows)
            # 找 ΔOI 最大的 strike
            if not delta_oi_df.empty:
                max_call_idx = delta_oi_df['call_delta_oi'].idxmax()
                max_put_idx = delta_oi_df['put_delta_oi'].idxmax()
                max_call_delta_oi_strike = delta_oi_df.loc[max_call_idx, 'strike']
                max_put_delta_oi_strike = delta_oi_df.loc[max_put_idx, 'strike']

    return GEXResult(
        df=df,
        total_gex=total_gex,
        call_gex=call_gex,
        put_gex=put_gex,
        gamma_flip=gamma_flip,
        atm_iv_pct=atm_iv_pct,
        missing_greeks=missing_greeks,
        missing_oi=missing_oi,
        invalid_contracts=invalid_contracts,
        call_wall=call_wall,
        put_wall=put_wall,
        positive_gamma=positive_gamma,
        delta_oi_df=delta_oi_df,
        max_call_delta_oi_strike=max_call_delta_oi_strike,
        max_put_delta_oi_strike=max_put_delta_oi_strike,
    )


def _calculate_walls(by_strike: pd.Series, spot: float) -> tuple[float | None, float | None]:
    """
    计算 Call Wall 和 Put Wall

    Call Wall: spot 上方正 GEX 最大的 strike（阻力位）
    Put Wall: spot 下方负 GEX 绝对值最大的 strike（支撑位）

    Args:
        by_strike: 按 strike 汇总的 GEX Series
        spot: 当前现货价格

    Returns:
        (call_wall, put_wall): 两个价格，可能为 None
    """
    if len(by_strike) == 0:
        return None, None

    # spot 上方的正 GEX（阻力）
    pos_above = {s: v for s, v in by_strike.items() if v > 0 and s >= spot}
    # spot 下方的负 GEX（支撑）
    neg_below = {s: v for s, v in by_strike.items() if v < 0 and s <= spot}

    call_wall = max(pos_above, key=pos_above.get) if pos_above else None
    put_wall = min(neg_below, key=neg_below.get) if neg_below else None

    return call_wall, put_wall


def _calculate_gamma_flip(by_strike: pd.Series, spot: float) -> float:
    """
    计算 Gamma Flip 价格（累积 GEX 穿越零点的位置）

    使用线性插值找到精确的零点位置，而不是简单取最近的 strike。

    Args:
        by_strike: 按 strike 汇总的 GEX Series
        spot: 当前现货价格

    Returns:
        Gamma flip 价格
    """
    if len(by_strike) == 0:
        return spot  # fallback to spot

    if len(by_strike) == 1:
        return float(by_strike.index[0])

    cumsum = by_strike.cumsum()

    # 检查是否有符号变化（穿越零点）
    signs = np.sign(cumsum.values)
    sign_changes = np.where(signs[:-1] * signs[1:] < 0)[0]

    if len(sign_changes) > 0:
        # 找到第一个符号变化位置，进行线性插值
        idx = sign_changes[0]
        strike_low = by_strike.index[idx]
        strike_high = by_strike.index[idx + 1]
        cumsum_low = cumsum.iloc[idx]
        cumsum_high = cumsum.iloc[idx + 1]

        # 线性插值: 找到 cumsum = 0 的位置
        # cumsum_low + (cumsum_high - cumsum_low) * t = 0
        # t = -cumsum_low / (cumsum_high - cumsum_low)
        if cumsum_high != cumsum_low:
            t = -cumsum_low / (cumsum_high - cumsum_low)
            gamma_flip = strike_low + (strike_high - strike_low) * t
        else:
            gamma_flip = (strike_low + strike_high) / 2
    else:
        # 没有穿越零点，返回累积绝对值最小的 strike
        gamma_flip = float(cumsum.abs().idxmin())

    return gamma_flip


def _calculate_atm_iv(df: pd.DataFrame, spot: float) -> float | None:
    """
    计算 ATM 隐含波动率

    Args:
        df: 期权数据 DataFrame
        spot: 当前现货价格

    Returns:
        ATM IV (百分比) 或 None
    """
    if df.empty:
        return None

    # 找最接近 spot 的 strike
    strikes = df['strike'].unique()
    atm_strike = min(strikes, key=lambda s: abs(s - spot))

    # 检查 ATM strike 是否足够接近 spot
    deviation = abs(atm_strike - spot) / spot
    if deviation > ATM_MAX_DEVIATION_PCT:
        return None  # strike 偏离太远，不够 ATM

    atm_rows = df[(df['strike'] == atm_strike) & df['iv'].notna()]
    if atm_rows.empty:
        return None

    # 分别计算 call 和 put IV，取平均
    call_iv = atm_rows[atm_rows.right == 'C']['iv'].mean()
    put_iv = atm_rows[atm_rows.right == 'P']['iv'].mean()
    ivs = [x for x in (call_iv, put_iv) if pd.notna(x)]

    if not ivs:
        return None

    atm_iv = np.mean(ivs)
    return float(atm_iv * 100)


def pick_expiry(chain, today_str: str) -> tuple[str | None, bool]:
    """
    选择最近的到期日

    Args:
        chain: IB 期权链
        today_str: 今日日期字符串 YYYYMMDD

    Returns:
        (expiry, is_true_0dte): 到期日字符串和是否为真 0DTE
    """
    future = sorted(e for e in chain.expirations if e >= today_str)
    if not future:
        return None, False
    chosen = future[0]
    return chosen, (chosen == today_str)
