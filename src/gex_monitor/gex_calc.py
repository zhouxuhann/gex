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


def calculate_gex(tickers, spot: float) -> GEXResult | None:
    """
    从 IB tickers 计算 GEX

    Args:
        tickers: IB ticker 列表
        spot: 当前现货价格

    Returns:
        GEXResult 或 None（无有效数据时）
    """
    rows = []
    missing_oi = 0
    missing_greeks = 0
    invalid_contracts = 0

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

        # 优先使用 volume（盘中实时），OI 仅作 fallback
        vol = getattr(t, 'volume', None)
        if vol is None or (isinstance(vol, float) and np.isnan(vol)) or vol <= 0:
            # fallback 到 OI
            if oi is None or (isinstance(oi, float) and np.isnan(oi)) or oi <= 0:
                # 无 volume 也无 OI，仍保留 strike（GEX=0），保持图表稳定
                quantity = 0
            else:
                quantity = oi
        else:
            quantity = vol

        # dealer 约定: +1 for calls, -1 for puts
        multiplier = int(c.multiplier) if c.multiplier else 100
        gex = sign * g.gamma * quantity * multiplier * spot ** 2 * 0.01

        rows.append({
            'strike': c.strike,
            'right': c.right,
            'gamma': g.gamma,
            'oi': quantity,  # 实际用的是 volume 或 OI
            'gex': gex,
            'iv': g.impliedVol,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # 按行权价汇总
    by_strike = df.groupby('strike')['gex'].sum().sort_index()

    # Gamma Flip: 累积 GEX 穿越零点的位置
    gamma_flip = _calculate_gamma_flip(by_strike, spot)

    total_gex = df['gex'].sum()
    call_gex = df[df.right == 'C']['gex'].sum()
    put_gex = df[df.right == 'P']['gex'].sum()

    # ATM IV
    atm_iv_pct = _calculate_atm_iv(df, spot)

    # Call Wall / Put Wall 计算
    call_wall, put_wall = _calculate_walls(by_strike, spot)

    # 是否正 Gamma 环境
    positive_gamma = total_gex > 0

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
