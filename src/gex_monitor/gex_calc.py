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
因此盘中 position GEX 的波动来自 gamma 和 spot^2。本模块另外计算
volume_gamma 作为当日成交活跃度，但绝不再用 volume 替代 OI。
"""
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ATM IV 计算时，strike 偏离 spot 的最大允许比例
ATM_MAX_DEVIATION_PCT = 0.02
ET = ZoneInfo("America/New_York")


@dataclass
class GEXResult:
    """GEX 计算结果"""
    df: pd.DataFrame           # 每个期权的详细数据
    total_gex: float           # 总 GEX
    call_gex: float            # Call GEX
    put_gex: float             # Put GEX
    gamma_flip: float | None   # 重新定价求得的 Gamma Flip；无零点时为 None
    atm_iv_pct: float | None   # ATM IV (百分比)
    missing_greeks: int        # 缺少 Greeks 的合约数
    missing_oi: int            # 缺少 OI 的合约数
    invalid_contracts: int     # 无效合约数（right 不是 C/P）
    # 新增字段
    call_wall: float | None    # Call Wall（spot 上方正 GEX 最大的 strike）
    put_wall: float | None     # Put Wall（spot 下方负 GEX 绝对值最大的 strike）
    positive_gamma: bool       # 是否正 Gamma 环境
    # Max Pain
    max_pain: float | None = None  # Max Pain 价格（期权卖方痛苦最小的价位）
    # ΔOI 相关
    delta_oi_df: pd.DataFrame | None = None  # ΔOI 数据 (strike, call_delta_oi, put_delta_oi)
    max_call_delta_oi_strike: float | None = None  # Call ΔOI 最大的 strike
    max_put_delta_oi_strike: float | None = None   # Put ΔOI 最大的 strike
    # 降级模式标记
    partial: bool = False  # True = 尾盘降级模式，仅 ATM 附近少量 strike
    # 成交量只描述活动，不代表未平仓 dealer 仓位。
    volume_gamma: float = 0.0
    call_volume_gamma: float = 0.0
    put_volume_gamma: float = 0.0
    gamma_flip_method: str = "unavailable"
    gross_gex: float = 0.0
    net_gex_ratio: float = 0.0
    gross_volume_gamma: float = 0.0


def calculate_gex(
    tickers,
    spot: float,
    oi_ready_threshold: float = 0.8,
    prev_oi: dict[float, dict] | None = None,
    as_of: datetime | None = None,
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
    if not np.isfinite(spot) or spot <= 0:
        return None
    oi_ready_threshold = float(np.clip(oi_ready_threshold, 0.0, 1.0))
    rows = []
    missing_oi = 0
    missing_greeks = 0
    invalid_contracts = 0
    total_with_greeks = 0  # 有 Greeks 的合约数
    total_with_oi = 0

    for t in tickers:
        if t is None:
            continue
        g = t.modelGreeks
        if not g or g.gamma is None or not np.isfinite(g.gamma) or g.gamma < 0:
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

        # OI 是 position GEX 的唯一仓位输入。
        try:
            oi_value = float(oi)
        except (TypeError, ValueError):
            oi_value = np.nan
        if not np.isfinite(oi_value) or oi_value <= 0:
            missing_oi += 1
            oi_qty = 0
        else:
            oi_qty = oi_value

        # Volume 仅用于独立的活动度指标，不能代替未平仓量。
        vol = getattr(t, 'volume', None)
        try:
            vol_value = float(vol)
        except (TypeError, ValueError):
            vol_value = np.nan
        has_volume = np.isfinite(vol_value) and vol_value > 0
        if has_volume:
            vol_qty = vol_value
        else:
            vol_qty = 0

        if oi_qty > 0:
            total_with_oi += 1

        # dealer 约定: +1 for calls, -1 for puts
        multiplier = int(c.multiplier) if c.multiplier else 100
        gex_oi = sign * g.gamma * oi_qty * multiplier * spot ** 2 * 0.01
        volume_gamma = sign * g.gamma * vol_qty * multiplier * spot ** 2 * 0.01

        expiry = getattr(c, 'lastTradeDateOrContractMonth', None)
        expiry = expiry if isinstance(expiry, str) else None
        con_id = getattr(c, 'conId', None)
        con_id = int(con_id) if isinstance(con_id, (int, np.integer)) else None

        rows.append({
            'strike': c.strike,
            'right': c.right,
            'gamma': g.gamma,
            'oi': oi_qty,
            'volume': vol_qty,
            'has_volume': has_volume,
            'gex_oi': gex_oi,    # OI-based GEX
            'gex': gex_oi,       # 主 GEX 始终是 OI-based
            'volume_gamma': volume_gamma,
            'iv': g.impliedVol,
            'multiplier': multiplier,
            'expiry': expiry,
            'con_id': con_id,
        })

    if not rows:
        return None

    # 检查 OI 就绪比例。成交量不能让 position GEX 通过就绪检查。
    if total_with_greeks > 0:
        oi_ready_ratio = total_with_oi / total_with_greeks
        if oi_ready_ratio < oi_ready_threshold:
            return None  # OI 数据未就绪，等待

    df = pd.DataFrame(rows)

    # Gamma Flip：在候选 spot 上用 Black-Scholes 重新计算各合约 gamma。
    gamma_flip = _calculate_repriced_gamma_flip(df, spot, as_of=as_of)
    gamma_flip_method = "repriced_oi" if gamma_flip is not None else "unavailable"

    # Position GEX 总量：只用 OI。
    total_gex = df['gex'].sum()
    call_gex = df[df.right == 'C']['gex'].sum()
    put_gex = df[df.right == 'P']['gex'].sum()
    gross_gex = abs(call_gex) + abs(put_gex)
    net_gex_ratio = total_gex / gross_gex if gross_gex > 0 else 0.0
    total_volume_gamma = df['volume_gamma'].sum()
    call_volume_gamma = df[df.right == 'C']['volume_gamma'].sum()
    put_volume_gamma = df[df.right == 'P']['volume_gamma'].sum()
    gross_volume_gamma = abs(call_volume_gamma) + abs(put_volume_gamma)

    # ATM IV
    atm_iv_pct = _calculate_atm_iv(df, spot)

    # Walls 分别从 call/put OI gamma 中计算，避免同 strike 相互抵消。
    call_wall, put_wall = _calculate_side_walls(df, spot)

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

        current_expiries = {x for x in df['expiry'].dropna().unique() if x}
        current_expiry = next(iter(current_expiries)) if len(current_expiries) == 1 else None
        # 只比较当前窗口内、且 expiry 明确相同的合约。缺失不能解释为 OI=0。
        for strike in sorted(set(call_oi_today.index) | set(put_oi_today.index)):
            call_today = call_oi_today.get(strike, 0)
            put_today = put_oi_today.get(strike, 0)
            prev = prev_oi.get(strike)
            same_contract = (
                prev is not None
                and current_expiry is not None
                and prev.get('expiry') == current_expiry
            )
            call_prev = prev.get('call_oi') if same_contract else np.nan
            put_prev = prev.get('put_oi') if same_contract else np.nan

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
                valid_call = delta_oi_df['call_delta_oi'].dropna()
                valid_put = delta_oi_df['put_delta_oi'].dropna()
                if not valid_call.empty:
                    max_call_delta_oi_strike = delta_oi_df.loc[valid_call.idxmax(), 'strike']
                if not valid_put.empty:
                    max_put_delta_oi_strike = delta_oi_df.loc[valid_put.idxmax(), 'strike']

    # Max Pain 计算（使用 OI）
    max_pain = _calculate_max_pain(df)

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
        max_pain=max_pain,
        delta_oi_df=delta_oi_df,
        max_call_delta_oi_strike=max_call_delta_oi_strike,
        max_put_delta_oi_strike=max_put_delta_oi_strike,
        volume_gamma=total_volume_gamma,
        call_volume_gamma=call_volume_gamma,
        put_volume_gamma=put_volume_gamma,
        gamma_flip_method=gamma_flip_method,
        gross_gex=gross_gex,
        net_gex_ratio=net_gex_ratio,
        gross_volume_gamma=gross_volume_gamma,
    )


def _calculate_side_walls(df: pd.DataFrame, spot: float) -> tuple[float | None, float | None]:
    """Return the strongest OI call above spot and OI put below spot."""
    calls = df[(df['right'] == 'C') & (df['strike'] >= spot)]
    puts = df[(df['right'] == 'P') & (df['strike'] <= spot)]
    call_by_strike = calls.groupby('strike')['gex'].sum()
    put_by_strike = puts.groupby('strike')['gex'].sum()
    call_wall = float(call_by_strike.idxmax()) if not call_by_strike.empty else None
    put_wall = float(put_by_strike.idxmin()) if not put_by_strike.empty else None
    return call_wall, put_wall


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


def _calculate_gamma_flip(by_strike: pd.Series, spot: float) -> float | None:
    """
    计算累积 GEX 平衡点（兼容旧调用；不是严格的 repriced flip）。

    使用线性插值找到精确的零点位置，而不是简单取最近的 strike。

    Args:
        by_strike: 按 strike 汇总的 GEX Series
        spot: 当前现货价格

    Returns:
        离 spot 最近的平衡点；没有穿越时返回 None
    """
    if by_strike is None or len(by_strike) == 0 or not np.isfinite(spot):
        return None

    if len(by_strike) == 1:
        return None

    cumsum = by_strike.cumsum()

    values = cumsum.to_numpy(dtype=float)
    exact = np.flatnonzero(np.isclose(values, 0.0, atol=1e-12))
    candidates = [float(cumsum.index[i]) for i in exact]
    signs = np.sign(values)
    sign_changes = np.where(signs[:-1] * signs[1:] < 0)[0]

    for idx in sign_changes:
        strike_low = by_strike.index[idx]
        strike_high = by_strike.index[idx + 1]
        cumsum_low = cumsum.iloc[idx]
        cumsum_high = cumsum.iloc[idx + 1]

        # 线性插值: 找到 cumsum = 0 的位置
        # cumsum_low + (cumsum_high - cumsum_low) * t = 0
        # t = -cumsum_low / (cumsum_high - cumsum_low)
        if cumsum_high != cumsum_low:
            t = -cumsum_low / (cumsum_high - cumsum_low)
            candidate = strike_low + (strike_high - strike_low) * t
        else:
            candidate = (strike_low + strike_high) / 2
        candidates.append(float(candidate))

    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - spot))


def _parse_expiry_close(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) < 8:
        return None
    try:
        day = datetime.strptime(value[:8], "%Y%m%d").date()
    except ValueError:
        return None
    return datetime.combine(day, time(16, 0), tzinfo=ET)


def _calculate_repriced_gamma_flip(
    df: pd.DataFrame,
    spot: float,
    *,
    as_of: datetime | None = None,
    risk_free_rate: float = 0.05,
    search_pct: float = 0.15,
    grid_points: int = 301,
) -> float | None:
    """Solve net OI gamma(S)=0 after re-pricing gamma across candidate spots.

    IV is held constant per contract. This is still a positioning estimate—the
    call-positive/put-negative dealer convention is an assumption—but unlike a
    strike cumsum it answers the actual spot-perturbation question.
    """
    required = {'strike', 'right', 'oi', 'iv', 'multiplier', 'expiry'}
    if df is None or df.empty or not required.issubset(df.columns):
        return None
    if not np.isfinite(spot) or spot <= 0 or grid_points < 3:
        return None

    now = as_of or datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)

    records = []
    for row in df[list(required)].to_dict('records'):
        expiry_close = _parse_expiry_close(row['expiry'])
        if expiry_close is None:
            continue
        t_years = (expiry_close - now).total_seconds() / (365.0 * 86400.0)
        try:
            strike = float(row['strike'])
            oi = float(row['oi'])
            iv = float(row['iv'])
            multiplier = float(row['multiplier'])
        except (TypeError, ValueError):
            continue
        if not (t_years > 0 and strike > 0 and oi > 0 and 0 < iv < 5 and multiplier > 0):
            continue
        sign = 1.0 if row['right'] == 'C' else -1.0 if row['right'] == 'P' else 0.0
        if sign:
            records.append((strike, oi, iv, multiplier, t_years, sign))

    if len(records) < 2:
        return None

    arr = np.asarray(records, dtype=float)
    strikes, oi, iv, multiplier, t_years, signs = arr.T
    # Never extrapolate a flip beyond the observed strike universe. A root at
    # the edge is evidence that the subscription window is too narrow, not a
    # reliable market level.
    lower = max(spot * (1.0 - search_pct), float(strikes.min()))
    upper = min(spot * (1.0 + search_pct), float(strikes.max()))
    if not lower < upper:
        return None
    spots = np.linspace(lower, upper, grid_points)
    sqrt_t = np.sqrt(t_years)[None, :]
    sigma = iv[None, :]
    spot_grid = spots[:, None]
    d1 = (
        np.log(spot_grid / strikes[None, :])
        + (risk_free_rate + 0.5 * sigma ** 2) * t_years[None, :]
    ) / (sigma * sqrt_t)
    normal_pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi)
    gamma = normal_pdf / (spot_grid * sigma * sqrt_t)
    exposures = (
        gamma * oi[None, :] * multiplier[None, :] * signs[None, :]
        * spot_grid ** 2 * 0.01
    ).sum(axis=1)

    exact = np.flatnonzero(np.isclose(exposures, 0.0, atol=1e-8))
    candidates = [float(spots[i]) for i in exact]
    changes = np.flatnonzero(exposures[:-1] * exposures[1:] < 0)
    for idx in changes:
        y0, y1 = exposures[idx], exposures[idx + 1]
        x0, x1 = spots[idx], spots[idx + 1]
        candidates.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))

    if not candidates:
        return None
    unique_strikes = np.unique(strikes)
    spacing = (
        float(np.median(np.diff(unique_strikes)))
        if len(unique_strikes) > 1 else (upper - lower)
    )
    interior = [x for x in candidates if lower + spacing <= x <= upper - spacing]
    if not interior:
        return None
    return min(interior, key=lambda x: abs(x - spot))


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

    iv_numeric = pd.to_numeric(df['iv'], errors='coerce')
    atm_rows = df[
        (df['strike'] == atm_strike)
        & iv_numeric.notna()
        & (iv_numeric > 0)
        & (iv_numeric < 5)
    ]
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


def _calculate_max_pain(df: pd.DataFrame) -> float | None:
    """
    计算 Max Pain（最大痛点）

    Max Pain 是使期权买方总内在价值最小的价格，
    即期权卖方（通常是做市商）痛苦最小、获利最大的价位。

    计算方法：
    - 遍历每个候选价格 P（使用所有 strike）
    - 对每个 P，计算所有期权到期时的总内在价值
    - 找出使总内在价值最小的 P

    Args:
        df: 期权数据 DataFrame，需包含 strike, right, oi 列

    Returns:
        Max Pain 价格，或 None（无有效数据时）
    """
    if df.empty or 'oi' not in df.columns:
        return None

    # 按 strike 和 right 聚合 OI
    call_oi = df[df['right'] == 'C'].groupby('strike')['oi'].sum()
    put_oi = df[df['right'] == 'P'].groupby('strike')['oi'].sum()

    if call_oi.empty and put_oi.empty:
        return None
    if float(call_oi.sum()) + float(put_oi.sum()) <= 0:
        return None

    # 获取所有 strikes 作为候选价格
    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))
    if not all_strikes:
        return None

    # 计算每个候选价格的总内在价值（买方总获利 = 卖方总亏损）
    min_pain = float('inf')
    max_pain_strike = None

    for P in all_strikes:
        total_intrinsic = 0.0

        # Call 内在价值: max(P - K, 0) * OI * 100
        for K, oi in call_oi.items():
            if P > K:
                total_intrinsic += (P - K) * oi * 100

        # Put 内在价值: max(K - P, 0) * OI * 100
        for K, oi in put_oi.items():
            if K > P:
                total_intrinsic += (K - P) * oi * 100

        if total_intrinsic < min_pain:
            min_pain = total_intrinsic
            max_pain_strike = P

    return max_pain_strike


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
