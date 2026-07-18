"""
对冲信号引擎

基于 skew surface + GEX regime 生成对冲建议。

决策矩阵:
┌──────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│ GEX Regime   │ Skew 便宜 (<30th)   │ Skew 正常 (30-70th) │ Skew 贵 (>70th)     │
├──────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ 负 Gamma     │ HEDGE_NOW (outright)│ HEDGE_SPREAD        │ HEDGE_SPREAD (宽)   │
│ 中性         │ HEDGE_NOW (spread)  │ MONITOR             │ SKIP                │
│ 正 Gamma     │ SKIP (省钱)         │ SKIP                │ SKIP                │
└──────────────┴─────────────────────┴─────────────────────┴─────────────────────┘

Term structure 修正:
  - Backwardation (近 > 远) → urgency +1 级
  - 强 backwardation → 至少 HEDGE_SPREAD

用法:
  from .hedge_signal import generate_hedge_signal, format_recommendation
  signal = generate_hedge_signal(surface, history_df, gex_regime)
  print(format_recommendation(signal))
"""
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 分位数阈值
CHEAP_THRESHOLD = 30   # percentile
EXPENSIVE_THRESHOLD = 70
MIN_HISTORY_DAYS = 20
HEDGE_DTE_MIN = 20
HEDGE_DTE_MAX = 55
HISTORY_DTE_TOLERANCE = 5

# urgency 权重
URGENCY_NEGATIVE_GAMMA = 0.4
URGENCY_CHEAP_SKEW = 0.3
URGENCY_BACKWARDATION = 0.3


@dataclass
class HedgeSignal:
    """对冲信号"""
    ts: datetime
    symbol: str
    action: str                    # HEDGE_NOW, HEDGE_SPREAD, MONITOR, SKIP, REDUCE
    urgency: float                 # 0.0 - 1.0
    skew_cheapness: float          # 0-100 分位数 (低=便宜)
    gex_regime: str                # positive / negative / neutral
    term_structure: str            # contango / backwardation / flat
    recommended_structure: str     # outright_put / put_spread / collar / none
    recommended_tenor: str         # 1W / 2W
    reasoning: str                 # 人类可读解释
    data_quality: str = 'good'
    hedge_tenor_dte: int | None = None
    rr_25_current: float | None = None
    history_days: int = 0

    def to_dict(self) -> dict:
        return {
            'ts': self.ts,
            'symbol': self.symbol,
            'action': self.action,
            'urgency': self.urgency,
            'skew_cheapness': self.skew_cheapness,
            'gex_regime': self.gex_regime,
            'term_structure': self.term_structure,
            'recommended_structure': self.recommended_structure,
            'recommended_tenor': self.recommended_tenor,
            'reasoning': self.reasoning,
            'data_quality': self.data_quality,
            'hedge_tenor_dte': self.hedge_tenor_dte,
            'rr_25_current': self.rr_25_current,
            'history_days': self.history_days,
        }


def generate_hedge_signal(
    surface,                          # SkewSurface (from skew_surface.py)
    history_df: pd.DataFrame | None,  # 历史 skew surface records
    gex_regime: str = 'neutral',      # positive / negative / neutral
    last_signal: dict | None = None,  # 上一次信号（用于 REDUCE 判断）
    macro=None,                       # MacroSnapshot（可选，宏观 urgency 调整）
) -> HedgeSignal:
    """
    生成对冲信号

    Args:
        surface: 当前 SkewSurface
        history_df: 过去 ~20 天的 skew surface 数据
        gex_regime: 当前 GEX regime
        last_signal: 上次的信号 dict（可选）
        macro: MacroSnapshot（可选，提供 VIX/MOVE/SOFR-OIS 宏观调整）

    Returns:
        HedgeSignal
    """
    now = surface.ts
    symbol = surface.symbol

    # 1. 只允许使用真正可执行的 20-55D tenor，不再 fallback 到 7D/0D。
    hedge_tenor = _select_hedge_tenor(surface)
    rr_25_current = hedge_tenor.rr_25 if hedge_tenor is not None else None
    target_dte = hedge_tenor.dte if hedge_tenor is not None else None
    cheapness, history_days = _compute_cheapness_details(
        rr_25_current, history_df, target_dte=target_dte,
        current_ts=surface.ts,
    )
    data_quality = 'good'
    if hedge_tenor is None:
        data_quality = 'insufficient_current_tenor'
    elif history_days < MIN_HISTORY_DAYS:
        data_quality = 'insufficient_same_tenor_history'

    # 2. Term structure 分类
    term_struct = _classify_term_structure(surface.term_spread_rr25, surface.term_spread_iv)

    # 3. 数据不足时只记录 MONITOR，禁止进入下单路径。
    if data_quality != 'good':
        action, structure = 'MONITOR', 'none'
    else:
        action, structure = _decide_action(
            cheapness, gex_regime, term_struct, last_signal
        )

    # 4. Urgency 计算（skew + GEX + term structure + 宏观）
    urgency = _compute_urgency(cheapness, gex_regime, term_struct)
    if macro is not None:
        urgency = min(1.0, max(0.0, urgency + macro.urgency_adjustment))

    # 5. Tenor 推荐
    tenor = _recommend_tenor(surface, hedge_tenor)

    # 6. 生成理由
    reasoning = _build_reasoning(
        action, cheapness, gex_regime, term_struct, rr_25_current, surface, macro,
        data_quality=data_quality, history_days=history_days,
    )

    return HedgeSignal(
        ts=now,
        symbol=symbol,
        action=action,
        urgency=urgency,
        skew_cheapness=cheapness,
        gex_regime=gex_regime,
        term_structure=term_struct,
        recommended_structure=structure,
        recommended_tenor=tenor,
        reasoning=reasoning,
        data_quality=data_quality,
        hedge_tenor_dte=target_dte,
        rr_25_current=rr_25_current,
        history_days=history_days,
    )


def _select_hedge_tenor(surface):
    """选择有效的 30-45DTE 对冲 tenor；绝不跨期限 fallback。"""
    candidates = [
        tenor for tenor in surface.tenors
        if HEDGE_DTE_MIN <= tenor.dte <= HEDGE_DTE_MAX
        and tenor.rr_25 is not None
        and getattr(tenor, 'quality', 'good') != 'bad'
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda tenor: abs(tenor.dte - 45))


def _get_current_rr25(surface) -> float | None:
    """向后兼容的 helper，仅返回合格中长期 tenor。"""
    tenor = _select_hedge_tenor(surface)
    return tenor.rr_25 if tenor is not None else None


def _compute_cheapness(
    rr_25_current: float | None,
    history_df: pd.DataFrame | None,
    target_dte: int | None = None,
) -> float:
    """
    计算 skew cheapness score (0-100)

    0 = 极便宜（put skew 很平，保险打折）
    100 = 极贵（put skew 很陡，保险溢价高）
    50 = 中位数

    没有历史数据时返回 50（中性）
    """
    cheapness, _ = _compute_cheapness_details(
        rr_25_current, history_df,
        target_dte=45 if target_dte is None else target_dte,
    )
    return cheapness


def _compute_cheapness_details(
    rr_25_current: float | None,
    history_df: pd.DataFrame | None,
    *,
    target_dte: int | None,
    current_ts=None,
) -> tuple[float, int]:
    """仅用同期限历史计算分位数，并返回独立历史日数。"""
    if rr_25_current is None or target_dte is None:
        return 50.0, 0
    if history_df is None or history_df.empty or 'dte' not in history_df:
        return 50.0, 0
    hist = history_df[
        history_df['dte'].between(
            target_dte - HISTORY_DTE_TOLERANCE,
            target_dte + HISTORY_DTE_TOLERANCE,
        )
    ].copy()
    hist = hist.dropna(subset=['rr_25'])
    if 'quality' in hist:
        hist = hist[hist['quality'] != 'bad']
    if 'ts' in hist:
        hist['_date'] = pd.to_datetime(hist['ts'], utc=True).dt.date
        if current_ts is not None:
            current_date = pd.Timestamp(current_ts).date()
            hist = hist[hist['_date'] != current_date]
        daily = hist.sort_values('ts').drop_duplicates('_date', keep='last')
        history_days = int(daily['_date'].nunique())
        hist_rr = daily['rr_25']
    else:
        hist_rr = hist['rr_25']
        history_days = int(len(hist_rr))
    if history_days < MIN_HISTORY_DAYS:
        return 50.0, history_days
    return float((hist_rr < rr_25_current).mean() * 100), history_days


def _classify_term_structure(
    term_spread_rr25: float | None,
    term_spread_iv: float | None,
) -> str:
    """
    分类 term structure

    - contango: 近端 < 远端（正常，远端溢价高）
    - backwardation: 近端 > 远端（异常，近端恐慌更重）
    - flat: 差异不大
    """
    if term_spread_rr25 is None and term_spread_iv is None:
        return 'unavailable'

    # 优先看 RR25 的 term structure
    spread = term_spread_rr25 if term_spread_rr25 is not None else term_spread_iv

    if spread is None:
        return 'flat'

    # 阈值: IV 差超过 1% 才算有方向
    if spread > 0.01:
        return 'backwardation'  # 近端 RR 更大（近端 put 更贵）
    elif spread < -0.01:
        return 'contango'       # 远端 RR 更大（正常）
    else:
        return 'flat'


def _decide_action(
    cheapness: float,
    gex_regime: str,
    term_struct: str,
    last_signal: dict | None,
) -> tuple[str, str]:
    """
    决策矩阵

    Returns:
        (action, recommended_structure)
    """
    is_cheap = cheapness < CHEAP_THRESHOLD
    is_expensive = cheapness > EXPENSIVE_THRESHOLD
    is_negative = gex_regime == 'negative'
    is_positive = gex_regime == 'positive'
    is_backwardation = term_struct == 'backwardation'

    # REDUCE: 之前建了对冲，现在环境好转
    if last_signal and last_signal.get('action') in ('HEDGE_NOW', 'HEDGE_SPREAD'):
        if is_positive and is_expensive:
            return 'REDUCE', 'none'

    # 负 Gamma: 尾部风险高，积极对冲
    if is_negative:
        if is_cheap:
            return 'HEDGE_NOW', 'outright_put'
        elif is_expensive:
            return 'HEDGE_SPREAD', 'put_spread'
        else:
            return 'HEDGE_SPREAD', 'put_spread'

    # 中性 Gamma
    if not is_positive and not is_negative:
        if is_cheap:
            return 'HEDGE_NOW', 'put_spread'
        elif is_backwardation:
            return 'HEDGE_SPREAD', 'put_spread'
        elif is_expensive:
            return 'SKIP', 'none'
        else:
            return 'MONITOR', 'none'

    # 正 Gamma: 市场 pinned，尾部风险低
    if is_positive:
        # 即使正 Gamma，backwardation 也要注意
        if is_backwardation and is_cheap:
            return 'HEDGE_SPREAD', 'put_spread'
        return 'SKIP', 'none'

    return 'MONITOR', 'none'


def _compute_urgency(
    cheapness: float,
    gex_regime: str,
    term_struct: str,
) -> float:
    """计算紧迫度 0.0 - 1.0"""
    u = 0.0

    # Gamma regime 贡献
    if gex_regime == 'negative':
        u += URGENCY_NEGATIVE_GAMMA
    elif gex_regime == 'neutral':
        u += URGENCY_NEGATIVE_GAMMA * 0.3

    # Skew cheapness 贡献（越便宜越该买）
    cheap_score = max(0, (100 - cheapness)) / 100  # 0=贵，1=便宜
    u += URGENCY_CHEAP_SKEW * cheap_score

    # Term structure 贡献
    if term_struct == 'backwardation':
        u += URGENCY_BACKWARDATION

    return min(1.0, u)


def _recommend_tenor(surface, selected_tenor=None) -> str:
    """
    推荐对冲 tenor

    默认 30-45DTE（theta 效率最优的甜蜜区间）。
    比较两个候选 tenor 的每日 IV 成本，推荐更划算的。
    """
    if selected_tenor is not None:
        return f'{selected_tenor.dte}D (exp {selected_tenor.expiry})'

    valid = [t for t in surface.tenors
             if HEDGE_DTE_MIN <= t.dte <= HEDGE_DTE_MAX
             and t.atm_iv is not None and t.rr_25 is not None]
    t30 = min(valid, key=lambda t: abs(t.dte - 30)) if valid else None
    t45 = min(valid, key=lambda t: abs(t.dte - 45)) if valid else None

    # 有 30/45DTE 数据时，比较每日 IV 成本
    if t30 is not None and t45 is not None:
        if t30.atm_iv is not None and t45.atm_iv is not None:
            iv_per_day_30 = t30.atm_iv / max(t30.dte, 1)
            iv_per_day_45 = t45.atm_iv / max(t45.dte, 1)
            # 45DTE 每日成本通常更低，但如果 30DTE 明显便宜就用 30DTE
            if iv_per_day_30 < iv_per_day_45 * 0.85:
                return f'30D (exp {t30.expiry})'
        return f'45D (exp {t45.expiry})'

    if t45 is not None:
        return f'45D (exp {t45.expiry})'
    if t30 is not None:
        return f'30D (exp {t30.expiry})'

    return 'unavailable'


def _build_reasoning(
    action: str,
    cheapness: float,
    gex_regime: str,
    term_struct: str,
    rr_25: float | None,
    surface,
    macro=None,
    data_quality: str = 'good',
    history_days: int = 0,
) -> str:
    """生成人类可读的理由"""
    parts = []

    if data_quality != 'good':
        quality_map = {
            'insufficient_current_tenor': '20-55DTE RR25 缺失',
            'insufficient_same_tenor_history': (
                f'同期限历史仅 {history_days} 天，需要 {MIN_HISTORY_DAYS} 天'
            ),
        }
        parts.append(f"数据质量不足: {quality_map.get(data_quality, data_quality)}")

    # Skew 状态
    rr_txt = f"{rr_25*100:.1f}%" if rr_25 is not None else "N/A"
    if cheapness < CHEAP_THRESHOLD:
        parts.append(f"Put skew 偏低 (RR={rr_txt}, {cheapness:.0f}th pct) — 保险便宜")
    elif cheapness > EXPENSIVE_THRESHOLD:
        parts.append(f"Put skew 偏高 (RR={rr_txt}, {cheapness:.0f}th pct) — 保险溢价")
    else:
        parts.append(f"Put skew 正常 (RR={rr_txt}, {cheapness:.0f}th pct)")

    # GEX regime
    regime_map = {
        'negative': '负 Gamma 环境 — 尾部风险较高',
        'neutral': '中性 Gamma',
        'positive': '正 Gamma 环境 — 市场趋于 pin',
    }
    parts.append(regime_map.get(gex_regime, f'Gamma: {gex_regime}'))

    # Term structure
    ts_map = {
        'backwardation': 'Term structure backwardation — 近端紧张',
        'contango': 'Term structure contango — 结构正常',
        'flat': 'Term structure 平坦',
        'unavailable': 'Term structure 数据不足',
    }
    parts.append(ts_map.get(term_struct, f'Term: {term_struct}'))

    # 宏观环境
    if macro is not None:
        parts.append(macro.summary())

    # 结论
    action_map = {
        'HEDGE_NOW': '建议立即对冲',
        'HEDGE_SPREAD': '建议用 spread 对冲（控制成本）',
        'MONITOR': '观望，暂不操作',
        'SKIP': '不建议对冲（成本不合理或风险低）',
        'REDUCE': '环境好转，可考虑减仓已有对冲',
    }
    parts.append(f"→ {action_map.get(action, action)}")

    return ' | '.join(parts)


# ============================================================
# 终端输出
# ============================================================

def format_recommendation(signal: HedgeSignal) -> str:
    """格式化对冲建议为终端可读文本"""
    ACTION_ICONS = {
        'HEDGE_NOW': '\033[91m🛡️  HEDGE NOW\033[0m',
        'HEDGE_SPREAD': '\033[93m🛡️  HEDGE (SPREAD)\033[0m',
        'MONITOR': '\033[94m👀 MONITOR\033[0m',
        'SKIP': '\033[92m✓  SKIP\033[0m',
        'REDUCE': '\033[95m📉 REDUCE\033[0m',
    }

    icon = ACTION_ICONS.get(signal.action, signal.action)
    urgency_bar = '█' * int(signal.urgency * 10) + '░' * (10 - int(signal.urgency * 10))

    lines = [
        '',
        f"{'='*60}",
        f"  {signal.symbol} Hedge Signal — {signal.ts.strftime('%Y-%m-%d %H:%M ET')}",
        f"{'='*60}",
        '',
        f"  Action:     {icon}",
        f"  Urgency:    [{urgency_bar}] {signal.urgency:.0%}",
        f"  Structure:  {signal.recommended_structure}",
        f"  Tenor:      {signal.recommended_tenor}",
        f"  Data:       {signal.data_quality} ({signal.history_days} history days)",
        '',
        f"  Skew:       {signal.skew_cheapness:.0f}th percentile "
        f"({'cheap' if signal.skew_cheapness < 30 else 'expensive' if signal.skew_cheapness > 70 else 'normal'})",
        f"  GEX:        {signal.gex_regime} gamma",
        f"  Term Str:   {signal.term_structure}",
        '',
        f"  {signal.reasoning}",
        '',
        f"{'='*60}",
    ]
    return '\n'.join(lines)
