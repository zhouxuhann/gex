"""
0DTE Skew 实时指标模块

三个核心度量:
  1. ATM IV — 基准波动率
  2. 25-delta Risk Reversal — IV(25Δ put) - IV(25Δ call)，衡量方向性恐慌
  3. OTM Skew Slope — (OTM put IV - OTM call IV) / ATM IV，归一化尾部偏斜

与 GEX 结合的信号:
  - 负 Gamma + RR z-score > 2.0 → BEARISH_ACCELERATION（对冲压力骤增）
  - 正 Gamma + RR z-score < -1.5 → PIN_BREAK_RISK（pin 失效风险）

注意:
  - IB modelGreeks 在 14:30 ET 后 0DTE 远端 IV 噪声很大
  - rolling z-score 需要至少 30 个 1 分钟采样点才有意义
"""
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# OTM strike 偏移比例（spot ±2%）
OTM_PUT_OFFSET = 0.98
OTM_CALL_OFFSET = 1.02

# 目标 delta
TARGET_DELTA_25 = 0.25

# z-score 信号阈值
BEARISH_ACCEL_THRESHOLD = 2.0
PIN_BREAK_THRESHOLD = -1.5


@dataclass
class SkewSnapshot:
    """单次 skew 计算结果"""
    atm_iv: float | None           # ATM IV（小数，如 0.25 = 25%）
    rr_25: float | None            # 25Δ risk reversal（put IV - call IV）
    skew_slope: float | None       # OTM skew slope，归一化
    rr_25_zscore: float | None     # RR 的日内 z-score
    signal: str | None             # GEX+skew 联合信号


def compute_skew(tickers, spot: float) -> SkewSnapshot | None:
    """
    从 IB tickers 计算 skew 指标

    复用 GEX 引擎已订阅的 tickers，不需要额外请求。

    Args:
        tickers: IB ticker 列表（和 calculate_gex 相同的输入）
        spot: 当前现货价格

    Returns:
        SkewSnapshot 或 None
    """
    if not tickers or spot <= 0:
        return None

    rows = []
    for t in tickers:
        if t is None:
            continue
        g = t.modelGreeks
        if not g or g.impliedVol is None or g.delta is None:
            continue
        c = t.contract
        if c.right not in ('C', 'P'):
            continue
        rows.append({
            'strike': c.strike,
            'right': c.right,
            'iv': g.impliedVol,
            'delta': g.delta,
        })

    if len(rows) < 4:  # 至少需要几个合约
        return None

    df = pd.DataFrame(rows)
    puts = df[df.right == 'P'].copy()
    calls = df[df.right == 'C'].copy()

    if puts.empty or calls.empty:
        return None

    # 1) ATM IV
    atm_iv = _calc_atm_iv(puts, calls, spot)

    # 2) 25-delta Risk Reversal
    rr_25 = _calc_risk_reversal(puts, calls, TARGET_DELTA_25)

    # 3) OTM Skew Slope
    skew_slope = _calc_skew_slope(puts, calls, spot, atm_iv)

    return SkewSnapshot(
        atm_iv=atm_iv,
        rr_25=rr_25,
        skew_slope=skew_slope,
        rr_25_zscore=None,  # SkewTracker 填充
        signal=None,        # SkewTracker 填充
    )


def _calc_atm_iv(puts: pd.DataFrame, calls: pd.DataFrame, spot: float) -> float | None:
    """ATM IV: 最接近 spot 的 put + call IV 平均"""
    atm_put = puts.iloc[(puts['strike'] - spot).abs().argsort()[:1]]
    atm_call = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]

    ivs = []
    if not atm_put.empty:
        ivs.append(atm_put.iloc[0]['iv'])
    if not atm_call.empty:
        ivs.append(atm_call.iloc[0]['iv'])

    return float(np.mean(ivs)) if ivs else None


def _calc_risk_reversal(
    puts: pd.DataFrame, calls: pd.DataFrame, target_delta: float
) -> float | None:
    """
    25-delta Risk Reversal: IV(25Δ put) - IV(25Δ call)

    正值 = put 比 call 贵 = 市场偏空恐慌
    """
    put_iv = _iv_at_delta(puts, target_delta)
    call_iv = _iv_at_delta(calls, target_delta)

    if put_iv is None or call_iv is None:
        return None
    return float(put_iv - call_iv)


def _iv_at_delta(side_df: pd.DataFrame, target_delta: float) -> float | None:
    """找到最接近目标 |delta| 的合约 IV"""
    valid = side_df.dropna(subset=['delta', 'iv'])
    if valid.empty:
        return None

    # delta 取绝对值匹配
    diffs = (valid['delta'].abs() - target_delta).abs()
    idx = diffs.idxmin()

    # 如果最近的 delta 偏差超过 0.15，认为数据不可靠
    if diffs.loc[idx] > 0.15:
        return None

    return float(valid.loc[idx, 'iv'])


def _calc_skew_slope(
    puts: pd.DataFrame, calls: pd.DataFrame, spot: float, atm_iv: float | None
) -> float | None:
    """
    OTM skew slope: (OTM put IV - OTM call IV) / ATM IV

    衡量尾部偏斜程度，归一化后可跨时间比较
    """
    if atm_iv is None or atm_iv < 1e-6:
        return None

    # OTM put: spot * 0.98 附近
    otm_put_strike = spot * OTM_PUT_OFFSET
    otm_put = puts.iloc[(puts['strike'] - otm_put_strike).abs().argsort()[:1]]

    # OTM call: spot * 1.02 附近
    otm_call_strike = spot * OTM_CALL_OFFSET
    otm_call = calls.iloc[(calls['strike'] - otm_call_strike).abs().argsort()[:1]]

    if otm_put.empty or otm_call.empty:
        return None

    otm_put_iv = otm_put.iloc[0]['iv']
    otm_call_iv = otm_call.iloc[0]['iv']

    if pd.isna(otm_put_iv) or pd.isna(otm_call_iv):
        return None

    return float((otm_put_iv - otm_call_iv) / atm_iv)


class SkewTracker:
    """
    维护 skew 日内 rolling 统计，计算 z-score 和联合信号

    在 IBWorker 中实例化，每次 tick 调用 update()
    """

    def __init__(self, window: int = 30):
        """
        Args:
            window: rolling 窗口大小（tick 数），默认 30
                    3s tick 间隔 × 30 ≈ 90s 窗口
        """
        self.window = window
        self._rr_history: deque[float] = deque(maxlen=window)

    def update(
        self, snapshot: SkewSnapshot | None, positive_gamma: bool
    ) -> SkewSnapshot | None:
        """
        更新 rolling 统计并生成信号

        Args:
            snapshot: 当前 skew 快照
            positive_gamma: 当前是否正 Gamma 环境（来自 GEX）

        Returns:
            enriched SkewSnapshot（填充了 z-score 和 signal）
        """
        if snapshot is None:
            return None

        # 更新 rolling 窗口
        if snapshot.rr_25 is not None:
            self._rr_history.append(snapshot.rr_25)

        # 计算 z-score
        zscore = self._calc_zscore()
        snapshot.rr_25_zscore = zscore

        # 联合信号
        snapshot.signal = self._classify_signal(zscore, positive_gamma)

        return snapshot

    def _calc_zscore(self) -> float | None:
        """计算 RR 的 rolling z-score"""
        if len(self._rr_history) < 10:  # 冷启动：至少 10 个点
            return None

        arr = np.array(self._rr_history)
        mean = arr.mean()
        std = arr.std()

        if std < 1e-8:
            return 0.0

        return float((arr[-1] - mean) / std)

    def _classify_signal(
        self, zscore: float | None, positive_gamma: bool
    ) -> str | None:
        """
        GEX + skew 联合信号分类

        - 负 Gamma + RR z > 2.0  → BEARISH_ACCELERATION
        - 正 Gamma + RR z < -1.5 → PIN_BREAK_RISK
        - 否则 → None
        """
        if zscore is None:
            return None

        if not positive_gamma and zscore > BEARISH_ACCEL_THRESHOLD:
            return 'BEARISH_ACCELERATION'
        elif positive_gamma and zscore < PIN_BREAK_THRESHOLD:
            return 'PIN_BREAK_RISK'

        return None

    def reset(self) -> None:
        """重置 rolling 窗口（新交易日调用）"""
        self._rr_history.clear()
