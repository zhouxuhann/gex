"""
Price Action 信号模块 — Brooks 风格 H2/L2 回调计数 + 信号棒评分

独立模块，无 IB 依赖。接收 OHLCV bar，输出信号。
可与 KDJ、GEX 等其他信号源组合使用。

用法：
    from pa_signal import PASignalEngine

    pa = PASignalEngine()

    # 每根 5min bar 闭合时调用
    signal = pa.on_bar(open, high, low, close, volume)
    if signal:
        print(signal)  # {'direction': 'LONG', 'setup': 'H2', 'score': 72, ...}
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def bar_range(self) -> float:
        return max(self.high - self.low, 1e-6)

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bull(self) -> bool:
        return self.close >= self.open

    @property
    def is_bear(self) -> bool:
        return self.close < self.open


class PASignalEngine:
    """
    Price Action 信号引擎

    核心逻辑：
    - EMA 判断 Always-In 方向（多/空趋势）
    - H 计数：多头趋势中，连续高点下降的 bar 计数（回调深度）
    - L 计数：空头趋势中，连续低点上升的 bar 计数
    - H2 = 多头回调第2脚 → 做多信号（经典回调买入）
    - L2 = 空头回调第2脚 → 做空信号
    - 信号棒六维度评分（实体比、收盘位、影线、gap、突破、成交量）
    """

    def __init__(self,
                 ema_len: int = 20,
                 atr_len: int = 14,
                 min_score: int = 55,
                 reset_bars: int = 30,
                 min_rr: float = 1.5):
        self.ema_len = ema_len
        self.atr_len = atr_len
        self.min_score = min_score
        self.reset_bars = reset_bars
        self.min_rr = min_rr

        self._bars: deque[Bar] = deque(maxlen=200)
        self.ema = 0.0
        self.atr = 0.0
        self.always_in_long = True
        self.h_count = 0
        self.l_count = 0
        self._h_last_bar = 0
        self._l_last_bar = 0
        self._bar_index = 0
        self._bars_from_ema = 0
        self._last_signal_bar = -1
        self.ready = False

    # ── Public API ──

    def on_bar(self, open_p: float, high: float, low: float,
               close: float, volume: float = 0) -> Optional[dict]:
        """
        每根 5min bar 闭合时调用。

        Returns:
            signal dict 或 None
            signal = {
                'direction': 'LONG' | 'SHORT',
                'setup': 'H2' | 'L2' | 'H1_SPIKE' | 'L1_SPIKE' | 'H3' | 'L3',
                'score': int (0-100),
                'entry': float,
                'stop': float,
                'target': float,
                'rr_ratio': float,
                'reason': str,
            }
        """
        bar = Bar(open_p, high, low, close, volume)
        self._bars.append(bar)
        self._bar_index += 1
        self._update_indicators(bar)
        self._update_counts(bar)

        if len(self._bars) < 3 or self.atr <= 0:
            return None

        if not self.ready and len(self._bars) >= self.ema_len:
            self.ready = True

        if not self.ready:
            return None

        return self._check_signal(bar)

    @property
    def state(self) -> dict:
        """当前状态（用于 UI 显示或日志）"""
        return {
            'ema': round(self.ema, 2),
            'atr': round(self.atr, 3),
            'always_in': 'LONG' if self.always_in_long else 'SHORT',
            'h_count': self.h_count,
            'l_count': self.l_count,
            'bars_from_ema': self._bars_from_ema,
        }

    # ── Indicators ──

    def _update_indicators(self, bar: Bar):
        # EMA
        if self.ema == 0.0:
            self.ema = bar.close
        else:
            k = 2.0 / (self.ema_len + 1)
            self.ema = bar.close * k + self.ema * (1 - k)

        # ATR
        bars = list(self._bars)
        if len(bars) >= 2:
            trs = []
            for i in range(1, min(self.atr_len + 1, len(bars))):
                b = bars[-i]
                pb = bars[-i - 1]
                tr = max(b.high, pb.close) - min(b.low, pb.close)
                trs.append(tr)
            self.atr = sum(trs) / len(trs) if trs else 0.0

        # 距 EMA 的 bar 数
        if self.atr > 0:
            dist = abs(bar.close - self.ema) / self.atr
            if dist > 1.0:
                self._bars_from_ema += 1
            else:
                self._bars_from_ema = 0

        # Always-In 方向
        if len(bars) >= 3:
            recent = [b.close for b in bars[-3:]]
            above = sum(1 for c in recent if c > self.ema)
            self.always_in_long = above >= 2

    # ── H/L Count ──

    def _update_counts(self, bar: Bar):
        bars = list(self._bars)

        # H 计数超时重置
        if self.h_count > 0 and (self._bar_index - self._h_last_bar) > self.reset_bars:
            self.h_count = 0

        # 新高重置 H 计数
        if len(bars) >= 10:
            recent_high = max(b.high for b in bars[-10:-1])
            if bar.high > recent_high:
                self.h_count = 0

        # H 计数递增（多头趋势中高点下降）
        if (self.always_in_long and len(bars) >= 2 and
                bar.high < bars[-2].high and
                self.h_count < 3 and
                (self.h_count == 0 or
                 (self._bar_index - self._h_last_bar) <= self.reset_bars)):
            self.h_count += 1
            self._h_last_bar = self._bar_index

        # L 计数超时重置
        if self.l_count > 0 and (self._bar_index - self._l_last_bar) > self.reset_bars:
            self.l_count = 0

        # 新低重置 L 计数
        if len(bars) >= 10:
            recent_low = min(b.low for b in bars[-10:-1])
            if bar.low < recent_low:
                self.l_count = 0

        # L 计数递增（空头趋势中低点上升）
        if (not self.always_in_long and len(bars) >= 2 and
                bar.low > bars[-2].low and
                self.l_count < 3 and
                (self.l_count == 0 or
                 (self._bar_index - self._l_last_bar) <= self.reset_bars)):
            self.l_count += 1
            self._l_last_bar = self._bar_index

    # ── Bar Scoring ──

    def _score_bull(self, bar: Bar, prev: Bar, avg_vol: float) -> int:
        score = 0.0
        # 实体占比（越大越好）
        score += min(25.0, (bar.body / bar.bar_range) * 31.25)
        # 收盘位置（越靠近 high 越好）
        score += ((bar.close - bar.low) / bar.bar_range) * 20.0
        # 上影线（越小越好）
        upper_wick = (bar.high - max(bar.close, bar.open)) / bar.bar_range
        score += max(0.0, 15.0 - upper_wick * 50.0)
        # Gap（与前根 close 的距离，越小越好）
        if self.atr > 0:
            gap = abs(bar.open - prev.close) / self.atr
            score += max(0.0, 10.0 - gap * 20.0)
            low_breach = max(0.0, prev.close - bar.low) / self.atr
            score += max(0.0, 10.0 - low_breach * 25.0)
        # 突破前根 close
        if bar.low >= prev.close:
            score += 5.0
        # 成交量
        if avg_vol > 0:
            score += min(10.0, bar.volume / avg_vol)
        return int(min(100, max(0, round(score))))

    def _score_bear(self, bar: Bar, prev: Bar, avg_vol: float) -> int:
        score = 0.0
        score += min(25.0, (bar.body / bar.bar_range) * 31.25)
        score += ((bar.high - bar.close) / bar.bar_range) * 20.0
        lower_wick = (min(bar.close, bar.open) - bar.low) / bar.bar_range
        score += max(0.0, 15.0 - lower_wick * 50.0)
        if self.atr > 0:
            gap = abs(bar.open - prev.close) / self.atr
            score += max(0.0, 10.0 - gap * 20.0)
            high_breach = max(0.0, bar.high - prev.close) / self.atr
            score += max(0.0, 10.0 - high_breach * 25.0)
        if bar.high <= prev.close:
            score += 5.0
        if avg_vol > 0:
            score += min(10.0, bar.volume / avg_vol)
        return int(min(100, max(0, round(score))))

    # ── Signal Check ──

    def _check_signal(self, bar: Bar) -> Optional[dict]:
        if self._bar_index == self._last_signal_bar:
            return None

        bars = list(self._bars)
        prev = bars[-2]
        direction = 'LONG' if self.always_in_long else 'SHORT'

        # 评分
        avg_vol = sum(b.volume for b in bars[-20:]) / min(20, len(bars))

        if direction == 'LONG' and bar.is_bull:
            score = self._score_bull(bar, prev, avg_vol)
        elif direction == 'SHORT' and bar.is_bear:
            score = self._score_bear(bar, prev, avg_vol)
        else:
            return None

        if score < self.min_score:
            return None

        # 判断 setup 类型
        is_spike = self._bars_from_ema >= 20 or self._bars_from_ema <= 3
        setup = None
        reason = ''
        ema_ok = True

        if direction == 'LONG':
            ema_ok = bar.close >= self.ema - self.atr * 0.3
            if self.h_count == 2 and ema_ok:
                setup = 'H2'
                reason = f'H2 多头回调第2脚（score={score}）'
            elif self.h_count == 1 and is_spike and ema_ok:
                setup = 'H1_SPIKE'
                reason = f'H1 尖峰回调（bars_from_ema={self._bars_from_ema}, score={score}）'
            elif self.h_count == 3 and ema_ok:
                setup = 'H3'
                reason = f'H3 楔形旗形（score={score}）'
        else:
            ema_ok = bar.close <= self.ema + self.atr * 0.3
            if self.l_count == 2 and ema_ok:
                setup = 'L2'
                reason = f'L2 空头回调第2脚（score={score}）'
            elif self.l_count == 1 and is_spike and ema_ok:
                setup = 'L1_SPIKE'
                reason = f'L1 尖峰回调（bars_from_ema={self._bars_from_ema}, score={score}）'
            elif self.l_count == 3 and ema_ok:
                setup = 'L3'
                reason = f'L3 楔形旗形（score={score}）'

        if setup is None:
            return None

        # 进场/止损/目标
        tick = 0.01
        if direction == 'LONG':
            entry = bar.high + tick
            stop = bar.low - tick
            target = entry + self.atr * 1.5
        else:
            entry = bar.low - tick
            stop = bar.high + tick
            target = entry - self.atr * 1.5

        risk = abs(entry - stop)
        if risk < 0.01:
            return None

        reward = abs(target - entry)
        rr = reward / risk
        if rr < self.min_rr:
            return None

        self._last_signal_bar = self._bar_index

        return {
            'direction': direction,
            'setup': setup,
            'score': score,
            'entry': round(entry, 2),
            'stop': round(stop, 2),
            'target': round(target, 2),
            'rr_ratio': round(rr, 2),
            'reason': reason,
            'ema': round(self.ema, 2),
            'atr': round(self.atr, 3),
            'h_count': self.h_count,
            'l_count': self.l_count,
        }
