"""
GEX + Price Action 综合刮头皮引擎（整合版）
====================================
本版整合内容：
  【严重 bug 修复】
  #1 删除 reqRealTimeBars 死代码
  #2 keepUpToDate=True 时 useRTH=False，自己按时间过滤
  #3 on_bar_update 用 bars[-2]（刚收盘那根）而不是 bars[-1]
  #5 修正 score_bull_bar 的 len([bar]) 死代码
  #6 Greeks 改为流式订阅 + 超时取消，提高成功率
  #12 GEX 目标方向一致性检查

  【逻辑改进】
  #4  节假日 / 无 0DTE 时 fallback 到下一个最近到期日
  #7  H/L 计数重置逻辑反转：创新高不再清零，而是在新回调低点时清零
  #8  is_spike 收紧为仅"远离 EMA"一种情况，不再把刚离开 EMA 也算
  #9  bias_direction 用 ATR 距离代替百分比距离
  #10 负 Gamma 环境下零线附近加缓冲，避免方向抖动
  #13 SNAP_BATCH 降到 8，避免 pacing violation
  #16 断线重连（disconnectedEvent + 主循环重连）
  #18 成交量评分收紧：2 倍均量才满分

依赖：
    pip install ib_insync numpy pandas

IB Gateway 端口：
    纸交易 7497 / 实盘 7496
"""

import asyncio
import sys
from datetime import datetime, time as dtime, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ib_insync import IB, Stock, Option, util

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
SYMBOL         = 'QQQ'
EXCHANGE       = 'SMART'
CURRENCY       = 'USD'
NUM_STRIKES    = 8
GEX_REFRESH    = 300
BAR_SIZE       = '5 mins'
EMA_LEN        = 20
ATR_LEN        = 14
SCORE_H2_MIN   = 55
RESET_BARS     = 30
IB_HOST        = '127.0.0.1'
IB_PORT        = 7497
IB_CLIENT_ID   = 11
GREEKS_TIMEOUT = 6.0
SNAP_BATCH     = 8

# 尖峰阈值：连续 N 根不碰 EMA 才算尖峰阶段
SPIKE_BARS     = 12

# 负 Gamma 环境下，价格距零线的最小 ATR 倍数（防止零线附近抖动）
ZERO_LINE_BUFFER_ATR = 0.3

# 节假日降级：如果当日无 0DTE，最多往后找几天
EXPIRY_LOOKAHEAD_DAYS = 5

# RTH 过滤（美东）
RTH_START = dtime(9, 30)
RTH_END   = dtime(16, 0)

# ─────────────────────────────────────────────
# 颜色
# ─────────────────────────────────────────────
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
RESET  = '\033[0m'

def cp(t, c=RESET): print(f"{c}{t}{RESET}")


def ema_dist(spot: float, ema: float) -> float:
    return abs(spot - ema)


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class GEXState:
    call_wall      : Optional[float] = None
    put_wall       : Optional[float] = None
    zero_line      : Optional[float] = None
    total_gex      : float = 0.0
    positive_gamma : bool = False
    last_update    : Optional[datetime] = None
    gex_by_strike  : dict = field(default_factory=dict)
    expiry_used    : Optional[str] = None

    @property
    def is_fresh(self) -> bool:
        if self.last_update is None:
            return False
        return (datetime.now() - self.last_update).total_seconds() < GEX_REFRESH * 2

    def bias_direction(self, spot: float, atr: float = 0.0) -> Optional[str]:
        """
        #9 #10 改进：
          - 正 Gamma 用 ATR 距离判断是否接近墙（而非固定百分比）
          - 负 Gamma 零线附近加缓冲，距离不足则返回中性
        """
        if not self.is_fresh or self.call_wall is None or self.put_wall is None:
            return None

        if not self.positive_gamma:
            if self.zero_line is None:
                return None
            buffer = atr * ZERO_LINE_BUFFER_ATR if atr > 0 else 0.0
            if spot > self.zero_line + buffer:
                return 'LONG'
            if spot < self.zero_line - buffer:
                return 'SHORT'
            return None

        # 正 Gamma 环境：接近墙时反向
        dist_thresh = atr * 0.5 if atr > 0 else self.call_wall * 0.002
        if (self.call_wall - spot) <= dist_thresh and spot <= self.call_wall:
            return 'SHORT'
        if (spot - self.put_wall) <= dist_thresh and spot >= self.put_wall:
            return 'LONG'
        return None

    def nearest_wall(self, spot: float, direction: str) -> Optional[float]:
        if direction == 'LONG' and self.call_wall:
            return self.call_wall
        if direction == 'SHORT' and self.put_wall:
            return self.put_wall
        return None


@dataclass
class Bar:
    time   : datetime
    open   : float
    high   : float
    low    : float
    close  : float
    volume : float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def bar_range(self) -> float:
        return self.high - self.low or 1e-9

    @property
    def is_bull(self) -> bool:
        return self.close > self.open

    @property
    def is_bear(self) -> bool:
        return self.close < self.open


@dataclass
class PAState:
    bars          : deque = field(default_factory=lambda: deque(maxlen=200))
    ema_vals      : deque = field(default_factory=lambda: deque(maxlen=200))
    atr_vals      : deque = field(default_factory=lambda: deque(maxlen=200))

    always_in_long  : bool = True
    h_count         : int = 0
    l_count         : int = 0
    h_last_bar      : int = 0
    l_last_bar      : int = 0
    bar_index       : int = 0
    bars_from_ema   : int = 0

    # #7 追踪最近的回调极点，用来判断"新回调"
    last_pullback_low  : Optional[float] = None
    last_pullback_high : Optional[float] = None

    last_signal_bar : int = -1

    @property
    def current_bar(self) -> Optional[Bar]:
        return self.bars[-1] if self.bars else None

    @property
    def ema(self) -> float:
        return self.ema_vals[-1] if self.ema_vals else 0.0

    @property
    def atr(self) -> float:
        return self.atr_vals[-1] if self.atr_vals else 0.0


@dataclass
class Signal:
    direction  : str
    setup      : str
    bar_score  : int
    pa_reason  : str
    gex_reason : str
    entry      : float
    stop       : float
    target     : float
    rr_ratio   : float
    confidence : str
    time       : datetime = field(default_factory=datetime.now)

    def __str__(self):
        conf_color = GREEN if self.confidence == 'HIGH' else YELLOW if self.confidence == 'MEDIUM' else DIM
        return (
            f"\n{'═'*56}\n"
            f"  {BOLD}{'▲ LONG' if self.direction=='LONG' else '▼ SHORT'} 信号{RESET}  "
            f"[{self.setup}]  {conf_color}{self.confidence}{RESET}\n"
            f"{'─'*56}\n"
            f"  入场 : {BOLD}{self.entry:.2f}{RESET}\n"
            f"  止损 : {RED}{self.stop:.2f}{RESET}  "
            f"（{abs(self.entry-self.stop):.2f} 点）\n"
            f"  目标 : {GREEN}{self.target:.2f}{RESET}  "
            f"（{abs(self.target-self.entry):.2f} 点）\n"
            f"  盈亏比: {BOLD}{self.rr_ratio:.1f}:1{RESET}\n"
            f"  信号棒: {self.bar_score} 分\n"
            f"{'─'*56}\n"
            f"  PA : {self.pa_reason}\n"
            f"  GEX: {self.gex_reason}\n"
            f"{'═'*56}"
        )


# ─────────────────────────────────────────────
# GEX 计算层
# ─────────────────────────────────────────────

def candidate_expiries(n_days: int = EXPIRY_LOOKAHEAD_DAYS) -> list:
    """#4 返回从今天开始往后 n 天的候选到期日（YYYYMMDD），跳过周末。"""
    out = []
    d = datetime.now().date()
    for i in range(n_days + 1):
        day = d + timedelta(days=i)
        if day.weekday() < 5:
            out.append(day.strftime('%Y%m%d'))
    return out


def build_contracts(spot: float, expiry: str, n: int) -> list:
    atm = round(spot)
    strikes = [atm + i for i in range(-n, n + 1)]
    contracts = []
    for s in strikes:
        for right in ['C', 'P']:
            contracts.append(
                Option(SYMBOL, expiry, float(s), right, EXCHANGE, currency=CURRENCY)
            )
    return contracts


async def find_valid_expiry(ib: IB, spot: float) -> tuple:
    """#4 依次尝试候选到期日，返回第一个能成功合成合约的 (expiry, qualified_contracts)。"""
    for expiry in candidate_expiries():
        contracts = build_contracts(spot, expiry, NUM_STRIKES)
        try:
            qualified = await ib.qualifyContractsAsync(*contracts)
        except Exception:
            continue
        valid = [c for c in qualified if c.conId > 0]
        if valid:
            return expiry, valid
    return None, []


async def fetch_greeks(ib: IB, contract, sem: asyncio.Semaphore):
    """#6 流式订阅 + 超时取消"""
    async with sem:
        ticker = None
        try:
            ticker = ib.reqMktData(contract, '100,101,104,106', snapshot=False)
            elapsed = 0.0
            while elapsed < GREEKS_TIMEOUT:
                await asyncio.sleep(0.25)
                elapsed += 0.25
                if ticker.modelGreeks is not None and ticker.modelGreeks.gamma is not None:
                    break
            g = ticker.modelGreeks
            if g is None or g.gamma is None:
                return None
            oi = ticker.openInterest or 0
            return (contract, max(oi, 0), g.gamma)
        except Exception:
            return None
        finally:
            if ticker is not None:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass


async def update_gex(ib: IB, gex_state: GEXState, spot: float):
    expiry, valid = await find_valid_expiry(ib, spot)
    if not valid:
        cp("  [GEX] ✗ 无有效期权合约（试过未来 5 个交易日）", RED)
        return

    if gex_state.expiry_used != expiry:
        cp(f"  [GEX] 使用到期日 {expiry}", CYAN)
        gex_state.expiry_used = expiry

    sem     = asyncio.Semaphore(SNAP_BATCH)
    tasks   = [fetch_greeks(ib, c, sem) for c in valid]
    results = [r for r in await asyncio.gather(*tasks) if r]

    if not results:
        cp("  [GEX] ✗ 未获取到 Greeks 数据", RED)
        return

    gex_raw = {}
    for contract, oi, gamma in results:
        s   = contract.strike
        r   = contract.right
        val = gamma * oi * 100 * spot
        if r == 'C':
            gex_raw[s] = gex_raw.get(s, 0) + val
        else:
            gex_raw[s] = gex_raw.get(s, 0) - val

    if not gex_raw:
        return

    pos   = {s: v for s, v in gex_raw.items() if v > 0}
    neg   = {s: v for s, v in gex_raw.items() if v < 0}
    total = sum(gex_raw.values())

    gex_state.gex_by_strike  = gex_raw
    gex_state.call_wall      = max(pos, key=pos.get)   if pos else None
    gex_state.put_wall       = min(neg, key=neg.get)   if neg else None
    gex_state.zero_line      = min(gex_raw, key=lambda s: abs(gex_raw[s]))
    gex_state.total_gex      = total
    gex_state.positive_gamma = total > 0
    gex_state.last_update    = datetime.now()

    cp(f"  [GEX] ✓ 更新完成  Call Wall={gex_state.call_wall}  "
       f"Put Wall={gex_state.put_wall}  "
       f"{'正Gamma' if gex_state.positive_gamma else '负Gamma'}  "
       f"合约={len(results)}/{len(valid)}",
       GREEN)


# ─────────────────────────────────────────────
# 价格行为计算层
# ─────────────────────────────────────────────

def calc_ema(prev_ema: float, price: float, length: int) -> float:
    k = 2 / (length + 1)
    return price * k + prev_ema * (1 - k)


def calc_atr(bars: deque, length: int) -> float:
    if len(bars) < 2:
        return 0.0
    trs = []
    bar_list = list(bars)
    for i in range(1, min(length + 1, len(bar_list))):
        b  = bar_list[-i]
        pb = bar_list[-i - 1]
        tr = max(b.high, pb.close) - min(b.low, pb.close)
        trs.append(tr)
    return sum(trs) / len(trs) if trs else 0.0


def _volume_score(vol: float, avg_vol: float) -> float:
    """#18 收紧：2 倍均量才满分 10"""
    if avg_vol <= 0:
        return 0.0
    ratio = vol / avg_vol
    return max(0.0, min(10.0, (ratio - 1.0) * 10.0))


def score_bull_bar(bar: Bar, prev_bar: Bar, atr: float, avg_vol: float, vol: float) -> int:
    score = 0.0

    body_ratio  = bar.body / bar.bar_range
    score      += min(25.0, body_ratio * 31.25)

    close_pos   = (bar.close - bar.low) / bar.bar_range
    score      += close_pos * 20.0

    upper_wick  = (bar.high - max(bar.close, bar.open)) / bar.bar_range
    score      += max(0.0, 15.0 - upper_wick * 50.0)

    if atr > 0:
        gap_ratio  = abs(bar.open - prev_bar.close) / atr
        score     += max(0.0, 10.0 - gap_ratio * 20.0)
        low_ratio  = max(0.0, prev_bar.close - bar.low) / atr
        score     += max(0.0, 10.0 - low_ratio * 25.0)

    # #5 去掉死代码
    if bar.low >= prev_bar.close:
        score += 5.0

    score += _volume_score(vol, avg_vol)

    return int(min(100, max(0, round(score))))


def score_bear_bar(bar: Bar, prev_bar: Bar, atr: float, avg_vol: float, vol: float) -> int:
    score = 0.0

    body_ratio  = bar.body / bar.bar_range
    score      += min(25.0, body_ratio * 31.25)

    close_pos   = (bar.close - bar.low) / bar.bar_range
    score      += (1.0 - close_pos) * 20.0

    lower_wick  = (min(bar.close, bar.open) - bar.low) / bar.bar_range
    score      += max(0.0, 15.0 - lower_wick * 50.0)

    if atr > 0:
        gap_ratio  = abs(bar.open - prev_bar.close) / atr
        score     += max(0.0, 10.0 - gap_ratio * 20.0)
        hi_ratio   = max(0.0, bar.high - prev_bar.close) / atr
        score     += max(0.0, 10.0 - hi_ratio * 25.0)

    if bar.high <= prev_bar.close:
        score += 5.0

    score += _volume_score(vol, avg_vol)

    return int(min(100, max(0, round(score))))


def update_pa_state(pa: PAState, new_bar: Bar):
    pa.bars.append(new_bar)
    pa.bar_index += 1
    bars = list(pa.bars)

    if len(pa.ema_vals) == 0:
        pa.ema_vals.append(new_bar.close)
    else:
        pa.ema_vals.append(calc_ema(pa.ema_vals[-1], new_bar.close, EMA_LEN))

    pa.atr_vals.append(calc_atr(pa.bars, ATR_LEN))

    ema = pa.ema

    above = 0
    below = 0
    for b in reversed(bars):
        if b.close > ema:
            if below > 0:
                break
            above += 1
        else:
            if above > 0:
                break
            below += 1
    prev_ai = pa.always_in_long
    pa.always_in_long = above > below

    # 方向切换时清零计数与回调极点
    if pa.always_in_long != prev_ai:
        pa.h_count = 0
        pa.l_count = 0
        pa.last_pullback_low  = None
        pa.last_pullback_high = None

    touched = (
        (new_bar.low <= ema <= new_bar.high) or
        (len(bars) >= 2 and (
            (bars[-2].close < ema and new_bar.close > ema) or
            (bars[-2].close > ema and new_bar.close < ema)
        ))
    )
    if touched:
        pa.bars_from_ema = 0
    else:
        pa.bars_from_ema += 1

    if len(bars) < 2:
        return
    prev_bar = bars[-2]

    # ── H 计数（多头）──
    # #7 新回调低点时清零；创新高不再清零
    if pa.always_in_long:
        if pa.h_count > 0 and (pa.bar_index - pa.h_last_bar) > RESET_BARS:
            pa.h_count = 0
            pa.last_pullback_low = None

        if pa.last_pullback_low is None:
            pa.last_pullback_low = new_bar.low
        else:
            if new_bar.low < pa.last_pullback_low - 1e-9:
                pa.last_pullback_low = new_bar.low
                pa.h_count = 0

        if (len(bars) >= 2 and
                new_bar.high > prev_bar.high and
                pa.h_count < 3):
            pa.h_count    += 1
            pa.h_last_bar  = pa.bar_index
    else:
        pa.h_count = 0

    # ── L 计数（空头）──
    if not pa.always_in_long:
        if pa.l_count > 0 and (pa.bar_index - pa.l_last_bar) > RESET_BARS:
            pa.l_count = 0
            pa.last_pullback_high = None

        if pa.last_pullback_high is None:
            pa.last_pullback_high = new_bar.high
        else:
            if new_bar.high > pa.last_pullback_high + 1e-9:
                pa.last_pullback_high = new_bar.high
                pa.l_count = 0

        if (len(bars) >= 2 and
                new_bar.low < prev_bar.low and
                pa.l_count < 3):
            pa.l_count    += 1
            pa.l_last_bar  = pa.bar_index
    else:
        pa.l_count = 0


# ─────────────────────────────────────────────
# 综合信号生成
# ─────────────────────────────────────────────

def generate_signal(pa: PAState, gex: GEXState) -> Optional[Signal]:
    bars = list(pa.bars)
    if len(bars) < 3:
        return None

    if pa.bar_index == pa.last_signal_bar:
        return None

    bar  = bars[-1]
    prev = bars[-2]

    ema = pa.ema
    atr = pa.atr
    spot = bar.close

    if atr <= 0:
        return None

    avg_vol = sum(b.volume for b in bars[-20:]) / min(20, len(bars))

    gex_dir = gex.bias_direction(spot, atr)
    pa_dir  = 'LONG' if pa.always_in_long else 'SHORT'

    if gex_dir is None:
        if gex.positive_gamma:
            return None
        if not gex.is_fresh:
            return None
        gex_dir = pa_dir  # 负 Gamma 且零线缓冲外刚好无 bias → 兜底

    if gex_dir != pa_dir:
        return None

    direction = pa_dir

    if direction == 'LONG' and bar.is_bull:
        score = score_bull_bar(bar, prev, atr, avg_vol, bar.volume)
    elif direction == 'SHORT' and bar.is_bear:
        score = score_bear_bar(bar, prev, atr, avg_vol, bar.volume)
    else:
        return None

    if score < SCORE_H2_MIN:
        return None

    # #8 仅当远离 EMA 一段时间才算尖峰
    is_spike = pa.bars_from_ema >= SPIKE_BARS
    setup    = None
    pa_reason = ""

    if direction == 'LONG':
        ema_ok = bar.close >= ema - atr * 0.3
        if pa.h_count == 2 and ema_ok:
            setup     = 'H2'
            pa_reason = f"H2 顺势多头（评分{score}）EMA距{ema_dist(spot, ema):.2f}"
        elif pa.h_count == 1 and is_spike and ema_ok:
            setup     = 'H1_SPIKE'
            pa_reason = f"H1 尖峰（缺口棒{pa.bars_from_ema}根 评分{score}）"
        elif pa.h_count == 3 and ema_ok:
            setup     = 'H3'
            pa_reason = f"H3 楔形旗形（评分{score}）"
    else:
        ema_ok = bar.close <= ema + atr * 0.3
        if pa.l_count == 2 and ema_ok:
            setup     = 'L2'
            pa_reason = f"L2 顺势空头（评分{score}）EMA距{ema_dist(spot, ema):.2f}"
        elif pa.l_count == 1 and is_spike and ema_ok:
            setup     = 'L1_SPIKE'
            pa_reason = f"L1 尖峰（缺口棒{pa.bars_from_ema}根 评分{score}）"
        elif pa.l_count == 3 and ema_ok:
            setup     = 'L3'
            pa_reason = f"L3 楔形旗形（评分{score}）"

    if gex.positive_gamma and setup is None:
        if (direction == 'SHORT' and gex.call_wall and
                abs(spot - gex.call_wall) <= atr * 0.5):
            setup     = 'WALL_FADE'
            pa_reason = f"Call Wall 压制反向（评分{score}）"
        elif (direction == 'LONG' and gex.put_wall and
                abs(spot - gex.put_wall) <= atr * 0.5):
            setup     = 'WALL_FADE'
            pa_reason = f"Put Wall 支撑反向（评分{score}）"

    if setup is None:
        return None

    if direction == 'LONG':
        entry = bar.high + 0.01
        stop  = bar.low  - 0.01
    else:
        entry = bar.low  - 0.01
        stop  = bar.high + 0.01

    risk = abs(entry - stop)
    if risk <= 0:
        return None

    # #12 目标方向一致性检查
    gex_target = gex.nearest_wall(spot, direction)
    use_gex_target = False
    if gex_target is not None:
        if direction == 'LONG' and gex_target > entry and (gex_target - entry) >= risk * 1.5:
            use_gex_target = True
        elif direction == 'SHORT' and gex_target < entry and (entry - gex_target) >= risk * 1.5:
            use_gex_target = True

    if use_gex_target:
        target = gex_target
        gex_reason = (
            f"{'Call' if direction=='LONG' else 'Put'} Wall "
            f"{gex_target:.1f}  "
            f"{'正' if gex.positive_gamma else '负'}Gamma环境"
        )
    else:
        target = entry + (1.5 * atr if direction == 'LONG' else -1.5 * atr)
        reason_note = "无有效 Wall 目标" if gex_target is None else "Wall 在反向/距离不足"
        gex_reason = (
            f"{reason_note}，用 1.5×ATR  "
            f"({'正' if gex.positive_gamma else '负'}Gamma)"
        )

    rr = abs(target - entry) / risk

    if gex.is_fresh and rr >= 2.0 and score >= 65:
        confidence = 'HIGH'
    elif rr >= 1.5 and score >= SCORE_H2_MIN:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    pa.last_signal_bar = pa.bar_index

    return Signal(
        direction  = direction,
        setup      = setup,
        bar_score  = score,
        pa_reason  = pa_reason,
        gex_reason = gex_reason,
        entry      = entry,
        stop       = stop,
        target     = target,
        rr_ratio   = rr,
        confidence = confidence,
    )


# ─────────────────────────────────────────────
# 实时状态面板
# ─────────────────────────────────────────────

def print_dashboard(pa: PAState, gex: GEXState):
    bar = pa.current_bar
    if bar is None:
        return

    spot      = bar.close
    ema       = pa.ema
    atr       = pa.atr
    ai_dir    = 'LONG  ▲' if pa.always_in_long else 'SHORT ▼'
    ai_color  = GREEN if pa.always_in_long else RED

    gex_env   = '正Gamma ↔ 区间' if gex.positive_gamma else '负Gamma → 趋势'
    gex_color = GREEN if gex.positive_gamma else RED
    gex_dir   = gex.bias_direction(spot, atr)
    gex_dir_s = gex_dir if gex_dir else '中性'

    now = datetime.now().strftime('%H:%M:%S')

    print(f"\n{CYAN}{'─'*58}{RESET}")
    print(
        f"{CYAN}[{now}]{RESET}  "
        f"现价 {BOLD}{spot:.2f}{RESET}  "
        f"EMA {ema:.2f}  "
        f"ATR {atr:.2f}"
    )
    print(
        f"  PA方向  : {ai_color}{BOLD}{ai_dir}{RESET}    "
        f"缺口棒 {pa.bars_from_ema} 根"
    )
    print(
        f"  H计数   : H{pa.h_count}    "
        f"L计数 : L{pa.l_count}"
    )
    print(
        f"  GEX环境 : {gex_color}{gex_env}{RESET}    "
        f"偏向 {gex_dir_s}"
    )
    if gex.is_fresh:
        cw = f"{gex.call_wall:.1f}" if gex.call_wall else "N/A"
        pw = f"{gex.put_wall:.1f}"  if gex.put_wall  else "N/A"
        zl = f"{gex.zero_line:.1f}" if gex.zero_line  else "N/A"
        print(
            f"  Call Wall {BOLD}{cw}{RESET}  "
            f"Put Wall {BOLD}{pw}{RESET}  "
            f"零线 {BOLD}{zl}{RESET}"
        )
    else:
        cp("  GEX 数据过期，等待下次更新", YELLOW)


# ─────────────────────────────────────────────
# GEX 定时刷新任务
# ─────────────────────────────────────────────

async def gex_refresh_loop(ib: IB, gex_state: GEXState, pa_state: PAState):
    while True:
        try:
            now = datetime.now()
            if dtime(9, 25) <= now.time() <= dtime(16, 5):
                if pa_state.current_bar:
                    spot = pa_state.current_bar.close
                else:
                    underlying = Stock(SYMBOL, EXCHANGE, CURRENCY)
                    await ib.qualifyContractsAsync(underlying)
                    ticker = ib.reqMktData(underlying, '', snapshot=False)
                    await asyncio.sleep(2.0)
                    spot = ticker.marketPrice()
                    ib.cancelMktData(underlying)

                if spot and spot > 0:
                    cp(f"\n[GEX] 开始刷新  现价={spot:.2f}", CYAN)
                    await update_gex(ib, gex_state, spot)
        except Exception as e:
            cp(f"  [GEX] 刷新异常: {e}", RED)

        now    = datetime.now()
        passed = (now.minute % 5) * 60 + now.second + now.microsecond / 1e6
        wait   = GEX_REFRESH - passed
        if wait < 10:
            wait += GEX_REFRESH
        await asyncio.sleep(wait)


# ─────────────────────────────────────────────
# K 线实时监听
# ─────────────────────────────────────────────

def is_in_rth(t) -> bool:
    if hasattr(t, 'time'):
        return RTH_START <= t.time() <= RTH_END
    return True


async def pa_bar_loop(ib: IB, pa_state: PAState, gex_state: GEXState):
    underlying = Stock(SYMBOL, EXCHANGE, CURRENCY)
    await ib.qualifyContractsAsync(underlying)

    cp("[PA] 拉取历史 K 线初始化...", CYAN)
    hist_bars = await ib.reqHistoricalDataAsync(
        underlying,
        endDateTime    = '',
        durationStr    = '2 D',
        barSizeSetting = BAR_SIZE,
        whatToShow     = 'TRADES',
        useRTH         = True,
        formatDate     = 1,
    )

    for hb in hist_bars[-(EMA_LEN * 3):]:
        bar = Bar(
            time   = hb.date,
            open   = hb.open,
            high   = hb.high,
            low    = hb.low,
            close  = hb.close,
            volume = hb.volume,
        )
        update_pa_state(pa_state, bar)

    pa_state.last_signal_bar = -1

    cp(f"[PA] ✓ 历史初始化完成  EMA={pa_state.ema:.2f}  ATR={pa_state.atr:.2f}", GREEN)

    # #1 #2 keepUpToDate=True 时必须 useRTH=False
    bars_live = ib.reqHistoricalData(
        underlying,
        endDateTime    = '',
        durationStr    = '1 D',
        barSizeSetting = BAR_SIZE,
        whatToShow     = 'TRADES',
        useRTH         = False,
        formatDate     = 1,
        keepUpToDate   = True,
    )

    last_processed_time = None

    def on_bar_update(bars, has_new_bar):
        nonlocal last_processed_time
        # #3 has_new_bar=True 时 bars[-1] 是新开的 tick 棒，bars[-2] 才是刚收盘的
        if not has_new_bar or len(bars) < 2:
            return

        ib_bar = bars[-2]

        if ib_bar.date == last_processed_time:
            return
        last_processed_time = ib_bar.date

        if not is_in_rth(ib_bar.date):
            return

        bar = Bar(
            time   = ib_bar.date,
            open   = ib_bar.open,
            high   = ib_bar.high,
            low    = ib_bar.low,
            close  = ib_bar.close,
            volume = ib_bar.volume,
        )

        update_pa_state(pa_state, bar)
        print_dashboard(pa_state, gex_state)

        signal = generate_signal(pa_state, gex_state)
        if signal:
            if signal.confidence in ('HIGH', 'MEDIUM'):
                print(signal)
            else:
                cp(f"  [低置信度信号跳过] {signal.setup} {signal.direction} "
                   f"分={signal.bar_score} 盈亏比={signal.rr_ratio:.1f}", DIM)
        else:
            cp("  → 无信号（等待合适设置）", DIM)

    bars_live.updateEvent += on_bar_update

    cp("[PA] ✓ 实时 K 线订阅成功，等待下一根 K 线...", GREEN)

    while True:
        await asyncio.sleep(10)


# ─────────────────────────────────────────────
# 主程序（#16 断线重连）
# ─────────────────────────────────────────────

async def run_session(ib: IB):
    gex_state = GEXState()
    pa_state  = PAState()

    disconnected = asyncio.Event()
    def on_disconnected():
        cp("\n  ⚠ IB 连接已断开，准备重连...", YELLOW)
        disconnected.set()
    ib.disconnectedEvent += on_disconnected

    tasks = [
        asyncio.create_task(gex_refresh_loop(ib, gex_state, pa_state)),
        asyncio.create_task(pa_bar_loop(ib, pa_state, gex_state)),
        asyncio.create_task(disconnected.wait()),
    ]

    try:
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                raise exc
    finally:
        try:
            ib.disconnectedEvent -= on_disconnected
        except Exception:
            pass


async def main():
    util.logToConsole(level=0)

    cp("=" * 58, CYAN)
    cp(f"  GEX + Price Action 综合刮头皮引擎（整合版） | {SYMBOL}", BOLD)
    cp(f"  GEX: 0DTE ATM ±{NUM_STRIKES} strikes  |  PA: EMA{EMA_LEN} H2/L2", CYAN)
    cp(f"  连接: {IB_HOST}:{IB_PORT}  clientId={IB_CLIENT_ID}", CYAN)
    cp("=" * 58, CYAN)

    backoff = 5
    while True:
        ib = IB()
        try:
            await ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            cp("  ✓ IB Gateway 连接成功", GREEN)
            backoff = 5
            await run_session(ib)
        except KeyboardInterrupt:
            cp("\n  用户中断，退出", YELLOW)
            break
        except Exception as e:
            cp(f"\n  ✗ 会话错误: {e}", RED)
            import traceback
            traceback.print_exc()
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass
            cp("  IB Gateway 已断开", CYAN)

        cp(f"  {backoff} 秒后重连...", YELLOW)
        try:
            await asyncio.sleep(backoff)
        except KeyboardInterrupt:
            break
        backoff = min(backoff * 2, 120)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
