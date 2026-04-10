"""
GEX + Price Action 综合刮头皮引擎（修订版 v2）
============================================
修订内容：
  #1 删除 reqRealTimeBars 死代码
  #2 keepUpToDate=True 时 useRTH=False，自己按时间过滤
  #3 on_bar_update 用 bars[-2]（刚收盘那根）而不是 bars[-1]
  #5 修正 score_bull_bar 的 len([bar]) 死代码
  #6 Greeks 改为流式订阅 + 超时取消，提高成功率
  #12 GEX 目标方向一致性检查

v2 修复：
  - 时区：使用 zoneinfo 转换为美东时间
  - 成交量评分：修正比例 (vol/avg_vol) * 5
  - put_wall：改为取绝对值最大的负 GEX strike
  - 现价获取：添加 NaN 检查
  - Magic numbers 提取为常量
  - 函数重命名 cp → cprint

依赖：
    pip install ib_insync numpy pandas

IB Gateway 端口：
    纸交易 7497 / 实盘 7496
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, time as dtime
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

from ib_insync import IB, Stock, Option, util

# ─────────────────────────────────────────────
# 时区
# ─────────────────────────────────────────────
ET = ZoneInfo('America/New_York')

def et_now() -> datetime:
    """返回当前美东时间"""
    return datetime.now(ET)

def to_et(dt: datetime) -> datetime:
    """将 datetime 转换为美东时间。naive datetime 假设已经是 ET（避免
    服务器本地时区不确定性导致的静默错误）。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
SYMBOL        = 'QQQ'
EXCHANGE      = 'SMART'
CURRENCY      = 'USD'
NUM_STRIKES   = 4
GEX_REFRESH   = 45   # 45 秒
BAR_SIZE      = '5 mins'
EMA_LEN       = 20
ATR_LEN       = 14
SCORE_H2_MIN  = 55
RESET_BARS    = 30
IB_HOST       = '127.0.0.1'
IB_PORT       = 7496
IB_CLIENT_ID  = 11
GREEKS_TIMEOUT = 15.0       # 流式订阅等待 Greeks 的超时
SNAP_BATCH    = 8           # 降到 8，避免 pacing violation

# RTH 过滤（美东时间 9:30-16:00）
RTH_START = dtime(9, 30)
RTH_END   = dtime(16, 0)

# ─────────────────────────────────────────────
# 交易参数常量（原 magic numbers）
# ─────────────────────────────────────────────
CALL_WALL_THRESHOLD   = 0.998   # 接近 call wall 阈值
PUT_WALL_THRESHOLD    = 1.002   # 接近 put wall 阈值
EMA_TOLERANCE_ATR     = 0.3     # EMA 容忍度（ATR 倍数）
WALL_PROXIMITY_ATR    = 0.5     # Wall 接近判定（ATR 倍数）
TARGET_ATR_MULT       = 1.5     # 默认目标（ATR 倍数）
MIN_RR_HIGH_CONF      = 2.0     # 高置信度最低盈亏比
MIN_RR_MEDIUM_CONF    = 1.5     # 中置信度最低盈亏比
MIN_SCORE_HIGH_CONF   = 65      # 高置信度最低评分
SPIKE_BARS_MIN        = 20      # 尖峰信号：连续远离 EMA 的最小棒数
VOL_SCORE_MULT        = 5.0     # 成交量评分乘数

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

def cprint(text: str, color: str = RESET) -> None:
    """带颜色的打印"""
    print(f"{color}{text}{RESET}")


def play_alert(confidence: str) -> None:
    """播放声音提醒（macOS）- 连续 3 次"""
    if confidence == 'HIGH':
        sound = '/System/Library/Sounds/Glass.aiff'
    else:
        sound = '/System/Library/Sounds/Pop.aiff'
    # 连续播放 3 次，间隔 0.5 秒，后台执行
    os.system(f'for i in 1 2 3; do afplay {sound}; sleep 0.5; done &')


def ema_dist(spot: float, ema: float) -> float:
    return abs(spot - ema)


def is_nan(value) -> bool:
    """检查是否为 NaN"""
    try:
        return value != value  # NaN != NaN
    except Exception:
        return True


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
    gex_by_strike  : dict[float, float] = field(default_factory=dict)
    prev_gex       : dict[float, float] = field(default_factory=dict)  # 上次 GEX
    max_change_strike : Optional[float] = None  # 变化最大的 strike
    max_change_value  : float = 0.0             # 变化量

    @property
    def is_fresh(self) -> bool:
        if self.last_update is None:
            return False
        return (et_now() - self.last_update).total_seconds() < GEX_REFRESH * 2

    def bias_direction(self, spot: float) -> Optional[str]:
        if not self.is_fresh or self.call_wall is None or self.put_wall is None:
            return None

        # 合理性检查：call_wall 应在 spot 上方，put_wall 应在 spot 下方
        # 否则说明 GEX 分布异常（例如尾部极端），跳过信号
        if self.call_wall <= spot or self.put_wall >= spot:
            return None

        if not self.positive_gamma:
            if self.zero_line and spot > self.zero_line:
                return 'LONG'
            elif self.zero_line and spot < self.zero_line:
                return 'SHORT'
            return None

        if spot >= self.call_wall * CALL_WALL_THRESHOLD:
            return 'SHORT'
        if spot <= self.put_wall * PUT_WALL_THRESHOLD:
            return 'LONG'
        return None

    def target_wall(self, spot: float, direction: str) -> Optional[float]:
        """返回目标方向上的 wall（LONG→call_wall，SHORT→put_wall），
        并且 wall 必须在入场方向的正确一侧。"""
        if direction == 'LONG' and self.call_wall and self.call_wall > spot:
            return self.call_wall
        if direction == 'SHORT' and self.put_wall and self.put_wall < spot:
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
        return max(self.high - self.low, 1e-9)

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

    always_in_long : bool = True
    h_count        : int = 0
    l_count        : int = 0
    h_last_bar     : int = 0
    l_last_bar     : int = 0
    bar_index      : int = 0
    bars_from_ema  : int = 0

    last_signal_bar : int = -1   # 去重用

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
    time       : datetime = field(default_factory=et_now)

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

def get_today_expiry() -> str:
    """返回今天的到期日字符串（美东时间）"""
    return et_now().strftime('%Y%m%d')


def build_0dte_contracts(spot: float, expiry: str, n: int) -> list:
    atm = round(spot)
    strikes = [atm + i for i in range(-n, n + 1)]
    contracts = []
    for s in strikes:
        for right in ['C', 'P']:
            contracts.append(
                Option(SYMBOL, expiry, float(s), right, EXCHANGE, currency=CURRENCY)
            )
    return contracts


async def fetch_greeks(ib: IB, contract, sem: asyncio.Semaphore):
    """
    #6 修订：改为流式订阅 + 超时取消
    generic tick 106 = 隐含波动率，会触发 Greeks 计算
    """
    async with sem:
        ticker = None
        try:
            ticker = ib.reqMktData(contract, '100,101,104,106', snapshot=False)
            elapsed = 0.0
            while elapsed < GREEKS_TIMEOUT:
                await asyncio.sleep(0.25)
                elapsed += 0.25
                # 等待 Greeks 和 volume 都到达
                has_greeks = ticker.modelGreeks is not None and ticker.modelGreeks.gamma is not None
                vol = getattr(ticker, 'volume', None) or getattr(ticker, 'lastSize', None)
                has_vol = vol is not None and not is_nan(vol) and vol > 0
                if has_greeks and has_vol:
                    break
                # 如果只有 Greeks 到了，继续等 volume（但不超时）
                if has_greeks and elapsed >= GREEKS_TIMEOUT * 0.8:
                    break
            g = ticker.modelGreeks
            if g is None or g.gamma is None:
                return None
            # 获取 volume：尝试多个属性
            vol = getattr(ticker, 'volume', None)
            if vol is None or is_nan(vol) or vol <= 0:
                vol = getattr(ticker, 'callVolume', None) or getattr(ticker, 'putVolume', None)
            if vol is None or is_nan(vol) or vol <= 0:
                vol = getattr(ticker, 'lastSize', None)
            if vol is None or is_nan(vol) or vol <= 0:
                vol = 0
            return (contract, max(int(vol), 0), g.gamma)
        except Exception:
            return None
        finally:
            if ticker is not None:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass


async def update_gex(ib: IB, gex_state: GEXState, spot: float):
    expiry    = get_today_expiry()
    contracts = build_0dte_contracts(spot, expiry, NUM_STRIKES)

    cprint(f"  [GEX] 到期日={expiry}  请求合约数={len(contracts)}", DIM)

    qualified = await ib.qualifyContractsAsync(*contracts)
    valid     = [c for c in qualified if c.conId > 0]

    cprint(f"  [GEX] 有效合约={len(valid)}/{len(contracts)}", DIM)

    if not valid:
        cprint("  [GEX] ✗ 无有效 0DTE 合约（非交易日或无当日到期）", RED)
        return

    sem     = asyncio.Semaphore(SNAP_BATCH)
    tasks   = [fetch_greeks(ib, c, sem) for c in valid]
    results = [r for r in await asyncio.gather(*tasks) if r]

    cprint(f"  [GEX] Greeks 返回={len(results)}/{len(valid)}", DIM)

    if not results:
        cprint("  [GEX] ✗ 未获取到 Greeks 数据（可能无期权数据订阅）", RED)
        return

    gex_raw: dict[float, float] = {}
    oi_zero_count = 0

    for contract, vol, gamma in sorted(results, key=lambda x: (x[0].strike, x[0].right)):
        s   = contract.strike
        r   = contract.right
        if vol == 0:
            oi_zero_count += 1
        val = gamma * vol * 100 * spot
        # 跳过 NaN 值
        if is_nan(val):
            continue
        if r == 'C':
            gex_raw[s] = gex_raw.get(s, 0) + val
        else:
            gex_raw[s] = gex_raw.get(s, 0) - val

    if oi_zero_count > 0:
        cprint(f"\n  [GEX] 注意: {oi_zero_count}/{len(results)} 个合约 Vol=0", YELLOW)

    if not gex_raw:
        return

    # 只在 spot 正确侧寻找墙（call wall 应在上方，put wall 应在下方）
    pos_above = {s: v for s, v in gex_raw.items() if v > 0 and s >= spot}
    neg_below = {s: v for s, v in gex_raw.items() if v < 0 and s <= spot}
    total = sum(gex_raw.values())

    # 计算变化最大的 strike
    max_change_strike = None
    max_change_value = 0.0
    if gex_state.prev_gex:
        for s in gex_raw:
            prev = gex_state.prev_gex.get(s, 0)
            change = abs(gex_raw[s] - prev)
            if change > max_change_value:
                max_change_value = change
                max_change_strike = s
    gex_state.prev_gex = gex_raw.copy()
    gex_state.max_change_strike = max_change_strike
    gex_state.max_change_value = max_change_value

    gex_state.gex_by_strike = gex_raw
    # Call Wall: spot 上方正 GEX 最大值（阻力）
    gex_state.call_wall = max(pos_above, key=pos_above.get) if pos_above else None
    # Put Wall: spot 下方负 GEX 绝对值最大（支撑）
    gex_state.put_wall = min(neg_below, key=neg_below.get) if neg_below else None

    # 零线：找 GEX 从正变负的插值点
    sorted_strikes = sorted(gex_raw.keys())
    zero_line = None
    for i in range(len(sorted_strikes) - 1):
        s1, s2 = sorted_strikes[i], sorted_strikes[i + 1]
        g1, g2 = gex_raw[s1], gex_raw[s2]
        # 检查是否跨零（一正一负）
        if g1 * g2 < 0:
            # 线性插值
            zero_line = s1 + (s2 - s1) * abs(g1) / (abs(g1) + abs(g2))
            break
    if zero_line is None:
        # 没有跨零点，用绝对值最小的 strike
        zero_line = min(gex_raw, key=lambda s: abs(gex_raw[s]))
    gex_state.zero_line = zero_line
    gex_state.total_gex = total
    gex_state.positive_gamma = total > 0
    gex_state.last_update = et_now()

    cprint(f"  [GEX] ✓ 更新完成  Call Wall={gex_state.call_wall}  "
           f"Put Wall={gex_state.put_wall}  "
           f"{'正Gamma' if gex_state.positive_gamma else '负Gamma'}  "
           f"合约={len(results)}/{len(valid)}",
           GREEN)

    # 打印详细 GEX 分布
    print_gex_chart(gex_state, spot)


def print_gex_chart(gex: GEXState, spot: float) -> None:
    """打印 GEX 柱状图"""
    if not gex.gex_by_strike:
        return

    BAR_WIDTH = 20  # 单侧柱子最大宽度
    valid_vals = [abs(v) for v in gex.gex_by_strike.values() if not is_nan(v)]
    if not valid_vals:
        return
    max_abs = max(valid_vals)
    if max_abs == 0:
        return

    print(f"\n  {'Strike':>8}  {'负 GEX (支撑)':^{BAR_WIDTH}}│{'正 GEX (阻力)':<{BAR_WIDTH}}  标记")
    print(f"  {'-'*8}  {'-'*BAR_WIDTH}┼{'-'*BAR_WIDTH}  {'-'*12}")

    for strike in sorted(gex.gex_by_strike.keys(), reverse=True):
        gex_val = gex.gex_by_strike[strike]
        if is_nan(gex_val):
            continue
        bar_len = int(abs(gex_val) / max_abs * BAR_WIDTH)

        if gex_val < 0:
            left_bar = ('█' * bar_len).rjust(BAR_WIDTH)
            right_bar = ' ' * BAR_WIDTH
        else:
            left_bar = ' ' * BAR_WIDTH
            right_bar = '█' * bar_len

        # 标记
        markers = []
        if strike == gex.call_wall:
            markers.append(f"Call Wall {gex.call_wall:.2f}")
        if strike == gex.put_wall:
            markers.append(f"Put Wall {gex.put_wall:.2f}")
        if gex.zero_line and abs(strike - gex.zero_line) < 0.5:
            markers.append(f"零线 {gex.zero_line:.2f}")
        if abs(strike - spot) < 0.5:
            markers.append(f"● 现价 {spot:.2f}")
        if gex.max_change_strike and strike == gex.max_change_strike:
            # 格式化变化量
            chg = gex.max_change_value
            if chg >= 1e6:
                chg_str = f"{chg/1e6:.1f}M"
            else:
                chg_str = f"{chg/1e3:.0f}K"
            markers.append(f"⚡变化最大 {gex.max_change_strike:.2f} Δ{chg_str}")

        marker_str = " ".join(markers)
        print(f"  {strike:>8.1f}  {RED}{left_bar}{RESET}│{GREEN}{right_bar}{RESET}  {marker_str}")


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

    # #5 修订：去掉 len([bar]) 死代码
    if bar.low >= prev_bar.close:
        score += 5.0

    # 修复：成交量评分比例调整
    if avg_vol > 0:
        score += min(10.0, (vol / avg_vol) * VOL_SCORE_MULT)

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

    # 修复：成交量评分比例调整
    if avg_vol > 0:
        score += min(10.0, (vol / avg_vol) * VOL_SCORE_MULT)

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
    pa.always_in_long = above > below

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

    if not pa.always_in_long:
        pa.h_count = 0
    else:
        if pa.h_count > 0 and (pa.bar_index - pa.h_last_bar) > RESET_BARS:
            pa.h_count = 0

        if (len(bars) >= 2 and
                new_bar.high > bars[-2].high and
                pa.h_count < 3 and
                (pa.h_count == 0 or (pa.bar_index - pa.h_last_bar) <= RESET_BARS)):
            pa.h_count    += 1
            pa.h_last_bar  = pa.bar_index

    if pa.always_in_long:
        pa.l_count = 0
    else:
        if pa.l_count > 0 and (pa.bar_index - pa.l_last_bar) > RESET_BARS:
            pa.l_count = 0

        if (len(bars) >= 2 and
                new_bar.low < bars[-2].low and
                pa.l_count < 3 and
                (pa.l_count == 0 or (pa.bar_index - pa.l_last_bar) <= RESET_BARS)):
            pa.l_count    += 1
            pa.l_last_bar  = pa.bar_index


# ─────────────────────────────────────────────
# 综合信号生成
# ─────────────────────────────────────────────

def generate_signal(pa: PAState, gex: GEXState) -> Optional[Signal]:
    bars = list(pa.bars)
    if len(bars) < 3:
        return None

    # 去重：同一根棒不重复出信号
    if pa.bar_index == pa.last_signal_bar:
        return None

    bar  = bars[-1]
    prev = bars[-2]

    ema = pa.ema
    atr = pa.atr
    spot = bar.close

    if atr <= 0:
        return None

    recent_bars = bars[-20:]
    avg_vol = sum(b.volume for b in recent_bars) / len(recent_bars)

    gex_dir = gex.bias_direction(spot)
    pa_dir = 'LONG' if pa.always_in_long else 'SHORT'

    # GEX 未就绪：完全不发信号（避免 PA-only 静默绕过 GEX 过滤）
    if not gex.is_fresh:
        return None

    if gex_dir is None:
        # GEX 新鲜但方向中性：仅在负 Gamma 趋势环境下允许顺 PA 方向
        if gex.positive_gamma:
            return None
        gex_dir = pa_dir

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

    # 尖峰 = 已连续远离 EMA 至少 SPIKE_BARS_MIN 根
    is_spike = pa.bars_from_ema >= SPIKE_BARS_MIN
    setup    = None
    pa_reason = ""

    if direction == 'LONG':
        ema_ok = bar.close >= ema - atr * EMA_TOLERANCE_ATR
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
        ema_ok = bar.close <= ema + atr * EMA_TOLERANCE_ATR
        if pa.l_count == 2 and ema_ok:
            setup     = 'L2'
            pa_reason = f"L2 顺势空头（评分{score}）EMA距{ema_dist(spot, ema):.2f}"
        elif pa.l_count == 1 and is_spike and ema_ok:
            setup     = 'L1_SPIKE'
            pa_reason = f"L1 尖峰（缺口棒{pa.bars_from_ema}根 评分{score}）"
        elif pa.l_count == 3 and ema_ok:
            setup     = 'L3'
            pa_reason = f"L3 楔形旗形（评分{score}）"

    # WALL_FADE 信号：在 wall 附近顺势交易（PA 和 GEX 都指向同一方向）
    # 注意：direction 已经等于 pa_dir，且 gex_dir == pa_dir
    # SHORT + 接近 call_wall = 在阻力位做空（fade 上涨）
    # LONG + 接近 put_wall = 在支撑位做多（fade 下跌）
    if gex.positive_gamma and setup is None:
        if (direction == 'SHORT' and gex.call_wall and
                abs(spot - gex.call_wall) <= atr * WALL_PROXIMITY_ATR):
            setup     = 'WALL_FADE'
            pa_reason = f"Call Wall 压制反向（评分{score}）"
        elif (direction == 'LONG' and gex.put_wall and
                abs(spot - gex.put_wall) <= atr * WALL_PROXIMITY_ATR):
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

    # #12 修订：目标必须在入场的正确方向，且距离足够
    gex_target = gex.target_wall(entry, direction)
    use_gex_target = False
    if gex_target is not None:
        if direction == 'LONG' and gex_target > entry and (gex_target - entry) >= risk * TARGET_ATR_MULT:
            use_gex_target = True
        elif direction == 'SHORT' and gex_target < entry and (entry - gex_target) >= risk * TARGET_ATR_MULT:
            use_gex_target = True

    if use_gex_target:
        target = gex_target
        gex_reason = (
            f"{'Call' if direction=='LONG' else 'Put'} Wall "
            f"{gex_target:.1f}  "
            f"{'正' if gex.positive_gamma else '负'}Gamma环境"
        )
    else:
        target = entry + (TARGET_ATR_MULT * atr if direction == 'LONG' else -TARGET_ATR_MULT * atr)
        reason_note = "无有效 Wall 目标" if gex_target is None else "Wall 在反向/距离不足"
        gex_reason = (
            f"{reason_note}，用 {TARGET_ATR_MULT}×ATR  "
            f"({'正' if gex.positive_gamma else '负'}Gamma)"
        )

    rr = abs(target - entry) / risk

    if gex_dir == pa_dir and gex.is_fresh and rr >= MIN_RR_HIGH_CONF and score >= MIN_SCORE_HIGH_CONF:
        confidence = 'HIGH'
    elif rr >= MIN_RR_MEDIUM_CONF and score >= SCORE_H2_MIN:
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
    gex_dir   = gex.bias_direction(spot)
    gex_dir_s = gex_dir if gex_dir else '中性'

    now = et_now().strftime('%H:%M:%S ET')

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
        print_gex_chart(gex, spot)
    else:
        cprint("  GEX 数据过期，等待下次更新", YELLOW)


# ─────────────────────────────────────────────
# GEX 定时刷新任务
# ─────────────────────────────────────────────

async def gex_refresh_loop(ib: IB, gex_state: GEXState, pa_state: PAState):
    while True:
        now = et_now()
        if dtime(9, 25) <= now.time() <= dtime(16, 5):
            if pa_state.current_bar:
                spot = pa_state.current_bar.close
            else:
                underlying = Stock(SYMBOL, EXCHANGE, CURRENCY)
                await ib.qualifyContractsAsync(underlying)
                ticker = ib.reqMktData(underlying, '', snapshot=False)
                spot = float('nan')
                elapsed = 0.0
                while elapsed < 5.0:
                    await asyncio.sleep(0.25)
                    elapsed += 0.25
                    price = ticker.marketPrice()
                    if price and not is_nan(price) and price > 0:
                        spot = price
                        break
                ib.cancelMktData(underlying)

            # 修复：添加 NaN 检查
            if spot and not is_nan(spot) and spot > 0:
                cprint(f"\n[GEX] 开始刷新  现价={spot:.2f}", CYAN)
                await update_gex(ib, gex_state, spot)
            else:
                cprint(f"\n[GEX] 无法获取有效现价，跳过刷新", YELLOW)

        now    = et_now()
        # 按 GEX_REFRESH 对齐：距离下一个 GEX_REFRESH 整点的秒数
        epoch_sec = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
        passed = epoch_sec % GEX_REFRESH
        wait   = GEX_REFRESH - passed
        if wait < 10:
            wait += GEX_REFRESH
        await asyncio.sleep(wait)


# ─────────────────────────────────────────────
# K 线实时监听
# ─────────────────────────────────────────────

def is_in_rth(t) -> bool:
    """按 bar 时间过滤 RTH（美东时间）"""
    if not hasattr(t, 'time'):
        return True
    # 每个 datetime 都有 tzinfo 属性（naive 时为 None），必须用 is not None 判断
    if t.tzinfo is not None:
        et_time = t.astimezone(ET).time()
    else:
        # naive：假设已经是 ET（IB formatDate=1 对 intraday 通常返回 aware，
        # 这里只是兜底）
        et_time = t.time()
    return RTH_START <= et_time <= RTH_END


async def pa_bar_loop(ib: IB, pa_state: PAState, gex_state: GEXState, live_holder: dict):
    underlying = Stock(SYMBOL, EXCHANGE, CURRENCY)
    await ib.qualifyContractsAsync(underlying)

    cprint("[PA] 拉取历史 K 线初始化...", CYAN)
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

    cprint(f"[PA] ✓ 历史初始化完成  EMA={pa_state.ema:.2f}  ATR={pa_state.atr:.2f}", GREEN)

    # #1 #2 修订：删除死代码，keepUpToDate=True 必须 useRTH=False
    bars_live = await ib.reqHistoricalDataAsync(
        underlying,
        endDateTime    = '',
        durationStr    = '1 D',
        barSizeSetting = BAR_SIZE,
        whatToShow     = 'TRADES',
        useRTH         = False,      # 关键：keepUpToDate 要求 False
        formatDate     = 1,
        keepUpToDate   = True,
    )
    live_holder['bars'] = bars_live

    last_processed_time = None

    def on_bar_update(bars, has_new_bar):
        nonlocal last_processed_time

        # #3 修订：has_new_bar=True 时 bars[-1] 是新开的 tick 棒，
        # bars[-2] 才是刚收盘的那根
        if not has_new_bar or len(bars) < 2:
            return

        ib_bar = bars[-2]

        if ib_bar.date == last_processed_time:
            return
        last_processed_time = ib_bar.date

        # #2 修订：自己按 RTH 过滤（使用美东时间）
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
                play_alert(signal.confidence)
            else:
                cprint(f"  [低置信度信号跳过] {signal.setup} {signal.direction} "
                       f"分={signal.bar_score} 盈亏比={signal.rr_ratio:.1f}", DIM)
        else:
            cprint("  → 无信号（等待合适设置）", DIM)

    bars_live.updateEvent += on_bar_update

    cprint("[PA] ✓ 实时 K 线订阅成功，等待下一根 K 线...", GREEN)

    while True:
        await asyncio.sleep(10)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

async def main():
    ib = IB()
    util.logToConsole(level=logging.CRITICAL)  # 只显示严重错误

    cprint("=" * 58, CYAN)
    cprint(f"  GEX + Price Action 综合刮头皮引擎（修订版 v2） | {SYMBOL}", BOLD)
    cprint(f"  GEX: 0DTE ATM ±{NUM_STRIKES} strikes  |  PA: EMA{EMA_LEN} H2/L2", CYAN)
    cprint(f"  连接: {IB_HOST}:{IB_PORT}  clientId={IB_CLIENT_ID}", CYAN)
    cprint("=" * 58, CYAN)

    try:
        await ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=True)
        cprint("  ✓ IB Gateway 连接成功", GREEN)
    except Exception as e:
        cprint(f"  ✗ 连接失败: {e}", RED)
        sys.exit(1)

    gex_state = GEXState()
    pa_state  = PAState()
    live_holder: dict = {}

    try:
        await asyncio.gather(
            gex_refresh_loop(ib, gex_state, pa_state),
            pa_bar_loop(ib, pa_state, gex_state, live_holder),
        )
    except KeyboardInterrupt:
        cprint("\n  用户中断，退出", YELLOW)
    except Exception as e:
        cprint(f"\n  ✗ 运行错误: {e}", RED)
        import traceback
        traceback.print_exc()
    finally:
        if 'bars' in live_holder:
            try:
                ib.cancelHistoricalData(live_holder['bars'])
            except Exception:
                pass
        ib.disconnect()
        cprint("  IB Gateway 已断开", CYAN)


if __name__ == '__main__':
    asyncio.run(main())
