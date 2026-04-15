"""
KDJ v4.8b Live Trader — Tick级实时信号 + Paper/Live下单

Tick → 30s bar → BarAggregator(1min+5min) → 信号引擎 → 下单
每个tick都检查模拟J值，实现bar内提前出场。

Safety:
  - Paper账户校验（DU前缀）
  - 日亏损上限（默认$500）
  - 日最大交易次数（默认20）
  - IB断线自动重连
  - 持仓状态持久化（JSON）
  - 盘前/盘后tick过滤
  - 交易记录CSV持久化

Usage:
  python -m gex_monitor.kdj_live_trader --dry-run
  python -m gex_monitor.kdj_live_trader --qty 100
"""

import sys
import os
import csv
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Callable

import pytz
from ib_insync import IB, Stock, MarketOrder, util

sys.path.insert(0, os.path.expanduser('~/Developer/mnq-trading-system'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_v48_live import KDJCalculator, ADXCalculator, ATRCalculator, SpeedDivergence, DEMACalculator

ET = pytz.timezone('America/New_York')

# 日志 + 数据目录
_data_dir = os.path.expanduser('~/Downloads/gex/logs')
os.makedirs(_data_dir, exist_ok=True)
_today = datetime.now(ET).strftime('%Y%m%d')
_log_file = os.path.join(_data_dir, f"kdj_trader_{_today}.log")
_csv_file = os.path.join(_data_dir, f"kdj_trades_{_today}.csv")
_state_file = os.path.join(_data_dir, "kdj_position_state.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [KDJ] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)
log.info(f"日志: {_log_file}")
log.info(f"交易记录: {_csv_file}")

# ═══════════════════════════════════════════════
#  风控参数
# ═══════════════════════════════════════════════
MAX_DAILY_LOSS = 500.0    # 日亏损上限（美元）
MAX_DAILY_TRADES = 20     # 日最大交易次数
RECONNECT_DELAY = 10      # 断线重连间隔（秒）
MAX_RECONNECT = 5         # 最大重连次数

# ═══════════════════════════════════════════════
#  Choppy Final 参数（Score 10.64）
# ═══════════════════════════════════════════════
PARAMS = {
    'b_atr_mult_sl': 0.6, 'b_atr_mult_tp': 2.0, 'b_j_neutral': 50,
    'a_enabled': False, 'a_atr_mult': 1.5,
    'a_long_j_stop': -10, 'a_short_j_stop': 110, 'a_time_stop': 8,
    'a_long_t1_j': 30, 'a_long_t2_j': 60, 'a_long_t3_j': 85,
    'a_short_t1_j': 75, 'a_short_t2_j': 50, 'a_short_t3_j': 20,
    'a_trail_gap': 0.4,
    'c_stop_loss': 1.0, 'c_t1_pts': 3.0, 'c_t2_pts': 5.0, 'c_trail_gap': 0.72,
    'div_len': 5, 'div_thresh': 1.5, 'div_j_bull_max': 30, 'div_j_bear_min': 70,
    'div_confirm_bars': 5,
    'adx_thresh': 25, 'adx_len': 14, 'atr_len': 14,
    't_enabled': False, 'trend_dema_fast': 8, 'trend_dema_slow': 34,
    'trend_atr_sl': 1.2, 'trend_trail_gap': 0.8,
    'dema_fast': 8, 'dema_slow': 34, 'kdj_period': 9,
    'session_start_min': 575, 'session_end_min': 950,
    'a_skip_14': True, 'bc_all_day': True, 'bc_end_min': 900,
}


# ═══════════════════════════════════════════════
#  Tick → Bar Aggregation
# ═══════════════════════════════════════════════

@dataclass
class Bar:
    symbol: str
    timeframe: str
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    closed: bool = False


TF_SECONDS = {'30 secs': 30, '1 min': 60, '5 mins': 300}


def _bar_period_start(dt, tf_secs):
    epoch = datetime(1970, 1, 1, tzinfo=pytz.utc)
    total = int((dt - epoch).total_seconds())
    return epoch + timedelta(seconds=(total // tf_secs) * tf_secs)


class TickBarBuilder:
    def __init__(self, symbol='QQQ', tf_secs=30):
        self.symbol = symbol
        self.tf_secs = tf_secs
        self._pending = None
        self.on_bar_closed: Optional[Callable] = None

    def on_tick(self, price: float, volume: int = 0):
        now = datetime.now(pytz.utc)
        period = _bar_period_start(now, self.tf_secs)
        price = float(price)
        if self._pending is None:
            self._pending = Bar(self.symbol, '30 secs', period,
                                price, price, price, price, volume, False)
            return
        if period == self._pending.dt:
            self._pending.high = max(self._pending.high, price)
            self._pending.low = min(self._pending.low, price)
            self._pending.close = price
            self._pending.volume += volume
            return
        self._pending.closed = True
        if self.on_bar_closed:
            self.on_bar_closed(self._pending)
        self._pending = Bar(self.symbol, '30 secs', period,
                            price, price, price, price, volume, False)


class BarAggregator:
    def __init__(self, timeframes=('1 min', '5 mins')):
        self.timeframes = timeframes
        self.tf_secs = {tf: TF_SECONDS[tf] for tf in timeframes}
        self._pending: Dict[str, Optional[Bar]] = {tf: None for tf in timeframes}
        self._closed: Dict[str, List[Bar]] = defaultdict(list)
        self.on_closed_bar: Optional[Callable] = None

    def update(self, raw_bar: Bar):
        newly_closed = []
        for tf in sorted(self.timeframes, key=lambda x: self.tf_secs[x]):
            closed = self._update_tf(tf, raw_bar)
            if closed:
                newly_closed.append(closed)
        if newly_closed and self.on_closed_bar:
            live = dict(self._pending)
            for cb in newly_closed:
                self.on_closed_bar(cb, live)

    def _update_tf(self, tf, raw_bar):
        tf_secs = self.tf_secs[tf]
        period = _bar_period_start(raw_bar.dt, tf_secs)
        pending = self._pending[tf]
        if pending is None:
            self._pending[tf] = Bar(raw_bar.symbol, tf, period,
                                    raw_bar.open, raw_bar.high, raw_bar.low,
                                    raw_bar.close, raw_bar.volume, False)
            return None
        if period == pending.dt:
            pending.high = max(pending.high, raw_bar.high)
            pending.low = min(pending.low, raw_bar.low)
            pending.close = raw_bar.close
            pending.volume += raw_bar.volume
            return None
        pending.closed = True
        self._closed[tf].append(pending)
        closed_bar = pending
        self._pending[tf] = Bar(raw_bar.symbol, tf, period,
                                raw_bar.open, raw_bar.high, raw_bar.low,
                                raw_bar.close, raw_bar.volume, False)
        return closed_bar


# ═══════════════════════════════════════════════
#  Position + CSV Logger
# ═══════════════════════════════════════════════

@dataclass
class Position:
    direction: int = 0
    qqq_entry: float = 0.0     # QQQ信号价（SL/TP/目标价基于此）
    tqqq_entry: float = 0.0    # TQQQ实际成交价（PnL基于此）
    sl_price: float = 0.0      # QQQ价格空间的SL
    phase: str = ''
    tranche: int = 0
    trail_high: float = None
    trail_low: float = None
    entry_bar: int = 0
    a_sl_atr: float = None
    b_tp_target: float = None   # QQQ价格空间的TP
    entry_time: str = ''


class TradeCSVLogger:
    """每笔交易写入CSV，方便后续分析。"""
    _FIELDS = ['timestamp', 'action', 'side', 'qty', 'phase', 'price',
               'fill_price', 'sl', 'tp', 'pnl', 'daily_pnl', 'reason',
               'j', 'adx', 'atr', 'mom_score']

    def __init__(self, filepath):
        self.filepath = filepath
        self._write_header()

    def _write_header(self):
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            with open(self.filepath, 'w', newline='') as f:
                csv.DictWriter(f, self._FIELDS).writeheader()

    def log_trade(self, **kwargs):
        row = {k: kwargs.get(k, '') for k in self._FIELDS}
        with open(self.filepath, 'a', newline='') as f:
            csv.DictWriter(f, self._FIELDS).writerow(row)


# ═══════════════════════════════════════════════
#  KDJ Live Trader
# ═══════════════════════════════════════════════

class KDJLiveTrader:

    def __init__(self, ib: IB, qty: int = 100, dry_run: bool = True):
        self.ib = ib
        self.qty = qty
        self._initial_qty = qty
        self.dry_run = dry_run
        self.p = PARAMS

        # 指标
        self.kdj = KDJCalculator(self.p['kdj_period'])
        self.adx = ADXCalculator(self.p['adx_len'])
        self.atr = ATRCalculator(self.p['atr_len'])
        self.div = SpeedDivergence(
            self.p['div_len'], self.p['div_thresh'],
            self.p['div_j_bull_max'], self.p['div_j_bear_min'],
        )
        self.trend_dema_f = DEMACalculator(self.p['trend_dema_fast'])
        self.trend_dema_s = DEMACalculator(self.p['trend_dema_slow'])

        from momentum_scorer import MomentumScorer
        self.momentum = MomentumScorer(
            fast_n=self.p['dema_fast'], slow_n=self.p['dema_slow'],
        )

        # 持仓
        self.pos = Position()
        self._bar5_count = 0
        self._prev_signals = {}
        self._prev_in_trend = False
        self._prev_trend_dir = 0
        self._prev_mom_score = 0

        # Tick → Bar
        self.tick_builder = TickBarBuilder('QQQ', tf_secs=30)
        self.agg = BarAggregator(('1 min', '5 mins'))
        self.tick_builder.on_bar_closed = self._on_30s_bar
        self.agg.on_closed_bar = self._on_agg_bar

        # IB — QQQ 信号源 + TQQQ 下单标的
        self.signal_contract = Stock('QQQ', 'SMART', 'USD')
        self.trade_contract = Stock('TQQQ', 'SMART', 'USD')
        self._last_price = 0.0
        self._processing = False  # 防止重入
        self._order_pending = False  # 防止重复下单

        # 风控
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self._halted = False

        # CSV logger
        self.csv_logger = TradeCSVLogger(_csv_file)
        self.trades_today = []

        # 恢复持仓
        self._load_state()

    # ── 持仓持久化 ──

    def _save_state(self):
        state = {
            'pos': asdict(self.pos),
            'qty': self.qty,
            'daily_pnl': self.daily_pnl,
            'daily_trade_count': self.daily_trade_count,
            'date': _today,
        }
        with open(_state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        if not os.path.exists(_state_file):
            return
        try:
            with open(_state_file) as f:
                state = json.load(f)
            if state.get('date') != _today:
                log.info("持仓状态过期（隔日），忽略")
                os.remove(_state_file)
                return
            self.pos = Position(**state['pos'])
            self.qty = state.get('qty', self._initial_qty)
            self.daily_pnl = state.get('daily_pnl', 0.0)
            self.daily_trade_count = state.get('daily_trade_count', 0)
            if self.pos.direction != 0:
                self.pos.entry_bar = -1  # 标记需要在预热后修复
                log.info(f"恢复持仓: {self.pos.phase} {'多' if self.pos.direction==1 else '空'} "
                    f"QQQ@${self.pos.qqq_entry:.2f} TQQQ@${self.pos.tqqq_entry:.2f} "
                    f"SL=${self.pos.sl_price:.2f}")
            log.info(f"恢复日统计: PnL=${self.daily_pnl:.2f} trades={self.daily_trade_count}")
        except Exception as e:
            log.info(f"恢复持仓失败: {e}")

    # ── 风控 ──

    def _check_risk(self) -> bool:
        """返回 True = 允许交易，False = 已触发风控停止。"""
        if self._halted:
            return False
        if self.daily_pnl <= -MAX_DAILY_LOSS:
            log.info(f"⛔ 日亏损上限! PnL=${self.daily_pnl:.2f} >= -${MAX_DAILY_LOSS}")
            self._halted = True
            return False
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            log.info(f"⛔ 日交易次数上限! {self.daily_trade_count} >= {MAX_DAILY_TRADES}")
            self._halted = True
            return False
        return True

    def _is_market_hours(self) -> bool:
        """只在 RTH 9:30-16:00 ET 处理tick。"""
        now = datetime.now(ET)
        cmins = now.hour * 60 + now.minute
        return 570 <= cmins <= 960  # 9:30 ~ 16:00

    # ── 启动 ──

    def start(self):
        self.ib.qualifyContracts(self.signal_contract)
        self.ib.qualifyContracts(self.trade_contract)

        log.info("拉取 QQQ 历史bar预热指标...")
        hist_5m = self.ib.reqHistoricalData(
            self.signal_contract, endDateTime='', durationStr='5 D',
            barSizeSetting='5 mins', whatToShow='TRADES',
            useRTH=True, keepUpToDate=False,
        )
        hist_1m = self.ib.reqHistoricalData(
            self.signal_contract, endDateTime='', durationStr='2 D',
            barSizeSetting='1 min', whatToShow='TRADES',
            useRTH=True, keepUpToDate=False,
        )

        for bar in hist_1m:
            dt = bar.date.astimezone(ET) if hasattr(bar.date, 'astimezone') else bar.date
            self.momentum.update(dt, bar.open, bar.high, bar.low, bar.close, bar.volume)

        for bar in hist_5m:
            self.kdj.update(bar.high, bar.low, bar.close)
            self.adx.update(bar.high, bar.low, bar.close)
            self.atr.update(bar.high, bar.low, bar.close)
            self.trend_dema_f.update(bar.close)
            self.trend_dema_s.update(bar.close)
            if self.kdj.ready:
                self.div.update(bar.high, bar.low, bar.close, self.kdj.j)
            self._bar5_count += 1

        log.info(f"预热完成: KDJ J={self.kdj.j:.1f} ADX={self.adx._adx:.1f} "
            f"ATR={self.atr.value:.3f} 5m={len(hist_5m)} 1m={len(hist_1m)}")

        # 修复重启后的 entry_bar（预热后 _bar5_count 才有意义）
        if self.pos.direction != 0 and self.pos.entry_bar == -1:
            self.pos.entry_bar = self._bar5_count
            log.info(f"  entry_bar 修复为 {self._bar5_count}")

        # 订阅 QQQ tick（信号源）
        self.ib.reqMktData(self.signal_contract, '', False, False)
        self.ib.pendingTickersEvent += self._on_ticker_update
        log.info(f"QQQ tick订阅启动! 下单标的=TQQQ qty={self.qty} dry_run={self.dry_run}")

    # ── Tick Events ──

    def _on_ticker_update(self, tickers):
        if self._processing:
            return  # 防重入
        self._processing = True
        try:
            for ticker in tickers:
                if ticker.contract != self.signal_contract:
                    continue
                price = ticker.last or ticker.close
                if not price or price <= 0:
                    continue
                if not self._is_market_hours():
                    continue

                vol = ticker.lastSize or 0
                self._last_price = price
                self.tick_builder.on_tick(price, vol)

                # tick级入场：有信号就立刻进（不等下一根5min bar闭合）
                if self.pos.direction == 0 and not self._order_pending and self._prev_signals and self._check_risk():
                    self._try_entry_tick(price)

                if self.pos.direction != 0 and self.kdj.ready:
                    sim_j = self.kdj.simulate_j(price)
                    self._check_j_early_exit(sim_j, price)

                if self.pos.direction != 0:
                    self._check_price_exits_tick(price)
        finally:
            self._processing = False

    def _on_30s_bar(self, bar_30s):
        self.agg.update(bar_30s)

    def _on_agg_bar(self, closed_bar, live_bars):
        if closed_bar.timeframe == '1 min':
            dt = closed_bar.dt.astimezone(ET) if closed_bar.dt.tzinfo else closed_bar.dt
            self.momentum.update(dt, closed_bar.open, closed_bar.high,
                                 closed_bar.low, closed_bar.close, closed_bar.volume)
            return
        if closed_bar.timeframe == '5 mins':
            self._process_5m_bar(closed_bar)

    # ── 5min 信号主逻辑 ──

    def _process_5m_bar(self, closed_bar):
        self._bar5_count += 1
        dt = closed_bar.dt.astimezone(ET) if closed_bar.dt.tzinfo else closed_bar.dt
        o, h, l, c, v = closed_bar.open, closed_bar.high, closed_bar.low, closed_bar.close, closed_bar.volume
        cmins = dt.hour * 60 + dt.minute
        p = self.p

        k, d, j = self.kdj.update(h, l, c)
        prev_j, prev_prev_j = self.kdj.prev_j, self.kdj.prev_prev_j
        adx_val = self.adx.update(h, l, c)
        atr_val = self.atr.update(h, l, c)
        self.trend_dema_f.update(c)
        self.trend_dema_s.update(c)

        in_trend = self.adx.ready and adx_val > p['adx_thresh']
        adx_breakout = in_trend and not self._prev_in_trend
        trend_dir = 0
        if self.trend_dema_f.ready and self.trend_dema_s.ready:
            trend_dir = 1 if self.trend_dema_f.value > self.trend_dema_s.value else -1

        bull_div, bear_div = False, False
        if self.kdj.ready and not in_trend:
            bull_div, bear_div = self.div.update(h, l, c, j)

        s_start, s_end = p['session_start_min'], p['session_end_min']
        in_session = s_start <= cmins <= s_end
        bc_end = p['bc_end_min']
        in_bc = in_session if p['bc_all_day'] else (s_start <= cmins < bc_end)
        in_pure_c = bc_end <= cmins <= s_end
        in_14 = 840 <= cmins < 900

        log.info(f"5m {dt.strftime('%H:%M')} | C={c:.2f} J={j:.1f} ADX={adx_val:.1f} "
            f"ATR={atr_val:.3f} mom={self.momentum.score} "
            f"pos={self.pos.phase or 'flat'} daily_pnl=${self.daily_pnl:.2f}")

        if self.pos.direction != 0 and adx_breakout:
            self._close_position(c, 'ADX_close')

        if self.pos.direction != 0 and cmins >= 955:
            self._close_position(c, 'EOD')
            self._prev_in_trend = in_trend
            self._prev_trend_dir = trend_dir
            self._prev_mom_score = self.momentum.score
            return

        if self.pos.direction == 1:
            if self.pos.trail_high is None or h > self.pos.trail_high:
                self.pos.trail_high = h
        elif self.pos.direction == -1:
            if self.pos.trail_low is None or l < self.pos.trail_low:
                self.pos.trail_low = l

        if self.pos.direction != 0:
            self._check_5m_exits(dt, h, l, c, j, prev_j, prev_prev_j, atr_val, in_trend)

        # 入场已移到tick级（_on_ticker_update），这里不再入场

        self._gen_signals(j, prev_j, prev_prev_j, c, in_trend, trend_dir,
                          bull_div, bear_div, in_session, in_bc, in_pure_c, in_14)

        self._prev_in_trend = in_trend
        self._prev_trend_dir = trend_dir
        self._prev_mom_score = self.momentum.score

    # ── Tick级出场 ──

    def _check_price_exits_tick(self, price):
        d = self.pos.direction
        if d == 0:
            return

        if self.pos.phase == 'B':
            if self.pos.b_tp_target and \
               ((d == 1 and price >= self.pos.b_tp_target) or (d == -1 and price <= self.pos.b_tp_target)):
                self._close_position(price, 'B_TP')
                return
            if (d == 1 and price <= self.pos.sl_price) or (d == -1 and price >= self.pos.sl_price):
                sl_p = self.pos.sl_price
                self._close_position(sl_p, 'B_SL→C')
                if self._check_risk():
                    new_dir = -d
                    self._execute_order(new_dir, sl_p, 'C', f"B反手C({'多' if new_dir==1 else '空'})")
                    self.pos.sl_price = sl_p - self.p['c_stop_loss'] * new_dir
                return

        elif self.pos.phase in ('C', '纯C'):
            p = self.p
            ep = self.pos.qqq_entry  # 所有目标价用QQQ价格空间
            if (d == 1 and price <= self.pos.sl_price) or (d == -1 and price >= self.pos.sl_price):
                self._close_position(price, 'C_SL')
                return
            t1 = ep + p['c_t1_pts'] * d
            t2 = ep + p['c_t2_pts'] * d
            if self.pos.tranche == 0:
                if (d == 1 and price >= t1) or (d == -1 and price <= t1):
                    self._partial_exit(0.4, price, 'C_T1')
                    self.pos.tranche = 1
                    self.pos.sl_price = ep  # breakeven at QQQ entry
                    self.pos.trail_high = self.pos.trail_low = None
            elif self.pos.tranche == 1:
                if (d == 1 and price >= t2) or (d == -1 and price <= t2):
                    self._partial_exit(0.35, price, 'C_T2')
                    self.pos.tranche = 2
                    self.pos.sl_price = ep + p['c_t1_pts'] * d  # lock T1 profit
                    self.pos.trail_high = self.pos.trail_low = None
            elif self.pos.tranche == 2:
                if d == 1:
                    if self.pos.trail_high is None or price > self.pos.trail_high:
                        self.pos.trail_high = price
                    if self.pos.trail_high and price <= self.pos.trail_high - p['c_trail_gap']:
                        self._close_position(price, 'C_T3_trail')
                else:
                    if self.pos.trail_low is None or price < self.pos.trail_low:
                        self.pos.trail_low = price
                    if self.pos.trail_low and price >= self.pos.trail_low + p['c_trail_gap']:
                        self._close_position(price, 'C_T3_trail')

        elif self.pos.phase == 'A' and self.pos.tranche == 0 and self.pos.a_sl_atr:
            if (d == 1 and price <= self.pos.a_sl_atr) or (d == -1 and price >= self.pos.a_sl_atr):
                self._close_position(price, 'A_ATR_stop')

        elif self.pos.phase == 'T':
            if (d == 1 and price <= self.pos.sl_price) or (d == -1 and price >= self.pos.sl_price):
                self._close_position(price, 'T_SL')

    def _check_j_early_exit(self, sim_j, price):
        p, d = self.p, self.pos.direction
        if d == 0:
            return
        price = float(price)
        if self.pos.phase == 'B':
            if (d == 1 and sim_j >= p['b_j_neutral']) or (d == -1 and sim_j <= p['b_j_neutral']):
                self._close_position(price, 'J_neutral')
        elif self.pos.phase == 'A':
            # J 止损
            if d == 1 and sim_j < p['a_long_j_stop']:
                self._close_position(price, 'J_stop')
            elif d == -1 and sim_j > p['a_short_j_stop']:
                self._close_position(price, 'J_stop')
            # J 止盈（T1：J已在目标位即出，不要求穿越）
            elif d == 1 and sim_j >= p['a_long_t1_j'] and self.pos.tranche == 0:
                self._close_position(price, 'A_T1')
            elif d == -1 and sim_j <= p['a_short_t1_j'] and self.pos.tranche == 0:
                self._close_position(price, 'A_T1')

    def _check_5m_exits(self, dt, h, l, c, j, prev_j, prev_prev_j, atr_val, in_trend):
        p, d = self.p, self.pos.direction
        if self.pos.phase == 'A':
            if self.pos.tranche == 0 and self.pos.entry_bar > 0:
                if (self._bar5_count - self.pos.entry_bar) >= p['a_time_stop']:
                    self._close_position(c, 'A_time_stop')
                    return
            if d == 1 and self.pos.tranche == 0 and prev_j >= p['a_long_t1_j'] and prev_prev_j < p['a_long_t1_j']:
                self._close_position(c, 'A_T1')
            elif d == -1 and self.pos.tranche == 0 and prev_j <= p['a_short_t1_j'] and prev_prev_j > p['a_short_t1_j']:
                self._close_position(c, 'A_T1')
        elif self.pos.phase == 'T':
            mom_rev = (d == 1 and self.momentum.trend_direction == -1) or \
                      (d == -1 and self.momentum.trend_direction == 1)
            adx_ret = not in_trend and self._prev_in_trend
            trail = (d == 1 and self.pos.trail_high and l <= self.pos.trail_high - p['trend_trail_gap']) or \
                    (d == -1 and self.pos.trail_low and h >= self.pos.trail_low + p['trend_trail_gap'])
            if mom_rev or trail or adx_ret:
                self._close_position(c, 'T_exit')

    # ── 信号生成 ──

    def _exhaust_allows(self, direction, score, prev):
        if direction == 1:
            return (score > prev) or (score > -3)
        return (score < prev) or (score < 3)

    def _gen_signals(self, j, prev_j, prev_prev_j, close, in_trend, trend_dir,
                     bull_div, bear_div, in_session, in_bc, in_pure_c, in_14):
        p = self.p
        ms, md, pms = self.momentum.score, self.momentum.trend_direction, self._prev_mom_score
        signals = {}
        if not self.kdj.ready:
            self._prev_signals = signals
            return

        if p.get('a_enabled', True):
            a_ok = not (p['a_skip_14'] and in_14) and in_session and not in_trend
            if a_ok:
                if prev_j < 0 and prev_prev_j >= 0 and self._exhaust_allows(1, ms, pms):
                    signals['A_LONG'] = close
                if prev_j > 100 and prev_prev_j <= 100 and self._exhaust_allows(-1, ms, pms):
                    signals['A_SHORT'] = close

        if in_bc and not in_trend:
            if bull_div and self._exhaust_allows(1, ms, pms):
                signals['B_LONG'] = close
            if bear_div and self._exhaust_allows(-1, ms, pms):
                signals['B_SHORT'] = close

        if p.get('t_enabled', True) and in_trend and in_session:
            ptd = self._prev_trend_dir
            if trend_dir == 1 and ptd != 1 and md >= 0:
                signals['T_LONG'] = close
            if trend_dir == -1 and ptd != -1 and md <= 0:
                signals['T_SHORT'] = close

        if signals:
            log.info(f"  ⚡ 信号: {list(signals.keys())}")
        self._prev_signals = signals

    # ── 入场 ──

    def _try_entry_tick(self, price):
        """tick级入场：5min bar闭合产生信号后，下一个tick立刻进场。"""
        atr_val = self.atr.value  # 用最新ATR
        p = self.p
        for key, phase, d in [
            ('B_LONG', 'B', 1), ('B_SHORT', 'B', -1),
            ('A_LONG', 'A', 1), ('A_SHORT', 'A', -1),
            ('T_LONG', 'T', 1), ('T_SHORT', 'T', -1),
        ]:
            if key not in self._prev_signals:
                continue
            self._execute_order(d, price, phase, f"{phase}{'做多' if d==1 else '做空'}")
            if phase == 'A':
                self.pos.a_sl_atr = price - atr_val * p['a_atr_mult'] * d
                self.pos.sl_price = self.pos.a_sl_atr
                self.pos.entry_bar = self._bar5_count
            elif phase == 'B':
                self.pos.sl_price = price - atr_val * p['b_atr_mult_sl'] * d
                self.pos.b_tp_target = price + atr_val * p['b_atr_mult_tp'] * d
                log.info(f"    SL=${self.pos.sl_price:.2f} TP=${self.pos.b_tp_target:.2f}")
            elif phase in ('C', '纯C'):
                self.pos.sl_price = price - p['c_stop_loss'] * d
            elif phase == 'T':
                self.pos.sl_price = price - atr_val * p['trend_atr_sl'] * d
            self._prev_signals = {}  # 信号已消费，清空防重复入场
            self._save_state()
            return

    def _try_entry(self, dt, open_price, atr_val):
        p = self.p
        for key, phase, d in [
            ('B_LONG', 'B', 1), ('B_SHORT', 'B', -1),
            ('A_LONG', 'A', 1), ('A_SHORT', 'A', -1),
            ('T_LONG', 'T', 1), ('T_SHORT', 'T', -1),
        ]:
            if key not in self._prev_signals:
                continue
            self._execute_order(d, open_price, phase, f"{phase}{'做多' if d==1 else '做空'}")
            if phase == 'A':
                self.pos.a_sl_atr = open_price - atr_val * p['a_atr_mult'] * d
                self.pos.sl_price = self.pos.a_sl_atr
                self.pos.entry_bar = self._bar5_count
            elif phase == 'B':
                self.pos.sl_price = open_price - atr_val * p['b_atr_mult_sl'] * d
                self.pos.b_tp_target = open_price + atr_val * p['b_atr_mult_tp'] * d
                log.info(f"    SL=${self.pos.sl_price:.2f} TP=${self.pos.b_tp_target:.2f}")
            elif phase in ('C', '纯C'):
                self.pos.sl_price = open_price - p['c_stop_loss'] * d
            elif phase == 'T':
                self.pos.sl_price = open_price - atr_val * p['trend_atr_sl'] * d
            self._save_state()
            return

    # ── 下单 ──

    def _execute_order(self, direction, price, phase, reason):
        """price = QQQ信号价，实际下单TQQQ。qqq_entry存QQQ价，tqqq_entry存TQQQ成交价。"""
        self._order_pending = True  # 立刻锁定，防止 tick 重复触发
        self._prev_signals = {}     # 立刻清空信号

        side = 'BUY' if direction == 1 else 'SELL'
        self.qty = self._initial_qty
        tqqq_fill = 0.0

        if self.dry_run:
            log.info(f"  ★ [DRY] {reason}: {side} {self.qty} TQQQ (QQQ信号@ ${price:.2f})")
            tqqq_fill = 0.0
        else:
            accounts = self.ib.managedAccounts()
            if not any(a.startswith('DU') for a in accounts):
                log.info(f"  BLOCKED: Not paper account ({accounts})")
                self._order_pending = False
                return
            order = MarketOrder(side, self.qty)
            trade = self.ib.placeOrder(self.trade_contract, order)
            log.info(f"  ★ {reason}: {side} {self.qty} TQQQ...")
            start = time.time()
            while time.time() - start < 30:
                self.ib.sleep(0.5)
                if trade.orderStatus.status == 'Filled':
                    tqqq_fill = trade.orderStatus.avgFillPrice
                    log.info(f"  ✓ FILLED TQQQ @ ${tqqq_fill:.2f}")
                    break
            else:
                log.info(f"  ✗ Fill timeout")
                self.ib.cancelOrder(order)
                self._order_pending = False
                return

        self.pos = Position(direction=direction, qqq_entry=price, tqqq_entry=tqqq_fill,
                            phase=phase, entry_time=datetime.now(ET).strftime('%H:%M:%S'))
        self.daily_trade_count += 1

        self.csv_logger.log_trade(
            timestamp=datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
            action='ENTRY', side=side, qty=self.qty, phase=phase,
            price=f"{price:.2f}", fill_price=f"{tqqq_fill:.2f}",
            sl=f"{self.pos.sl_price:.2f}" if self.pos.sl_price else '',
            reason=reason,
            j=f"{self.kdj.j:.1f}", adx=f"{self.adx._adx:.1f}",
            atr=f"{self.atr.value:.3f}", mom_score=str(self.momentum.score),
        )

        self.trades_today.append({
            'time': datetime.now(ET).strftime('%H:%M:%S'),
            'action': side, 'phase': phase, 'price': tqqq_fill,
            'reason': reason, 'qqq_signal_price': price,
        })
        self._order_pending = False  # 下单完成，解锁

    def _close_position(self, price, reason):
        """price = QQQ触发价（用于日志），实际平TQQQ，PnL用TQQQ成交价算。"""
        if self.pos.direction == 0 or self._order_pending:
            return
        self._order_pending = True
        d = self.pos.direction
        side = 'SELL' if d == 1 else 'BUY'
        tqqq_fill = 0.0
        pnl = 0.0

        if self.dry_run:
            log.info(f"  ✗ [DRY] {reason}: {side} {self.qty} TQQQ (QQQ触发@ ${price:.2f})")
        else:
            order = MarketOrder(side, self.qty)
            trade = self.ib.placeOrder(self.trade_contract, order)
            start = time.time()
            while time.time() - start < 30:
                self.ib.sleep(0.5)
                if trade.orderStatus.status == 'Filled':
                    tqqq_fill = trade.orderStatus.avgFillPrice
                    pnl = (tqqq_fill - self.pos.tqqq_entry) * d * self.qty
                    log.info(f"  ✓ CLOSED TQQQ @ ${tqqq_fill:.2f} PnL=${pnl:.2f}")
                    break
            else:
                log.info(f"  ✗ 平仓超时，取消订单，持仓保持不变")
                self.ib.cancelOrder(order)
                return  # 不清空持仓

        self.daily_pnl += pnl

        self.csv_logger.log_trade(
            timestamp=datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
            action='EXIT', side=side, qty=self.qty, phase=self.pos.phase,
            price=f"{price:.2f}", fill_price=f"{tqqq_fill:.2f}",
            pnl=f"{pnl:.2f}", daily_pnl=f"{self.daily_pnl:.2f}",
            reason=reason,
            j=f"{self.kdj.j:.1f}", adx=f"{self.adx._adx:.1f}",
            atr=f"{self.atr.value:.3f}", mom_score=str(self.momentum.score),
        )

        self.trades_today.append({
            'time': datetime.now(ET).strftime('%H:%M:%S'),
            'action': side, 'phase': self.pos.phase, 'price': tqqq_fill,
            'reason': reason, 'pnl': pnl, 'qqq_trigger_price': price,
        })
        self.pos = Position()
        self._order_pending = False
        self._save_state()

    def _partial_exit(self, pct, price, reason):
        """price = QQQ触发价，实际部分平TQQQ。"""
        partial_qty = max(1, int(self.qty * pct))
        d = self.pos.direction
        side = 'SELL' if d == 1 else 'BUY'
        tqqq_fill = 0.0
        pnl = 0.0

        if self.dry_run:
            log.info(f"  △ [DRY] {reason}: {side} {partial_qty}/{self.qty} TQQQ (QQQ@ ${price:.2f})")
        else:
            order = MarketOrder(side, partial_qty)
            trade = self.ib.placeOrder(self.trade_contract, order)
            start = time.time()
            while time.time() - start < 30:
                self.ib.sleep(0.5)
                if trade.orderStatus.status == 'Filled':
                    tqqq_fill = trade.orderStatus.avgFillPrice
                    pnl = (tqqq_fill - self.pos.tqqq_entry) * d * partial_qty
                    log.info(f"  △ {reason}: {side} {partial_qty} TQQQ @ ${tqqq_fill:.2f} PnL=${pnl:.2f}")
                    break

        self.daily_pnl += pnl
        self.qty -= partial_qty

        self.csv_logger.log_trade(
            timestamp=datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
            action='PARTIAL', side=side, qty=partial_qty, phase=self.pos.phase,
            price=f"{price:.2f}", fill_price=f"{tqqq_fill:.2f}",
            pnl=f"{pnl:.2f}", daily_pnl=f"{self.daily_pnl:.2f}",
            reason=reason,
            j=f"{self.kdj.j:.1f}", adx=f"{self.adx._adx:.1f}",
            atr=f"{self.atr.value:.3f}", mom_score=str(self.momentum.score),
        )

        self.trades_today.append({
            'time': datetime.now(ET).strftime('%H:%M:%S'),
            'action': f'{side}({pct:.0%})', 'phase': self.pos.phase,
            'price': tqqq_fill, 'reason': reason, 'pnl': pnl,
        })
        self._save_state()

    def print_summary(self):
        if not self.trades_today:
            log.info("今日无交易")
            return
        total_pnl = sum(t.get('pnl', 0) for t in self.trades_today)
        log.info(f"\n{'='*60}")
        log.info(f"  今日: {len(self.trades_today)}笔  PnL=${total_pnl:.2f}")
        log.info(f"{'='*60}")
        for t in self.trades_today:
            pnl_s = f" PnL=${t['pnl']:.2f}" if 'pnl' in t else ""
            log.info(f"  {t['time']} {t['action']:10s} {t['phase']:3s} "
                f"${t['price']:.2f} {t['reason']}{pnl_s}")


# ═══════════════════════════════════════════════
#  Main with reconnection
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='KDJ v4.8b Live Trader (Tick级)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=4002, help='4002=Paper, 4001=Live')
    parser.add_argument('--client-id', type=int, default=30)
    parser.add_argument('--qty', type=int, default=500)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-daily-loss', type=float, default=500.0)
    parser.add_argument('--max-daily-trades', type=int, default=20)
    args = parser.parse_args()

    global MAX_DAILY_LOSS, MAX_DAILY_TRADES
    MAX_DAILY_LOSS = args.max_daily_loss
    MAX_DAILY_TRADES = args.max_daily_trades

    log.info(f"KDJ v4.8b Tick Live Trader")
    log.info(f"  IB: {args.host}:{args.port} clientId={args.client_id}")
    log.info(f"  Qty: {args.qty}  {'DRY RUN' if args.dry_run else 'LIVE ORDERS'}")
    log.info(f"  Risk: max_loss=${MAX_DAILY_LOSS} max_trades={MAX_DAILY_TRADES}")

    reconnect_count = 0
    trader = None

    while reconnect_count <= MAX_RECONNECT:
        ib = IB()
        try:
            ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)
            log.info(f"Connected. Accounts: {ib.managedAccounts()}")
            reconnect_count = 0  # 连接成功重置计数

            trader = KDJLiveTrader(ib, qty=args.qty, dry_run=args.dry_run)
            trader.start()

            log.info("运行中... (Ctrl+C 停止)")
            while True:
                ib.sleep(1)

        except KeyboardInterrupt:
            log.info("\n用户停止")
            if trader:
                trader.print_summary()
            break

        except Exception as e:
            reconnect_count += 1
            log.info(f"连接断开: {e}")
            if reconnect_count <= MAX_RECONNECT:
                log.info(f"  {RECONNECT_DELAY}秒后重连 ({reconnect_count}/{MAX_RECONNECT})...")
                time.sleep(RECONNECT_DELAY)
            else:
                log.info(f"  重连次数耗尽，退出")
        finally:
            try:
                ib.disconnect()
            except:
                pass

    if trader:
        trader.print_summary()
    log.info("已退出.")


if __name__ == '__main__':
    main()
