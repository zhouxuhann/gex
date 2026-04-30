#!/usr/bin/env python3
"""
KDJ v4.8 回测引擎

用 IBKR 本地数据库的 QQQ 30 秒 K 线（2022-12-30 ~ 2026-03-27）回放
live trader 的完整信号+入场+出场逻辑。

- 重用 engine_v48_live 的所有 indicator 类（KDJ/ADX/ATR/DEMA/SpeedDivergence/PureCState）
- 信号/入场/出场逻辑镜像 kdj_live_trader._gen_signals / _check_*_exits
- 无 IB/订单/tick threading，单线程顺序回放
- PnL 用 TQQQ ≈ 3× QQQ intraday 的简化近似

Usage:
    python3 kdj_backtest.py --start 2026-03-20 --end 2026-03-27
    python3 kdj_backtest.py --start 2024-01-01 --end 2024-03-31 --out results.csv
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import pytz

# 复用 indicator 类（不碰 live code，只导入）
from engine_v48_live import (
    KDJCalculator, ADXCalculator, ATRCalculator, DEMACalculator,
    SpeedDivergence, PureCState,
)
from gex_regime_reader import GEXRegimeReader
from vix_regime_reader import VIXRegimeReader

ET = pytz.timezone('America/New_York')

# ═══════════════════════════════════════════════
#  回测参数（镜像 kdj_live_trader.PARAMS）
# ═══════════════════════════════════════════════
PARAMS = {
    'b_atr_mult_sl': 0.6, 'b_atr_mult_t1': 1.0, 'b_atr_mult_t2': 2.0, 'b_j_neutral': 50,
    'a_enabled': True,  # 回测默认打开，方便对比
    'a_atr_mult': 1.5,
    'a_long_j_stop': -10, 'a_short_j_stop': 110, 'a_time_stop': 8,
    'a_long_t1_j': 30, 'a_short_t1_j': 75,
    'c_stop_loss': 0.5, 'c_t1_pts': 0.8, 'c_t2_pts': 1.6, 'c_trail_gap': 0.6,
    'div_len': 5, 'div_thresh': 1.5, 'div_j_bull_max': 30, 'div_j_bear_min': 70,
    'div_confirm_bars': 5,
    'pure_c_enabled': True,
    'adx_thresh': 25, 'adx_len': 14, 'atr_len': 14,
    't_enabled': True, 'trend_dema_fast': 8, 'trend_dema_slow': 34,
    'trend_atr_sl': 1.2, 'trend_trail_gap': 0.8,
    'dema_fast': 8, 'dema_slow': 34, 'kdj_period': 9,
    'session_start_min': 575, 'session_end_min': 950,  # 09:35 - 15:50
    'a_skip_14': True, 'bc_all_day': True, 'bc_end_min': 900,  # 纯 C 从 15:00 起
    't_confirm_2bar': True,   # T 信号 2-bar 确认
    'b_reverse_c': True,      # B 止损后立即反手 C（v4.8 原版行为）
    # GEX regime gate: +γ 只开 A/B, -γ 只开 T, 纯C 不 gate
    'gex_gate_enabled': False,  # backtest 默认关，--gex-gate 打开
    'gex_gate_symbol': 'QQQ',
    'gex_stale_sec_max': 180,
    # VIX regime gate: 用前一日 VIX close 分三档，低→反转, 高→趋势, 中→全开
    'vix_gate_enabled': False,
    'vix_low_thresh': 15.0,
    'vix_high_thresh': 25.0,
    # 回测专用
    'tqqq_leverage': 3.0,     # TQQQ ≈ 3× QQQ intraday 收益
    'qty': 500,               # 每次交易数量
    'slippage_bps': 2.0,      # 单边滑点 (bps on QQQ)
}


# ═══════════════════════════════════════════════
#  简化版 MomentumScorer（backtest only）
# ═══════════════════════════════════════════════
class SimpleMomentumScorer:
    """
    简化版 1-min 动量评分器。不做 VWAP/Kalman，只用 DEMA cross。
    score ∈ {-1, 0, +1}，不会触及 |score|==2 或 3，因此 _exhaust_allows 永远 True。

    这保证回测不会被 momentum filter 误挡信号，结果反映 v4.8 核心逻辑。
    live 版有完整 3-signal score 会更严，但我们先看 baseline。
    """
    def __init__(self, fast_n=8, slow_n=34):
        self.dema_f = DEMACalculator(fast_n)
        self.dema_s = DEMACalculator(slow_n)
        self.score = 0
        self.trend_direction = 0

    def update_1m(self, close):
        self.dema_f.update(close)
        self.dema_s.update(close)
        if self.dema_f.ready and self.dema_s.ready:
            if self.dema_f.value > self.dema_s.value:
                self.score = 1
                self.trend_direction = 1
            elif self.dema_f.value < self.dema_s.value:
                self.score = -1
                self.trend_direction = -1
            else:
                self.score = 0
                self.trend_direction = 0


# ═══════════════════════════════════════════════
#  Position 状态
# ═══════════════════════════════════════════════
@dataclass
class BacktestPosition:
    direction: int = 0
    qqq_entry: float = 0.0
    sl_price: float = 0.0
    phase: str = ''
    tranche: int = 0       # 0=full, 1=after T1, 2=after T2
    trail_high: Optional[float] = None
    trail_low: Optional[float] = None
    entry_bar: int = 0
    a_sl_atr: Optional[float] = None
    b_t1_target: Optional[float] = None
    b_t2_target: Optional[float] = None
    entry_time: str = ''
    qty: int = 0           # 初始仓量 (当 T1 部分出后 qty 减少)
    initial_qty: int = 0   # 最初入场的仓量


@dataclass
class TradeRecord:
    date: str
    action: str            # ENTRY / PARTIAL / EXIT
    phase: str
    signal: str
    dir: int
    qqq_entry: float = 0.0
    qqq_exit: float = 0.0
    qty: int = 0
    reason: str = ''
    pnl_bps: float = 0.0   # QQQ 点数的 bps
    pnl_usd: float = 0.0   # 按 TQQQ leverage 估算美元
    j: float = 0.0
    adx: float = 0.0
    hold_bars: int = 0     # 持仓 5min bar 数
    time: str = ''


# ═══════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════
def load_30s_bars(start_date: str, end_date: str, symbol='QQQ') -> pd.DataFrame:
    """从 IBKR PostgreSQL DB 加载 30 秒 bar 数据。返回按时间升序的 DataFrame (UTC)。"""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5433)),
        database=os.getenv('DB_NAME', 'ibkr_market_data'),
        user=os.getenv('DB_USER', 'ibkr_user'),
        password=os.getenv('DB_PASSWORD', 'ibkr_secure_password_2026'),
    )
    q = """
        SELECT datetime, open, high, low, close, volume
        FROM market_data_bars
        WHERE symbol=%s AND bar_size=%s
          AND datetime >= %s AND datetime < %s
        ORDER BY datetime ASC
    """
    # DB times are UTC-naive in this schema. QQQ RTH in UTC = 13:30 - 20:00
    df = pd.read_sql_query(
        q, conn,
        params=[symbol, '30 secs', start_date, end_date + ' 23:59:59'],
    )
    conn.close()
    if len(df) == 0:
        return df
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC')
    df['dt_et'] = df['datetime'].dt.tz_convert(ET)
    # Keep only RTH + cast to floats
    df['hhmm'] = df['dt_et'].dt.hour * 60 + df['dt_et'].dt.minute
    df = df[(df['hhmm'] >= 9 * 60 + 30) & (df['hhmm'] < 16 * 60)].copy()
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df['volume'] = df['volume'].astype(int)
    return df


# ═══════════════════════════════════════════════
#  回测核心 (one day at a time)
# ═══════════════════════════════════════════════
class KDJBacktester:

    def __init__(self, params=None, gex_reader=None, vix_reader=None):
        self.p = params or PARAMS
        self.gex_reader = gex_reader   # 可选：GEXRegimeReader（historical loaded）
        self.vix_reader = vix_reader   # 可选：VIXRegimeReader（historical loaded）
        self._reset_for_new_day()
        self.all_trades: list[TradeRecord] = []
        self.daily_stats: list[dict] = []
        self.gate_log = []  # 记录每次 gate 决策 (合并 gex + vix)

    def _reset_for_new_day(self):
        p = self.p
        self.kdj = KDJCalculator(p['kdj_period'])
        self.adx = ADXCalculator(p['adx_len'])
        self.atr = ATRCalculator(p['atr_len'])
        self.div = SpeedDivergence(p['div_len'], p['div_thresh'],
                                    p['div_j_bull_max'], p['div_j_bear_min'])
        self.pure_c = PureCState(p['div_confirm_bars'])
        self.trend_dema_f = DEMACalculator(p['trend_dema_fast'])
        self.trend_dema_s = DEMACalculator(p['trend_dema_slow'])
        self.momentum = SimpleMomentumScorer(p['dema_fast'], p['dema_slow'])

        self.pos = BacktestPosition()
        self._bar5_count = 0
        self._prev_signals = {}
        self._prev_in_trend = False
        self._prev_trend_dir = 0
        self._prev_prev_trend_dir = 0
        self._pending_1m = []   # 30s bars accumulator → 1 min
        self._pending_5m = []   # 30s bars accumulator → 5 min
        self.day_trades: list[TradeRecord] = []
        self.daily_pnl_bps = 0.0

    def run_day(self, date_str: str, bars_30s: pd.DataFrame):
        """回放一个交易日的 30s bars。"""
        self._reset_for_new_day()

        # Group 30s bars into 1min + 5min aligned buckets
        for _, row in bars_30s.iterrows():
            dt_et = row['dt_et']
            hhmm = dt_et.hour * 60 + dt_et.minute

            # 累积到 1-min / 5-min
            self._pending_1m.append(row)
            self._pending_5m.append(row)

            # 每 1min bar 闭合时（偶数秒数完成）
            # 5min bar 闭合时 (minute % 5 == 4 且 second == 30)
            # 简化：在 30 秒 bar 的 close second 判定是否是 1m/5m bar 边界
            sec = dt_et.second
            minute = dt_et.minute

            # 1-min bar: 闭合发生在 second 恰好为 30（上一个 30s bar 的终点对应整分钟末）
            if sec == 30:  # 这根 30s bar 覆盖 xx:30-xx:59 → 结束时是下一个 minute 的开始
                self._on_1m_close()

            # 5-min bar: 闭合在 minute % 5 == 4 且这根是 xx:59 结尾 (sec=30 of minute ending in 4)
            if sec == 30 and minute % 5 == 4:
                self._on_5m_close(dt_et)
                continue  # 5m bar 处理会覆盖 tick-level 出入场

            # Tick-level exits / entries (每 30s bar close 当成 tick)
            price = float(row['close'])
            self._on_tick(price, dt_et)

        # 日终强平（safety net，通常 15:50 之前 5m bar 处理会触发 EOD）
        if self.pos.direction != 0:
            last_price = float(bars_30s.iloc[-1]['close'])
            self._close_position(last_price, 'EOD_safety')

        # 汇总日统计
        self.all_trades.extend(self.day_trades)
        wins = [t for t in self.day_trades if t.action == 'EXIT' and t.pnl_bps > 0]
        losses = [t for t in self.day_trades if t.action == 'EXIT' and t.pnl_bps < 0]
        entries = [t for t in self.day_trades if t.action == 'ENTRY']
        exits = [t for t in self.day_trades if t.action == 'EXIT']

        stats = {
            'date': date_str,
            'n_entries': len(entries),
            'n_exits': len(exits),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(exits) * 100) if exits else 0,
            'day_pnl_bps': self.daily_pnl_bps,
            'day_pnl_usd': self.daily_pnl_bps / 10000 * 500 *
                           (bars_30s.iloc[0]['close'] if len(bars_30s) else 500) *
                           self.p['tqqq_leverage'] / self.p['tqqq_leverage'],  # rough
        }
        # Cleaner usd calc from records
        stats['day_pnl_usd'] = sum(t.pnl_usd for t in self.day_trades)
        self.daily_stats.append(stats)
        return stats

    # ── 1-min bar close ──
    def _on_1m_close(self):
        if len(self._pending_1m) < 2:
            return
        bars = self._pending_1m[-2:]  # last two 30s bars form one 1min bar
        c = float(bars[-1]['close'])
        self.momentum.update_1m(c)

    # ── 5-min bar close ──
    def _on_5m_close(self, dt_et):
        # Pull last 10 30s bars (= one 5min bar)
        if len(self._pending_5m) < 10:
            return
        last10 = self._pending_5m[-10:]
        o = float(last10[0]['open'])
        h = float(max(b['high'] for b in last10))
        l = float(min(b['low'] for b in last10))
        c = float(last10[-1]['close'])
        self._process_5m_bar(dt_et, o, h, l, c)

    def _process_5m_bar(self, dt, o, h, l, c):
        self._bar5_count += 1
        p = self.p
        cmins = dt.hour * 60 + dt.minute

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

        pure_c_long, pure_c_short = False, False
        if self.kdj.ready and p.get('pure_c_enabled', True):
            pure_c_long, pure_c_short = self.pure_c.update(
                bull_div, bear_div, c, in_pure_c, in_trend,
            )

        # ADX breakout → 强平
        if self.pos.direction != 0 and adx_breakout:
            self._close_position(c, 'ADX_close')

        # EOD 强平
        if self.pos.direction != 0 and cmins >= 955:
            self._close_position(c, 'EOD')
            self._prev_in_trend = in_trend
            self._prev_prev_trend_dir = self._prev_trend_dir
            self._prev_trend_dir = trend_dir
            return

        # 更新 trailing
        if self.pos.direction == 1:
            if self.pos.trail_high is None or h > self.pos.trail_high:
                self.pos.trail_high = h
        elif self.pos.direction == -1:
            if self.pos.trail_low is None or l < self.pos.trail_low:
                self.pos.trail_low = l

        # 5-min 级出场（A time_stop / T momentum_reverse）
        if self.pos.direction != 0:
            self._check_5m_exits(dt, h, l, c, j, prev_j, prev_prev_j, atr_val, in_trend)

        # 生成下根 bar 开始时的信号
        self._gen_signals(j, prev_j, prev_prev_j, c, in_trend, trend_dir,
                          bull_div, bear_div, pure_c_long, pure_c_short,
                          in_session, in_bc, in_pure_c, in_14, bar_ts=dt)

        # 5-min 级的入场可以在这里立即执行（close price 近似 next-tick 价格）
        if self._prev_signals and self.pos.direction == 0:
            slip = c * self.p['slippage_bps'] / 10000
            self._try_entry(c + slip * (1 if 'LONG' in list(self._prev_signals)[0] else -1),
                           atr_val, dt)

        self._prev_in_trend = in_trend
        self._prev_prev_trend_dir = self._prev_trend_dir
        self._prev_trend_dir = trend_dir

    # ── Signal generation (镜像 live) ──
    def _exhaust_allows(self, direction, score):
        if direction == 1:
            return score > -2
        return score < 2

    def _gen_signals(self, j, prev_j, prev_prev_j, close, in_trend, trend_dir,
                     bull_div, bear_div, pure_c_long, pure_c_short,
                     in_session, in_bc, in_pure_c, in_14, bar_ts=None):
        p = self.p
        ms = self.momentum.score
        signals = {}
        if not self.kdj.ready:
            self._prev_signals = signals
            return

        if p.get('a_enabled', True):
            a_ok = not (p['a_skip_14'] and in_14) and in_session and not in_trend
            if a_ok:
                if prev_j < 0 and prev_prev_j >= 0 and self._exhaust_allows(1, ms):
                    signals['A_LONG'] = close
                if prev_j > 100 and prev_prev_j <= 100 and self._exhaust_allows(-1, ms):
                    signals['A_SHORT'] = close

        if in_bc and not in_trend:
            if bull_div and self._exhaust_allows(1, ms):
                signals['B_LONG'] = close
            if bear_div and self._exhaust_allows(-1, ms):
                signals['B_SHORT'] = close

        if p.get('pure_c_enabled', True):
            if pure_c_long and self._exhaust_allows(1, ms):
                signals['PURE_C_LONG'] = close
            if pure_c_short and self._exhaust_allows(-1, ms):
                signals['PURE_C_SHORT'] = close

        if p.get('t_enabled', True) and in_trend and in_session:
            ptd, pptd = self._prev_trend_dir, self._prev_prev_trend_dir
            md = self.momentum.trend_direction
            if p.get('t_confirm_2bar', True):
                if trend_dir == 1 and ptd == 1 and pptd != 1 and md >= 0:
                    signals['T_LONG'] = close
                if trend_dir == -1 and ptd == -1 and pptd != -1 and md <= 0:
                    signals['T_SHORT'] = close
            else:
                if trend_dir == 1 and ptd != 1 and md >= 0:
                    signals['T_LONG'] = close
                if trend_dir == -1 and ptd != -1 and md <= 0:
                    signals['T_SHORT'] = close

        # GEX regime gate
        if signals and p.get('gex_gate_enabled', False) and self.gex_reader is not None:
            filtered = {}
            for k, v in signals.items():
                ok, reason = self.gex_reader.allows(k, ts=bar_ts)
                self.gate_log.append({
                    'ts': bar_ts, 'gate': 'gex', 'signal': k,
                    'allowed': ok, 'reason': reason,
                })
                if ok:
                    filtered[k] = v
            signals = filtered

        # VIX regime gate (可与 GEX gate 叠加)
        if signals and p.get('vix_gate_enabled', False) and self.vix_reader is not None:
            filtered = {}
            td = bar_ts.date() if bar_ts is not None else None
            for k, v in signals.items():
                ok, reason = self.vix_reader.allows(k, trade_date=td)
                self.gate_log.append({
                    'ts': bar_ts, 'gate': 'vix', 'signal': k,
                    'allowed': ok, 'reason': reason,
                })
                if ok:
                    filtered[k] = v
            signals = filtered

        self._prev_signals = signals

    # ── Entry (simplified: close-price entry) ──
    def _try_entry(self, price, atr_val, dt):
        p = self.p
        for key, phase, d in [
            ('B_LONG', 'B', 1), ('B_SHORT', 'B', -1),
            ('A_LONG', 'A', 1), ('A_SHORT', 'A', -1),
            ('PURE_C_LONG', '纯C', 1), ('PURE_C_SHORT', '纯C', -1),
            ('T_LONG', 'T', 1), ('T_SHORT', 'T', -1),
        ]:
            if key not in self._prev_signals:
                continue
            qty = p['qty']
            self.pos = BacktestPosition(
                direction=d, qqq_entry=price, phase=phase,
                entry_time=dt.strftime('%H:%M:%S'),
                qty=qty, initial_qty=qty, entry_bar=self._bar5_count,
            )
            if phase == 'A':
                self.pos.a_sl_atr = price - atr_val * p['a_atr_mult'] * d
                self.pos.sl_price = self.pos.a_sl_atr
            elif phase == 'B':
                self.pos.sl_price = price - atr_val * p['b_atr_mult_sl'] * d
                self.pos.b_t1_target = price + atr_val * p['b_atr_mult_t1'] * d
                self.pos.b_t2_target = price + atr_val * p['b_atr_mult_t2'] * d
            elif phase in ('C', '纯C'):
                self.pos.sl_price = price - p['c_stop_loss'] * d
            elif phase == 'T':
                self.pos.sl_price = price - atr_val * p['trend_atr_sl'] * d

            self.day_trades.append(TradeRecord(
                date=dt.strftime('%Y-%m-%d'),
                action='ENTRY', phase=phase, signal=key, dir=d,
                qqq_entry=price, qty=qty, time=dt.strftime('%H:%M:%S'),
                j=self.kdj.j, adx=self.adx._adx if hasattr(self.adx, '_adx') else 0.0,
                reason=key,
            ))
            self._prev_signals = {}
            return

    # ── Tick-level exits (每 30s bar close 当作 tick) ──
    def _on_tick(self, price, dt):
        d = self.pos.direction
        if d == 0:
            return
        p = self.p

        if self.pos.phase == 'B':
            # SL → (可选) 反手 C
            if (d == 1 and price <= self.pos.sl_price) or (d == -1 and price >= self.pos.sl_price):
                sl_p = self.pos.sl_price
                reason = 'B_SL→C' if p.get('b_reverse_c', True) else 'B_SL'
                self._close_position(sl_p, reason, dt=dt)
                if p.get('b_reverse_c', True):
                    # 反手 C
                    new_dir = -d
                    qty = p['qty']
                    self.pos = BacktestPosition(
                        direction=new_dir, qqq_entry=sl_p, phase='C',
                        entry_time=dt.strftime('%H:%M:%S'),
                        qty=qty, initial_qty=qty, entry_bar=self._bar5_count,
                    )
                    self.pos.sl_price = sl_p - p['c_stop_loss'] * new_dir
                    self.day_trades.append(TradeRecord(
                        date=dt.strftime('%Y-%m-%d'),
                        action='ENTRY', phase='C', signal='B_REVERSE', dir=new_dir,
                        qqq_entry=sl_p, qty=qty, time=dt.strftime('%H:%M:%S'),
                        j=self.kdj.j, reason='B反手C',
                    ))
                return
            # T1 → 60% 出
            if self.pos.tranche == 0 and self.pos.b_t1_target and \
               ((d == 1 and price >= self.pos.b_t1_target) or (d == -1 and price <= self.pos.b_t1_target)):
                self._partial_exit(0.6, price, 'B_T1', dt)
                self.pos.tranche = 1
                self.pos.sl_price = self.pos.qqq_entry   # breakeven
                self.pos.trail_high = self.pos.trail_low = None
                return
            # T2 → 全出
            if self.pos.tranche == 1 and self.pos.b_t2_target and \
               ((d == 1 and price >= self.pos.b_t2_target) or (d == -1 and price <= self.pos.b_t2_target)):
                self._close_position(price, 'B_T2_atr', dt=dt)
                return

        elif self.pos.phase in ('C', '纯C'):
            ep = self.pos.qqq_entry
            if (d == 1 and price <= self.pos.sl_price) or (d == -1 and price >= self.pos.sl_price):
                self._close_position(price, 'C_SL', dt=dt)
                return
            t1 = ep + p['c_t1_pts'] * d
            t2 = ep + p['c_t2_pts'] * d
            if self.pos.tranche == 0:
                if (d == 1 and price >= t1) or (d == -1 and price <= t1):
                    self._partial_exit(0.4, price, 'C_T1', dt)
                    self.pos.tranche = 1
                    self.pos.sl_price = ep
                    self.pos.trail_high = self.pos.trail_low = None
            elif self.pos.tranche == 1:
                if (d == 1 and price >= t2) or (d == -1 and price <= t2):
                    self._partial_exit(0.35, price, 'C_T2', dt)
                    self.pos.tranche = 2
                    self.pos.sl_price = ep + p['c_t1_pts'] * d
                    self.pos.trail_high = self.pos.trail_low = None
            elif self.pos.tranche == 2:
                if d == 1:
                    if self.pos.trail_high is None or price > self.pos.trail_high:
                        self.pos.trail_high = price
                    if self.pos.trail_high and price <= self.pos.trail_high - p['c_trail_gap']:
                        self._close_position(price, 'C_T3_trail', dt=dt)
                else:
                    if self.pos.trail_low is None or price < self.pos.trail_low:
                        self.pos.trail_low = price
                    if self.pos.trail_low and price >= self.pos.trail_low + p['c_trail_gap']:
                        self._close_position(price, 'C_T3_trail', dt=dt)

        elif self.pos.phase == 'A' and self.pos.tranche == 0 and self.pos.a_sl_atr:
            if (d == 1 and price <= self.pos.a_sl_atr) or (d == -1 and price >= self.pos.a_sl_atr):
                self._close_position(price, 'A_ATR_stop', dt=dt)

        elif self.pos.phase == 'T':
            if (d == 1 and price <= self.pos.sl_price) or (d == -1 and price >= self.pos.sl_price):
                self._close_position(price, 'T_SL', dt=dt)

    # ── 5-min 级出场 ──
    def _check_5m_exits(self, dt, h, l, c, j, prev_j, prev_prev_j, atr_val, in_trend):
        p, d = self.p, self.pos.direction
        if self.pos.phase == 'A':
            # Time stop
            if self.pos.tranche == 0 and self.pos.entry_bar > 0:
                if (self._bar5_count - self.pos.entry_bar) >= p['a_time_stop']:
                    self._close_position(c, 'A_time_stop', dt=dt)
                    return
            # J 止损 & T1
            if d == 1:
                if j < p['a_long_j_stop']:
                    self._close_position(c, 'A_J_stop', dt=dt); return
                if self.pos.tranche == 0 and prev_j >= p['a_long_t1_j'] and prev_prev_j < p['a_long_t1_j']:
                    self._close_position(c, 'A_T1', dt=dt); return
            else:
                if j > p['a_short_j_stop']:
                    self._close_position(c, 'A_J_stop', dt=dt); return
                if self.pos.tranche == 0 and prev_j <= p['a_short_t1_j'] and prev_prev_j > p['a_short_t1_j']:
                    self._close_position(c, 'A_T1', dt=dt); return
        elif self.pos.phase == 'B':
            # T2 J-neutral (only after T1)
            if self.pos.tranche == 1:
                if (d == 1 and j >= p['b_j_neutral']) or (d == -1 and j <= p['b_j_neutral']):
                    self._close_position(c, 'B_T2_J', dt=dt); return
        elif self.pos.phase == 'T':
            mom_rev = (d == 1 and self.momentum.trend_direction == -1) or \
                      (d == -1 and self.momentum.trend_direction == 1)
            adx_ret = not in_trend and self._prev_in_trend
            trail = (d == 1 and self.pos.trail_high and l <= self.pos.trail_high - p['trend_trail_gap']) or \
                    (d == -1 and self.pos.trail_low and h >= self.pos.trail_low + p['trend_trail_gap'])
            if mom_rev or trail or adx_ret:
                self._close_position(c, 'T_exit', dt=dt)

    # ── 平仓 ──
    def _partial_exit(self, pct, price, reason, dt):
        if self.pos.direction == 0:
            return
        d = self.pos.direction
        qty_out = max(1, int(self.pos.initial_qty * pct))
        qty_out = min(qty_out, self.pos.qty)  # 不超过剩余
        pnl_qqq_per_share = (price - self.pos.qqq_entry) * d
        pnl_bps = pnl_qqq_per_share / self.pos.qqq_entry * 10000
        # 减去 slippage 2×（入场+出场）
        pnl_bps -= 2 * self.p['slippage_bps']
        # TQQQ PnL 美元: qty × entry_tqqq × (pnl_bps/10000) × leverage
        # 简化: qty × qqq_entry × 3× × ratio
        pnl_usd = qty_out * self.pos.qqq_entry * (pnl_bps / 10000) * self.p['tqqq_leverage']

        self.pos.qty -= qty_out
        self.daily_pnl_bps += pnl_bps * (qty_out / self.pos.initial_qty)

        self.day_trades.append(TradeRecord(
            date=dt.strftime('%Y-%m-%d'),
            action='PARTIAL', phase=self.pos.phase, signal=reason,
            dir=d, qqq_entry=self.pos.qqq_entry, qqq_exit=price,
            qty=qty_out, pnl_bps=pnl_bps, pnl_usd=pnl_usd,
            time=dt.strftime('%H:%M:%S'), reason=reason,
            hold_bars=self._bar5_count - self.pos.entry_bar,
        ))

    def _close_position(self, price, reason, dt=None):
        if self.pos.direction == 0:
            return
        d = self.pos.direction
        qty_out = self.pos.qty
        pnl_qqq_per_share = (price - self.pos.qqq_entry) * d
        pnl_bps = pnl_qqq_per_share / self.pos.qqq_entry * 10000
        pnl_bps -= 2 * self.p['slippage_bps']
        pnl_usd = qty_out * self.pos.qqq_entry * (pnl_bps / 10000) * self.p['tqqq_leverage']

        self.daily_pnl_bps += pnl_bps * (qty_out / self.pos.initial_qty)

        time_str = dt.strftime('%H:%M:%S') if dt else ''
        date_str = dt.strftime('%Y-%m-%d') if dt else ''
        self.day_trades.append(TradeRecord(
            date=date_str, action='EXIT', phase=self.pos.phase, signal=reason,
            dir=d, qqq_entry=self.pos.qqq_entry, qqq_exit=price,
            qty=qty_out, pnl_bps=pnl_bps, pnl_usd=pnl_usd,
            time=time_str, reason=reason,
            hold_bars=self._bar5_count - self.pos.entry_bar,
        ))
        self.pos = BacktestPosition()


# ═══════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='YYYY-MM-DD (inclusive)')
    ap.add_argument('--out', default=None, help='output trades CSV path')
    ap.add_argument('--daily', default=None, help='output daily stats CSV path')
    ap.add_argument('--disable-a', action='store_true')
    ap.add_argument('--disable-t', action='store_true')
    ap.add_argument('--disable-pure-c', action='store_true')
    ap.add_argument('--no-2bar-confirm', action='store_true',
                    help='Use old single-bar DEMA cross for T signal')
    ap.add_argument('--no-reverse-c', action='store_true',
                    help='Disable B 止损→反手 C 机制，B 止损就止损')
    ap.add_argument('--gex-gate', action='store_true',
                    help='启用 GEX regime gate (+γ→A/B, -γ→T)')
    ap.add_argument('--gex-symbol', default='QQQ',
                    help='GEX 数据来源的 symbol (默认 QQQ)')
    ap.add_argument('--vix-gate', action='store_true',
                    help='启用 VIX regime gate (低→A/B, 高→T, 中→全开)')
    ap.add_argument('--vix-low', type=float, default=15.0,
                    help='VIX 低波阈值 (默认 15)')
    ap.add_argument('--vix-high', type=float, default=25.0,
                    help='VIX 高波阈值 (默认 25)')
    args = ap.parse_args()

    p = dict(PARAMS)
    if args.disable_a:
        p['a_enabled'] = False
    if args.disable_t:
        p['t_enabled'] = False
    if args.disable_pure_c:
        p['pure_c_enabled'] = False
    if args.no_2bar_confirm:
        p['t_confirm_2bar'] = False
    if args.no_reverse_c:
        p['b_reverse_c'] = False
    if args.gex_gate:
        p['gex_gate_enabled'] = True
        p['gex_gate_symbol'] = args.gex_symbol
    if args.vix_gate:
        p['vix_gate_enabled'] = True
        p['vix_low_thresh'] = args.vix_low
        p['vix_high_thresh'] = args.vix_high

    # 初始化 GEX reader（如启用 gate）
    gex_reader = None
    if p.get('gex_gate_enabled'):
        gex_reader = GEXRegimeReader(symbol=p['gex_gate_symbol'])
        n_gex = gex_reader.load_historical(args.start, args.end)
        print(f"GEX gate: loaded {n_gex} historical snapshots for {p['gex_gate_symbol']}")
        if n_gex == 0:
            print("⚠ 没有 GEX 历史数据，gate 将 fail-open (等同未启用)")

    # 初始化 VIX reader
    vix_reader = None
    if p.get('vix_gate_enabled'):
        vix_reader = VIXRegimeReader(
            low_thresh=p['vix_low_thresh'],
            high_thresh=p['vix_high_thresh'],
        )
        n_vix = vix_reader.load_historical(args.start, args.end)
        print(f"VIX gate: loaded {n_vix} daily bars, thresholds low={args.vix_low} high={args.vix_high}")
        if n_vix == 0:
            print("⚠ 没有 VIX 日线数据，gate 将 fail-open")
        else:
            dist = vix_reader._daily_df['regime'].value_counts().to_dict()
            print(f"  regime 分布: {dist}")

    bt = KDJBacktester(p, gex_reader=gex_reader, vix_reader=vix_reader)

    print(f"加载 QQQ 30s bars: {args.start} ~ {args.end}")
    df = load_30s_bars(args.start, args.end)
    if len(df) == 0:
        print("✗ 没找到数据")
        return
    print(f"  取到 {len(df)} 根 30s bar")

    # Group by ET date
    df['et_date'] = df['dt_et'].dt.strftime('%Y-%m-%d')
    dates = sorted(df['et_date'].unique())
    print(f"  覆盖 {len(dates)} 个交易日")

    print(f"\n{'date':12} {'entries':>7} {'exits':>6} {'win_rate':>9} {'pnl_bps':>9} {'pnl_usd':>10}")
    print('-' * 60)

    for d in dates:
        day_bars = df[df['et_date'] == d].sort_values('datetime').reset_index(drop=True)
        if len(day_bars) < 50:
            continue
        stats = bt.run_day(d, day_bars)
        print(f"{stats['date']:12} {stats['n_entries']:>7} {stats['n_exits']:>6} "
              f"{stats['win_rate']:>8.0f}% {stats['day_pnl_bps']:>+9.1f} "
              f"{stats['day_pnl_usd']:>+10.2f}")

    # Aggregate summary
    if not bt.daily_stats:
        print("\n无结果")
        return

    total_entries = sum(s['n_entries'] for s in bt.daily_stats)
    total_exits = sum(s['n_exits'] for s in bt.daily_stats)
    total_wins = sum(s['wins'] for s in bt.daily_stats)
    total_pnl_bps = sum(s['day_pnl_bps'] for s in bt.daily_stats)
    total_pnl_usd = sum(s['day_pnl_usd'] for s in bt.daily_stats)
    avg_daily_bps = total_pnl_bps / len(bt.daily_stats)
    days_pos = sum(1 for s in bt.daily_stats if s['day_pnl_bps'] > 0)
    days_neg = sum(1 for s in bt.daily_stats if s['day_pnl_bps'] < 0)

    print(f"\n{'='*60}")
    print(f"总交易日数:    {len(bt.daily_stats)}")
    print(f"总入场数:      {total_entries}")
    print(f"总平仓数:      {total_exits}")
    print(f"总胜率:        {total_wins}/{total_exits} = {(total_wins/total_exits*100 if total_exits else 0):.1f}%")
    print(f"总 PnL:        {total_pnl_bps:+.1f} bps  ≈  ${total_pnl_usd:+.2f}")
    print(f"日均 PnL:      {avg_daily_bps:+.2f} bps/天")
    print(f"盈利日:亏损日: {days_pos} : {days_neg}")

    # 按信号类型汇总
    sig_stats = {}
    for t in bt.all_trades:
        if t.action != 'EXIT':
            continue
        sig = t.phase
        if sig not in sig_stats:
            sig_stats[sig] = {'n': 0, 'wins': 0, 'pnl_bps': 0.0, 'pnl_usd': 0.0, 'holds': []}
        sig_stats[sig]['n'] += 1
        sig_stats[sig]['pnl_bps'] += t.pnl_bps
        sig_stats[sig]['pnl_usd'] += t.pnl_usd
        sig_stats[sig]['holds'].append(t.hold_bars)
        if t.pnl_bps > 0:
            sig_stats[sig]['wins'] += 1

    print(f"\n按 phase 汇总:")
    print(f"  {'phase':6} {'n':>4} {'win%':>6} {'avg_bps':>8} {'sum_bps':>8} {'sum_usd':>10} {'avg_hold':>9}")
    for phase, s in sorted(sig_stats.items()):
        avg_bps = s['pnl_bps'] / s['n']
        avg_hold = sum(s['holds']) / len(s['holds']) if s['holds'] else 0
        print(f"  {phase:6} {s['n']:>4} {s['wins']/s['n']*100:>5.0f}% "
              f"{avg_bps:>+8.2f} {s['pnl_bps']:>+8.1f} {s['pnl_usd']:>+10.2f} "
              f"{avg_hold:>8.1f}")

    # Gate 汇总 (按 gate 分组)
    if bt.gate_log:
        from collections import Counter, defaultdict
        by_gate = defaultdict(list)
        for g in bt.gate_log:
            by_gate[g.get('gate', 'unknown')].append(g)
        for gate_name, logs in by_gate.items():
            n_total = len(logs)
            n_allowed = sum(1 for g in logs if g['allowed'])
            n_blocked = n_total - n_allowed
            print(f"\n{gate_name.upper()} gate: {n_total} 次判定，通过 {n_allowed} ({n_allowed/n_total*100:.0f}%)，阻挡 {n_blocked}")
            blocked = Counter((g['signal'], g['reason']) for g in logs if not g['allowed'])
            for (sig, reason), n in blocked.most_common(8):
                print(f"  阻挡 {sig:12} × {n:4}  ({reason})")

    if args.out:
        df_out = pd.DataFrame([t.__dict__ for t in bt.all_trades])
        df_out.to_csv(args.out, index=False)
        print(f"\n✓ 交易明细: {args.out}")

    if args.daily:
        pd.DataFrame(bt.daily_stats).to_csv(args.daily, index=False)
        print(f"✓ 日统计: {args.daily}")


if __name__ == '__main__':
    main()
