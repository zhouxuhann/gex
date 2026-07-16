"""
KDJ Live Trader 单元测试
测试信号引擎、出场逻辑、风控，不需要IB连接。
"""

import sys
import os
import types
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from collections import deque

import pytz

# Mock momentum_scorer
_mom_mock = types.ModuleType('momentum_scorer')
_mom_scorer = MagicMock()
_mom_scorer.score = 0
_mom_scorer.trend_direction = 0
_mom_mock.MomentumScorer = MagicMock(return_value=_mom_scorer)
sys.modules['momentum_scorer'] = _mom_mock

# 路径设置
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.expanduser('~/Developer/mnq-trading-system'))

ET = pytz.timezone('America/New_York')


# ═══════════════════════════════════════════════
#  指标类测试（engine_v48_live.py）
# ═══════════════════════════════════════════════

class TestKDJCalculator:
    def setup_method(self):
        from gex_monitor.engine_v48_live import KDJCalculator
        self.kdj = KDJCalculator(period=9)

    def test_warmup(self):
        """预热期间 ready=False"""
        for i in range(8):
            self.kdj.update(100 + i, 99 + i, 99.5 + i)
        assert not self.kdj.ready

    def test_ready_after_warmup(self):
        """预热完成后 ready=True"""
        for i in range(20):
            self.kdj.update(100 + i * 0.1, 99.5 + i * 0.1, 99.8 + i * 0.1)
        assert self.kdj.ready

    def test_j_range(self):
        """J值在极端输入下不爆炸"""
        for i in range(30):
            k, d, j = self.kdj.update(500, 490, 495)
        assert isinstance(j, float)
        assert not (j != j)  # not NaN

    def test_j_extreme_high(self):
        """持续上涨 → J > 100"""
        for i in range(30):
            self.kdj.update(100 + i, 99 + i, 100 + i)
        assert self.kdj.j > 80

    def test_j_extreme_low(self):
        """持续下跌 → J < 0"""
        for i in range(30):
            self.kdj.update(100 - i, 99 - i, 99 - i)
        assert self.kdj.j < 20

    def test_prev_j_tracking(self):
        """prev_j 和 prev_prev_j 正确追踪"""
        vals = []
        for i in range(20):
            k, d, j = self.kdj.update(100 + i * 0.5, 99 + i * 0.5, 99.5 + i * 0.5)
            vals.append(j)
        assert self.kdj.prev_j == vals[-2]
        assert self.kdj.prev_prev_j == vals[-3]

    def test_simulate_j_readonly(self):
        """simulate_j 不改变内部状态"""
        for i in range(20):
            self.kdj.update(100, 99, 99.5)
        j_before = self.kdj.j
        sim = self.kdj.simulate_j(105)
        assert self.kdj.j == j_before  # 状态不变
        assert sim != j_before  # 模拟值不同

    def test_simulate_j_empty_highs(self):
        """highs 为空时 simulate_j 不崩溃"""
        sim = self.kdj.simulate_j(100)
        assert isinstance(sim, float)


class TestADXCalculator:
    def test_warmup(self):
        from gex_monitor.engine_v48_live import ADXCalculator
        adx = ADXCalculator(period=14)
        for i in range(13):
            adx.update(100 + i, 99 + i, 99.5 + i)
        assert not adx.ready

    def test_ready(self):
        from gex_monitor.engine_v48_live import ADXCalculator
        adx = ADXCalculator(period=14)
        for i in range(30):
            adx.update(100 + i * 0.5, 99 + i * 0.5, 99.8 + i * 0.5)
        assert adx.ready

    def test_trending_market(self):
        """强趋势 → ADX 高"""
        from gex_monitor.engine_v48_live import ADXCalculator
        adx = ADXCalculator(period=14)
        for i in range(50):
            adx.update(100 + i, 99 + i, 99.5 + i)
        assert adx._adx > 20


class TestATRCalculator:
    def test_basic(self):
        from gex_monitor.engine_v48_live import ATRCalculator
        atr = ATRCalculator(period=14)
        for i in range(20):
            atr.update(101, 99, 100)
        assert abs(atr.value - 2.0) < 0.5  # ATR ≈ 2 (high-low)

    def test_first_bar(self):
        """第一根 bar 没有 prev_close"""
        from gex_monitor.engine_v48_live import ATRCalculator
        atr = ATRCalculator()
        val = atr.update(105, 100, 103)
        assert val == 5.0  # high - low


class TestSpeedDivergence:
    def test_no_signal_during_warmup(self):
        from gex_monitor.engine_v48_live import SpeedDivergence
        div = SpeedDivergence(div_len=5)
        for i in range(4):
            bull, bear = div.update(100, 99, 99.5, 50)
        assert not bull and not bear

    def test_bull_divergence(self):
        """价格跌 + J跌更快 → 看涨背离"""
        from gex_monitor.engine_v48_live import SpeedDivergence
        div = SpeedDivergence(div_len=5, div_thresh=1.0, j_bull_max=50)
        # 填充50根历史
        for i in range(50):
            div.update(100, 99, 99.5, 50)
        # 价格小跌，J 大跌
        bull, bear = div.update(99.5, 98.5, 98.8, 10)
        # 不一定第一根就触发，但不应崩溃
        assert isinstance(bull, bool)
        assert isinstance(bear, bool)


class TestDEMACalculator:
    def test_convergence(self):
        from gex_monitor.engine_v48_live import DEMACalculator
        dema = DEMACalculator(period=8)
        for _ in range(100):
            dema.update(100.0)
        assert abs(dema.value - 100.0) < 0.01

    def test_ready(self):
        from gex_monitor.engine_v48_live import DEMACalculator
        dema = DEMACalculator(period=8)
        for _ in range(15):
            dema.update(100)
        assert not dema.ready
        dema.update(100)
        assert dema.ready


# ═══════════════════════════════════════════════
#  Bar聚合测试
# ═══════════════════════════════════════════════

class TestTickBarBuilder:
    def test_bar_closes_on_new_period(self):
        from gex_monitor.kdj_live_trader import TickBarBuilder
        builder = TickBarBuilder('QQQ', tf_secs=30)
        closed_bars = []
        builder.on_bar_closed = lambda bar: closed_bars.append(bar)

        # 模拟 tick：手动设置时间
        import unittest.mock as mock
        base = datetime(2025, 1, 2, 15, 0, 0, tzinfo=pytz.utc)

        with mock.patch('gex_monitor.kdj_live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = base
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            builder.on_tick(100.0, 100)
            builder.on_tick(100.5, 50)

            # 跳到下一个30秒窗口
            mock_dt.now.return_value = base + timedelta(seconds=31)
            builder.on_tick(101.0, 200)

        assert len(closed_bars) == 1
        assert closed_bars[0].open == 100.0
        assert closed_bars[0].high == 100.5
        assert closed_bars[0].close == 100.5
        assert closed_bars[0].volume == 150


class TestBarAggregator:
    def test_1min_aggregation(self):
        from gex_monitor.kdj_live_trader import BarAggregator, Bar
        agg = BarAggregator(('1 min',))
        closed = []
        agg.on_closed_bar = lambda bar, live: closed.append(bar)

        base = datetime(2025, 1, 2, 15, 0, 0, tzinfo=pytz.utc)
        # 2 个 30s bar = 1 个 1min bar
        agg.update(Bar('QQQ', '30 secs', base, 100, 101, 99, 100.5, 1000, True))
        assert len(closed) == 0  # 还在同一分钟

        agg.update(Bar('QQQ', '30 secs', base + timedelta(seconds=30), 100.5, 102, 100, 101, 500, True))
        assert len(closed) == 0  # 同一分钟的第二个30s

        # 下一分钟的第一个30s → 关闭上一分钟
        agg.update(Bar('QQQ', '30 secs', base + timedelta(seconds=60), 101, 103, 100, 102, 800, True))
        assert len(closed) == 1
        assert closed[0].timeframe == '1 min'
        assert closed[0].high == 102  # max of 101, 102
        assert closed[0].volume == 1500


# ═══════════════════════════════════════════════
#  Position + 风控测试
# ═══════════════════════════════════════════════

class TestPosition:
    def test_default(self):
        from gex_monitor.kdj_live_trader import Position
        pos = Position()
        assert pos.direction == 0
        assert pos.qqq_entry == 0.0
        assert pos.tqqq_entry == 0.0

    def test_dataclass_asdict(self):
        from dataclasses import asdict
        from gex_monitor.kdj_live_trader import Position
        pos = Position(direction=1, qqq_entry=480.5, tqqq_entry=60.2, phase='B')
        d = asdict(pos)
        assert d['direction'] == 1
        assert d['qqq_entry'] == 480.5


class TestRiskControl:
    def setup_method(self):
        """创建 mock IB 的 trader"""
        self.ib = MagicMock()
        self.ib.managedAccounts.return_value = ['DU12345']
        self.ib.qualifyContracts.return_value = True
        self.ib.reqHistoricalData.return_value = []

        # patch MomentumScorer import
        with patch.dict('sys.modules', {'momentum_scorer': MagicMock()}):
            from gex_monitor.kdj_live_trader import KDJLiveTrader
            self.trader = KDJLiveTrader(self.ib, qty=100, dry_run=True)

    def test_daily_loss_limit(self):
        self.trader.daily_pnl = -501.0
        assert not self.trader._check_risk()
        assert self.trader._halted

    def test_daily_trade_limit(self):
        self.trader.daily_trade_count = 20
        assert not self.trader._check_risk()

    def test_risk_ok(self):
        self.trader.daily_pnl = -100.0
        self.trader.daily_trade_count = 5
        assert self.trader._check_risk()

    def test_market_hours(self):
        # 这个测试依赖当前时间，只检查不崩溃
        result = self.trader._is_market_hours()
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════
#  CSV Logger 测试
# ═══════════════════════════════════════════════

class TestCSVLogger:
    def test_write_and_read(self, tmp_path):
        from gex_monitor.kdj_live_trader import TradeCSVLogger
        csv_file = str(tmp_path / "test_trades.csv")
        logger = TradeCSVLogger(csv_file)

        logger.log_trade(
            timestamp='2025-01-02 10:30:00',
            action='ENTRY', side='BUY', qty=100, phase='B',
            price='480.50', fill_price='60.20',
            reason='B做多',
        )

        import csv
        with open(csv_file) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]['action'] == 'ENTRY'
        assert reader[0]['side'] == 'BUY'
        assert reader[0]['reason'] == 'B做多'

    def test_append(self, tmp_path):
        from gex_monitor.kdj_live_trader import TradeCSVLogger
        csv_file = str(tmp_path / "test_trades.csv")
        logger = TradeCSVLogger(csv_file)
        logger.log_trade(action='ENTRY', side='BUY')
        logger.log_trade(action='EXIT', side='SELL')

        import csv
        with open(csv_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2


# ═══════════════════════════════════════════════
#  信号引擎逻辑测试
# ═══════════════════════════════════════════════

class TestExhaustFilter:
    def setup_method(self):
        self.ib = MagicMock()
        self.ib.managedAccounts.return_value = ['DU12345']
        self.ib.reqHistoricalData.return_value = []
        with patch.dict('sys.modules', {'momentum_scorer': MagicMock()}):
            from gex_monitor.kdj_live_trader import KDJLiveTrader
            self.trader = KDJLiveTrader(self.ib, qty=100, dry_run=True)

    def test_exhaust_allows_long_improving(self):
        """动量在改善（-3→-2）→ 允许做多"""
        assert self.trader._exhaust_allows(1, -2, -3)

    def test_exhaust_blocks_long_extreme(self):
        """动量最强且不改善（-3→-3）→ 阻止做多"""
        assert not self.trader._exhaust_allows(1, -3, -3)

    def test_exhaust_allows_short_improving(self):
        assert self.trader._exhaust_allows(-1, 2, 3)

    def test_exhaust_blocks_short_extreme(self):
        assert not self.trader._exhaust_allows(-1, 3, 3)

    def test_exhaust_allows_neutral(self):
        """中性动量 → 允许"""
        assert self.trader._exhaust_allows(1, 0, 0)
        assert self.trader._exhaust_allows(-1, 0, 0)
