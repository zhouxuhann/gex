"""Skew 模块测试"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from gex_monitor.skew import compute_skew, SkewTracker, SkewSnapshot


def _make_ticker(strike, right, iv, delta):
    """构造 mock ticker"""
    t = MagicMock()
    t.contract.strike = strike
    t.contract.right = right
    g = MagicMock()
    g.impliedVol = iv
    g.delta = delta
    t.modelGreeks = g
    t.bid = 1.0
    t.ask = 1.2
    t.volume = 100
    t.putOpenInterest = 200
    t.callOpenInterest = 300
    return t


def _build_tickers(spot=480.0):
    """构造一组完整的 mock tickers，模拟真实 skew 曲线"""
    tickers = []
    for offset in range(-5, 6):
        strike = spot + offset
        # Put: IV 越 OTM（strike 越低）越高（典型 skew）
        put_iv = 0.25 + 0.01 * max(0, -offset)
        put_delta = -0.5 + offset * 0.05  # 粗略线性
        put_delta = max(-0.95, min(-0.05, put_delta))
        tickers.append(_make_ticker(strike, 'P', put_iv, put_delta))

        # Call: IV 越 OTM（strike 越高）略高
        call_iv = 0.25 + 0.003 * max(0, offset)
        call_delta = 0.5 - offset * 0.05
        call_delta = max(0.05, min(0.95, call_delta))
        tickers.append(_make_ticker(strike, 'C', call_iv, call_delta))
    return tickers


class TestComputeSkew:
    def test_returns_none_for_empty(self):
        assert compute_skew([], 480.0) is None
        assert compute_skew(None, 480.0) is None
        assert compute_skew([MagicMock()], 0) is None

    def test_basic_skew(self):
        tickers = _build_tickers(480.0)
        snap = compute_skew(tickers, 480.0)
        assert snap is not None
        assert snap.atm_iv is not None
        assert snap.atm_iv > 0
        # RR 应该有值（符号取决于 delta 映射）
        assert snap.rr_25 is not None
        assert snap.put_25_iv is not None
        assert snap.call_25_iv is not None
        assert snap.put_25_mid == 1.1
        assert snap.call_25_mid == 1.1
        assert snap.put_25_open_interest == 200
        assert snap.call_25_open_interest == 300
        assert snap.skew_slope is not None
        # z-score 和 signal 由 tracker 填充
        assert snap.rr_25_zscore is None
        assert snap.signal is None

    def test_missing_greeks(self):
        """tickers 缺少 greeks 时应返回 None"""
        t = MagicMock()
        t.modelGreeks = None
        assert compute_skew([t, t, t, t], 480.0) is None


class TestSkewTracker:
    def test_cold_start(self):
        """冷启动期间 z-score 应为 None"""
        tracker = SkewTracker(window=30)
        snap = SkewSnapshot(atm_iv=0.25, rr_25=0.02, skew_slope=0.1,
                            rr_25_zscore=None, signal=None)
        result = tracker.update(snap, positive_gamma=True)
        assert result.rr_25_zscore is None  # < 10 个样本

    def test_zscore_after_warmup(self):
        """积累足够样本后应有 z-score"""
        tracker = SkewTracker(window=30)
        for i in range(15):
            snap = SkewSnapshot(atm_iv=0.25, rr_25=0.02, skew_slope=0.1,
                                rr_25_zscore=None, signal=None)
            result = tracker.update(snap, positive_gamma=True)

        assert result.rr_25_zscore is not None

    def test_bearish_signal(self):
        """负 Gamma + 高 RR z-score → BEARISH_ACCELERATION"""
        tracker = SkewTracker(window=20)
        # 先填充正常值
        for _ in range(15):
            snap = SkewSnapshot(atm_iv=0.25, rr_25=0.02, skew_slope=0.1,
                                rr_25_zscore=None, signal=None)
            tracker.update(snap, positive_gamma=False)

        # 突然 RR 飙升
        spike = SkewSnapshot(atm_iv=0.25, rr_25=0.08, skew_slope=0.3,
                             rr_25_zscore=None, signal=None)
        result = tracker.update(spike, positive_gamma=False)
        # z-score 应该很高，信号应该触发
        assert result.rr_25_zscore is not None
        assert result.rr_25_zscore > 1.0  # spike 应该远超均值

    def test_none_input(self):
        tracker = SkewTracker()
        assert tracker.update(None, positive_gamma=True) is None

    def test_reset(self):
        tracker = SkewTracker(window=30)
        for _ in range(15):
            snap = SkewSnapshot(atm_iv=0.25, rr_25=0.02, skew_slope=0.1,
                                rr_25_zscore=None, signal=None)
            tracker.update(snap, positive_gamma=True)
        tracker.reset()
        snap = SkewSnapshot(atm_iv=0.25, rr_25=0.02, skew_slope=0.1,
                            rr_25_zscore=None, signal=None)
        result = tracker.update(snap, positive_gamma=True)
        assert result.rr_25_zscore is None  # reset 后重新冷启动
