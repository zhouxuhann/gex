"""Skew surface + hedge signal 测试"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from gex_monitor.skew_surface import (
    pick_tenor_expiries, _compute_tenor_skew, SkewSurface, SkewTenorSnapshot,
)
from gex_monitor.hedge_signal import (
    generate_hedge_signal, _compute_cheapness, _classify_term_structure,
    _decide_action, _get_current_rr25, HedgeSignal,
)


# ============================================================
# skew_surface tests
# ============================================================

class TestPickTenorExpiries:
    def test_basic_selection(self):
        """选择 0DTE, ~7DTE, ~14DTE, ~30DTE, ~45DTE"""
        chain = MagicMock()
        chain.expirations = [
            '20260413', '20260414', '20260415',
            '20260418', '20260420', '20260425', '20260427',
            '20260511', '20260515', '20260525', '20260530',
        ]
        result = pick_tenor_expiries(chain, '20260413')
        assert len(result) >= 4  # 0DTE + ~7 + ~14 + ~30 + ~45 (some may dedup)
        # 第一个应该是 0DTE
        assert result[0] == ('20260413', 0)

    def test_no_0dte(self):
        """没有 0DTE 时回退到最近的"""
        chain = MagicMock()
        chain.expirations = ['20260414', '20260418', '20260425']
        result = pick_tenor_expiries(chain, '20260413')
        assert len(result) >= 2
        assert result[0][0] == '20260414'  # 最近的

    def test_empty_chain(self):
        chain = MagicMock()
        chain.expirations = []
        assert pick_tenor_expiries(chain, '20260413') == []

    def test_deduplication(self):
        """如果 7DTE 和 14DTE 选到同一个 expiry，应去重"""
        chain = MagicMock()
        chain.expirations = ['20260413', '20260420']  # 只有两个
        result = pick_tenor_expiries(chain, '20260413')
        expiries = [r[0] for r in result]
        assert len(expiries) == len(set(expiries))  # 无重复


class TestComputeTenorSkew:
    def _make_ticker(self, strike, right, iv, delta):
        t = MagicMock()
        t.contract.strike = strike
        t.contract.right = right
        g = MagicMock()
        g.impliedVol = iv
        g.delta = delta
        t.modelGreeks = g
        return t

    def test_basic(self):
        tickers = []
        for offset in range(-3, 4):
            strike = 480 + offset
            put_iv = 0.25 + 0.01 * max(0, -offset)
            call_iv = 0.25 + 0.003 * max(0, offset)
            tickers.append(self._make_ticker(strike, 'P', put_iv, -0.5 + offset * 0.07))
            tickers.append(self._make_ticker(strike, 'C', call_iv, 0.5 - offset * 0.07))

        snap = _compute_tenor_skew(tickers, 480.0, '20260413', 7)
        assert snap.expiry == '20260413'
        assert snap.dte == 7
        assert snap.atm_iv is not None
        assert snap.n_contracts == 14

    def test_empty_tickers(self):
        snap = _compute_tenor_skew([], 480.0, '20260413', 7)
        assert snap.atm_iv is None
        assert snap.n_contracts == 0


class TestSkewSurface:
    def test_to_records(self):
        surface = SkewSurface(
            ts=datetime(2026, 4, 13, 15, 30),
            symbol='QQQ',
            spot=480.0,
            tenors=[
                SkewTenorSnapshot('20260413', 0, 0.30, 0.05, 0.08, 0.15, 20),
                SkewTenorSnapshot('20260420', 7, 0.25, 0.03, 0.06, 0.12, 18),
            ],
            term_spread_rr25=0.02,
            term_spread_iv=0.05,
        )
        records = surface.to_records()
        assert len(records) == 2
        assert records[0]['dte'] == 0
        assert records[1]['dte'] == 7
        assert records[0]['term_spread_rr25'] == 0.02
        assert 'put_25_strike' in records[0]
        assert 'quality' in records[0]

    def test_get_tenor(self):
        surface = SkewSurface(
            ts=datetime(2026, 4, 13),
            symbol='QQQ',
            spot=480.0,
            tenors=[
                SkewTenorSnapshot('20260413', 0, 0.30, None, None, None, 0),
                SkewTenorSnapshot('20260420', 7, 0.25, None, None, None, 0),
                SkewTenorSnapshot('20260427', 14, 0.22, None, None, None, 0),
                SkewTenorSnapshot('20260513', 30, 0.20, None, None, None, 0),
                SkewTenorSnapshot('20260528', 45, 0.19, None, None, None, 0),
            ],
        )
        assert surface.get_tenor(0).dte == 0
        assert surface.get_tenor(7).dte == 7
        assert surface.get_tenor(14).dte == 14
        assert surface.get_tenor(30).dte == 30
        assert surface.get_tenor(45).dte == 45
        assert surface.get_tenor(10).dte == 7   # closest
        assert surface.get_tenor(35).dte == 30  # closest

    def test_get_tenor_empty(self):
        surface = SkewSurface(ts=datetime.now(), symbol='QQQ', spot=480.0)
        assert surface.get_tenor(7) is None


# ============================================================
# hedge_signal tests
# ============================================================

def _make_surface(rr_25_30d=0.03, atm_iv_0d=0.30, atm_iv_45d=0.20,
                  term_spread_rr25=0.02, term_spread_iv=0.08):
    """构造测试用 surface（含 30/45 DTE tenor）"""
    return SkewSurface(
        ts=datetime(2026, 4, 13, 15, 30),
        symbol='QQQ',
        spot=480.0,
        tenors=[
            SkewTenorSnapshot('20260413', 0, atm_iv_0d, 0.05, 0.08, 0.15, 20),
            SkewTenorSnapshot('20260420', 7, 0.25, rr_25_30d * 0.7, 0.06, 0.12, 18),
            SkewTenorSnapshot('20260427', 14, 0.23, rr_25_30d * 0.8, 0.06, 0.11, 16),
            SkewTenorSnapshot('20260513', 30, 0.21, rr_25_30d, 0.05, 0.10, 14),
            SkewTenorSnapshot('20260528', 45, atm_iv_45d, rr_25_30d * 1.05, 0.05, 0.09, 12),
        ],
        term_spread_rr25=term_spread_rr25,
        term_spread_iv=term_spread_iv,
    )


def _make_history(rr_25_mean=0.03, n_days=20, n_per_day=5):
    """构造历史数据（含 30/45 DTE tenor）"""
    rows = []
    base_date = datetime(2026, 3, 24)
    for d in range(n_days):
        dt = base_date + timedelta(days=d)
        for dte in [0, 7, 14, 30, 45]:
            for _ in range(n_per_day):
                rr = rr_25_mean + np.random.normal(0, 0.005)
                rows.append({
                    'ts': dt,
                    'symbol': 'QQQ',
                    'spot': 480.0,
                    'expiry': (dt + timedelta(days=dte)).strftime('%Y%m%d'),
                    'dte': dte,
                    'atm_iv': 0.25 - dte * 0.001,  # 远端 IV 略低
                    'rr_25': rr,
                    'rr_10': rr * 1.5,
                    'skew_slope': 0.1,
                    'n_contracts': 20,
                    'term_spread_rr25': 0.02,
                    'term_spread_iv': 0.05,
                })
    return pd.DataFrame(rows)


class TestComputeCheapness:
    def test_no_history(self):
        assert _compute_cheapness(0.03, None) == 50.0
        assert _compute_cheapness(0.03, pd.DataFrame()) == 50.0

    def test_cheap(self):
        """当前 RR 低于历史 → cheapness 低"""
        hist = _make_history(rr_25_mean=0.05)
        cheapness = _compute_cheapness(0.01, hist)
        assert cheapness < 30  # 应该很便宜

    def test_expensive(self):
        """当前 RR 高于历史 → cheapness 高"""
        hist = _make_history(rr_25_mean=0.02)
        cheapness = _compute_cheapness(0.06, hist)
        assert cheapness > 70  # 应该很贵

    def test_none_current(self):
        assert _compute_cheapness(None, _make_history()) == 50.0


class TestClassifyTermStructure:
    def test_backwardation(self):
        assert _classify_term_structure(0.02, 0.05) == 'backwardation'

    def test_contango(self):
        assert _classify_term_structure(-0.02, -0.05) == 'contango'

    def test_flat(self):
        assert _classify_term_structure(0.005, 0.003) == 'flat'

    def test_none(self):
        assert _classify_term_structure(None, None) == 'unavailable'


class TestDecideAction:
    def test_negative_gamma_cheap(self):
        action, struct = _decide_action(10, 'negative', 'flat', None)
        assert action == 'HEDGE_NOW'
        assert struct == 'outright_put'

    def test_negative_gamma_expensive(self):
        action, struct = _decide_action(80, 'negative', 'flat', None)
        assert action == 'HEDGE_SPREAD'
        assert struct == 'put_spread'

    def test_positive_gamma_skip(self):
        action, _ = _decide_action(50, 'positive', 'flat', None)
        assert action == 'SKIP'

    def test_positive_gamma_backwardation_cheap(self):
        """正 Gamma 但 backwardation + 便宜 → 还是要对冲"""
        action, _ = _decide_action(10, 'positive', 'backwardation', None)
        assert action == 'HEDGE_SPREAD'

    def test_reduce_signal(self):
        """之前 HEDGE_NOW + 现在正 Gamma + 贵 → REDUCE"""
        last = {'action': 'HEDGE_NOW'}
        action, _ = _decide_action(80, 'positive', 'flat', last)
        assert action == 'REDUCE'

    def test_neutral_cheap(self):
        action, struct = _decide_action(10, 'neutral', 'flat', None)
        assert action == 'HEDGE_NOW'
        assert struct == 'put_spread'

    def test_neutral_normal(self):
        action, _ = _decide_action(50, 'neutral', 'flat', None)
        assert action == 'MONITOR'


class TestGenerateHedgeSignal:
    def test_full_flow(self):
        surface = _make_surface(rr_25_30d=0.01)
        history = _make_history(rr_25_mean=0.04)
        signal = generate_hedge_signal(surface, history, 'negative')
        assert isinstance(signal, HedgeSignal)
        assert signal.action in ('HEDGE_NOW', 'HEDGE_SPREAD', 'MONITOR', 'SKIP', 'REDUCE')
        assert 0 <= signal.urgency <= 1
        assert signal.reasoning  # not empty

    def test_no_history(self):
        """无历史数据时应返回中性信号"""
        surface = _make_surface()
        signal = generate_hedge_signal(surface, None, 'neutral')
        assert signal.skew_cheapness == 50.0

    def test_to_dict(self):
        surface = _make_surface()
        signal = generate_hedge_signal(surface, None, 'neutral')
        d = signal.to_dict()
        assert 'action' in d
        assert 'urgency' in d
        assert 'reasoning' in d
        assert 'data_quality' in d

    def test_short_tenor_never_substitutes_for_hedge_tenor(self):
        surface = _make_surface()
        for tenor in surface.tenors:
            if tenor.dte >= 20:
                tenor.rr_25 = None
                tenor.atm_iv = None
                tenor.quality = 'bad'
        assert _get_current_rr25(surface) is None
        signal = generate_hedge_signal(surface, _make_history(), 'negative')
        assert signal.action == 'MONITOR'
        assert signal.recommended_structure == 'none'
        assert signal.data_quality == 'insufficient_current_tenor'

    def test_same_tenor_history_gate_blocks_execution(self):
        surface = _make_surface()
        history = _make_history(n_days=10)
        signal = generate_hedge_signal(surface, history, 'negative')
        assert signal.action == 'MONITOR'
        assert signal.data_quality == 'insufficient_same_tenor_history'
        assert signal.history_days == 10
