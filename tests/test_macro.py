"""macro 模块测试"""
from gex_monitor.macro import MacroSnapshot


class TestMacroSnapshot:
    def test_urgency_stressed(self):
        """VIX > 25 + MOVE > 120 → +0.2"""
        m = MacroSnapshot(vix=30.0, move=130.0, sofr_ois_spread_bps=2.0)
        assert m.urgency_adjustment == 0.2
        assert m.regime_label == 'stressed'

    def test_urgency_calm(self):
        """VIX < 15 + MOVE < 80 → -0.1"""
        m = MacroSnapshot(vix=12.0, move=70.0, sofr_ois_spread_bps=1.0)
        assert m.urgency_adjustment == -0.1
        assert m.regime_label == 'calm'

    def test_urgency_normal(self):
        m = MacroSnapshot(vix=20.0, move=100.0, sofr_ois_spread_bps=2.0)
        assert m.urgency_adjustment == 0.0
        assert m.regime_label == 'normal'

    def test_urgency_one_elevated(self):
        """VIX > 25 but MOVE < 120 → +0.1"""
        m = MacroSnapshot(vix=28.0, move=90.0, sofr_ois_spread_bps=0.0)
        assert m.urgency_adjustment == 0.1

    def test_sofr_stress(self):
        """SOFR-OIS > 5bps adds +0.1"""
        m = MacroSnapshot(vix=20.0, move=100.0, sofr_ois_spread_bps=8.0)
        assert m.urgency_adjustment == 0.1

    def test_combined_max(self):
        """All stressed: +0.2 + 0.1 = +0.3"""
        m = MacroSnapshot(vix=35.0, move=150.0, sofr_ois_spread_bps=10.0)
        assert abs(m.urgency_adjustment - 0.3) < 1e-9

    def test_none_values(self):
        """Missing data → no adjustment"""
        m = MacroSnapshot(vix=None, move=None, sofr_ois_spread_bps=None)
        assert m.urgency_adjustment == 0.0
        assert m.regime_label == 'unknown'

    def test_summary(self):
        m = MacroSnapshot(vix=20.0, move=100.0, sofr_ois_spread_bps=3.0)
        s = m.summary()
        assert 'VIX=20.0' in s
        assert 'MOVE=100.0' in s
        assert 'normal' in s
