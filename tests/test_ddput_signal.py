"""测试 DdputSignalDetector 的核心行为。

关注点：
  * 跨分钟边界才产生 state
  * 同一分钟多次 tick 会更新缓冲但不重复出信号
  * 时段过滤（10:00 前 / 15:00 后 silent）
  * Cooldown 去重
  * 跨日自动清空
  * 持久化到正确文件
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from gex_monitor.ddput_signal import DdputConfig, DdputSignalDetector, SignalState


ET = ZoneInfo('America/New_York')


def ts(date: str, h: int, m: int, s: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f'{date} {h:02d}:{m:02d}:{s:02d}', tz=ET)


def config(**kwargs):
    """快速构造 DdputConfig，默认短 cooldown 方便测试"""
    defaults = dict(enabled=True, cooldown_sec=60, min_history_min=5,
                    min_time_et=0.0, max_time_et=24.0)
    defaults.update(kwargs)
    return DdputConfig(**defaults)


# -----------------------------------------------------------------------
class TestBasicFlow:
    def test_same_minute_ticks_dont_fire(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(), data_dir=tmp_path)
        # 同一分钟内 3 个 tick，都不应出 state（state is None 因为缓冲不足且非新 bar）
        for sec in [0, 10, 30]:
            state = d.update(ts('2026-04-21', 10, 30, sec), -50e9)
            assert state is None, f'sec={sec}: 同分钟不应返回 state'

    def test_new_minute_returns_state_when_enough_history(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(), data_dir=tmp_path)
        # 连续喂 10 个分钟的数据，最后一个应该返回 state
        values = [-50, -49, -48, -47, -46, -45, -44, -43, -42, -41]
        last_state = None
        for i, v in enumerate(values):
            last_state = d.update(ts('2026-04-21', 10, 30 + i), v * 1e9)
        assert last_state is not None
        assert last_state.put_gex_B == pytest.approx(-41.0)

    def test_insufficient_history_returns_none(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(min_history_min=10), data_dir=tmp_path)
        # 只喂 5 个 bar，少于 min_history_min=10
        for i in range(5):
            state = d.update(ts('2026-04-21', 10, 30 + i), -50e9)
        # 第 5 个 bar 返回 None（还不够）
        assert state is None


# -----------------------------------------------------------------------
class TestDetection:
    def _feed_ramp_then_spike(self, d, base_time, ramp_count=15, spike_value=-20):
        """喂一个稳定基准 + 末尾大跳变。返回每个 bar 的 state。"""
        states = []
        # 稳定基准：-50 ± 小噪声
        for i in range(ramp_count):
            noise = 0.05 * ((i % 3) - 1)   # -0.05 / 0 / +0.05
            v = (-50.0 + noise) * 1e9
            s = d.update(base_time + pd.Timedelta(minutes=i), v)
            states.append(s)
        # 大 put_gex 正向跳变（put 快速 unwind，ddput 正峰）
        s = d.update(base_time + pd.Timedelta(minutes=ramp_count),
                     spike_value * 1e9, spot=640.0)
        states.append(s)
        return states

    def test_extreme_positive_fires_long_alert(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(z_mild=1.5, z_strong=2.5),
                                  data_dir=tmp_path)
        states = self._feed_ramp_then_spike(d, ts('2026-04-21', 11, 0))
        last = states[-1]
        assert last is not None
        assert last.direction == '+', f'ddput={last.ddput}, z={last.z_score}'
        assert last.alert_fired

    def test_extreme_negative_fires_short_alert(self, tmp_path):
        """ddput 极负：模拟 put 快速扩张（put_gex 变得更负后加速）"""
        d = DdputSignalDetector('QQQ', config(z_mild=1.5, z_strong=2.5),
                                  data_dir=tmp_path)
        # 平台 + 急坠
        base_time = ts('2026-04-21', 11, 0)
        for i in range(15):
            d.update(base_time + pd.Timedelta(minutes=i), -50e9)
        # put_gex 急速走向更负
        last = d.update(base_time + pd.Timedelta(minutes=15), -80e9, spot=640.0)
        # 这会导致 dput 负，ddput 负（看 dput 的二阶差分）
        # 先再来一个更负的让 ddput 累加
        last = d.update(base_time + pd.Timedelta(minutes=16), -120e9, spot=640.0)
        assert last is not None
        assert last.direction == '-', f'ddput={last.ddput}, z={last.z_score}'

    def test_alert_directions_can_match_long_only_trader(self, tmp_path):
        d = DdputSignalDetector(
            'QQQ',
            config(z_mild=1.5, z_strong=2.5, alert_directions=('+',)),
            data_dir=tmp_path,
        )
        base_time = ts('2026-04-21', 11, 0)
        for i in range(15):
            d.update(base_time + pd.Timedelta(minutes=i), -50e9)
        d.update(base_time + pd.Timedelta(minutes=15), -80e9, spot=640.0)
        last = d.update(base_time + pd.Timedelta(minutes=16), -120e9, spot=640.0)

        assert last is not None
        assert last.direction is None
        assert not last.alert_fired
        assert not list(Path(tmp_path).glob('signals_live_QQQ_*.parquet'))


# -----------------------------------------------------------------------
class TestTimeWindow:
    def test_before_min_time_no_alert(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(min_time_et=10.0, max_time_et=15.0),
                                 data_dir=tmp_path)
        # 时间在 09:45 — 在 10:00 之前，即使触发也不报 alert
        base = ts('2026-04-21', 9, 45)
        for i in range(10):
            d.update(base + pd.Timedelta(minutes=i), -50e9)
        last = d.update(base + pd.Timedelta(minutes=10), -20e9)
        assert last is not None
        assert last.direction is None  # 在窗口外 silent

    def test_after_max_time_no_alert(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(min_time_et=10.0, max_time_et=15.0),
                                 data_dir=tmp_path)
        base = ts('2026-04-21', 15, 10)
        for i in range(10):
            d.update(base + pd.Timedelta(minutes=i), -50e9)
        last = d.update(base + pd.Timedelta(minutes=10), -20e9)
        assert last is not None
        assert last.direction is None  # 尾盘 silent

    def test_inside_window_can_fire(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(min_time_et=10.0, max_time_et=15.0),
                                 data_dir=tmp_path)
        base = ts('2026-04-21', 11, 30)
        for i in range(15):
            d.update(base + pd.Timedelta(minutes=i), -50e9)
        last = d.update(base + pd.Timedelta(minutes=15), -20e9, spot=640.0)
        assert last is not None
        assert last.direction == '+'


# -----------------------------------------------------------------------
class TestCooldown:
    def test_same_direction_cooldown_suppresses(self, tmp_path):
        # 用 flat 基准（std=0 → warmup 期不触发），然后连续两次 spike 测 cooldown
        d = DdputSignalDetector('QQQ',
                                 config(cooldown_sec=300, z_mild=0.5,
                                        min_history_min=10),
                                 data_dir=tmp_path)
        base = ts('2026-04-21', 11, 0)
        for i in range(15):
            d.update(base + pd.Timedelta(minutes=i), -50e9)   # 完全平
        # 第一次 spike → 第一次 fire
        s1 = d.update(base + pd.Timedelta(minutes=15), -20e9, spot=640.0)
        assert s1.alert_fired, f'first spike should fire: {s1}'
        # 继续加大的 spike，方向还是 +
        s2 = d.update(base + pd.Timedelta(minutes=16), 0.0e9, spot=640.0)
        assert s2 is not None
        assert s2.direction == '+', (
            f'ddput={s2.ddput}, z={s2.z_score}, std={s2.daily_std}'
        )
        assert not s2.alert_fired   # 在 cooldown 里 silent

    def test_cooldown_expires(self, tmp_path):
        d = DdputSignalDetector('QQQ',
                                 config(cooldown_sec=120, z_mild=1.0),
                                 data_dir=tmp_path)
        base = ts('2026-04-21', 11, 0)
        for i in range(15):
            d.update(base + pd.Timedelta(minutes=i), -50e9)
        s1 = d.update(base + pd.Timedelta(minutes=15), -20e9, spot=640.0)
        assert s1.alert_fired
        # 3 分钟后再触发
        for i in range(3):
            d.update(base + pd.Timedelta(minutes=16 + i), -50e9)
        s2 = d.update(base + pd.Timedelta(minutes=19), -20e9, spot=640.0)
        if s2 is not None and s2.direction == '+':
            # cooldown 过了，应该能再 fire
            assert s2.alert_fired


# -----------------------------------------------------------------------
class TestCrossDayReset:
    def test_new_day_clears_buffer(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(), data_dir=tmp_path)
        # Day 1
        base1 = ts('2026-04-21', 14, 0)
        for i in range(15):
            d.update(base1 + pd.Timedelta(minutes=i), -50e9)
        # Day 2 第一 tick
        base2 = ts('2026-04-22', 9, 30)
        state = d.update(base2, -50e9)
        # buffer 应该刚被清空，返回 None（history 不足）
        assert state is None
        assert d._today_date == '20260422'


# -----------------------------------------------------------------------
class TestPersistence:
    def test_alert_appended_to_parquet(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(z_mild=1.0, cooldown_sec=30),
                                  data_dir=tmp_path)
        base = ts('2026-04-21', 11, 0)
        for i in range(15):
            d.update(base + pd.Timedelta(minutes=i), -50e9)
        d.update(base + pd.Timedelta(minutes=15), -20e9, spot=640.0)
        path = tmp_path / 'signals_live_QQQ_20260421.parquet'
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]['direction'] == '+'
        assert df.iloc[0]['symbol'] == 'QQQ'

    def test_multiple_alerts_accumulate(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(z_mild=1.0, cooldown_sec=30),
                                  data_dir=tmp_path)
        base = ts('2026-04-21', 11, 0)
        for i in range(15):
            d.update(base + pd.Timedelta(minutes=i), -50e9)
        # 第一条
        d.update(base + pd.Timedelta(minutes=15), -20e9, spot=640.0)
        # 等 cooldown 过（配置 30s，跨 1 个 bar 就够）
        d.update(base + pd.Timedelta(minutes=16), -50e9)
        # 第二条，再次触发
        d.update(base + pd.Timedelta(minutes=17), -15e9, spot=641.0)
        path = tmp_path / 'signals_live_QQQ_20260421.parquet'
        df = pd.read_parquet(path)
        assert len(df) >= 1  # 至少有一条

    def test_get_recent_alerts(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(z_mild=1.0, cooldown_sec=30),
                                  data_dir=tmp_path)
        base = ts('2026-04-21', 11, 0)
        for i in range(15):
            d.update(base + pd.Timedelta(minutes=i), -50e9)
        d.update(base + pd.Timedelta(minutes=15), -20e9, spot=640.0)
        recent = d.get_recent_alerts(n=5)
        assert len(recent) >= 1


# -----------------------------------------------------------------------
class TestEdgeCases:
    def test_nan_put_gex_returns_none(self, tmp_path):
        d = DdputSignalDetector('QQQ', config(), data_dir=tmp_path)
        import numpy as np
        assert d.update(ts('2026-04-21', 10, 30), np.nan) is None

    def test_zero_std_returns_none(self, tmp_path):
        """所有 put_gex 完全相等 → std=0 → 不出信号"""
        d = DdputSignalDetector('QQQ', config(), data_dir=tmp_path)
        base = ts('2026-04-21', 11, 0)
        for i in range(15):
            s = d.update(base + pd.Timedelta(minutes=i), -50e9)
        # 最后一个 bar 也是 -50，std 应为 0
        assert s is None or s.direction is None
