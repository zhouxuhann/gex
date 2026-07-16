"""ddput 实时信号检测器 (L2: 观察提醒 + 自动记录)

**不是可交易信号**。输出仅为"历史 in-sample 偏多/偏空观察"，基于 4 天全涨日样本，
regime-gated 假设未验证。见 memory/project_ddput_signal_hypothesis.md。

流程：
1. 每个 GEX 计算结果调 `update(ts, put_gex)` 一次（3 秒间隔）
2. 在新 1-min bar 产生时触发 `_compute_signal()`：
   - trailing 5-min 平滑（不用 center=True，因为实时场景没有未来数据）
   - 两次差分得到 ddput
   - 用"今日 open 至今"的 ddput 标准差算 z-score
   - 按 z 阈值 + 冷却时间 + 时段过滤决定是否 fire alert
3. 每次触发 append 一行到 data/signals_live_{symbol}_{date}.parquet
4. 返回 SignalState 给调用方用于 UI 显示

实时和离线分析器的差异：
- 离线 `gex_tools.py` 用 rolling(5, center=True, min_periods=3) —— 用了未来数据，
  不适合实时
- 这里用 rolling(5, min_periods=3) —— 纯 trailing，信号延后约 2-3 分钟但不泄露未来
- 阈值用 z-score 基于今日 from-open 累计 std，而非全天 static 分位
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class DdputConfig:
    """ddput 信号的可调参数"""
    enabled: bool = False
    z_mild: float = 1.5
    z_strong: float = 2.5
    min_time_et: float = 10.0       # 开盘前 30min 噪声大，10:00 后才评估
    max_time_et: float = 15.0       # 15:00 后 0DTE gamma 塌陷，信号是机械噪声
    cooldown_sec: int = 300         # 同方向 5 分钟内只报一次；0 = 不去重
    alert_directions: tuple[str, ...] = ('+', '-')  # 哪些方向触发 alert；paper trade 用 ('+',)
    smooth_window: int = 5          # trailing rolling mean
    buffer_minutes: int = 60        # 保留多少分钟 put_gex 历史
    min_history_min: int = 10       # 少于这么多分钟不评估
    # std 计算方式：
    #   None  = from-open expanding std（从当日 09:30 累计到当前）
    #   N>0   = rolling N 分钟 std（只看最近 N 分钟）
    # rolling 的好处：对 regime 变化更敏感，不被开盘数据永久拖累
    std_window_min: int | None = None


@dataclass
class SignalState:
    """单次 `update()` 后返回的当前状态"""
    ts: pd.Timestamp
    put_gex_B: float                 # 最新原始 put_gex (B$)
    put_sm_B: float                  # 5-min trailing 平滑
    dput: float                      # 一阶差分 (B$/min)
    ddput: float                     # 二阶差分 (B$/min²)
    z_score: float                   # 今日 ddput 累计 z-score
    daily_std: float                 # 今日 ddput std
    direction: str | None            # '+' / '-' / None
    strength: str | None             # 'mild' / 'strong' / None
    alert_fired: bool                # 本次 tick 是否触发 alert（过了 cooldown）
    spot: float | None = None        # 用于写盘记录

    def to_row(self) -> dict:
        d = asdict(self)
        d['ts'] = self.ts.isoformat()
        return d


class DdputSignalDetector:
    """单标的实时 ddput 检测器"""

    def __init__(
        self,
        symbol: str,
        config: DdputConfig | None = None,
        data_dir: Path | str = Path('data'),
        email_notifier=None,
    ):
        self.symbol = symbol
        self.config = config or DdputConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.email_notifier = email_notifier

        # (ts_minute, put_gex_B) 的时间序列缓冲
        self._bar_buffer: dict[pd.Timestamp, float] = {}
        self._current_minute: pd.Timestamp | None = None
        self._current_minute_latest_value: float | None = None

        # 冷却
        self._last_alert: dict[str, pd.Timestamp | None] = {'+': None, '-': None}

        # 今日 ddput 序列，用于增量 std 计算
        self._today_date: str | None = None

    # --------------------------------------------------------------
    def update(
        self,
        ts: pd.Timestamp,
        put_gex: float,
        spot: float | None = None,
    ) -> SignalState | None:
        """Tick 级调用。仅当新 1-min bar 产生时返回非 None state。"""
        if ts is None or pd.isna(put_gex):
            return None

        # 兼容 datetime.datetime 和 pd.Timestamp（et_now() 返回前者，
        # 但内部 pandas 操作需要后者；datetime 没有 .tz 只有 .tzinfo）
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        ts_et = ts.tz_convert('America/New_York') if ts.tz is not None else ts
        bar_ts = ts_et.floor('1min')

        # 跨日重置
        date_str = bar_ts.strftime('%Y%m%d')
        if self._today_date != date_str:
            self._today_date = date_str
            self._bar_buffer.clear()
            self._last_alert = {'+': None, '-': None}
            self._current_minute = None

        put_gex_B = put_gex / 1e9

        if bar_ts == self._current_minute:
            # 同一分钟，只更新最新值（最后一个 tick 的值会被采纳）
            self._bar_buffer[bar_ts] = put_gex_B
            self._current_minute_latest_value = put_gex_B
            return None  # 不在新 bar 边界上，不出信号

        # 新 bar
        self._bar_buffer[bar_ts] = put_gex_B
        self._current_minute = bar_ts
        self._current_minute_latest_value = put_gex_B

        # 修剪 buffer
        cutoff = bar_ts - pd.Timedelta(minutes=self.config.buffer_minutes)
        self._bar_buffer = {t: v for t, v in self._bar_buffer.items() if t > cutoff}

        return self._compute_signal(bar_ts, spot)

    # --------------------------------------------------------------
    def _compute_signal(
        self, now_ts: pd.Timestamp, spot: float | None
    ) -> SignalState | None:
        if len(self._bar_buffer) < self.config.min_history_min:
            return None

        s = pd.Series(self._bar_buffer).sort_index()
        # Trailing rolling mean (NOT center)
        sm = s.rolling(self.config.smooth_window, min_periods=3).mean()
        dput = sm.diff()
        ddput = dput.diff()

        current_ddput = ddput.iloc[-1]
        if pd.isna(current_ddput):
            return None

        # std 计算：from-open expanding 或 rolling N-min
        if self.config.std_window_min is None:
            today_start = now_ts.normalize() + pd.Timedelta(hours=9, minutes=30)
            sample = ddput.loc[ddput.index >= today_start].dropna()
        else:
            # rolling: 只用最近 N 个 bar
            sample = ddput.dropna().iloc[-self.config.std_window_min:]

        if len(sample) < self.config.min_history_min:
            return None
        std = sample.std()
        if std is None or std == 0 or pd.isna(std):
            return None

        z = float(current_ddput / std)

        # 时段过滤
        hour_et = now_ts.hour + now_ts.minute / 60.0
        in_active_window = (hour_et >= self.config.min_time_et
                             and hour_et < self.config.max_time_et)

        direction = None
        strength = None
        if in_active_window:
            if z >= self.config.z_strong:
                direction, strength = '+', 'strong'
            elif z >= self.config.z_mild:
                direction, strength = '+', 'mild'
            elif z <= -self.config.z_strong:
                direction, strength = '-', 'strong'
            elif z <= -self.config.z_mild:
                direction, strength = '-', 'mild'

        if direction and direction not in self.config.alert_directions:
            direction, strength = None, None

        alert_fired = False
        if direction:
            if self.config.cooldown_sec <= 0:
                # cooldown 关闭：每个超阈值都 fire
                alert_fired = True
                self._last_alert[direction] = now_ts
            else:
                last = self._last_alert.get(direction)
                if (last is None
                    or (now_ts - last).total_seconds() >= self.config.cooldown_sec):
                    alert_fired = True
                    self._last_alert[direction] = now_ts

        state = SignalState(
            ts=now_ts,
            put_gex_B=float(s.iloc[-1]),
            put_sm_B=float(sm.iloc[-1]) if not pd.isna(sm.iloc[-1]) else float('nan'),
            dput=float(dput.iloc[-1]) if not pd.isna(dput.iloc[-1]) else float('nan'),
            ddput=float(current_ddput),
            z_score=z,
            daily_std=float(std),
            direction=direction,
            strength=strength,
            alert_fired=alert_fired,
            spot=float(spot) if spot is not None else None,
        )

        if alert_fired:
            self._persist_alert(state)
            self._email_alert(state)

        return state

    # --------------------------------------------------------------
    def _persist_alert(self, state: SignalState) -> None:
        """Append 单条 alert 到 data/signals_live_{symbol}_{date}.parquet"""
        path = self.data_dir / f'signals_live_{self.symbol}_{self._today_date}.parquet'
        row = state.to_row()
        row['symbol'] = self.symbol
        df_new = pd.DataFrame([row])
        try:
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_parquet(path, index=False)
            spot_str = f'{state.spot:.2f}' if state.spot is not None else '?'
            log.info(f'[{self.symbol}] ddput {state.direction}{state.strength} '
                     f'z={state.z_score:+.2f} ddput={state.ddput:+.3f} '
                     f'spot={spot_str}')
        except Exception as e:
            log.error(f'[{self.symbol}] failed to persist ddput alert: {e}')

    # --------------------------------------------------------------
    def _email_alert(self, state: SignalState) -> None:
        """失败不影响主流程"""
        if self.email_notifier is None:
            return
        try:
            self.email_notifier.send_ddput_alert(
                symbol=self.symbol,
                direction=state.direction,
                strength=state.strength,
                z_score=state.z_score,
                ddput=state.ddput,
                put_gex_B=state.put_gex_B,
                spot=state.spot,
                ts_et=state.ts.strftime('%Y-%m-%d %H:%M:%S ET'),
            )
        except Exception as e:
            log.error(f'[{self.symbol}] email_alert failed: {e}')

    # --------------------------------------------------------------
    def get_recent_alerts(self, n: int = 10) -> pd.DataFrame:
        """读出今日文件里最近 n 条 alerts（给 UI 用）"""
        if self._today_date is None:
            return pd.DataFrame()
        path = self.data_dir / f'signals_live_{self.symbol}_{self._today_date}.parquet'
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        return df.tail(n)
