"""状态管理模块"""
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .time_utils import et_now

log = logging.getLogger(__name__)


@dataclass
class GEXSnapshot:
    """单次 GEX 数据快照"""
    ts: datetime
    spot: float
    total_gex: float
    flip: float
    call_gex: float
    put_gex: float
    atm_iv_pct: float | None
    # 新增字段
    call_wall: float | None = None
    put_wall: float | None = None
    positive_gamma: bool = False
    # Regime 分类
    regime_code: str | None = None
    regime_tags: dict | None = None


@dataclass
class OHLCBar:
    """OHLC K 线"""
    ts: datetime
    open: float
    high: float
    low: float
    close: float


class StateManager:
    """
    单个标的的状态管理器

    线程安全，所有读写都通过锁保护
    """

    def __init__(self, symbol: str, max_history: int = 8000, max_logs: int = 30):
        self.symbol = symbol
        self.max_history = max_history
        self.max_logs = max_logs

        self._lock = threading.Lock()

        # 实时状态
        self._spot: float = 0
        self._total_gex: float = 0
        self._gamma_flip: float = 0
        self._atm_iv_pct: float | None = None
        self._expiry: str | None = None
        self._is_true_0dte: bool = False
        self._df: pd.DataFrame = pd.DataFrame()
        self._updated: str = '未连接'
        self._last_update_ts: datetime | None = None
        self._market_open: bool = False
        self._connected: bool = False
        # 新增字段
        self._call_wall: float | None = None
        self._put_wall: float | None = None
        self._positive_gamma: bool = False
        # Regime 分类
        self._regime_code: str | None = None
        self._regime_tags: dict | None = None

        # 历史数据
        self._history: deque = deque(maxlen=max_history)
        self._ohlc_minute: deque = deque(maxlen=max_history)
        self._last_minute_bar: dict | None = None
        self._history_version: int = 0

        # strike-level 数据（用于回放，每分钟采样一次）
        # 乘数 50 假设每分钟约 50 个 strike (ATM ±4% 范围，$1 间距)
        self._strikes_per_minute_estimate = 50
        self._strikes_history: deque = deque(maxlen=max_history * self._strikes_per_minute_estimate)
        self._last_strike_minute: datetime | None = None

        # 日志
        self._logs: deque = deque(maxlen=max_logs)

        # resample 缓存
        self._cache_lock = threading.Lock()
        self._resample_cache = {'version': -1, 'df': None}

    def update(self, spot: float, total_gex: float, gamma_flip: float,
               call_gex: float, put_gex: float, atm_iv_pct: float | None,
               expiry: str, is_true_0dte: bool, df: pd.DataFrame,
               call_wall: float | None = None, put_wall: float | None = None,
               positive_gamma: bool = False,
               regime_code: str | None = None, regime_tags: dict | None = None) -> None:
        """更新实时状态"""
        now = et_now()
        minute = now.replace(second=0, microsecond=0)

        with self._lock:
            self._spot = spot
            self._total_gex = total_gex
            self._gamma_flip = gamma_flip
            self._atm_iv_pct = atm_iv_pct
            self._expiry = expiry
            self._is_true_0dte = is_true_0dte
            self._df = df
            self._updated = now.strftime('%H:%M:%S ET')
            self._last_update_ts = now
            self._market_open = True
            self._connected = True
            # 新增字段
            self._call_wall = call_wall
            self._put_wall = put_wall
            self._positive_gamma = positive_gamma
            # Regime 分类
            self._regime_code = regime_code
            self._regime_tags = regime_tags

            # 追加历史
            self._history.append({
                'ts': now,
                'spot': spot,
                'total_gex': total_gex,
                'flip': gamma_flip,
                'call_gex': call_gex,
                'put_gex': put_gex,
                'atm_iv_pct': atm_iv_pct,
                'call_wall': call_wall,
                'put_wall': put_wall,
                'positive_gamma': positive_gamma,
            })

            # OHLC
            lb = self._last_minute_bar
            if lb is None or lb['ts'] != minute:
                if lb is not None:
                    self._ohlc_minute.append(lb)
                self._last_minute_bar = {
                    'ts': minute, 'open': spot,
                    'high': spot, 'low': spot, 'close': spot
                }
            else:
                lb['high'] = max(lb['high'], spot)
                lb['low'] = min(lb['low'], spot)
                lb['close'] = spot

            # strike-level 数据（每分钟采样一次，用于回放）
            if self._last_strike_minute is None or self._last_strike_minute != minute:
                self._last_strike_minute = minute
                if not df.empty:
                    # 使用 to_dict('records') 替代 iterrows，性能更好
                    for row in df.to_dict('records'):
                        self._strikes_history.append({
                            'ts': minute,
                            'strike': row['strike'],
                            'right': row['right'],
                            'gex': row['gex'],
                            'gamma': row.get('gamma', 0),
                            'oi': row.get('oi', 0),
                            'iv': row.get('iv', None),
                        })

            self._history_version += 1

    def set_status(self, market_open: bool | None = None, connected: bool | None = None,
                   updated: str | None = None) -> None:
        """设置连接状态"""
        with self._lock:
            if market_open is not None:
                self._market_open = market_open
            if connected is not None:
                self._connected = connected
            if updated is not None:
                self._updated = updated

    def log(self, level: str, msg: str) -> None:
        """记录日志"""
        with self._lock:
            self._logs.append((level, et_now(), str(msg)))
        getattr(log, level)(f"[{self.symbol}] {msg}")

    def get_snapshot(self) -> dict:
        """获取当前状态快照（不含历史数据）"""
        with self._lock:
            return {
                'spot': self._spot,
                'total_gex': self._total_gex,
                'gamma_flip': self._gamma_flip,
                'atm_iv_pct': self._atm_iv_pct,
                'expiry': self._expiry,
                'is_true_0dte': self._is_true_0dte,
                'updated': self._updated,
                'last_update_ts': self._last_update_ts,
                'market_open': self._market_open,
                'connected': self._connected,
                # 新增字段
                'call_wall': self._call_wall,
                'put_wall': self._put_wall,
                'positive_gamma': self._positive_gamma,
                # Regime 分类
                'regime_code': self._regime_code,
                'regime_tags': self._regime_tags,
            }

    def get_df(self) -> pd.DataFrame:
        """获取当前期权数据 DataFrame"""
        with self._lock:
            return self._df.copy()

    def get_history_for_resample(self) -> tuple[list, int]:
        """获取历史数据副本和版本号（用于 resample）

        Returns:
            (history_copy, version): 历史数据的浅拷贝和当前版本号
        """
        with self._lock:
            # 返回副本而非内部引用，确保线程安全
            return list(self._history), self._history_version

    def get_logs(self) -> list:
        """获取日志列表"""
        with self._lock:
            return list(self._logs)

    def get_persist_data(self) -> tuple[list, list, list]:
        """获取需要持久化的数据 (hist, ohlc, strikes)"""
        with self._lock:
            hist = [dict(h) for h in self._history]
            ohlc = [dict(b) for b in self._ohlc_minute]
            if self._last_minute_bar is not None:
                ohlc.append(dict(self._last_minute_bar))
            strikes = [dict(s) for s in self._strikes_history]
            return hist, ohlc, strikes

    def resample_history(self, rule: str) -> pd.DataFrame:
        """重采样历史数据"""
        history, version = self.get_history_for_resample()
        if len(history) < 2:
            return pd.DataFrame()

        with self._cache_lock:
            if (self._resample_cache['version'] == version
                    and self._resample_cache['df'] is not None):
                df = self._resample_cache['df']
            else:
                # history 已经是 list，无需再 list()
                df = pd.DataFrame(history).set_index('ts')
                self._resample_cache['version'] = version
                self._resample_cache['df'] = df

            return df.resample(rule).last().dropna(subset=['total_gex'])


class StateRegistry:
    """
    多标的状态注册表

    管理所有标的的 StateManager 实例
    """

    def __init__(self):
        self._managers: dict[str, StateManager] = {}
        self._lock = threading.Lock()

    def register(self, symbol: str, max_history: int = 8000) -> StateManager:
        """注册一个标的"""
        with self._lock:
            if symbol not in self._managers:
                self._managers[symbol] = StateManager(symbol, max_history)
            return self._managers[symbol]

    def get(self, symbol: str) -> StateManager | None:
        """获取指定标的的状态管理器"""
        with self._lock:
            return self._managers.get(symbol)

    def list_symbols(self) -> list[str]:
        """列出所有已注册的标的"""
        with self._lock:
            return list(self._managers.keys())

    def get_all(self) -> dict[str, StateManager]:
        """获取所有状态管理器"""
        with self._lock:
            return dict(self._managers)


# 全局注册表
registry = StateRegistry()
