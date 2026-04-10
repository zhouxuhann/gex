"""IB 连接与数据采集模块"""
import asyncio
import logging
import time

import numpy as np
from ib_insync import IB, Stock, Index, Option

from .config import TimingConfig
from .gex_calc import calculate_gex, pick_expiry
from .features import compute_realtime_features
from .state import StateManager
from .storage import StorageManager
from .time_utils import (
    et_now, trading_date_str, is_market_open, seconds_until_next_open
)

log = logging.getLogger(__name__)

# IB market data generic ticks
GENERIC_TICKS = '100,101,104,106'


class IBWorker:
    """
    IB 数据采集 Worker

    每个标的一个实例，在独立线程运行
    """

    def __init__(
        self,
        symbol: str,
        trading_class: str,
        state: StateManager,
        storage: StorageManager,
        ib_host: str = '127.0.0.1',
        ib_port: int = 4002,
        client_id: int = 10,
        strike_range: float = 0.04,
        spot_sanity_pct: float = 0.01,
        sec_type: str = 'STK',
        connect_timeout: int = 20,
        max_retries: int = 3,
        timing: TimingConfig | None = None,
    ):
        self.symbol = symbol
        self.trading_class = trading_class
        self.state = state
        self.storage = storage
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.client_id = client_id
        self.strike_range = strike_range
        self.spot_sanity_pct = spot_sanity_pct
        self.sec_type = sec_type
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.timing = timing or TimingConfig()

        self.ib: IB | None = None
        self.underlying = None
        self.chain = None
        self.current_key: tuple | None = None
        self.current_contracts: list = []
        self.last_persist: float = 0
        self.last_expiry_seen: str | None = None
        self.last_good_spot: float | None = None
        self._running: bool = True

    def _log(self, level: str, msg: str) -> None:
        """记录日志到 state 和 logger"""
        self.state.log(level, msg)

    def _sleep(self, sec: float) -> None:
        """睡眠，同时推进 IB event loop"""
        if self.ib is not None and self.ib.isConnected():
            self.ib.sleep(sec)
        else:
            time.sleep(sec)

    def _connect(self) -> None:
        """建立 IB 连接（带重试和超时）"""
        if self.ib is not None:
            try:
                self.ib.disconnect()
            except Exception:
                pass

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.ib = IB()
                self.ib.connect(
                    self.ib_host, self.ib_port, clientId=self.client_id,
                    timeout=self.connect_timeout
                )
                break  # 连接成功
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = attempt * 2  # 指数退避: 2s, 4s, 6s
                    self._log('warning',
                              f"IB connect attempt {attempt}/{self.max_retries} failed: {e}, "
                              f"retrying in {delay}s")
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"IB connect failed after {self.max_retries} attempts: {last_error}"
                    )

        # 创建 underlying
        if self.sec_type == 'IND':
            self.underlying = Index(self.symbol, 'CBOE', 'USD')
        else:
            self.underlying = Stock(self.symbol, 'SMART', 'USD')

        self.ib.qualifyContracts(self.underlying)

        # 获取期权链
        chains = self.ib.reqSecDefOptParams(
            self.underlying.symbol, '', self.underlying.secType, self.underlying.conId
        )
        self.chain = next((c for c in chains if c.exchange == 'SMART'), None)
        if self.chain is None:
            raise RuntimeError(
                f"No SMART option chain for {self.symbol}, "
                f"available exchanges: {[c.exchange for c in chains]}"
            )

        # 订阅 underlying 行情
        self.ib.reqMktData(self.underlying, genericTickList='', snapshot=False)

        self.current_key = None
        self.current_contracts = []
        self.last_good_spot = None

        self.state.set_status(connected=True, market_open=True)
        self._log('info', f"IB connected (host={self.ib_host}, port={self.ib_port})")
        self.ib.sleep(1)

    def _subscribe_options(self, expiry: str, strikes: list[float]) -> None:
        """订阅期权行情"""
        key = (expiry, tuple(strikes))
        if key == self.current_key:
            return

        # 取消旧订阅
        if self.current_contracts:
            for c in self.current_contracts:
                try:
                    self.ib.cancelMktData(c)
                except Exception:
                    pass

        # 创建新合约
        raw = [
            Option(self.symbol, expiry, s, r, 'SMART',
                   tradingClass=self.trading_class)
            for s in strikes for r in ['C', 'P']
        ]
        self.current_contracts = self.ib.qualifyContracts(*raw)

        # 订阅行情
        for c in self.current_contracts:
            self.ib.reqMktData(c, genericTickList=GENERIC_TICKS, snapshot=False)

        self.current_key = key
        self._log('info', f"订阅 {len(self.current_contracts)} 个合约 "
                          f"expiry={expiry} strikes={len(strikes)}")
        self.ib.sleep(2)

    def _process_tick(self) -> bool:
        """处理一次 tick"""
        # 获取 spot
        u_ticker = self.ib.ticker(self.underlying)
        spot = u_ticker.marketPrice() if u_ticker else None
        if not spot or np.isnan(spot) or spot <= 0:
            return False

        # spot sanity check
        if self.last_good_spot is not None:
            drift = abs(spot - self.last_good_spot) / self.last_good_spot
            if drift > self.spot_sanity_pct:
                self._log('warning',
                          f"丢弃异常 spot={spot:.2f} "
                          f"(上次={self.last_good_spot:.2f}, 漂移 {drift:.1%})")
                return False
        self.last_good_spot = spot

        # 选择 expiry
        today_str = trading_date_str()
        expiry, is_true_0dte = pick_expiry(self.chain, today_str)
        if expiry is None:
            self._log('error', '无可用 expiry')
            return False

        # 选择 strikes：ATM 前后各 10 个整数 strike
        all_strikes = sorted(s for s in self.chain.strikes if s == int(s))
        below = [s for s in all_strikes if s <= spot][-10:]  # ATM 及以下 10 个
        above = [s for s in all_strikes if s > spot][:10]    # ATM 以上 10 个
        strikes = sorted(set(below + above))

        # 订阅期权
        self._subscribe_options(expiry, strikes)

        # 记录 expiry 变化
        if expiry != self.last_expiry_seen:
            self.last_expiry_seen = expiry
            if is_true_0dte:
                self._log('info', f"当前 expiry: {expiry} (真 0DTE)")
            else:
                self._log('warning',
                          f"⚠️ 今日无 0DTE 合约，回退到 {expiry} — GEX 语义与 0DTE 不同")

        # 计算 GEX
        tickers = [self.ib.ticker(c) for c in self.current_contracts]
        result = calculate_gex(tickers, spot)

        if result is None:
            self._log('warning',
                      f'No valid data from {len(self.current_contracts)} contracts — '
                      'check market data subscription')
            return False

        if result.missing_greeks > 0 or result.missing_oi > 0:
            # 仅在数据较多缺失时警告
            total = len(self.current_contracts)
            missing = result.missing_greeks + result.missing_oi
            if missing > total * 0.5:
                self._log('warning',
                          f'数据缺失较多: missing_greeks={result.missing_greeks} '
                          f'missing_oi={result.missing_oi}')

        # 计算 regime 特征
        try:
            history, _ = self.state.get_history_for_resample()
            _, regime_code, regime_tags = compute_realtime_features(
                result.df, spot, history
            )
        except Exception as e:
            log.debug(f"Regime 计算失败: {e}")
            regime_code, regime_tags = None, None

        # 更新状态
        self.state.update(
            spot=spot,
            total_gex=result.total_gex,
            gamma_flip=result.gamma_flip,
            call_gex=result.call_gex,
            put_gex=result.put_gex,
            atm_iv_pct=result.atm_iv_pct,
            expiry=expiry,
            is_true_0dte=is_true_0dte,
            df=result.df,
            call_wall=result.call_wall,
            put_wall=result.put_wall,
            positive_gamma=result.positive_gamma,
            regime_code=regime_code,
            regime_tags=regime_tags,
        )

        # 定期持久化
        if time.time() - self.last_persist > self.timing.persist_interval_sec:
            hist, ohlc, strikes = self.state.get_persist_data()
            self.storage.persist_async(self.symbol, hist, ohlc, strikes)
            self.last_persist = time.time()

        return True

    def run(self) -> None:
        """主循环（在独立线程调用）"""
        asyncio.set_event_loop(asyncio.new_event_loop())

        while self._running:
            # 非交易时段
            if not is_market_open():
                self.state.set_status(
                    market_open=False,
                    updated=f"非交易时段 ({et_now().strftime('%H:%M ET')})"
                )

                if self.ib is not None and self.ib.isConnected():
                    # 盘后落盘
                    try:
                        hist, ohlc, strikes = self.state.get_persist_data()
                        self.storage.persist_async(self.symbol, hist, ohlc, strikes)
                    except Exception as e:
                        self._log('error', f"盘后 persist 失败: {e}")

                    # 断开连接
                    try:
                        for c in self.current_contracts:
                            try:
                                self.ib.cancelMktData(c)
                            except Exception:
                                pass
                        self.ib.disconnect()
                    except Exception:
                        pass
                    self.ib = None
                    self.current_key = None
                    self.current_contracts = []
                    self.last_good_spot = None

                # 等待下一个交易日
                try:
                    sleep_sec = max(seconds_until_next_open() - 60, 30)
                    self._log('info', f"Market closed, next check in {sleep_sec:.0f}s")
                    time.sleep(min(sleep_sec, self.timing.max_sleep_sec))
                except RuntimeError as e:
                    self._log('error', f"{e}; retrying in {self.timing.market_closed_check_sec}s")
                    time.sleep(self.timing.market_closed_check_sec)
                continue

            # 确保连接
            if self.ib is None or not self.ib.isConnected():
                try:
                    self._connect()
                except Exception as e:
                    self.state.set_status(connected=False)
                    self._log('error', f"IB connect failed: {e}")
                    time.sleep(self.timing.reconnect_delay_sec)
                    continue

            # 主循环
            try:
                self._process_tick()
            except Exception as e:
                self._log('error', f"Main loop error: {e}")
                if self.ib is not None and not self.ib.isConnected():
                    self.state.set_status(connected=False)

            self._sleep(self.timing.tick_interval_sec)

    def stop(self) -> None:
        """停止 worker"""
        self._running = False
        if self.ib is not None and self.ib.isConnected():
            try:
                self.ib.disconnect()
            except Exception:
                pass
