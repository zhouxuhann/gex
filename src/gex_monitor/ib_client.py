"""IB 连接与数据采集模块"""
import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from statistics import median

import numpy as np
from ib_insync import IB, Index, Option, Stock

from .config import (
    IntradayTurningPointShadowConfig,
    IntradayVRPConfig,
    TimingConfig,
)
from .data_quality import evaluate_tick_quality
from .db_storage import GEXDBStorage
from .features import compute_realtime_features
from .gex_calc import calculate_gex, pick_expiry
from .hedge_executor import HedgeExecutor, format_trade_result
from .hedge_signal import format_recommendation, generate_hedge_signal
from .intraday_turning_point_shadow import IntradayTurningPointShadow
from .intraday_vrp_monitor import IntradayVRPMonitor
from .macro import fetch_macro_snapshot
from .skew import SkewTracker, compute_skew
from .skew_surface import collect_skew_surface
from .state import StateManager
from .storage import SkewSurfaceStorage, StorageManager
from .strike_selector import select_strikes
from .time_utils import (
    et_now,
    is_extended_hours,
    is_market_open,
    option_expiry_date_str,
    seconds_until_next_open,
    seconds_until_next_session,
    should_connect,
    trading_date_str,
)
from .vrp_context import market_vol_context, standardized_term_structure_context

log = logging.getLogger(__name__)

# IB market data generic ticks
GENERIC_TICKS = '100,101,104,106'
RECOVERABLE_IB_ERROR_CODES = {10197, 1100, 1101}


def select_option_chain(chains, trading_class: str):
    """Select the option chain matching the configured trading class.

    Fallback order:
      1. SMART exchange + exact tradingClass  (best: 0DTE weekly chains)
      2. Any exchange  + exact tradingClass   (e.g. CBOE+SPXW)
    Never fall back to a different tradingClass — SPX monthly vs SPXW weekly
    have different strike intervals and expiry sets.
    """
    return (
        next((c for c in chains
              if c.exchange == 'SMART' and c.tradingClass == trading_class), None)
        or next((c for c in chains if c.tradingClass == trading_class), None)
    )


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
        market_data_stale_sec: int = 60,
        quality_min_contracts: int = 20,
        quality_max_missing_ratio: float = 0.25,
        db_storage: GEXDBStorage | None = None,
        hedge_enabled: bool = False,
        hedge_dry_run: bool = False,
        hedge_qty: int = 1,
        ib_error_watcher=None,
        extended_hours: bool = False,
        intraday_vrp_config: IntradayVRPConfig | None = None,
        intraday_turning_point_shadow_config: (
            IntradayTurningPointShadowConfig | None
        ) = None,
    ):
        self.symbol = symbol
        self.trading_class = trading_class
        self.extended_hours = extended_hours
        self.state = state
        self.storage = storage
        self.db_storage = db_storage
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.client_id = client_id
        self.strike_range = strike_range
        self.spot_sanity_pct = spot_sanity_pct
        self.sec_type = sec_type
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.timing = timing or TimingConfig()
        self.market_data_stale_sec = max(
            int(market_data_stale_sec),
            self.timing.tick_interval_sec * 3,
        )
        self.quality_min_contracts = max(1, int(quality_min_contracts))
        self.quality_max_missing_ratio = float(quality_max_missing_ratio)

        self.ib: IB | None = None
        self.underlying = None
        self.chain = None
        self.current_key: tuple | None = None
        self.current_contracts: list = []
        self._invalid_contract_cache: set[tuple[str, float, str]] = set()
        self.last_persist: float = 0
        self.last_expiry_seen: str | None = None
        self.last_good_spot: float | None = None
        self._connected_at_ts: float = 0.0
        self._last_success_ts: float = 0.0
        self._last_market_data_marker: float | None = None
        self._reconnect_requested_reason: str | None = None
        self._last_quality_reasons: tuple[str, ...] = ()
        self._running: bool = True

        # ΔOI 相关
        self.prev_oi: dict[float, dict] | None = None  # 前一交易日 OI
        self.today_oi: dict[float, dict] = {}  # 今日 OI（用于收盘保存）
        self._load_prev_oi()

        # Strike 选择 hysteresis：spot 偏移超过此值才重选 strike
        self._last_strike_spot: float | None = None

        # Flip 平滑（滑动中位数，防止单 tick 跳变）
        self._flip_buffer: deque[float] = deque(maxlen=20)

        # Skew tracker
        self.skew_tracker = SkewTracker(window=30)

        # Daily skew surface + hedge signal (auto at 15:30 ET)
        self._skew_surface_captured_today: bool = False
        self._skew_surface_storage = SkewSurfaceStorage(storage.data_dir)
        self._hedge_enabled = hedge_enabled
        self._hedge_dry_run = hedge_dry_run
        self._hedge_qty = hedge_qty
        self._ib_error_watcher = ib_error_watcher
        self._vrp_monitor = None
        self._intraday_vrp_config = intraday_vrp_config
        if (intraday_vrp_config is not None
                and intraday_vrp_config.enabled
                and self.symbol in intraday_vrp_config.symbols):
            self._vrp_monitor = IntradayVRPMonitor(
                self.symbol, self.storage, intraday_vrp_config
            )
            paper_modes = []
            if intraday_vrp_config.paper_execution_enabled:
                paper_modes.append('Iron Fly')
            if intraday_vrp_config.paper_straddle_execution_enabled:
                paper_modes.append('short Straddle')
            mode = ('observation + PAPER ' + ' + '.join(paper_modes) + ' execution'
                    if paper_modes else 'observation only')
            self._log('info', f'Intraday VRP enabled ({mode})')
        self._turning_point_shadow = None
        self._intraday_turning_point_shadow_config = intraday_turning_point_shadow_config
        if (
            intraday_turning_point_shadow_config is not None
            and intraday_turning_point_shadow_config.enabled
            and self.symbol in intraday_turning_point_shadow_config.symbols
        ):
            self._turning_point_shadow = IntradayTurningPointShadow(
                self.symbol,
                self.storage,
                intraday_turning_point_shadow_config,
            )
            self._log(
                'info',
                'Intraday turning-point shadow enabled (observation only, no orders)',
            )

    def _load_prev_oi(self) -> None:
        """加载前一交易日的 OI 快照"""
        today = option_expiry_date_str()
        prev_date = self.storage.get_previous_trading_day(today, self.symbol)
        if prev_date:
            self.prev_oi = self.storage.load_oi_snapshot(self.symbol, prev_date)
            if self.prev_oi:
                log.info(
                    f"[{self.symbol}] Loaded prev OI from {prev_date}: "
                    f"{len(self.prev_oi)} strikes"
                )
            else:
                log.info(f"[{self.symbol}] No prev OI found for {prev_date}")
        else:
            log.info(f"[{self.symbol}] No previous trading day OI snapshot found")

    def _log(self, level: str, msg: str) -> None:
        """记录日志到 state 和 logger"""
        self.state.log(level, msg)

    def _detach_ib_handlers(self, ib) -> None:
        if self._ib_error_watcher is not None:
            self._ib_error_watcher.detach(ib)
        try:
            ib.errorEvent -= self._on_ib_error
        except Exception:
            pass

    def _request_reconnect(self, reason: str) -> None:
        if self._reconnect_requested_reason is None:
            self._reconnect_requested_reason = reason
            self._log('warning', f"{reason}; 将重建 IB 行情连接")

    def _on_ib_error(self, reqId, errorCode, errorString, contract) -> None:
        if errorCode not in RECOVERABLE_IB_ERROR_CODES:
            return
        if errorCode == 10197:
            reason = "IB 10197: 实时行情被另一端会话占用"
        elif errorCode == 1100:
            self.state.set_status(connected=False)
            reason = "IB 1100: Gateway/TWS 连接断开"
        else:
            reason = f"IB {errorCode}: 连接恢复/行情 reset，需要重新订阅"
        self._request_reconnect(reason)

    def _force_reconnect(self, reason: str) -> None:
        self._log('warning', f"{reason}; 正在断开并重新订阅行情")
        self.state.set_status(connected=False, updated=f"重连中: {reason}")
        if self.ib is not None:
            try:
                self._detach_ib_handlers(self.ib)
                for c in self.current_contracts:
                    try:
                        self.ib.cancelMktData(c)
                    except Exception:
                        pass
                if self.underlying is not None:
                    try:
                        self.ib.cancelMktData(self.underlying)
                    except Exception:
                        pass
                self.ib.disconnect()
            except Exception as e:
                self._log('warning', f"IB reconnect cleanup failed: {e}")
        self.ib = None
        self.current_key = None
        self.current_contracts = []
        self._last_strike_spot = None
        self.last_good_spot = None
        self._connected_at_ts = 0.0
        self._last_market_data_marker = None
        self._reconnect_requested_reason = None

    def _stale_reconnect_reason(self) -> str | None:
        if self.ib is None or not self.ib.isConnected():
            return None
        now_ts = time.time()
        ref_ts = self._last_success_ts or self._connected_at_ts
        if ref_ts <= 0:
            return None
        age = now_ts - ref_ts
        if age >= self.market_data_stale_sec:
            return f"行情 {age:.0f}s 未成功更新，疑似订阅失效"
        return None

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
                self._detach_ib_handlers(self.ib)
                self.ib.disconnect()
            except Exception:
                pass

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.ib = IB()
                if self._ib_error_watcher is not None:
                    self._ib_error_watcher.attach(self.ib)
                self.ib.errorEvent += self._on_ib_error
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
        self.chain = select_option_chain(chains, self.trading_class)
        if self.chain is None:
            raise RuntimeError(
                f"No option chain for {self.symbol}/{self.trading_class}, "
                f"available: {[(c.exchange, c.tradingClass) for c in chains]}"
            )

        # 订阅 underlying 行情
        self.ib.reqMktData(self.underlying, genericTickList='', snapshot=False)

        self.current_key = None
        self.current_contracts = []
        self.last_good_spot = None
        self._connected_at_ts = time.time()
        self._last_success_ts = 0.0
        self._last_market_data_marker = None
        self._reconnect_requested_reason = None

        # 重新加载 ΔOI 基线：进程跨天长跑时，"前一交易日"会变，
        # 只在 __init__ 加载一次会让基线越来越陈旧
        self._load_prev_oi()

        self.state.set_status(connected=True, market_open=True)
        self._log('info', f"IB connected (host={self.ib_host}, port={self.ib_port})")
        self.ib.sleep(1)

        # 补结算仅依赖历史 Parquet；失败不影响实时 GEX 连接。
        if self._vrp_monitor is not None:
            try:
                recovered = self._vrp_monitor.recover_unsettled()
                if recovered:
                    self._log('info', f'VRP recovered {recovered} historical observations')
            except Exception as e:
                self._log('warning', f'VRP recovery failed: {e}')

        # 连接后执行预热，确保数据就绪
        self._warmup()

    def _warmup(self) -> None:
        """
        连接后预热：等待 spot 就绪并完成初始期权订阅

        确保在开始正式数据采集前，所有合约都已正确订阅
        """
        self._log('info', "开始预热...")

        # 1. 等待 spot 就绪
        spot = None
        for attempt in range(10):
            self.ib.sleep(1)
            u_ticker = self.ib.ticker(self.underlying)
            spot = u_ticker.marketPrice() if u_ticker else None
            if spot and not np.isnan(spot) and spot > 0:
                self._log('info', f"Spot 就绪: {spot:.2f}")
                self.last_good_spot = spot
                break
            self._log('info', f"等待 spot... (attempt {attempt + 1}/10)")
        else:
            self._log('warning', "预热: spot 未就绪，将在主循环中重试")
            return

        # 2. 获取 expiry
        today_str = option_expiry_date_str()
        expiry, _ = pick_expiry(self.chain, today_str)
        if expiry is None:
            self._log('warning', "预热: 无可用 expiry")
            return

        # 3. 选择 strikes
        strikes = select_strikes(
            self.chain.strikes, spot, self.strike_range,
            include_half_dollar=True,
        )
        expected_contracts = len(strikes) * 2  # C + P

        # 4. 订阅期权（带重试）
        for attempt in range(3):
            self._subscribe_options(expiry, strikes, validate=True)
            actual = len(self.current_contracts)

            if actual >= expected_contracts * 0.9:  # 允许 10% 容差
                self._log('info',
                          f"预热完成: {actual}/{expected_contracts} 合约就绪, "
                          f"strikes={len(strikes)}, expiry={expiry}")
                return

            self._log('warning',
                      f"预热: 合约不足 {actual}/{expected_contracts}, 重试 ({attempt + 1}/3)")
            self.current_key = None  # 强制重新订阅
            self.ib.sleep(2)

        self._log('warning',
                  f"预热: 合约订阅未达预期 ({len(self.current_contracts)}/{expected_contracts}), "
                  "继续运行")

    def _update_warmup(self) -> None:
        """
        预热期更新：检查 spot 变化，必要时重新订阅 strikes

        在开盘前持续调用，确保 strikes 跟随盘前价格变化
        """
        # 获取当前 spot
        u_ticker = self.ib.ticker(self.underlying)
        spot = u_ticker.marketPrice() if u_ticker else None
        if not spot or np.isnan(spot) or spot <= 0:
            return

        # 检查 spot 是否变化超过 $1
        if self.last_good_spot is not None:
            change = abs(spot - self.last_good_spot)
            if change < 1.0:  # 变化 < $1，不需要更新
                return
            self._log(
                'info',
                f"预热: spot 变化 {self.last_good_spot:.2f} → {spot:.2f} (${change:.2f})",
            )

        self.last_good_spot = spot

        # 获取 expiry
        today_str = option_expiry_date_str()
        expiry, _ = pick_expiry(self.chain, today_str)
        if expiry is None:
            return

        # 重新计算 strikes
        new_strikes = select_strikes(
            self.chain.strikes, spot, self.strike_range,
            include_half_dollar=True,
        )

        # 检查 strikes 是否变化
        new_key = (expiry, tuple(new_strikes))
        if new_key == self.current_key:
            return  # strikes 没变

        self._log('info', f"预热: 重新订阅 strikes (spot={spot:.2f})")
        self.current_key = None  # 强制重新订阅
        self._subscribe_options(expiry, new_strikes, validate=True)

    def _subscribe_options(self, expiry: str, strikes: list[float],
                           validate: bool = False) -> None:
        """订阅期权行情

        Args:
            expiry: 到期日
            strikes: 行权价列表
            validate: 是否验证合约数量（预热时使用）
        """
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
            if (expiry, float(s), r) not in self._invalid_contract_cache
        ]
        expected = len(raw)
        if expected > 80:
            self._log(
                'warning',
                f'期权行情订阅 {expected} 行，可能超过 IB 行情额度；'
                '请按账户额度调整 strike_range/启用标的数',
            )
        if not raw:
            self.current_contracts = []
            self.current_key = key
            self._log('warning', f"订阅跳过：{expiry} 所有候选合约均已确认无效")
            return
        self.current_contracts = self.ib.qualifyContracts(*raw)
        actual = len(self.current_contracts)

        # 验证合约数量
        if validate and actual < expected:
            # 记录哪些合约验证失败
            qualified_keys = {(c.strike, c.right) for c in self.current_contracts}
            missing = [(s, r) for s in strikes for r in ['C', 'P']
                       if (s, r) not in qualified_keys]
            if missing:
                self._log('warning',
                          f"合约验证失败: {len(missing)} 个 - {missing[:5]}...")
                # 只在大部分合约都能 qualify 时缓存少数无效项。
                # 如果整批大面积失败，更可能是 IB/网络暂时故障，不能永久记黑。
                if actual >= expected * 0.5:
                    self._invalid_contract_cache.update(
                        (expiry, float(s), r) for s, r in missing
                    )

        # 订阅行情
        for c in self.current_contracts:
            self.ib.reqMktData(c, genericTickList=GENERIC_TICKS, snapshot=False)

        self.current_key = key
        self._log('info', f"订阅 {actual}/{expected} 个合约 "
                          f"expiry={expiry} strikes={len(strikes)}")
        self.ib.sleep(2)

    def _latest_market_data_marker(self) -> float | None:
        """返回 IB ticker 中最新一次真实行情事件时间。

        ib_insync 在订阅停止后仍会保留 Ticker 对象和旧值，因此不能把
        marketPrice()/Greeks 仍可读当成“行情有更新”。
        """
        if self.ib is None:
            return None
        contracts = ([self.underlying] if self.underlying is not None else [])
        contracts += list(self.current_contracts)
        markers: list[float] = []
        for contract in contracts:
            ticker = self.ib.ticker(contract)
            ticker_time = getattr(ticker, 'time', None) if ticker is not None else None
            if isinstance(ticker_time, datetime):
                markers.append(ticker_time.timestamp())
            elif isinstance(ticker_time, (int, float)) and ticker_time > 0:
                markers.append(float(ticker_time))
        return max(markers) if markers else None

    def _process_tick(self) -> bool:
        """处理一次 tick"""
        market_data_marker = self._latest_market_data_marker()
        if (market_data_marker is not None
                and market_data_marker == self._last_market_data_marker):
            return False

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
        today_str = option_expiry_date_str()
        expiry, is_true_0dte = pick_expiry(self.chain, today_str)
        if expiry is None:
            self._log('error', '无可用 expiry')
            return False

        # 选择 strikes：严格使用配置的百分比范围，并保留半美元行权价。
        # Hysteresis: spot 偏移超过 $1 才重选，避免边界抖动触发重新订阅
        all_strikes = sorted(float(s) for s in self.chain.strikes if s and float(s) > 0)
        need_reselect = (
            self._last_strike_spot is None
            or abs(spot - self._last_strike_spot) >= 1.0
            or self.current_key is None
            or self.current_key[0] != expiry
        )
        if need_reselect:
            strikes = select_strikes(
                self.chain.strikes, spot, self.strike_range,
                include_half_dollar=True,
            )
            self._subscribe_options(expiry, strikes)
            self._last_strike_spot = spot

        # 记录 expiry 变化
        if expiry != self.last_expiry_seen:
            self.last_expiry_seen = expiry
            if is_true_0dte:
                self._log('info', f"当前 expiry: {expiry} (真 0DTE)")
            else:
                self._log('warning',
                          f"⚠️ 今日无 0DTE 合约，回退到 {expiry} — GEX 语义与 0DTE 不同")

        # 计算 GEX（传入前一日 OI 用于计算 ΔOI）
        tickers = [self.ib.ticker(c) for c in self.current_contracts]
        result = calculate_gex(tickers, spot, prev_oi=self.prev_oi)

        if result is None:
            # 尾盘降级：15:00 ET 之后（最后一小时），缩窄到 ATM ±3 strike 重试
            now_et = et_now()
            if now_et.hour >= 15:
                narrow_below = [s for s in all_strikes if s <= spot][-3:]
                narrow_above = [s for s in all_strikes if s > spot][:3]
                narrow_strikes = sorted(set(narrow_below + narrow_above))
                self._subscribe_options(expiry, narrow_strikes)
                tickers = [self.ib.ticker(c) for c in self.current_contracts]
                result = calculate_gex(tickers, spot, oi_ready_threshold=0.0,
                                       prev_oi=self.prev_oi)
                if result is not None:
                    result.partial = True
                    self._log('info',
                              f'尾盘降级模式: {len(narrow_strikes)} strikes, '
                              f'partial GEX={result.total_gex:.0f}')
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

        quality_reasons = evaluate_tick_quality(
            result,
            len(self.current_contracts),
            min_contracts=self.quality_min_contracts,
            max_missing_ratio=self.quality_max_missing_ratio,
        )
        if quality_reasons:
            result.partial = True
        quality_key = tuple(quality_reasons)
        if quality_key != self._last_quality_reasons:
            if quality_reasons:
                self._log('warning', '快照质量降级: ' + '; '.join(quality_reasons))
            elif self._last_quality_reasons:
                self._log('info', '快照质量恢复正常')
            self._last_quality_reasons = quality_key

        # 收集今日 OI（用于收盘保存）
        for _, row in result.df.iterrows():
            strike = row['strike']
            if strike not in self.today_oi:
                self.today_oi[strike] = {
                    'call_oi': 0,
                    'put_oi': 0,
                    'expiry': row.get('expiry'),
                }
            if row['right'] == 'C':
                self.today_oi[strike]['call_oi'] = int(row['oi'])
            else:
                self.today_oi[strike]['put_oi'] = int(row['oi'])

        # Flip 平滑：只接纳 solver 确实找到的有限零点。
        raw_flip = result.gamma_flip
        flip_is_valid = raw_flip is not None and np.isfinite(raw_flip)
        if not flip_is_valid or abs(result.total_gex) < 5e8:
            # GEX < 0.5B: flip 不可靠，沿用上次值
            if self._flip_buffer:
                result.gamma_flip = median(self._flip_buffer)
                result.gamma_flip_method = 'smoothed_previous'
            else:
                result.gamma_flip = None
            # 不把不可靠的值放进 buffer
        else:
            self._flip_buffer.append(raw_flip)
            result.gamma_flip = median(self._flip_buffer)

        # 计算 regime 特征
        try:
            history, _ = self.state.get_history_for_resample()
            _, regime_code, regime_tags = compute_realtime_features(
                result.df, spot, history
            )
        except Exception as e:
            log.debug(f"Regime 计算失败: {e}")
            regime_code, regime_tags = None, None

        # 计算 skew 指标
        rr_25 = skew_slope = rr_25_zscore = skew_signal = None
        drr_25 = drr_25_zscore = skew_alert_level = skew_alert_score = None
        rr_10 = butterfly_25 = put_25_iv = call_25_iv = None
        put_25_richness = call_25_richness = None
        wing_curvature_asymmetry = None
        skew_node_context = {}
        try:
            skew_snap = compute_skew(tickers, spot)
            skew_snap = self.skew_tracker.update(skew_snap, result.positive_gamma)
            if skew_snap is not None:
                rr_25 = skew_snap.rr_25
                skew_slope = skew_snap.skew_slope
                rr_25_zscore = skew_snap.rr_25_zscore
                skew_signal = skew_snap.signal
                drr_25 = skew_snap.drr_25
                drr_25_zscore = skew_snap.drr_25_zscore
                skew_alert_level = skew_snap.alert_level
                skew_alert_score = skew_snap.alert_score
                rr_10 = skew_snap.rr_10
                butterfly_25 = skew_snap.butterfly_25
                put_25_iv = skew_snap.put_25_iv
                call_25_iv = skew_snap.call_25_iv
                put_25_richness = skew_snap.put_25_richness
                call_25_richness = skew_snap.call_25_richness
                wing_curvature_asymmetry = skew_snap.wing_curvature_asymmetry
                for prefix in ("put_25", "call_25"):
                    for suffix in (
                        "strike", "delta", "bid", "ask", "mid",
                        "volume", "open_interest",
                    ):
                        field = f"{prefix}_{suffix}"
                        skew_node_context[field] = getattr(skew_snap, field, None)
        except Exception as e:
            log.debug(f"Skew 计算失败: {e}")

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
            max_pain=result.max_pain,
            regime_code=regime_code,
            regime_tags=regime_tags,
            rr_25=rr_25,
            skew_slope=skew_slope,
            rr_25_zscore=rr_25_zscore,
            skew_signal=skew_signal,
            partial=getattr(result, 'partial', False),
            quality_reasons=quality_reasons,
            volume_gamma=result.volume_gamma,
            gamma_flip_method=result.gamma_flip_method,
            gross_gex=result.gross_gex,
            net_gex_ratio=result.net_gex_ratio,
            gross_volume_gamma=result.gross_volume_gamma,
        )

        if self._turning_point_shadow is not None:
            try:
                shadow_row = self._turning_point_shadow.on_update(
                    now=et_now(),
                    input_provider=lambda: self.state.get_turning_point_inputs(
                        self._intraday_turning_point_shadow_config.lookback_minutes
                    ),
                )
                if shadow_row is not None:
                    self._log(
                        'info',
                        'Turning-point shadow '
                        f"{shadow_row['watch_level']} "
                        f"{shadow_row['setup_direction']} "
                        f"bias={shadow_row['score_bias']} "
                        f"event={shadow_row['event_id']}",
                    )
            except Exception as e:
                self._log('warning', f'Turning-point shadow failed: {e}')

        if self._vrp_monitor is not None:
            try:
                vrp_state = dict(self.state.get_snapshot())
                vrp_state.update({
                    'rr_25': rr_25,
                    'skew_slope': skew_slope,
                    'rr_25_zscore': rr_25_zscore,
                    'skew_signal': skew_signal,
                    'drr_25': drr_25,
                    'drr_25_zscore': drr_25_zscore,
                    'skew_alert_level': skew_alert_level,
                    'skew_alert_score': skew_alert_score,
                    'rr_10': rr_10,
                    'butterfly_25': butterfly_25,
                    'put_25_iv': put_25_iv,
                    'call_25_iv': call_25_iv,
                    'put_25_richness': put_25_richness,
                    'call_25_richness': call_25_richness,
                    'wing_curvature_asymmetry': wing_curvature_asymmetry,
                })
                vrp_state.update(skew_node_context)
                self._vrp_monitor.on_gex_update(
                    self.ib,
                    self.current_contracts,
                    now=et_now(),
                    spot=spot,
                    expiry=expiry,
                    is_true_0dte=is_true_0dte,
                    gex_state=vrp_state,
                    # 只在固定采样点调用，避免每个 3 秒 tick 都复制状态。
                    intraday_bars_provider=lambda: self.state.get_persist_data()[1],
                    market_context_provider=lambda: market_vol_context(
                        self.ib, et_now(), self._intraday_vrp_config.vix_cache_seconds
                    ),
                    term_structure_provider=lambda: standardized_term_structure_context(
                        self.ib, self.symbol, et_now(), spot,
                        self._intraday_vrp_config.standardized_iv_dtes,
                        self._intraday_vrp_config.term_structure_cache_seconds,
                    ),
                    ib_port=self.ib_port,
                )
            except Exception as e:
                self._log('warning', f'VRP observation failed: {e}')

        # 缓冲到 DB
        if self.db_storage is not None:
            self.db_storage.buffer_snapshot({
                'symbol': self.symbol,
                'ts': et_now(),
                'spot': spot,
                'total_gex': result.total_gex,
                'call_gex': result.call_gex,
                'put_gex': result.put_gex,
                'flip': result.gamma_flip,
                'call_wall': result.call_wall,
                'put_wall': result.put_wall,
                'max_pain': result.max_pain,
                'atm_iv_pct': result.atm_iv_pct,
                'positive_gamma': result.positive_gamma,
                'regime_code': regime_code,
                'rr_25': rr_25,
                'skew_slope': skew_slope,
                'rr_25_zscore': rr_25_zscore,
                'skew_signal': skew_signal,
                'partial': getattr(result, 'partial', False),
            })

        # 定期持久化
        if time.time() - self.last_persist > self.timing.persist_interval_sec:
            hist, ohlc, strikes = self.state.get_persist_data()
            self.storage.persist_async(self.symbol, hist, ohlc, strikes)
            # DB flush
            if self.db_storage is not None:
                try:
                    self.db_storage.flush()
                except Exception as e:
                    log.warning(f"DB flush error: {e}")
            self.last_persist = time.time()

        self._last_market_data_marker = market_data_marker
        return True

    def run(self) -> None:
        """主循环（在独立线程调用）"""
        asyncio.set_event_loop(asyncio.new_event_loop())

        while self._running:
            now = et_now()
            market_open = is_market_open(now)
            extended = self.extended_hours and is_extended_hours(now)
            should_conn = should_connect(now, warmup_minutes=5,
                                         include_extended=self.extended_hours)

            # 非连接时段（收盘后且不在预热期）
            if not should_conn:
                self.state.set_status(
                    market_open=False,
                    updated=f"非交易时段 ({now.strftime('%H:%M ET')})"
                )

                if self.ib is not None and self.ib.isConnected():
                    # 盘后落盘
                    today = trading_date_str(now)
                    try:
                        hist, ohlc, strikes = self.state.get_persist_data()
                        # 收盘路径允许阻塞：先等周期任务，再同步追加最后一批，
                        # 防止 persist_async 正忙时跳过收盘快照。
                        self.storage.wait_for_persist(self.symbol)
                        self.storage.persist_sync(self.symbol, hist, ohlc, strikes)
                    except Exception as e:
                        self._log('error', f"盘后 persist 失败: {e}")

                    if self._vrp_monitor is not None and now.hour >= 16:
                        try:
                            self._vrp_monitor.settle_date(today)
                        except Exception as e:
                            self._log('error', f'VRP settlement failed: {e}')

                    # 保存今日 OI 快照（用于明天计算 ΔOI）
                    try:
                        if self.today_oi:
                            self.storage.save_oi_snapshot(self.symbol, today, self.today_oi)
                            self._log('info', f"Saved OI snapshot: {len(self.today_oi)} strikes")
                            # 清空，否则跨天长跑时次日快照混入今天的 stale strikes
                            self.today_oi = {}
                    except Exception as e:
                        self._log('error', f"保存 OI 快照失败: {e}")

                    if self.db_storage is not None:
                        try:
                            self.db_storage.flush()
                        except Exception as e:
                            self._log('error', f"盘后 DB flush 失败: {e}")

                    # GTH 早盘 09:25 的短暂停顿不能提前生成“空白日”报告。
                    if now.hour >= 16:
                        try:
                            report = self.storage.finalize_day(self.symbol, today)
                            level = 'info' if report.status == 'good' else 'warning'
                            self._log(
                                level,
                                f"日终数据质量 {report.status.upper()} "
                                f"score={report.score} coverage={report.gex_coverage:.1%}; "
                                f"{'; '.join(report.reasons[:4]) or '无异常'}",
                            )
                        except Exception as e:
                            self._log('error', f"生成日终质量报告失败: {e}")

                    # 重置每日 skew surface 标志
                    self._skew_surface_captured_today = False

                    # 断开连接
                    try:
                        self._detach_ib_handlers(self.ib)
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

                # 等待下一个数据时段。延伸标的不能只算常规开盘，
                # 否则 16:00 后的 30min cap 会越过 16:15 Curb 开始。
                try:
                    if self.extended_hours:
                        sleep_sec = max(
                            seconds_until_next_session(include_extended=True) - 5,
                            5,
                        )
                    else:
                        sleep_sec = max(seconds_until_next_open() - 60, 30)
                    self._log('info', f"Market closed, next session check in {sleep_sec:.0f}s")
                    time.sleep(min(sleep_sec, self.timing.max_sleep_sec))
                except RuntimeError as e:
                    self._log('error', f"{e}; retrying in {self.timing.market_closed_check_sec}s")
                    time.sleep(self.timing.market_closed_check_sec)
                continue

            # 确保连接（在预热期或交易时段）
            if self.ib is None or not self.ib.isConnected():
                try:
                    self._connect()
                except Exception as e:
                    self.state.set_status(connected=False)
                    self._log('error', f"IB connect failed: {e}")
                    time.sleep(self.timing.reconnect_delay_sec)
                    continue

            # 延伸时段（GTH 20:15-9:25 / Curb 16:15-17:00）：直接跑 tick
            if extended and not market_open:
                if self._reconnect_requested_reason:
                    self._force_reconnect(self._reconnect_requested_reason)
                    time.sleep(self.timing.reconnect_delay_sec)
                    continue

                self.state.set_status(
                    # UI 的 market_open 实际表示“当前有 live data session”。
                    # 延伸时段若继续设 False，Dash 会主动返回休市空图。
                    market_open=True,
                    updated=f"延伸时段 ({now.strftime('%H:%M:%S ET')})"
                )
                try:
                    if (self._process_tick()
                            and self._last_market_data_marker is not None):
                        self._last_success_ts = time.time()
                except Exception as e:
                    self._log('error', f"Extended hours tick error: {e}")
                    if self.ib is not None and not self.ib.isConnected():
                        self.state.set_status(connected=False)
                stale_reason = self._stale_reconnect_reason()
                if stale_reason:
                    self._force_reconnect(stale_reason)
                    time.sleep(self.timing.reconnect_delay_sec)
                    continue
                self._sleep(self.timing.tick_interval_sec)
                continue

            # 预热期：已连接但市场未开，持续更新 strikes
            if not market_open:
                self.state.set_status(
                    market_open=False,
                    updated=f"预热中，等待开盘 ({now.strftime('%H:%M:%S ET')})"
                )
                try:
                    self._update_warmup()
                except Exception as e:
                    self._log('warning', f"预热更新失败: {e}")
                self._sleep(1)
                continue

            # 主循环：市场已开
            if self._reconnect_requested_reason:
                self._force_reconnect(self._reconnect_requested_reason)
                time.sleep(self.timing.reconnect_delay_sec)
                continue

            try:
                if (self._process_tick()
                        and self._last_market_data_marker is not None):
                    self._last_success_ts = time.time()
            except Exception as e:
                self._log('error', f"Main loop error: {e}")
                if self.ib is not None and not self.ib.isConnected():
                    self.state.set_status(connected=False)
            stale_reason = self._stale_reconnect_reason()
            if stale_reason:
                self._force_reconnect(stale_reason)
                time.sleep(self.timing.reconnect_delay_sec)
                continue

            # 15:30 ET 自动采集 multi-tenor skew surface
            self._maybe_capture_skew_surface(now)

            self._sleep(self.timing.tick_interval_sec)

    def _maybe_capture_skew_surface(self, now) -> None:
        """
        15:30 ET 自动执行（每日一次）:
          1. 采集 multi-tenor skew surface
          2. 生成 hedge signal
          3. 如果 hedge_enabled: 自动下单
        """
        if self._skew_surface_captured_today:
            return
        if now.hour != 15 or now.minute < 30:
            return
        if self.ib is None or not self.ib.isConnected():
            return

        self._skew_surface_captured_today = True
        self._log('info', '=== 15:30 Daily Hedge Routine ===')

        # Step 1: 采集 skew surface
        surface = None
        try:
            surface = collect_skew_surface(
                self.ib, self.symbol, self.trading_class, self.sec_type,
                spot_override=self.state.get_snapshot().get('spot'),
                # Multi-tenor surface must rebuild and merge all exchange chain
                # fragments; the GEX worker chain can be a sparse SMART subset.
                chain_override=None,
            )
            if surface is not None:
                self._skew_surface_storage.save_surface(surface.to_records())
                self._log('info',
                          f'Skew surface saved: {len(surface.tenors)} tenors, '
                          f'term_spread_rr25={surface.term_spread_rr25}')
            else:
                self._log('warning', 'Skew surface collection returned None')
                return
        except Exception as e:
            self._log('error', f'Skew surface capture failed: {e}')
            self._skew_surface_captured_today = False
            return

        # Step 2: 生成 hedge signal
        try:
            history_df = self._skew_surface_storage.load_surface_history(
                self.symbol, n_days=90
            )

            # GEX regime from current state
            snapshot = self.state.get_snapshot()
            total_gex = snapshot.get('total_gex', 0)
            if total_gex > 0:
                gex_regime = 'positive'
            elif total_gex < 0:
                gex_regime = 'negative'
            else:
                gex_regime = 'neutral'

            last_signal = self._skew_surface_storage.get_last_signal(self.symbol)

            # 采集宏观快照（VIX/MOVE/SOFR-OIS）
            try:
                macro = fetch_macro_snapshot(self.ib)
            except Exception as e:
                self._log('warning', f'Macro snapshot failed: {e}')
                macro = None

            signal = generate_hedge_signal(
                surface=surface,
                history_df=history_df,
                gex_regime=gex_regime,
                last_signal=last_signal,
                macro=macro,
            )

            # 存储信号
            self._skew_surface_storage.save_hedge_signal(signal.to_dict())

            # 日志输出
            self._log('info',
                      f'Hedge signal: {signal.action} '
                      f'(urgency={signal.urgency:.0%}, '
                      f'skew={signal.skew_cheapness:.0f}pct, '
                      f'gex={gex_regime}, '
                      f'struct={signal.recommended_structure}, '
                      f'tenor={signal.recommended_tenor})')

            # 输出到 stdout（log 文件也能看到）
            log.info(f'\n{format_recommendation(signal)}')

        except Exception as e:
            self._log('error', f'Hedge signal generation failed: {e}')
            return

        # Step 3: 自动执行（如果启用）
        if not self._hedge_enabled:
            self._log('info', 'Hedge execution disabled (use --hedge to enable)')
            return

        if signal.action not in ('HEDGE_NOW', 'HEDGE_SPREAD', 'REDUCE'):
            self._log('info', f'Signal is {signal.action}, no trade needed')
            return

        try:
            from .config import DatabaseConfig
            db_config = DatabaseConfig()

            executor = HedgeExecutor(
                ib=self.ib,
                db_config=db_config,
                qty=self._hedge_qty,
                dry_run=self._hedge_dry_run,
            )

            if signal.action == 'REDUCE':
                results = executor.execute_reduce(self.symbol)
                for r in results:
                    self._log('info', f"Closed trade #{r['trade_id']}")
            else:
                record = executor.execute(signal, surface.spot)
                if record:
                    self._log('info', f'Trade executed: {record.status}')
                    log.info(f'\n{format_trade_result(record)}')
                else:
                    self._log('warning', 'Trade execution returned None')

            executor.shutdown()

        except Exception as e:
            self._log('error', f'Hedge execution failed: {e}')

    def stop(self) -> None:
        """停止 worker"""
        self._running = False
        if self.ib is not None and self.ib.isConnected():
            try:
                self._detach_ib_handlers(self.ib)
                self.ib.disconnect()
            except Exception:
                pass
