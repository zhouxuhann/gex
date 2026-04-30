"""IB 连接与数据采集模块"""
import asyncio
import logging
import time
from collections import deque
from statistics import median

import numpy as np
from ib_insync import IB, Stock, Index, Option

from .config import TimingConfig
from .ddput_signal import DdputConfig, DdputSignalDetector
from .email_notifier import EmailNotifier
from .gex_calc import calculate_gex, pick_expiry
from .strike_selector import select_strikes
from .features import compute_realtime_features
from .skew import compute_skew, SkewTracker
from .skew_surface import collect_skew_surface
from .hedge_signal import generate_hedge_signal, format_recommendation
from .hedge_executor import HedgeExecutor, format_trade_result
from .macro import fetch_macro_snapshot
from .db_storage import GEXDBStorage
from .state import StateManager
from .storage import StorageManager, SkewSurfaceStorage
from .time_utils import (
    et_now, trading_date_str, is_market_open, should_connect,
    seconds_until_next_open, format_countdown_to_open,
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
        db_storage: GEXDBStorage | None = None,
        hedge_enabled: bool = False,
        hedge_dry_run: bool = False,
        hedge_qty: int = 1,
        ddput_signal_config: DdputConfig | None = None,
        ddput_email_notifier: EmailNotifier | None = None,
        ib_error_watcher=None,
    ):
        self.symbol = symbol
        self.trading_class = trading_class
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

        self.ib: IB | None = None
        self.underlying = None
        self.chain = None
        self.current_key: tuple | None = None
        self.current_contracts: list = []
        self.last_persist: float = 0
        self.last_expiry_seen: str | None = None
        self.last_good_spot: float | None = None
        self._running: bool = True

        # ΔOI 相关
        self.prev_oi: dict[float, dict] | None = None  # 前一交易日 OI
        self.today_oi: dict[float, dict] = {}  # 今日 OI（用于收盘保存）
        self._load_prev_oi()

        # Strike 选择 hysteresis：spot 偏移超过此值才重选 strike
        self._last_strike_spot: float | None = None

        # Stale spot 检测：连续 N 个 tick spot 不变时跳过写入
        self._stale_spot_count: int = 0
        self._stale_spot_threshold: int = 10  # 连续 10 tick (~30s) 不变即 stale

        # IB error watcher (10197 手机登录冲突等关键错误监听)
        self._ib_error_watcher = ib_error_watcher

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

        # ddput 实时信号检测器（L2 观察提醒 + 自动记录）
        self._ddput_detector: DdputSignalDetector | None = None
        if ddput_signal_config is not None and ddput_signal_config.enabled:
            self._ddput_detector = DdputSignalDetector(
                symbol=self.symbol,
                config=ddput_signal_config,
                data_dir=self.storage.data_dir,
                email_notifier=ddput_email_notifier,
            )
            email_status = ('email on' if ddput_email_notifier and
                            ddput_email_notifier.config.enabled else 'email off')
            log.info(f'[{self.symbol}] ddput signal detector enabled '
                     f'(z_mild={ddput_signal_config.z_mild}, '
                     f'z_strong={ddput_signal_config.z_strong}, '
                     f'window {ddput_signal_config.min_time_et:.1f}-'
                     f'{ddput_signal_config.max_time_et:.1f} ET, '
                     f'{email_status})')

    def _load_prev_oi(self) -> None:
        """加载前一交易日的 OI 快照"""
        today = trading_date_str()
        prev_date = self.storage.get_previous_trading_day(today)
        if prev_date:
            self.prev_oi = self.storage.load_oi_snapshot(self.symbol, prev_date)
            if self.prev_oi:
                log.info(f"[{self.symbol}] Loaded prev OI from {prev_date}: {len(self.prev_oi)} strikes")
            else:
                log.info(f"[{self.symbol}] No prev OI found for {prev_date}")
        else:
            log.info(f"[{self.symbol}] No previous trading day OI snapshot found")

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

        # 挂 IB 关键 error 监听 (10197 等)
        if self._ib_error_watcher is not None:
            self._ib_error_watcher.attach(self.ib)

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
        # 同时按 exchange=SMART 和 trading_class 过滤
        # SPX 会返回多个 chain：SMART+SPX(月度)、SMART+SPXW(周度/0DTE)、CBOE+... 等
        # 只按 exchange 过滤会命中第一个（SMART+SPX 月度），导致拿不到 0DTE 和 $5 间距 strike
        self.chain = next(
            (c for c in chains
             if c.exchange == 'SMART' and c.tradingClass == self.trading_class),
            None,
        )
        if self.chain is None:
            available = [(c.exchange, c.tradingClass) for c in chains]
            raise RuntimeError(
                f"No SMART option chain for {self.symbol} "
                f"with trading_class={self.trading_class}. "
                f"Available: {available}"
            )

        # 订阅 underlying 行情
        self.ib.reqMktData(self.underlying, genericTickList='', snapshot=False)

        self.current_key = None
        self.current_contracts = []
        self.last_good_spot = None

        self.state.set_status(connected=True, market_open=True)
        self._log('info', f"IB connected (host={self.ib_host}, port={self.ib_port})")
        self.ib.sleep(1)

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
        today_str = trading_date_str()
        expiry, _ = pick_expiry(self.chain, today_str)
        if expiry is None:
            self._log('warning', "预热: 无可用 expiry")
            return

        # 3. 选择 strikes（按 config.strike_range 覆盖 ±N%）
        strikes = select_strikes(
            self.chain.strikes, spot, self.strike_range,
            include_half_dollar=False,
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
            self._log('info', f"预热: spot 变化 {self.last_good_spot:.2f} → {spot:.2f} (${change:.2f})")

        self.last_good_spot = spot

        # 获取 expiry
        today_str = trading_date_str()
        expiry, _ = pick_expiry(self.chain, today_str)
        if expiry is None:
            return

        # 重新计算 strikes
        new_strikes = select_strikes(
            self.chain.strikes, spot, self.strike_range,
            include_half_dollar=False,
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
        ]
        expected = len(raw)
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

        # 订阅行情
        for c in self.current_contracts:
            self.ib.reqMktData(c, genericTickList=GENERIC_TICKS, snapshot=False)

        self.current_key = key
        self._log('info', f"订阅 {actual}/{expected} 个合约 "
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
        # Stale spot 检测：IB 行情断开时 spot 会冻结，避免写入无效快照
        if self.last_good_spot is not None and spot == self.last_good_spot:
            self._stale_spot_count += 1
            if self._stale_spot_count >= self._stale_spot_threshold:
                if self._stale_spot_count == self._stale_spot_threshold:
                    self._log('warning',
                              f'Spot 冻结 {self._stale_spot_count} tick '
                              f'({spot:.2f}) — 疑似行情中断，暂停写入')
                return False
        else:
            if self._stale_spot_count >= self._stale_spot_threshold:
                self._log('info',
                          f'Spot 恢复更新 ({self.last_good_spot:.2f} → {spot:.2f})，'
                          f'跳过了 {self._stale_spot_count} tick')
            self._stale_spot_count = 0

        self.last_good_spot = spot

        # 选择 expiry
        today_str = trading_date_str()
        expiry, is_true_0dte = pick_expiry(self.chain, today_str)
        if expiry is None:
            self._log('error', '无可用 expiry')
            return False

        # 选择 strikes：按 config.strike_range 覆盖 ±N%
        # Hysteresis: spot 漂移超过 max($1, spot*0.002) 才重选（保证阈值
        # 随 spot 缩放，避免高价标的的最外层 strike 未跟上 spot 漂移）
        hysteresis = max(1.0, spot * 0.002)
        need_reselect = (
            self._last_strike_spot is None
            or abs(spot - self._last_strike_spot) >= hysteresis
        )
        if need_reselect:
            strikes = select_strikes(
                self.chain.strikes, spot, self.strike_range,
                include_half_dollar=False,
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
            # （沿用 select_strikes 的 min_strikes_each_side floor 机制：
            #  传极小 strike_range 强制走 floor 路径，拿到 ATM 上下最近各 3 个）
            now_et = et_now()
            if now_et.hour >= 15:
                narrow_strikes = select_strikes(
                    self.chain.strikes, spot, strike_range=1e-6,
                    include_half_dollar=False,
                    min_strikes_each_side=3,
                )
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

        # 收集今日 OI（用于收盘保存）
        for _, row in result.df.iterrows():
            strike = row['strike']
            if strike not in self.today_oi:
                self.today_oi[strike] = {'call_oi': 0, 'put_oi': 0}
            if row['right'] == 'C':
                self.today_oi[strike]['call_oi'] = int(row['oi'])
            else:
                self.today_oi[strike]['put_oi'] = int(row['oi'])

        # Flip 平滑：滑动中位数 + 低 GEX 锁定
        # 新版 _calculate_gamma_flip 可能返回 None（无 cumsum 穿越时）
        raw_flip = result.gamma_flip
        valid_buffer = [f for f in self._flip_buffer if f is not None]
        if raw_flip is None:
            # 算法诚实地说无 flip（dealer 在采样窗口内全程同号），
            # 沿用 buffer 里最近的有效值；若 buffer 也空，保持 None
            result.gamma_flip = median(valid_buffer) if valid_buffer else None
        elif abs(result.total_gex) < 5e8:
            # GEX < 0.5B: flip 不可靠，沿用上次值
            if valid_buffer:
                result.gamma_flip = median(valid_buffer)
            # 不把不可靠的值放进 buffer
        else:
            self._flip_buffer.append(raw_flip)
            valid_buffer = [f for f in self._flip_buffer if f is not None]
            result.gamma_flip = median(valid_buffer) if valid_buffer else raw_flip

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
        try:
            skew_snap = compute_skew(tickers, spot)
            skew_snap = self.skew_tracker.update(skew_snap, result.positive_gamma)
            if skew_snap is not None:
                rr_25 = skew_snap.rr_25
                skew_slope = skew_snap.skew_slope
                rr_25_zscore = skew_snap.rr_25_zscore
                skew_signal = skew_snap.signal
        except Exception as e:
            log.debug(f"Skew 计算失败: {e}")

        # ddput 实时信号（L2 观察级，失败不影响主采集但要能看到错误）
        if self._ddput_detector is not None:
            try:
                self._ddput_detector.update(
                    ts=et_now(),
                    put_gex=result.put_gex,
                    spot=spot,
                )
            except Exception as e:
                # 用 warning 而不是 debug，避免重蹈"17 小时 silent fail"覆辙
                log.warning(f'[{self.symbol}] ddput signal update failed: {e}')

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
        )

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

        return True

    def run(self) -> None:
        """主循环（在独立线程调用）"""
        asyncio.set_event_loop(asyncio.new_event_loop())

        while self._running:
            now = et_now()
            market_open = is_market_open(now)
            should_conn = should_connect(now, warmup_minutes=5)

            # 非连接时段（收盘后且不在预热期）
            # UI 端会每 4s 自己重算 countdown（见 callbacks.py），
            # 这里只维护 market_open 标志位即可，倒计时不在这里算
            if not should_conn:
                self.state.set_status(
                    market_open=False,
                    updated=format_countdown_to_open(now)
                )

                if self.ib is not None and self.ib.isConnected():
                    # 盘后落盘
                    try:
                        hist, ohlc, strikes = self.state.get_persist_data()
                        self.storage.persist_async(self.symbol, hist, ohlc, strikes)
                    except Exception as e:
                        self._log('error', f"盘后 persist 失败: {e}")

                    # 保存今日 OI 快照（用于明天计算 ΔOI）
                    try:
                        if self.today_oi:
                            today = trading_date_str()
                            self.storage.save_oi_snapshot(self.symbol, today, self.today_oi)
                            self._log('info', f"Saved OI snapshot: {len(self.today_oi)} strikes")
                    except Exception as e:
                        self._log('error', f"保存 OI 快照失败: {e}")

                    # 重置每日 skew surface 标志
                    self._skew_surface_captured_today = False

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

            # 确保连接（在预热期或交易时段）
            if self.ib is None or not self.ib.isConnected():
                try:
                    self._connect()
                except Exception as e:
                    self.state.set_status(connected=False)
                    self._log('error', f"IB connect failed: {e}")
                    time.sleep(self.timing.reconnect_delay_sec)
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
            try:
                self._process_tick()
            except Exception as e:
                self._log('error', f"Main loop error: {e}")
                if self.ib is not None and not self.ib.isConnected():
                    self.state.set_status(connected=False)

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
                self.ib, self.symbol, self.trading_class, self.sec_type
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
                self.symbol, n_days=20
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
                self.ib.disconnect()
            except Exception:
                pass
