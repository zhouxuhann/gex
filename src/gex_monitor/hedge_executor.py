"""
对冲交易执行器

职责:
  1. 根据 HedgeSignal 选择具体合约（expiry + strike by delta）
  2. 在 IB Paper 账户下单
  3. 记录交易到 PostgreSQL
  4. 跟踪持仓，支持 REDUCE 平仓

安全设计:
  - 仅限 Paper 账户（检查 accountType）
  - 最大持仓上限
  - dry-run 模式
"""
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from ib_insync import ComboLeg, Contract, IB, LimitOrder, Option, Stock

from .config import DatabaseConfig
from .hedge_signal import HedgeSignal
from .time_utils import et_now, ET

log = logging.getLogger(__name__)

# 默认参数
DEFAULT_TARGET_DELTA = 0.25       # -0.25Δ put
SPREAD_SHORT_DELTA = 0.10         # -0.10Δ put (short leg)
DEFAULT_QTY = 1                   # 1 合约
MAX_OPEN_POSITIONS = 5            # 最大同时持有的对冲头寸
ORDER_TIMEOUT_SEC = 30            # 订单超时
ORDER_RECONCILE_SEC = 5           # 取消后继续等待服务器成交回报
TIF_PRESET_WARNING_CODE = 10349
MAX_DELTA_ERROR = 0.06
CONTRACT_SEARCH_RANGE_PCT = 0.20
MAX_CONTRACT_CANDIDATES = 61
QUOTE_BATCH_SIZE = 30


@dataclass
class TradeRecord:
    """交易记录"""
    signal: HedgeSignal
    legs: list[dict] = field(default_factory=list)
    entry_ts: datetime | None = None
    entry_spot: float | None = None
    entry_cost: float = 0.0
    status: str = 'PENDING'

    def to_db_dict(self) -> dict:
        """转为 DB 写入格式"""
        s = self.signal
        return {
            'signal_ts': s.ts,
            'symbol': s.symbol,
            'action': s.action,
            'structure': s.recommended_structure,
            'entry_ts': self.entry_ts,
            'entry_spot': self.entry_spot,
            'entry_cost': self.entry_cost,
            'legs': json.dumps(self.legs),
            'status': self.status,
            'urgency': s.urgency,
            'skew_cheapness': s.skew_cheapness,
            'gex_regime': s.gex_regime,
            'term_structure': s.term_structure,
            'rr_25_at_entry': None,  # filled by caller if available
        }


class HedgeExecutor:
    """对冲交易执行器"""

    def __init__(
        self,
        ib: IB,
        db_config: DatabaseConfig | None = None,
        qty: int = DEFAULT_QTY,
        max_positions: int = MAX_OPEN_POSITIONS,
        dry_run: bool = False,
    ):
        self.ib = ib
        self.qty = qty
        self.max_positions = max_positions
        self.dry_run = dry_run

        # DB 连接（记录用）
        self._conn = None
        if db_config and not dry_run:
            self._init_db(db_config)

    def _init_db(self, config: DatabaseConfig) -> None:
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=config.host, port=config.port,
                dbname=config.dbname, user=config.user,
                password=config.password, connect_timeout=5,
            )
            self._conn.autocommit = True
        except Exception as e:
            log.warning(f"Executor DB connect failed: {e}")

    def verify_paper_account(self) -> bool:
        """验证是 Paper 账户"""
        accounts = self.ib.managedAccounts()
        if not accounts:
            log.error("No accounts found")
            return False
        # Paper 账户通常以 'DU' 开头
        for acc in accounts:
            if acc.startswith('DU'):
                log.info(f"Paper account verified: {acc}")
                return True
        log.error(f"Not a paper account! Accounts: {accounts}")
        return False

    def get_open_positions(self, symbol: str) -> int:
        """获取当前 symbol 的对冲持仓数"""
        if self._conn is None:
            return 0
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM hedge_trades WHERE symbol=%s AND status='OPEN'",
                    [symbol]
                )
                return cur.fetchone()[0]
        except Exception:
            return 0

    def execute(self, signal: HedgeSignal, spot: float) -> TradeRecord | None:
        """
        执行对冲交易

        Args:
            signal: 对冲信号
            spot: 当前 spot 价格

        Returns:
            TradeRecord 或 None（跳过/失败）
        """
        if signal.action not in ('HEDGE_NOW', 'HEDGE_SPREAD'):
            log.info(f"Signal is {signal.action}, no trade needed")
            return None

        # 安全检查
        if not self.dry_run and not self.verify_paper_account():
            log.error("BLOCKED: Not a paper account")
            return None
        if not self.dry_run and self.qty != 1:
            log.error("BLOCKED: atomic hedge executor currently supports qty=1 only")
            return None

        # 持仓上限检查
        open_count = self.get_open_positions(signal.symbol)
        if open_count >= self.max_positions:
            log.warning(f"[{signal.symbol}] Max positions reached ({open_count}/{self.max_positions})")
            return None

        # 解析 tenor 推荐中的 expiry
        expiry = self._parse_expiry(signal)
        if not expiry:
            log.error("Cannot determine expiry from signal")
            return None

        # 选择合约
        if signal.recommended_structure == 'outright_put':
            return self._execute_outright_put(signal, spot, expiry)
        elif signal.recommended_structure == 'put_spread':
            return self._execute_put_spread(signal, spot, expiry)
        else:
            log.warning(f"Unsupported structure: {signal.recommended_structure}")
            return None

    def execute_reduce(self, symbol: str) -> list[dict]:
        """
        平仓所有 OPEN 的对冲头寸

        Returns:
            平仓记录列表
        """
        if self._conn is None:
            log.warning("No DB connection, cannot track positions to reduce")
            return []

        # 获取 OPEN 头寸
        try:
            import pandas as pd
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT id, legs FROM hedge_trades WHERE symbol=%s AND status='OPEN'",
                    [symbol]
                )
                rows = cur.fetchall()
        except Exception as e:
            log.error(f"Failed to query open positions: {e}")
            return []

        if not rows:
            log.info(f"[{symbol}] No open positions to reduce")
            return []

        results = []
        for trade_id, legs_json in rows:
            legs = json.loads(legs_json) if legs_json else []
            closed = self._close_legs(symbol, legs)

            if closed or self.dry_run:
                # 更新 DB
                self._update_trade_closed(trade_id, closed)
                results.append({'trade_id': trade_id, 'legs_closed': len(legs)})

        return results

    # ==================== 内部方法 ====================

    def _parse_expiry(self, signal: HedgeSignal) -> str | None:
        """从 tenor 推荐中解析 expiry"""
        # 格式: "45D (exp 20260528)" 或 "30-45D"
        tenor = signal.recommended_tenor
        if 'exp ' in tenor:
            return tenor.split('exp ')[-1].rstrip(')')
        # fallback: 用 IB 链找最接近的
        return None

    def _find_contract_by_delta(
        self, symbol: str, expiry: str, target_delta: float, right: str = 'P',
        spot_override: float | None = None,
    ) -> tuple | None:
        """
        找到最接近目标 delta 的合约

        Returns:
            (Option contract, actual_delta, iv) 或 None
        """
        # 获取 underlying
        underlying = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(underlying)

        # 获取链
        chains = self.ib.reqSecDefOptParams(
            underlying.symbol, '', underlying.secType, underlying.conId
        )
        chain = (
            next((c for c in chains if c.exchange == 'SMART'
                  and c.tradingClass == symbol), None)
            or next((c for c in chains if c.tradingClass == symbol), None)
        )
        if not chain:
            return None

        # 获取 spot
        spot = spot_override
        if spot is None:
            snapshots = self.ib.reqTickers(underlying)
            u_ticker = snapshots[0] if snapshots else None
            spot = u_ticker.marketPrice() if u_ticker else None

        if not spot or np.isnan(spot):
            return None

        # 选 strikes
        all_strikes = sorted({float(s) for s in chain.strikes if float(s) > 0})
        lo = spot * (1 - CONTRACT_SEARCH_RANGE_PCT)
        hi = spot * (1 + CONTRACT_SEARCH_RANGE_PCT)
        strikes = [s for s in all_strikes if lo <= s <= hi]
        if len(strikes) > MAX_CONTRACT_CANDIDATES:
            indices = np.linspace(0, len(strikes) - 1, MAX_CONTRACT_CANDIDATES)
            strikes = sorted({strikes[int(round(i))] for i in indices})

        # 创建合约并请求数据
        contracts = [
            Option(symbol, expiry, s, right, 'SMART', tradingClass=symbol)
            for s in strikes
        ]
        qualified = self.ib.qualifyContracts(*contracts)
        tickers = []
        for start in range(0, len(qualified), QUOTE_BATCH_SIZE):
            tickers.extend(self.ib.reqTickers(*qualified[start:start + QUOTE_BATCH_SIZE]))

        # 找最接近 target_delta 的
        best = None
        best_diff = float('inf')
        for t in tickers:
            if t is None or t.modelGreeks is None:
                continue
            c = t.contract
            delta = t.modelGreeks.delta
            iv = t.modelGreeks.impliedVol
            if delta is None:
                continue
            diff = abs(abs(delta) - target_delta)
            if diff < best_diff:
                best_diff = diff
                best = (c, delta, iv)

        if best and best_diff <= MAX_DELTA_ERROR:
            return best
        return None

    def _execute_outright_put(
        self, signal: HedgeSignal, spot: float, expiry: str
    ) -> TradeRecord | None:
        """执行 outright put 买入"""
        symbol = signal.symbol
        log.info(f"[{symbol}] Finding -0.25Δ put, expiry={expiry}...")

        result = self._find_contract_by_delta(
            symbol, expiry, DEFAULT_TARGET_DELTA, 'P', spot_override=spot
        )
        if result is None:
            log.error(f"[{symbol}] Cannot find suitable put contract")
            return None

        contract, delta, iv = result
        log.info(f"[{symbol}] Selected: {contract.strike}P (Δ={delta:.3f}, IV={iv:.1%})")

        record = TradeRecord(signal=signal, entry_spot=spot, entry_ts=et_now())

        if self.dry_run:
            log.info(f"[DRY RUN] Would buy {self.qty}x {symbol} {contract.strike}P exp={expiry}")
            record.legs = [{
                'strike': contract.strike, 'right': 'P', 'expiry': expiry,
                'side': 'BUY', 'qty': self.qty, 'fill_price': None,
                'delta': delta, 'iv': iv,
            }]
            record.status = 'DRY_RUN'
            self._save_trade(record)
            return record

        # 显式 DAY + 限价，避免 IB Gateway preset 10349 和市价滑点。
        ticker = self._snapshot_quotes(contract)[0]
        limit_price = self._single_leg_limit(ticker, 'BUY')
        if limit_price is None:
            log.error(f"[{symbol}] Cannot build safe limit price for {contract.strike}P")
            return None
        order = LimitOrder('BUY', self.qty, limit_price, tif='DAY')
        trade = self.ib.placeOrder(contract, order)
        log.info(f"[{symbol}] DAY limit order placed @ ${limit_price:.2f}, waiting for fill...")

        # 等待成交
        filled = self._wait_fill(trade)
        if not filled:
            log.error(f"[{symbol}] Order not filled; cancelling and reconciling")
            if not self._cancel_and_reconcile(trade):
                return None
        if not self._is_fully_filled(trade):
            return None

        fill_price = trade.orderStatus.avgFillPrice
        record.entry_cost = fill_price * self.qty * 100  # 期权乘数
        record.legs = [{
            'strike': contract.strike, 'right': 'P', 'expiry': expiry,
            'side': 'BUY', 'qty': self.qty, 'fill_price': fill_price,
            'delta': delta, 'iv': iv,
        }]
        record.status = 'OPEN'

        log.info(f"[{symbol}] FILLED: {self.qty}x {contract.strike}P @ ${fill_price:.2f} "
                 f"(cost=${record.entry_cost:.2f})")

        self._save_trade(record)
        return record

    def _execute_put_spread(
        self, signal: HedgeSignal, spot: float, expiry: str
    ) -> TradeRecord | None:
        """执行 put spread (买 -0.25Δ put, 卖 -0.10Δ put)"""
        symbol = signal.symbol
        log.info(f"[{symbol}] Finding put spread, expiry={expiry}...")

        # 长腿: -0.25Δ
        long_result = self._find_contract_by_delta(
            symbol, expiry, DEFAULT_TARGET_DELTA, 'P', spot_override=spot
        )
        if long_result is None:
            log.error(f"[{symbol}] Cannot find long put")
            return None
        long_contract, long_delta, long_iv = long_result

        # 短腿: -0.10Δ
        short_result = self._find_contract_by_delta(
            symbol, expiry, SPREAD_SHORT_DELTA, 'P', spot_override=spot
        )
        if short_result is None:
            log.error(f"[{symbol}] Cannot find short put")
            return None
        short_contract, short_delta, short_iv = short_result

        # 确保短腿 strike < 长腿 strike (put spread: 买高卖低)
        if short_contract.strike >= long_contract.strike:
            log.error(f"[{symbol}] Invalid spread: short {short_contract.strike} >= long {long_contract.strike}")
            return None

        log.info(f"[{symbol}] Spread: BUY {long_contract.strike}P (Δ={long_delta:.3f}) / "
                 f"SELL {short_contract.strike}P (Δ={short_delta:.3f})")

        record = TradeRecord(signal=signal, entry_spot=spot, entry_ts=et_now())

        if self.dry_run:
            log.info(f"[DRY RUN] Would buy {long_contract.strike}/{short_contract.strike} put spread")
            record.legs = [
                {'strike': long_contract.strike, 'right': 'P', 'expiry': expiry,
                 'side': 'BUY', 'qty': self.qty, 'fill_price': None,
                 'delta': long_delta, 'iv': long_iv},
                {'strike': short_contract.strike, 'right': 'P', 'expiry': expiry,
                 'side': 'SELL', 'qty': self.qty, 'fill_price': None,
                 'delta': short_delta, 'iv': short_iv},
            ]
            record.status = 'DRY_RUN'
            self._save_trade(record)
            return record

        # 单个 SMART BAG 限价单：两条腿原子化，不会留下裸露单腿。
        quotes = self._snapshot_quotes(long_contract, short_contract)
        limit_debit = self._spread_limit_debit(quotes[0], quotes[1])
        if limit_debit is None:
            log.error(f"[{symbol}] Cannot build safe spread limit from NBBO")
            return None
        bag = self._build_bag(symbol, long_contract, short_contract)
        order = LimitOrder('BUY', self.qty, limit_debit, tif='DAY')
        trade = self.ib.placeOrder(bag, order)
        log.info(
            f"[{symbol}] BAG DAY limit placed @ ${limit_debit:.2f} debit, "
            "waiting for atomic fill..."
        )

        if not self._wait_fill(trade):
            log.warning(f"[{symbol}] BAG not filled; cancelling and reconciling")
            if not self._cancel_and_reconcile(trade):
                return None
        if not self._is_fully_filled(trade):
            return None

        combo_fill = float(trade.orderStatus.avgFillPrice)
        net_debit = combo_fill * self.qty * 100
        long_price = self._leg_fill_price(trade, long_contract.conId)
        short_price = self._leg_fill_price(trade, short_contract.conId)

        record.entry_cost = net_debit
        record.legs = [
            {'strike': long_contract.strike, 'right': 'P', 'expiry': expiry,
             'side': 'BUY', 'qty': self.qty, 'fill_price': long_price,
             'delta': long_delta, 'iv': long_iv},
            {'strike': short_contract.strike, 'right': 'P', 'expiry': expiry,
             'side': 'SELL', 'qty': self.qty, 'fill_price': short_price,
             'delta': short_delta, 'iv': short_iv},
        ]
        record.status = 'OPEN'

        log.info(
            f"[{symbol}] BAG FILLED: {long_contract.strike}/{short_contract.strike}P "
            f"@ ${combo_fill:.2f} debit (total=${net_debit:.2f})"
        )

        self._save_trade(record)
        return record

    def _close_legs(self, symbol: str, legs: list[dict]) -> bool:
        """平仓所有 legs"""
        if self.dry_run:
            for leg in legs:
                side = 'SELL' if leg['side'] == 'BUY' else 'BUY'
                log.info(f"[DRY RUN] Would {side} {leg['qty']}x "
                         f"{symbol} {leg['strike']}{leg['right']} exp={leg['expiry']}")
            return True

        # 垂直价差必须作为一个 BAG 整体平仓。
        if len(legs) == 2:
            long_leg = next((leg for leg in legs if leg.get('side') == 'BUY'), None)
            short_leg = next((leg for leg in legs if leg.get('side') == 'SELL'), None)
            if long_leg is None or short_leg is None:
                log.error(f"[{symbol}] Cannot identify spread legs for atomic close")
                return False
            contracts = [
                Option(symbol, long_leg['expiry'], long_leg['strike'], long_leg['right'],
                       'SMART', tradingClass=symbol),
                Option(symbol, short_leg['expiry'], short_leg['strike'], short_leg['right'],
                       'SMART', tradingClass=symbol),
            ]
            qualified = self.ib.qualifyContracts(*contracts)
            if len(qualified) != 2:
                return False
            long_contract, short_contract = qualified
            quotes = self._snapshot_quotes(long_contract, short_contract)
            limit_credit = self._spread_limit_credit(quotes[0], quotes[1])
            if limit_credit is None:
                log.error(f"[{symbol}] Cannot build safe close credit from NBBO")
                return False
            bag = self._build_bag(symbol, long_contract, short_contract)
            qty = min(int(long_leg.get('qty', 1)), int(short_leg.get('qty', 1)))
            order = LimitOrder('SELL', qty, limit_credit, tif='DAY')
            trade = self.ib.placeOrder(bag, order)
            if not self._wait_fill(trade) and not self._cancel_and_reconcile(trade):
                return False
            return self._is_fully_filled(trade)

        if len(legs) != 1:
            log.error(f"[{symbol}] Unsupported close structure with {len(legs)} legs")
            return False
        leg = legs[0]
        close_side = 'SELL' if leg['side'] == 'BUY' else 'BUY'
        contract = Option(symbol, leg['expiry'], leg['strike'], leg['right'],
                          'SMART', tradingClass=symbol)
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            return False
        contract = qualified[0]
        ticker = self._snapshot_quotes(contract)[0]
        limit_price = self._single_leg_limit(ticker, close_side)
        if limit_price is None:
            return False
        order = LimitOrder(close_side, leg['qty'], limit_price, tif='DAY')
        trade = self.ib.placeOrder(contract, order)
        if not self._wait_fill(trade) and not self._cancel_and_reconcile(trade):
            return False
        return self._is_fully_filled(trade)

    @staticmethod
    def _build_bag(symbol: str, long_contract, short_contract) -> Contract:
        return Contract(
            secType='BAG', symbol=symbol, exchange='SMART', currency='USD',
            comboLegs=[
                ComboLeg(conId=long_contract.conId, ratio=1,
                         action='BUY', exchange='SMART'),
                ComboLeg(conId=short_contract.conId, ratio=1,
                         action='SELL', exchange='SMART'),
            ],
        )

    def _snapshot_quotes(self, *contracts):
        tickers = self.ib.reqTickers(*contracts)
        by_con_id = {ticker.contract.conId: ticker for ticker in tickers}
        result = []
        for contract in contracts:
            ticker = by_con_id.get(contract.conId)
            if ticker is None:
                raise RuntimeError(f'No snapshot quote for conId={contract.conId}')
            result.append(ticker)
        return result

    @staticmethod
    def _valid_price(value, *, allow_zero: bool = False) -> float | None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            return None
        return value

    @classmethod
    def _single_leg_limit(cls, ticker, action: str) -> float | None:
        value = ticker.ask if action == 'BUY' else ticker.bid
        price = cls._valid_price(value)
        return math.ceil(price * 100) / 100 if price is not None else None

    @classmethod
    def _spread_limit_debit(cls, long_ticker, short_ticker) -> float | None:
        long_bid = cls._valid_price(long_ticker.bid, allow_zero=True)
        long_ask = cls._valid_price(long_ticker.ask)
        short_bid = cls._valid_price(short_ticker.bid, allow_zero=True)
        short_ask = cls._valid_price(short_ticker.ask)
        if None in (long_bid, long_ask, short_bid, short_ask):
            return None
        natural = long_ask - short_bid
        midpoint = ((long_bid + long_ask) - (short_bid + short_ask)) / 2
        if natural <= 0:
            return None
        # 从 midpoint 向 natural 让价 25%，不超过当时最差可接受 debit。
        price = midpoint + 0.25 * (natural - midpoint)
        price = min(max(price, 0.01), natural)
        return math.ceil(price * 100) / 100

    @classmethod
    def _spread_limit_credit(cls, long_ticker, short_ticker) -> float | None:
        long_bid = cls._valid_price(long_ticker.bid, allow_zero=True)
        long_ask = cls._valid_price(long_ticker.ask)
        short_bid = cls._valid_price(short_ticker.bid, allow_zero=True)
        short_ask = cls._valid_price(short_ticker.ask)
        if None in (long_bid, long_ask, short_bid, short_ask):
            return None
        natural = long_bid - short_ask
        midpoint = ((long_bid + long_ask) - (short_bid + short_ask)) / 2
        if natural <= 0:
            return None
        price = midpoint - 0.25 * (midpoint - natural)
        price = max(min(price, midpoint), natural)
        return max(0.01, math.floor(price * 100) / 100)

    @staticmethod
    def _leg_fill_price(trade, con_id: int) -> float | None:
        fills = [fill for fill in getattr(trade, 'fills', [])
                 if getattr(fill.contract, 'conId', None) == con_id]
        quantities = [float(fill.execution.shares) for fill in fills]
        total = sum(quantities)
        if total <= 0:
            return None
        return sum(float(fill.execution.price) * qty
                   for fill, qty in zip(fills, quantities)) / total

    @staticmethod
    def _has_tif_preset_warning(trade) -> bool:
        return any(getattr(entry, 'errorCode', None) == TIF_PRESET_WARNING_CODE
                   for entry in getattr(trade, 'log', []))

    @staticmethod
    def _is_fully_filled(trade) -> bool:
        status = getattr(trade.orderStatus, 'status', '')
        if status == 'Filled':
            return True
        try:
            filled = float(getattr(trade.orderStatus, 'filled', 0) or 0)
            quantity = float(getattr(trade.order, 'totalQuantity', 0) or 0)
            return quantity > 0 and filled >= quantity
        except (TypeError, ValueError):
            return False

    def _wait_fill(self, trade, timeout: int = ORDER_TIMEOUT_SEC) -> bool:
        """等待订单成交；10349 是 preset 通知，不视为服务器拒单。"""
        start = time.time()
        while time.time() - start < timeout:
            self.ib.sleep(0.5)
            if self._is_fully_filled(trade):
                return True
            if trade.orderStatus.status in ('Cancelled', 'ApiCancelled'):
                if self._has_tif_preset_warning(trade):
                    continue
                return False
        return False

    def _cancel_and_reconcile(self, trade) -> bool:
        """取消后继续消化成交回报，防止把延迟 fill 误判为未成交。"""
        if self._is_fully_filled(trade):
            return True
        try:
            self.ib.cancelOrder(trade.order)
        except Exception as exc:
            log.warning(f"Cancel request failed: {exc}")
        try:
            self.ib.reqOpenOrders()
        except Exception:
            pass
        deadline = time.time() + ORDER_RECONCILE_SEC
        while time.time() < deadline:
            self.ib.sleep(0.25)
            if self._is_fully_filled(trade):
                log.warning("Order filled during cancel reconciliation")
                return True
        return self._is_fully_filled(trade)

    def _save_trade(self, record: TradeRecord) -> None:
        """保存交易记录到 DB"""
        if self._conn is None:
            log.info("No DB connection, trade not persisted (printed only)")
            return

        d = record.to_db_dict()
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO hedge_trades (
                        signal_ts, symbol, action, structure,
                        entry_ts, entry_spot, entry_cost, legs, status,
                        urgency, skew_cheapness, gex_regime, term_structure
                    ) VALUES (
                        %(signal_ts)s, %(symbol)s, %(action)s, %(structure)s,
                        %(entry_ts)s, %(entry_spot)s, %(entry_cost)s,
                        %(legs)s::jsonb, %(status)s,
                        %(urgency)s, %(skew_cheapness)s, %(gex_regime)s, %(term_structure)s
                    )
                """, d)
            log.info(f"Trade saved to DB: {d['action']} {d['symbol']}")
        except Exception as e:
            log.error(f"Failed to save trade: {e}")

    def _update_trade_closed(self, trade_id: int, closed_legs: bool) -> None:
        """更新交易为 CLOSED"""
        if self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    UPDATE hedge_trades
                    SET status = 'CLOSED', exit_ts = %s, updated_at = %s
                    WHERE id = %s
                """, [et_now(), et_now(), trade_id])
        except Exception as e:
            log.error(f"Failed to update trade {trade_id}: {e}")

    def shutdown(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


def format_trade_result(record: TradeRecord) -> str:
    """格式化交易结果"""
    s = record.signal
    lines = [
        '',
        f"  {'='*50}",
        f"  Trade Executed: {s.symbol} {s.action}",
        f"  {'='*50}",
        f"  Status:  {record.status}",
        f"  Spot:    ${record.entry_spot:.2f}" if record.entry_spot else "",
    ]

    if record.entry_cost:
        lines.append(f"  Cost:    ${record.entry_cost:.2f}")

    for i, leg in enumerate(record.legs):
        price_txt = f"@ ${leg['fill_price']:.2f}" if leg.get('fill_price') else "(unfilled)"
        lines.append(
            f"  Leg {i+1}:  {leg['side']} {leg.get('qty', 1)}x "
            f"{leg['strike']}{leg['right']} exp={leg['expiry']} "
            f"{price_txt} (Δ={leg.get('delta', 0):.3f})"
        )

    lines.append(f"  {'='*50}")
    return '\n'.join(l for l in lines if l)
