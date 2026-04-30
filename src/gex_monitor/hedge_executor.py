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
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from ib_insync import IB, Stock, Index, Option, LimitOrder, MarketOrder

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
        self, symbol: str, expiry: str, target_delta: float, right: str = 'P'
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
        chain = next((c for c in chains if c.exchange == 'SMART'), None)
        if not chain:
            return None

        # 获取 spot
        self.ib.reqMktData(underlying, genericTickList='', snapshot=False)
        self.ib.sleep(2)
        u_ticker = self.ib.ticker(underlying)
        spot = u_ticker.marketPrice() if u_ticker else None
        self.ib.cancelMktData(underlying)

        if not spot or np.isnan(spot):
            return None

        # 选 strikes
        all_strikes = sorted(s for s in chain.strikes if s == int(s))
        lo, hi = spot * 0.90, spot * 1.10
        strikes = [s for s in all_strikes if lo <= s <= hi]

        # 创建合约并请求数据
        contracts = [
            Option(symbol, expiry, s, right, 'SMART', tradingClass=symbol)
            for s in strikes
        ]
        qualified = self.ib.qualifyContracts(*contracts)
        for c in qualified:
            self.ib.reqMktData(c, genericTickList='106', snapshot=False)
        self.ib.sleep(5)

        # 找最接近 target_delta 的
        best = None
        best_diff = float('inf')
        for c in qualified:
            t = self.ib.ticker(c)
            if t is None or t.modelGreeks is None:
                continue
            delta = t.modelGreeks.delta
            iv = t.modelGreeks.impliedVol
            if delta is None:
                continue
            diff = abs(abs(delta) - target_delta)
            if diff < best_diff:
                best_diff = diff
                best = (c, delta, iv)

        # 清理
        for c in qualified:
            try:
                self.ib.cancelMktData(c)
            except Exception:
                pass

        if best and best_diff < 0.15:
            return best
        return None

    def _execute_outright_put(
        self, signal: HedgeSignal, spot: float, expiry: str
    ) -> TradeRecord | None:
        """执行 outright put 买入"""
        symbol = signal.symbol
        log.info(f"[{symbol}] Finding -0.25Δ put, expiry={expiry}...")

        result = self._find_contract_by_delta(symbol, expiry, DEFAULT_TARGET_DELTA, 'P')
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

        # 下单
        order = MarketOrder('BUY', self.qty)
        trade = self.ib.placeOrder(contract, order)
        log.info(f"[{symbol}] Order placed, waiting for fill...")

        # 等待成交
        filled = self._wait_fill(trade)
        if not filled:
            log.error(f"[{symbol}] Order not filled within timeout")
            self.ib.cancelOrder(order)
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
        long_result = self._find_contract_by_delta(symbol, expiry, DEFAULT_TARGET_DELTA, 'P')
        if long_result is None:
            log.error(f"[{symbol}] Cannot find long put")
            return None
        long_contract, long_delta, long_iv = long_result

        # 短腿: -0.10Δ
        short_result = self._find_contract_by_delta(symbol, expiry, SPREAD_SHORT_DELTA, 'P')
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

        # 分别下单（简单实现，combo order 更优但更复杂）
        long_order = MarketOrder('BUY', self.qty)
        short_order = MarketOrder('SELL', self.qty)

        long_trade = self.ib.placeOrder(long_contract, long_order)
        short_trade = self.ib.placeOrder(short_contract, short_order)

        log.info(f"[{symbol}] Spread orders placed, waiting for fills...")

        long_filled = self._wait_fill(long_trade)
        short_filled = self._wait_fill(short_trade)

        if not long_filled or not short_filled:
            log.error(f"[{symbol}] Spread not fully filled")
            # 尝试取消未成交的
            if not long_filled:
                self.ib.cancelOrder(long_order)
            if not short_filled:
                self.ib.cancelOrder(short_order)
            return None

        long_price = long_trade.orderStatus.avgFillPrice
        short_price = short_trade.orderStatus.avgFillPrice
        net_debit = (long_price - short_price) * self.qty * 100

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

        log.info(f"[{symbol}] SPREAD FILLED: "
                 f"BUY {long_contract.strike}P @ ${long_price:.2f} / "
                 f"SELL {short_contract.strike}P @ ${short_price:.2f} "
                 f"(net debit=${net_debit:.2f})")

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

        for leg in legs:
            close_side = 'SELL' if leg['side'] == 'BUY' else 'BUY'
            contract = Option(symbol, leg['expiry'], leg['strike'],
                              leg['right'], 'SMART', tradingClass=symbol)
            self.ib.qualifyContracts(contract)
            order = MarketOrder(close_side, leg['qty'])
            trade = self.ib.placeOrder(contract, order)
            filled = self._wait_fill(trade)
            if filled:
                price = trade.orderStatus.avgFillPrice
                log.info(f"[{symbol}] Closed: {close_side} {leg['strike']}{leg['right']} @ ${price:.2f}")
            else:
                log.error(f"[{symbol}] Failed to close {leg['strike']}{leg['right']}")
                return False
        return True

    def _wait_fill(self, trade, timeout: int = ORDER_TIMEOUT_SEC) -> bool:
        """等待订单成交"""
        start = time.time()
        while time.time() - start < timeout:
            self.ib.sleep(0.5)
            if trade.orderStatus.status == 'Filled':
                return True
            if trade.orderStatus.status in ('Cancelled', 'ApiCancelled'):
                return False
        return False

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
