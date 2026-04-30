"""
Multi-tenor Skew Surface 采集模块

每日收盘前快照 0DTE / ~7DTE / ~14DTE / ~30DTE / ~45DTE 的 IV surface，
用于对冲时机决策。30-45DTE 是对冲甜蜜区间（theta 效率最优）。

设计原则:
  - 纯函数 + dataclass，不持有 IB 连接
  - 与现有 GEX 引擎完全解耦
  - 输出结构可直接序列化为 parquet
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, Index, Option

from .time_utils import et_now, ET

log = logging.getLogger(__name__)

# 目标 tenor（天数）
# 0/7/14 = 日内 + 近期监控, 30/45 = 对冲甜蜜区间
TARGET_DTES = [0, 7, 14, 30, 45]

# strike 范围：spot ±5%
STRIKE_RANGE_PCT = 0.05

# 数据等待时间（秒）
DATA_WAIT_SEC = 8

# delta 匹配容差
DELTA_TOLERANCE = 0.15


@dataclass
class SkewTenorSnapshot:
    """单个 tenor 的 skew 度量"""
    expiry: str              # YYYYMMDD
    dte: int                 # 距到期天数
    atm_iv: float | None     # ATM IV（小数）
    rr_25: float | None      # 25Δ risk reversal
    rr_10: float | None      # 10Δ risk reversal（深 OTM）
    skew_slope: float | None  # OTM skew slope（归一化）
    n_contracts: int          # 有效合约数


@dataclass
class SkewSurface:
    """多 tenor skew surface 快照"""
    ts: datetime
    symbol: str
    spot: float
    tenors: list[SkewTenorSnapshot] = field(default_factory=list)
    # 衍生指标
    term_spread_rr25: float | None = None  # rr_25(0DTE) - rr_25(2W)
    term_spread_iv: float | None = None    # atm_iv(0DTE) - atm_iv(2W)

    def to_records(self) -> list[dict]:
        """转为 flat records，便于 parquet 存储"""
        rows = []
        for t in self.tenors:
            rows.append({
                'ts': self.ts,
                'symbol': self.symbol,
                'spot': self.spot,
                'expiry': t.expiry,
                'dte': t.dte,
                'atm_iv': t.atm_iv,
                'rr_25': t.rr_25,
                'rr_10': t.rr_10,
                'skew_slope': t.skew_slope,
                'n_contracts': t.n_contracts,
                'term_spread_rr25': self.term_spread_rr25,
                'term_spread_iv': self.term_spread_iv,
            })
        return rows

    def get_tenor(self, target_dte: int) -> SkewTenorSnapshot | None:
        """获取最接近目标 DTE 的 tenor"""
        if not self.tenors:
            return None
        return min(self.tenors, key=lambda t: abs(t.dte - target_dte))


def pick_tenor_expiries(chain, today_str: str) -> list[tuple[str, int]]:
    """
    从期权链中选择 0DTE / ~7DTE / ~14DTE 三个 expiry

    Returns:
        [(expiry_str, dte), ...] 去重后的列表
    """
    future = sorted(e for e in chain.expirations if e >= today_str)
    if not future:
        return []

    today_dt = datetime.strptime(today_str, '%Y%m%d')
    result = []
    seen = set()

    for target_dte in TARGET_DTES:
        target_dt = today_dt + timedelta(days=target_dte)
        target_str = target_dt.strftime('%Y%m%d')

        # 找最接近目标日期的 expiry
        closest = min(future, key=lambda e: abs(
            (datetime.strptime(e, '%Y%m%d') - target_dt).days
        ))
        actual_dte = (datetime.strptime(closest, '%Y%m%d') - today_dt).days

        if closest not in seen:
            seen.add(closest)
            result.append((closest, actual_dte))

    return result


def collect_skew_surface(
    ib: IB,
    symbol: str,
    trading_class: str | None = None,
    sec_type: str = 'STK',
) -> SkewSurface | None:
    """
    采集多 tenor skew surface

    使用已有的 IB 连接，请求快照数据。

    Args:
        ib: 已连接的 IB 实例
        symbol: 标的代码（QQQ, SPY）
        trading_class: 期权交易类（默认同 symbol）
        sec_type: 标的类型（STK 或 IND）

    Returns:
        SkewSurface 或 None
    """
    trading_class = trading_class or symbol
    now = et_now()

    # 1. 获取 underlying 价格
    if sec_type == 'IND':
        underlying = Index(symbol, 'CBOE', 'USD')
    else:
        underlying = Stock(symbol, 'SMART', 'USD')

    ib.qualifyContracts(underlying)
    ib.reqMktData(underlying, genericTickList='', snapshot=False)
    ib.sleep(2)

    u_ticker = ib.ticker(underlying)
    spot = u_ticker.marketPrice() if u_ticker else None
    if not spot or np.isnan(spot) or spot <= 0:
        log.error(f"[{symbol}] 无法获取 spot 价格")
        ib.cancelMktData(underlying)
        return None

    # 2. 获取期权链
    chains = ib.reqSecDefOptParams(
        underlying.symbol, '', underlying.secType, underlying.conId
    )
    chain = next((c for c in chains if c.exchange == 'SMART'), None)
    if chain is None:
        log.error(f"[{symbol}] 无 SMART 期权链")
        ib.cancelMktData(underlying)
        return None

    # 3. 选择 tenor expiries
    today_str = now.strftime('%Y%m%d')
    expiries = pick_tenor_expiries(chain, today_str)
    if not expiries:
        log.error(f"[{symbol}] 无可用 expiry")
        ib.cancelMktData(underlying)
        return None

    log.info(f"[{symbol}] Spot={spot:.2f}, expiries={expiries}")

    # 4. 选择 strikes
    all_strikes = sorted(s for s in chain.strikes if s == int(s))
    lo = spot * (1 - STRIKE_RANGE_PCT)
    hi = spot * (1 + STRIKE_RANGE_PCT)
    strikes = [s for s in all_strikes if lo <= s <= hi]

    if len(strikes) < 5:
        log.error(f"[{symbol}] strikes 不足: {len(strikes)}")
        ib.cancelMktData(underlying)
        return None

    # 5. 批量创建合约并订阅
    all_contracts = []
    expiry_contracts: dict[str, list] = {}  # expiry -> contracts

    for expiry, dte in expiries:
        contracts = [
            Option(symbol, expiry, s, r, 'SMART', tradingClass=trading_class)
            for s in strikes for r in ['C', 'P']
        ]
        qualified = ib.qualifyContracts(*contracts)
        expiry_contracts[expiry] = qualified
        all_contracts.extend(qualified)

    log.info(f"[{symbol}] 订阅 {len(all_contracts)} 个合约 ({len(expiries)} tenor × {len(strikes)} strikes × 2)")

    # 订阅行情
    for c in all_contracts:
        ib.reqMktData(c, genericTickList='100,101,104,106', snapshot=False)

    # 等待数据填充
    ib.sleep(DATA_WAIT_SEC)

    # 6. 读取数据并计算每个 tenor 的 skew
    tenors = []
    for expiry, dte in expiries:
        contracts = expiry_contracts.get(expiry, [])
        tickers = [ib.ticker(c) for c in contracts]
        tenor_snap = _compute_tenor_skew(tickers, spot, expiry, dte)
        tenors.append(tenor_snap)

    # 7. 取消订阅
    for c in all_contracts:
        try:
            ib.cancelMktData(c)
        except Exception:
            pass
    ib.cancelMktData(underlying)

    # 8. 计算 term structure spread
    term_spread_rr25 = None
    term_spread_iv = None

    if len(tenors) >= 2:
        near = tenors[0]  # 最短 tenor
        far = tenors[-1]  # 最长 tenor
        if near.rr_25 is not None and far.rr_25 is not None:
            term_spread_rr25 = near.rr_25 - far.rr_25
        if near.atm_iv is not None and far.atm_iv is not None:
            term_spread_iv = near.atm_iv - far.atm_iv

    surface = SkewSurface(
        ts=now,
        symbol=symbol,
        spot=spot,
        tenors=tenors,
        term_spread_rr25=term_spread_rr25,
        term_spread_iv=term_spread_iv,
    )

    log.info(
        f"[{symbol}] Surface collected: "
        f"{len(tenors)} tenors, "
        f"term_spread_rr25={term_spread_rr25}, "
        f"term_spread_iv={term_spread_iv}"
    )

    return surface


def _compute_tenor_skew(
    tickers, spot: float, expiry: str, dte: int
) -> SkewTenorSnapshot:
    """计算单个 tenor 的 skew 度量"""
    rows = []
    for t in tickers:
        if t is None:
            continue
        g = t.modelGreeks
        if not g or g.impliedVol is None or g.delta is None:
            continue
        c = t.contract
        if c.right not in ('C', 'P'):
            continue
        rows.append({
            'strike': c.strike,
            'right': c.right,
            'iv': g.impliedVol,
            'delta': g.delta,
        })

    if len(rows) < 4:
        return SkewTenorSnapshot(
            expiry=expiry, dte=dte,
            atm_iv=None, rr_25=None, rr_10=None,
            skew_slope=None, n_contracts=len(rows),
        )

    df = pd.DataFrame(rows)
    puts = df[df.right == 'P']
    calls = df[df.right == 'C']

    # ATM IV
    atm_iv = _calc_atm_iv(puts, calls, spot)

    # Risk Reversals
    rr_25 = _calc_risk_reversal(puts, calls, 0.25)
    rr_10 = _calc_risk_reversal(puts, calls, 0.10)

    # Skew slope
    skew_slope = _calc_skew_slope(puts, calls, spot, atm_iv)

    return SkewTenorSnapshot(
        expiry=expiry,
        dte=dte,
        atm_iv=atm_iv,
        rr_25=rr_25,
        rr_10=rr_10,
        skew_slope=skew_slope,
        n_contracts=len(rows),
    )


def _calc_atm_iv(puts: pd.DataFrame, calls: pd.DataFrame, spot: float) -> float | None:
    atm_put = puts.iloc[(puts['strike'] - spot).abs().argsort()[:1]]
    atm_call = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]
    ivs = []
    if not atm_put.empty:
        ivs.append(atm_put.iloc[0]['iv'])
    if not atm_call.empty:
        ivs.append(atm_call.iloc[0]['iv'])
    return float(np.mean(ivs)) if ivs else None


def _calc_risk_reversal(
    puts: pd.DataFrame, calls: pd.DataFrame, target_delta: float
) -> float | None:
    put_iv = _iv_at_delta(puts, target_delta)
    call_iv = _iv_at_delta(calls, target_delta)
    if put_iv is None or call_iv is None:
        return None
    return float(put_iv - call_iv)


def _iv_at_delta(side_df: pd.DataFrame, target_delta: float) -> float | None:
    valid = side_df.dropna(subset=['delta', 'iv'])
    if valid.empty:
        return None
    diffs = (valid['delta'].abs() - target_delta).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > DELTA_TOLERANCE:
        return None
    return float(valid.loc[idx, 'iv'])


def _calc_skew_slope(
    puts: pd.DataFrame, calls: pd.DataFrame, spot: float, atm_iv: float | None
) -> float | None:
    if atm_iv is None or atm_iv < 1e-6:
        return None

    otm_put_target = spot * 0.98
    otm_call_target = spot * 1.02

    otm_put = puts.iloc[(puts['strike'] - otm_put_target).abs().argsort()[:1]]
    otm_call = calls.iloc[(calls['strike'] - otm_call_target).abs().argsort()[:1]]

    if otm_put.empty or otm_call.empty:
        return None

    otm_put_iv = otm_put.iloc[0]['iv']
    otm_call_iv = otm_call.iloc[0]['iv']

    if pd.isna(otm_put_iv) or pd.isna(otm_call_iv):
        return None

    return float((otm_put_iv - otm_call_iv) / atm_iv)
