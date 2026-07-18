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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, Index, Option

from .time_utils import et_now, ET

log = logging.getLogger(__name__)

# 目标 tenor（天数）
# 0/7/14 = 日内 + 近期监控, 30/45 = 对冲甜蜜区间
TARGET_DTES = [0, 7, 14, 30, 45]

# 按 DTE 扩展 strike 范围，使 30-45DTE 的 10/25 delta 合约也能被覆盖。
STRIKE_RANGE_BY_DTE = (
    (1, 0.06),
    (8, 0.10),
    (16, 0.13),
    (60, 0.20),
)
MAX_STRIKES_PER_TENOR = 49
SNAPSHOT_STRIKES_PER_BATCH = 24

# delta 匹配容差
DELTA_TOLERANCE = 0.08
MIN_GOOD_CONTRACTS = 8


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
    atm_strike: float | None = None
    put_25_strike: float | None = None
    put_25_delta: float | None = None
    put_25_bid: float | None = None
    put_25_ask: float | None = None
    call_25_strike: float | None = None
    call_25_delta: float | None = None
    call_25_bid: float | None = None
    call_25_ask: float | None = None
    put_10_strike: float | None = None
    put_10_delta: float | None = None
    put_10_bid: float | None = None
    put_10_ask: float | None = None
    call_10_strike: float | None = None
    call_10_delta: float | None = None
    call_10_bid: float | None = None
    call_10_ask: float | None = None
    quality: str = 'unknown'
    quality_reasons: str = ''


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
            row = asdict(t)
            row.update({
                'ts': self.ts,
                'symbol': self.symbol,
                'spot': self.spot,
                'term_spread_rr25': self.term_spread_rr25,
                'term_spread_iv': self.term_spread_iv,
            })
            rows.append(row)
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
    spot_override: float | None = None,
    chain_override=None,
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

    # 1. 获取 underlying 价格。Worker 传入 override 时不碰它的实时订阅。
    if sec_type == 'IND':
        underlying = Index(symbol, 'CBOE', 'USD')
    else:
        underlying = Stock(symbol, 'SMART', 'USD')

    spot = spot_override
    chain = chain_override
    if spot is None or chain is None:
        ib.qualifyContracts(underlying)
    if spot is None:
        tickers = ib.reqTickers(underlying)
        u_ticker = tickers[0] if tickers else None
        spot = u_ticker.marketPrice() if u_ticker else None
    if not spot or np.isnan(spot) or spot <= 0:
        log.error(f"[{symbol}] 无法获取 spot 价格")
        return None

    # 2. 获取期权链
    if chain is None:
        chains = ib.reqSecDefOptParams(
            underlying.symbol, '', underlying.secType, underlying.conId
        )
        chain = (
            next((c for c in chains if c.exchange == 'SMART'
                  and c.tradingClass == trading_class), None)
            or next((c for c in chains if c.tradingClass == trading_class), None)
        )
    if chain is None:
        log.error(f"[{symbol}] 无匹配 tradingClass={trading_class} 的期权链")
        return None

    # 3. 选择 tenor expiries
    today_str = now.strftime('%Y%m%d')
    expiries = pick_tenor_expiries(chain, today_str)
    if not expiries:
        log.error(f"[{symbol}] 无可用 expiry")
        return None

    log.info(f"[{symbol}] Spot={spot:.2f}, expiries={expiries}")

    # 4. 按 tenor 顺序请求快照，避免五个期限同时占满 IB 行情额度。
    all_strikes = sorted({float(s) for s in chain.strikes if float(s) > 0})
    tenors = []
    for expiry, dte in expiries:
        strikes = _select_strikes(all_strikes, spot, dte)
        if len(strikes) < 5:
            log.error(f"[{symbol}] {expiry} strikes 不足: {len(strikes)}")
            tenors.append(_compute_tenor_skew([], spot, expiry, dte))
            continue
        tickers = []
        for start in range(0, len(strikes), SNAPSHOT_STRIKES_PER_BATCH):
            batch = strikes[start:start + SNAPSHOT_STRIKES_PER_BATCH]
            contracts = [
                Option(symbol, expiry, s, r, 'SMART', tradingClass=trading_class)
                for s in batch for r in ('C', 'P')
            ]
            qualified = ib.qualifyContracts(*contracts)
            if not qualified:
                continue
            try:
                tickers.extend(ib.reqTickers(*qualified))
            except Exception as exc:
                log.warning(f"[{symbol}] {expiry} snapshot batch failed: {exc}")
        tenor_snap = _compute_tenor_skew(tickers, spot, expiry, dte)
        tenors.append(tenor_snap)
        log.info(
            f"[{symbol}] {expiry} ({dte}D) skew quality={tenor_snap.quality} "
            f"contracts={tenor_snap.n_contracts} rr25={tenor_snap.rr_25}"
        )

    # 8. 计算 term structure spread
    term_spread_rr25 = None
    term_spread_iv = None

    near_candidates = [t for t in tenors if t.dte <= 8 and t.quality != 'bad']
    far_candidates = [t for t in tenors if 20 <= t.dte <= 55 and t.quality != 'bad']
    if near_candidates and far_candidates:
        near = min(near_candidates, key=lambda t: t.dte)
        far = min(far_candidates, key=lambda t: abs(t.dte - 45))
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


def _strike_range_pct(dte: int) -> float:
    for max_dte, pct in STRIKE_RANGE_BY_DTE:
        if dte <= max_dte:
            return pct
    return STRIKE_RANGE_BY_DTE[-1][1]


def _select_strikes(all_strikes: list[float], spot: float, dte: int) -> list[float]:
    """在动态范围内均匀抽样，同时保留 ATM 和边界。"""
    pct = _strike_range_pct(dte)
    selected = [s for s in all_strikes if spot * (1 - pct) <= s <= spot * (1 + pct)]
    if len(selected) <= MAX_STRIKES_PER_TENOR:
        return selected
    indices = np.linspace(0, len(selected) - 1, MAX_STRIKES_PER_TENOR)
    chosen = {selected[int(round(i))] for i in indices}
    chosen.add(min(selected, key=lambda s: abs(s - spot)))
    return sorted(chosen)


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
            'bid': _finite_quote(getattr(t, 'bid', None)),
            'ask': _finite_quote(getattr(t, 'ask', None)),
        })

    if len(rows) < 4:
        return SkewTenorSnapshot(
            expiry=expiry, dte=dte,
            atm_iv=None, rr_25=None, rr_10=None,
            skew_slope=None, n_contracts=len(rows), quality='bad',
            quality_reasons='insufficient_contracts',
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

    atm_candidates = df.iloc[(df['strike'] - spot).abs().argsort()[:2]]
    atm_strike = (float(atm_candidates.iloc[0]['strike'])
                  if not atm_candidates.empty else None)
    put25 = _row_at_delta(puts, 0.25)
    call25 = _row_at_delta(calls, 0.25)
    put10 = _row_at_delta(puts, 0.10)
    call10 = _row_at_delta(calls, 0.10)

    reasons = []
    if len(rows) < MIN_GOOD_CONTRACTS:
        reasons.append('low_contract_count')
    if atm_iv is None:
        reasons.append('missing_atm_iv')
    if rr_25 is None:
        reasons.append('missing_rr25')
    if rr_10 is None:
        reasons.append('missing_rr10')
    quality = 'good' if not reasons else ('partial' if atm_iv is not None else 'bad')

    return SkewTenorSnapshot(
        expiry=expiry,
        dte=dte,
        atm_iv=atm_iv,
        rr_25=rr_25,
        rr_10=rr_10,
        skew_slope=skew_slope,
        n_contracts=len(rows),
        atm_strike=atm_strike,
        put_25_strike=_row_value(put25, 'strike'),
        put_25_delta=_row_value(put25, 'delta'),
        put_25_bid=_row_value(put25, 'bid'),
        put_25_ask=_row_value(put25, 'ask'),
        call_25_strike=_row_value(call25, 'strike'),
        call_25_delta=_row_value(call25, 'delta'),
        call_25_bid=_row_value(call25, 'bid'),
        call_25_ask=_row_value(call25, 'ask'),
        put_10_strike=_row_value(put10, 'strike'),
        put_10_delta=_row_value(put10, 'delta'),
        put_10_bid=_row_value(put10, 'bid'),
        put_10_ask=_row_value(put10, 'ask'),
        call_10_strike=_row_value(call10, 'strike'),
        call_10_delta=_row_value(call10, 'delta'),
        call_10_bid=_row_value(call10, 'bid'),
        call_10_ask=_row_value(call10, 'ask'),
        quality=quality,
        quality_reasons=';'.join(reasons),
    )


def _finite_quote(value) -> float | None:
    value = _finite_number(value)
    return value if value is not None and value >= 0 else None


def _finite_number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _row_at_delta(side_df: pd.DataFrame, target_delta: float):
    valid = side_df.dropna(subset=['delta', 'iv'])
    if valid.empty:
        return None
    diffs = (valid['delta'].abs() - target_delta).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > DELTA_TOLERANCE:
        return None
    return valid.loc[idx]


def _row_value(row, column: str) -> float | None:
    if row is None:
        return None
    return _finite_number(row.get(column))


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
    row = _row_at_delta(side_df, target_delta)
    return float(row['iv']) if row is not None else None


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
