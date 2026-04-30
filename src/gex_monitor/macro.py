"""
宏观环境快照 — 三大类监测框架

1. 真实利率 (Real Yield)
   - 10Y TIPS yield (DFII10) — 真实融资成本
   - 5Y5Y forward breakeven (T5YIFR) — 通胀预期
2. 美元强度 (Dollar Strength)
   - DXY 指数 (IB Gateway)
   - 广义美元指数 DTWEXBGS (FRED, weekly)
3. 融资条件 (Funding Stress)
   - VIX — 股市隐含波动率 (IB / FRED fallback)
   - MOVE — 债市波动率 (IB)
   - SOFR-OIS spread — 短端融资压力 (FRED)
   - HY OAS (BAMLH0A0HYM2) — 信用利差 (FRED)

数据源优先级: IB Gateway 实时 → FRED T-1 日度 → None

用法:
  from .macro import fetch_macro_snapshot
  macro = fetch_macro_snapshot(ib)
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import requests
from dotenv import load_dotenv
from ib_insync import IB, Index

load_dotenv()

log = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# --- 融资条件阈值 ---
VIX_CALM = 15.0
VIX_ELEVATED = 25.0
MOVE_CALM = 80.0
MOVE_ELEVATED = 120.0
SOFR_OIS_STRESS_BPS = 5.0
HY_OAS_ELEVATED = 400.0   # HY OAS 走阔（bps）
HY_OAS_STRESS = 500.0

# --- 真实利率阈值 ---
TIPS_10Y_ALERT = 2.2       # 10Y TIPS yield 警觉线
TIPS_10Y_DANGER = 2.4      # 窗口基本关闭


@dataclass
class MacroSnapshot:
    """宏观环境快照 — 三大类框架"""

    # --- 融资条件 (Funding Stress) ---
    vix: float | None = None
    move: float | None = None
    sofr_ois_spread_bps: float | None = None
    hy_oas_bps: float | None = None          # BAMLH0A0HYM2, HY OAS spread

    # --- 真实利率 (Real Yield) ---
    tips_10y: float | None = None            # DFII10, 10Y TIPS yield
    breakeven_5y5y: float | None = None      # T5YIFR, 5Y5Y forward inflation expectation

    # --- 美元强度 (Dollar Strength) ---
    dxy: float | None = None                 # DXY 指数 (IB)
    broad_dollar: float | None = None        # DTWEXBGS 广义美元指数 (FRED, weekly)

    # ── Sub-scores ──────────────────────────────────────────

    @property
    def real_yield_score(self) -> float:
        """真实利率 sub-score: 0 ~ +0.2"""
        score = 0.0
        if self.tips_10y is not None:
            if self.tips_10y > TIPS_10Y_DANGER:
                score += 0.2
            elif self.tips_10y > TIPS_10Y_ALERT:
                score += 0.1
        return score

    @property
    def dollar_score(self) -> float:
        """美元强度 sub-score — 首版只采集展示，不参与 urgency（看形态不看绝对值）"""
        return 0.0

    @property
    def funding_score(self) -> float:
        """融资条件 sub-score: -0.1 ~ +0.4"""
        score = 0.0
        if self.vix is not None and self.move is not None:
            if self.vix > VIX_ELEVATED and self.move > MOVE_ELEVATED:
                score += 0.2
            elif self.vix > VIX_ELEVATED or self.move > MOVE_ELEVATED:
                score += 0.1
            elif self.vix < VIX_CALM and self.move < MOVE_CALM:
                score -= 0.1

        if self.sofr_ois_spread_bps is not None:
            if self.sofr_ois_spread_bps > SOFR_OIS_STRESS_BPS:
                score += 0.1

        if self.hy_oas_bps is not None:
            if self.hy_oas_bps > HY_OAS_STRESS:
                score += 0.1
            elif self.hy_oas_bps > HY_OAS_ELEVATED:
                score += 0.05

        return score

    # ── Aggregate ───────────────────────────────────────────

    @property
    def urgency_adjustment(self) -> float:
        """汇总三大类 sub-score"""
        return self.real_yield_score + self.dollar_score + self.funding_score

    @property
    def regime_label(self) -> str:
        """人类可读的宏观环境标签"""
        if self.vix is None:
            return 'unknown'
        if self.vix > VIX_ELEVATED:
            return 'stressed'
        if self.vix < VIX_CALM:
            return 'calm'
        return 'normal'

    def summary(self) -> str:
        def _f(v, fmt=".1f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        parts = [
            f"[{self.regime_label}] urgency_adj={self.urgency_adjustment:+.2f}",
            f"  RealYield: TIPS10Y={_f(self.tips_10y, '.2f')}% BkEven5Y5Y={_f(self.breakeven_5y5y, '.2f')}%"
            f"  (score={self.real_yield_score:+.1f})",
            f"  Dollar: DXY={_f(self.dxy)} BroadUSD={_f(self.broad_dollar)}"
            f"  (score={self.dollar_score:+.1f})",
            f"  Funding: VIX={_f(self.vix)} MOVE={_f(self.move)}"
            f" SOFR-OIS={_f(self.sofr_ois_spread_bps, '+.1f')}bps"
            f" HY-OAS={_f(self.hy_oas_bps, '.0f')}bps"
            f"  (score={self.funding_score:+.1f})",
        ]
        return "Macro:\n" + "\n".join(parts)


def fetch_macro_snapshot(ib: IB | None = None) -> MacroSnapshot:
    """
    采集宏观快照 — 三大类指标（并行请求，~3s 完成）

    优先级: IB Gateway（实时） → FRED（T-1 日度） → Yahoo → None
    任何子指标拿不到都不阻塞，graceful degradation。
    """
    # IB 请求必须串行（共用一个连接），先快速尝试
    vix_ib = _fetch_ib_index(ib, 'VIX', 'CBOE')
    move_ib = _fetch_ib_index(ib, 'MOVE', 'CBOE')
    dxy_ib = _fetch_dxy(ib)

    # FRED + Yahoo 请求并行
    results = {}
    tasks = {
        'vix_fred': _fetch_fred_vix,
        'sofr_ois': _fetch_sofr_ois_spread,
        'hy_oas': _fetch_hy_oas,
        'tips_10y': _fetch_tips_10y,
        'breakeven_5y5y': _fetch_breakeven_5y5y,
        'broad_dollar': _fetch_broad_dollar,
        'move_yahoo': lambda: _fetch_yahoo('^MOVE', 'MOVE'),
        'dxy_yahoo': lambda: _fetch_yahoo('DX-Y.NYB', 'DXY'),
    }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                log.warning(f"[Macro] {key} failed: {e}")
                results[key] = None

    snap = MacroSnapshot(
        vix=vix_ib or results.get('vix_fred'),
        move=move_ib or results.get('move_yahoo'),
        sofr_ois_spread_bps=results.get('sofr_ois'),
        hy_oas_bps=results.get('hy_oas'),
        tips_10y=results.get('tips_10y'),
        breakeven_5y5y=results.get('breakeven_5y5y'),
        dxy=dxy_ib or results.get('dxy_yahoo'),
        broad_dollar=results.get('broad_dollar'),
    )
    log.info(snap.summary())
    return snap


def _fetch_ib_index(ib: IB | None, symbol: str, exchange: str) -> float | None:
    """从 IB Gateway 获取指数实时价格"""
    if ib is None or not ib.isConnected():
        return None

    try:
        contract = Index(symbol, exchange, 'USD')
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            log.info(f"[Macro] {symbol}@{exchange}: contract not found in IB")
            return None
        ib.reqMktData(contract, genericTickList='', snapshot=False)
        ib.sleep(2)

        ticker = ib.ticker(contract)
        price = ticker.marketPrice() if ticker else None
        ib.cancelMktData(contract)

        if price is not None and not np.isnan(price) and price > 0:
            log.info(f"[Macro] {symbol} = {price:.2f} (IB)")
            return float(price)

        log.info(f"[Macro] {symbol} price unavailable from IB")
        return None
    except Exception as e:
        log.warning(f"[Macro] {symbol} IB fetch failed: {e}")
        return None


def _fetch_yahoo(symbol: str, label: str) -> float | None:
    """Yahoo Finance fallback — 用于 IB/FRED 都拿不到的指标（MOVE, DXY）"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = requests.get(url, params={"range": "1d", "interval": "1d"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return None
        meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is not None and price > 0:
            log.info(f"[Macro] {label} = {price:.2f} (Yahoo)")
            return float(price)
        return None
    except Exception as e:
        log.warning(f"[Macro] {label} Yahoo fetch failed: {e}")
        return None


def _fetch_fred_vix() -> float | None:
    """VIX fallback: FRED VIXCLS 系列（T-1 日度数据）"""
    if not FRED_API_KEY:
        return None
    try:
        val = _fetch_fred_latest("VIXCLS")
        if val is not None:
            log.info(f"[Macro] VIX = {val:.2f} (FRED fallback, T-1)")
        return val
    except Exception as e:
        log.warning(f"[Macro] VIX FRED fallback failed: {e}")
        return None


def _fetch_sofr_ois_spread() -> float | None:
    """
    SOFR - OIS(1M SOFR avg) spread in basis points

    正值 = 融资压力（SOFR 高于均值）
    """
    if not FRED_API_KEY:
        return None
    try:
        sofr = _fetch_fred_latest("SOFR")
        ois = _fetch_fred_latest("SOFR30DAYAVG")
        if sofr is not None and ois is not None:
            return round((sofr - ois) * 100, 1)  # 转 bps
        return None
    except Exception as e:
        log.warning(f"SOFR-OIS spread fetch failed: {e}")
        return None


def _fetch_tips_10y() -> float | None:
    """10Y TIPS yield (DFII10) — 真实融资成本"""
    if not FRED_API_KEY:
        return None
    try:
        val = _fetch_fred_latest("DFII10")
        if val is not None:
            log.info(f"[Macro] 10Y TIPS = {val:.2f}% (FRED)")
        return val
    except Exception as e:
        log.warning(f"[Macro] 10Y TIPS fetch failed: {e}")
        return None


def _fetch_breakeven_5y5y() -> float | None:
    """5Y5Y forward breakeven inflation (T5YIFR)"""
    if not FRED_API_KEY:
        return None
    try:
        val = _fetch_fred_latest("T5YIFR")
        if val is not None:
            log.info(f"[Macro] 5Y5Y BkEven = {val:.2f}% (FRED)")
        return val
    except Exception as e:
        log.warning(f"[Macro] 5Y5Y breakeven fetch failed: {e}")
        return None


def _fetch_dxy(ib: IB | None = None) -> float | None:
    """DXY 美元指数 — IB realtime only"""
    return _fetch_ib_index(ib, 'DX-Y.NYB', 'NYBOT')


def _fetch_broad_dollar() -> float | None:
    """广义美元指数 DTWEXBGS (FRED, weekly)"""
    if not FRED_API_KEY:
        return None
    try:
        val = _fetch_fred_latest("DTWEXBGS")
        if val is not None:
            log.info(f"[Macro] Broad USD = {val:.2f} (FRED)")
        return val
    except Exception as e:
        log.warning(f"[Macro] Broad USD fetch failed: {e}")
        return None


def _fetch_hy_oas() -> float | None:
    """HY OAS spread (BAMLH0A0HYM2) in bps — 信用利差"""
    if not FRED_API_KEY:
        return None
    try:
        val = _fetch_fred_latest("BAMLH0A0HYM2")
        if val is not None:
            # FRED 返回的是百分比单位，乘 100 转 bps
            val_bps = val * 100
            log.info(f"[Macro] HY OAS = {val_bps:.0f}bps (FRED)")
            return val_bps
        return None
    except Exception as e:
        log.warning(f"[Macro] HY OAS fetch failed: {e}")
        return None


def _fetch_fred_latest(series_id: str) -> float | None:
    """从 FRED 获取最新观测值"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "limit": 5,
        "sort_order": "desc",
        "file_type": "json",
    }, timeout=10)
    for obs in r.json().get("observations", []):
        if obs["value"] != ".":
            return float(obs["value"])
    return None


ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/anthropic/v1/messages"


def interpret_macro(snap: MacroSnapshot) -> str:
    """调用 GLM-5 解读当前宏观环境，返回中文分析文本"""
    if not ZHIPU_API_KEY:
        return "⚠️ 未配置 ZHIPU_API_KEY，无法调用 AI 解读"

    prompt = f"""你是一位宏观交易员的助手。请用简洁的中文解读以下宏观数据快照，
重点分析：1) 当前宏观环境对股票/期权交易的影响 2) 需要警惕的风险信号 3) 对冲建议。

当前数据:
- 10Y TIPS (真实利率): {snap.tips_10y if snap.tips_10y is not None else 'N/A'}%  (阈值: >2.2% 警觉, >2.4% 危险)
- 5Y5Y Breakeven (通胀预期): {snap.breakeven_5y5y if snap.breakeven_5y5y is not None else 'N/A'}%
- DXY (美元指数): {snap.dxy if snap.dxy is not None else 'N/A'}
- 广义美元 DTWEXBGS: {snap.broad_dollar if snap.broad_dollar is not None else 'N/A'}
- VIX (股市波动率): {snap.vix if snap.vix is not None else 'N/A'}  (阈值: <15 平静, >25 紧张)
- MOVE (债市波动率): {snap.move if snap.move is not None else 'N/A'}  (阈值: <80 平静, >120 紧张)
- SOFR-OIS spread: {snap.sofr_ois_spread_bps if snap.sofr_ois_spread_bps is not None else 'N/A'} bps  (>5bps 融资压力)
- HY OAS (高收益债利差): {snap.hy_oas_bps if snap.hy_oas_bps is not None else 'N/A'} bps  (>400 警觉, >500 危险)

评分:
- 真实利率 score: {snap.real_yield_score:+.1f}
- 融资条件 score: {snap.funding_score:+.1f}
- 总 urgency adjustment: {snap.urgency_adjustment:+.2f}
- 宏观环境: {snap.regime_label}

请在200字以内给出判断，直接说结论，不要重复数据。"""

    try:
        r = requests.post(
            ZHIPU_BASE_URL,
            headers={
                "x-api-key": ZHIPU_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "glm-5",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if r.status_code != 200:
            return f"⚠️ API 错误 ({r.status_code}): {r.text[:200]}"

        content = r.json().get("content", [])
        for block in content:
            if block.get("type") == "text":
                return block["text"]
        return "⚠️ API 返回格式异常"
    except requests.exceptions.Timeout:
        return "⚠️ AI 解读超时，请稍后重试"
    except Exception as e:
        return f"⚠️ AI 解读失败: {e}"
