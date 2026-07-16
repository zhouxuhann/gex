"""Tech theme rotation monitor backed by IB Gateway daily bars.

This is a Python port of ``Tech Theme Rotation Monitor v1``.  The calculation
intentionally follows the Pine script's original scoring semantics rather than
trying to "improve" missing-data behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ib_insync import IB, Stock


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "tech_theme_rotation"
ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThemeConfig:
    name: str
    proxy: str
    symbols: tuple[str, ...]
    is_basket: bool


@dataclass(frozen=True)
class MonitorParams:
    trend_len: int = 20
    slow_len: int = 50
    vol_len: int = 20
    vol_confirm: float = 1.30
    crowd_rel_60: float = 18.0
    crowd_dist_50: float = 8.0


DEFAULT_BENCHMARK = "QQQ"
DEFAULT_THEMES: tuple[ThemeConfig, ...] = (
    ThemeConfig("半导体", "SMH", ("SMH",), False),
    ThemeConfig("软件", "IGV", ("IGV",), False),
    ThemeConfig("云计算", "SKYY", ("SKYY",), False),
    ThemeConfig("网络安全", "CIBR", ("CIBR",), False),
    ThemeConfig("机器人", "BOTZ", ("BOTZ",), False),
    ThemeConfig("AI综合", "AIQ", ("AIQ",), False),
    ThemeConfig("光模块", "Basket", ("ANET", "CIEN", "COHR", "LITE", "AAOI"), True),
    ThemeConfig("储存", "Basket", ("MU", "WDC", "STX"), True),
    ThemeConfig("数据中心电力散热", "Basket", ("VRT", "ETN", "PWR", "CEG", "GEV"), True),
    ThemeConfig("云巨头", "Basket", ("MSFT", "AMZN", "GOOGL", "META"), True),
    ThemeConfig("AI硬件卖方", "Basket", ("NVDA", "AVGO", "AMD", "ANET"), True),
)


TREND_TEXT = {
    2: "强",
    1: "偏强",
    -1: "偏弱",
    -2: "弱",
    0: "中性",
}
STATE_TEXT = {
    3: "拥挤主升",
    2: "确认进入",
    1: "早期轮动",
    0: "中性观察",
    -1: "资金撤出",
    -2: "派发/撤出",
    9: "无数据",
}


def _clean_float(value, *, positive: bool = False) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if positive and out <= 0:
        return None
    return out


def _bar_day(value) -> date:
    if isinstance(value, datetime):
        return value.astimezone(ET).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            source = text[:8] if fmt == "%Y%m%d" else text[:10]
            return datetime.strptime(source, fmt).date()
        except ValueError:
            pass
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(ET).date() if parsed.tzinfo else parsed.date()


def _normalize_symbol(symbol: str) -> str:
    text = symbol.strip().upper()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text


def _unique_symbols(benchmark: str, themes: Iterable[ThemeConfig]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for symbol in [benchmark, *(sym for theme in themes for sym in theme.symbols)]:
        normalized = _normalize_symbol(symbol)
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _connect(args: argparse.Namespace) -> "IB":
    try:
        from ib_insync import IB
    except ImportError as exc:
        raise RuntimeError("ib_insync is required to fetch data from IB Gateway") from exc

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True, timeout=args.timeout)
    ib.reqMarketDataType(args.market_data_type)
    return ib


def _stock_contract(ib: "IB", symbol: str) -> "Stock":
    try:
        from ib_insync import Stock
    except ImportError as exc:
        raise RuntimeError("ib_insync is required to qualify IB stock contracts") from exc

    contract = Stock(_normalize_symbol(symbol), "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"Could not qualify stock contract for {symbol}")
    return qualified[0]


def _duration_for_bars(calc_bars: int) -> str:
    # Calendar days: enough cushion for weekends and market holidays.
    return f"{max(120, int(math.ceil(calc_bars * 1.8)))} D"


def fetch_daily_bars(
    ib: "IB",
    symbols: Iterable[str],
    *,
    calc_bars: int,
    duration: str | None,
    use_rth: bool,
    symbol_sleep: float,
) -> dict[str, pd.DataFrame]:
    """Fetch daily close/volume data from IB Gateway."""
    out: dict[str, pd.DataFrame] = {}
    duration_str = duration or _duration_for_bars(calc_bars)

    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        try:
            contract = _stock_contract(ib, normalized)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            log.warning("[%s] historical fetch failed: %s", normalized, exc)
            out[normalized] = pd.DataFrame(columns=["close", "volume"])
            continue

        rows: list[dict] = []
        for bar in bars:
            close = _clean_float(getattr(bar, "close", None), positive=True)
            if close is None:
                continue
            volume = _clean_float(getattr(bar, "volume", None))
            rows.append({"date": _bar_day(getattr(bar, "date")), "close": close, "volume": volume})

        df = pd.DataFrame(rows)
        if df.empty:
            out[normalized] = pd.DataFrame(columns=["close", "volume"])
        else:
            df = df.drop_duplicates("date").sort_values("date").tail(calc_bars)
            out[normalized] = df.set_index("date")[["close", "volume"]]
        log.info("[%s] fetched %d daily bars", normalized, len(out[normalized]))

        if symbol_sleep > 0:
            time.sleep(symbol_sleep)

    return out


def align_field(bars: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Outer-align a close or volume field across symbols."""
    series_map = {
        symbol: df[field].astype(float)
        for symbol, df in bars.items()
        if field in df.columns and not df.empty
    }
    if not series_map:
        return pd.DataFrame()
    return pd.DataFrame(series_map).sort_index()


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a / b
    return result.where(a.notna() & b.notna() & (b != 0.0))


def pct_change_like_pine(x: pd.Series, length: int) -> pd.Series:
    prev = x.shift(length)
    return (100.0 * (x / prev - 1.0)).where(x.notna() & prev.notna() & (prev != 0.0))


def ret_one(c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    return (c / prev - 1.0).where(c.notna() & prev.notna() & (prev != 0.0))


def avg_ret_like_pine(close_df: pd.DataFrame, symbols: tuple[str, ...]) -> pd.Series:
    returns = [ret_one(close_df.get(symbol, pd.Series(index=close_df.index, dtype=float))) for symbol in symbols]
    ret_df = pd.concat(returns, axis=1)
    return ret_df.mean(axis=1, skipna=True).fillna(0.0)


def basket_index_like_pine(basket_ret: pd.Series) -> pd.Series:
    values: list[float] = []
    prev = np.nan
    for value in basket_ret.fillna(0.0):
        current = 100.0 if pd.isna(prev) else prev * (1.0 + value)
        values.append(current)
        prev = current
    return pd.Series(values, index=basket_ret.index, dtype=float)


def dvol_like_pine(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (close * volume).fillna(0.0)


def above_fast_trend(c: pd.Series, trend_len: int) -> pd.Series:
    ma = c.rolling(trend_len, min_periods=trend_len).mean()
    return c.notna() & (c > ma)


def breadth5_like_pine(close_df: pd.DataFrame, symbols: tuple[str, ...], trend_len: int) -> pd.Series:
    n = pd.Series(0, index=close_df.index, dtype=float)
    a = pd.Series(0, index=close_df.index, dtype=float)
    for symbol in symbols:
        c = close_df.get(symbol, pd.Series(index=close_df.index, dtype=float))
        present = c.notna()
        n = n + present.astype(float)
        a = a + (present & above_fast_trend(c, trend_len)).astype(float)
    return (100.0 * a / n).where(n > 0)


def breadth_etf_like_pine(c: pd.Series, trend_len: int) -> pd.Series:
    ma = c.rolling(trend_len, min_periods=trend_len).mean()
    return pd.Series(np.where(c.isna(), np.nan, np.where(c > ma, 70.0, 30.0)), index=c.index)


def _trend_text(code: int | float) -> str:
    return TREND_TEXT.get(int(code), "中性") if not pd.isna(code) else "中性"


def _state_text(code: int | float) -> str:
    return STATE_TEXT.get(int(code), "中性") if not pd.isna(code) else "无数据"


def calc_metrics(
    idx: pd.Series,
    dollar_vol: pd.Series,
    breadth_pct: pd.Series,
    benchmark_close: pd.Series,
    params: MonitorParams,
) -> pd.DataFrame:
    ratio = safe_div(idx, benchmark_close)

    rel1 = pct_change_like_pine(ratio, 1)
    rel5 = pct_change_like_pine(ratio, 5)
    rel20 = pct_change_like_pine(ratio, 20)
    rel60 = pct_change_like_pine(ratio, 60)

    ma20 = ratio.rolling(params.trend_len, min_periods=params.trend_len).mean()
    ma50 = ratio.rolling(params.slow_len, min_periods=params.slow_len).mean()
    vol_ma = dollar_vol.rolling(params.vol_len, min_periods=params.vol_len).mean()
    vol_r = safe_div(dollar_vol, vol_ma)
    dist50 = (100.0 * (ratio / ma50 - 1.0)).where(ratio.notna() & ma50.notna() & (ma50 != 0.0))

    valid = ratio.notna() & benchmark_close.notna()

    trend_score = (
        (ratio > ma20).astype(float) * 10.0
        + (ratio > ma50).astype(float) * 10.0
        + (ma20 > ma20.shift(5)).astype(float) * 10.0
    )

    acc_score = (
        (rel5 > 0).astype(float) * 8.0
        + (rel5 > rel20 / 4.0).astype(float) * 9.0
        + (rel5 > rel5.shift(5)).astype(float) * 8.0
    )

    vol_score = pd.Series(
        np.where(
            (rel1 > 0) & (vol_r >= params.vol_confirm),
            20.0,
            np.where((rel1 > 0) & (vol_r >= 1.0), 10.0, 0.0),
        ),
        index=idx.index,
    )

    br_score = pd.Series(
        np.where(
            breadth_pct.isna(),
            0.0,
            np.where(breadth_pct >= 70.0, 15.0, np.where(breadth_pct >= 50.0, 8.0, 0.0)),
        ),
        index=idx.index,
    )

    over_ext = ((rel60.notna()) & (rel60 > params.crowd_rel_60)) | (
        (dist50.notna()) & (dist50 > params.crowd_dist_50)
    )
    crowd_score = pd.Series(np.where(over_ext, 0.0, 10.0), index=idx.index)

    raw_score = trend_score + acc_score + vol_score + br_score + crowd_score
    score = raw_score.where(valid)

    distribution = valid & (ratio < ma20) & (rel5 < 0) & (vol_r > params.vol_confirm) & (rel1 < 0)

    trend_code = pd.Series(0, index=idx.index, dtype=int)
    trend_code = trend_code.mask((ratio > ma20) & (ratio > ma50), 2)
    trend_code = trend_code.mask(~((ratio > ma20) & (ratio > ma50)) & (ratio > ma20), 1)
    trend_code = trend_code.mask((ratio < ma20) & (ratio < ma50), -2)
    trend_code = trend_code.mask(~((ratio < ma20) & (ratio < ma50)) & (ratio < ma20), -1)

    state = pd.Series(0, index=idx.index, dtype=int)
    state = state.mask(~valid, 9)
    state = state.mask(valid & distribution, -2)
    state = state.mask(valid & ~distribution & over_ext & (score >= 60), 3)
    state = state.mask(valid & ~distribution & ~(over_ext & (score >= 60)) & (score >= 75), 2)
    state = state.mask(
        valid
        & ~distribution
        & ~(over_ext & (score >= 60))
        & ~(score >= 75)
        & (score >= 60),
        1,
    )
    state = state.mask(
        valid
        & ~distribution
        & ~(over_ext & (score >= 60))
        & ~(score >= 75)
        & ~(score >= 60)
        & (score < 45)
        & (rel5 < 0)
        & (rel20 < 0),
        -1,
    )

    return pd.DataFrame(
        {
            "rel1": rel1,
            "rel5": rel5,
            "rel20": rel20,
            "rel60": rel60,
            "vol_r": vol_r,
            "breadth": breadth_pct,
            "score": score,
            "state": state,
            "trend": trend_code,
        }
    )


def compute_theme_rotation(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    themes: tuple[ThemeConfig, ...] = DEFAULT_THEMES,
    params: MonitorParams = MonitorParams(),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute the latest dashboard rows and full metric history."""
    benchmark = _normalize_symbol(benchmark)
    if benchmark not in close_df:
        raise ValueError(f"Benchmark {benchmark} is missing from close data")

    close_df = close_df.sort_index()
    volume_df = volume_df.reindex(close_df.index).sort_index()
    benchmark_close = close_df[benchmark]
    histories: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []

    for theme in themes:
        symbols = tuple(_normalize_symbol(symbol) for symbol in theme.symbols)
        if theme.is_basket:
            theme_ret = avg_ret_like_pine(close_df, symbols)
            idx = basket_index_like_pine(theme_ret)
            dollar_vol = sum(
                dvol_like_pine(
                    close_df.get(symbol, pd.Series(index=close_df.index, dtype=float)),
                    volume_df.get(symbol, pd.Series(index=close_df.index, dtype=float)),
                )
                for symbol in symbols
            )
            breadth = breadth5_like_pine(close_df, symbols, params.trend_len)
        else:
            symbol = symbols[0]
            idx = close_df.get(symbol, pd.Series(index=close_df.index, dtype=float))
            dollar_vol = dvol_like_pine(
                idx,
                volume_df.get(symbol, pd.Series(index=close_df.index, dtype=float)),
            )
            breadth = breadth_etf_like_pine(idx, params.trend_len)

        metrics = calc_metrics(idx, dollar_vol, breadth, benchmark_close, params)
        histories[theme.name] = metrics
        latest = metrics.iloc[-1] if not metrics.empty else pd.Series(dtype=float)
        rows.append(
            {
                "主题": theme.name,
                "代理": theme.proxy,
                "1D相对%": latest.get("rel1", np.nan),
                "5D相对%": latest.get("rel5", np.nan),
                "20D相对%": latest.get("rel20", np.nan),
                "60D相对%": latest.get("rel60", np.nan),
                "趋势": _trend_text(latest.get("trend", 0)),
                "量能比": latest.get("vol_r", np.nan),
                "广度%": latest.get("breadth", np.nan),
                "分数": latest.get("score", np.nan),
                "状态": _state_text(latest.get("state", 9)),
                "state_code": int(latest.get("state", 9)) if not pd.isna(latest.get("state", 9)) else 9,
                "trend_code": int(latest.get("trend", 0)) if not pd.isna(latest.get("trend", 0)) else 0,
            }
        )

    return pd.DataFrame(rows), histories


def _fmt_pct(value: float) -> str:
    return "na" if pd.isna(value) else f"{value:+.2f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    return "na" if pd.isna(value) else f"{value:.{digits}f}"


def _fmt_score(value: float) -> str:
    return "na" if pd.isna(value) else f"{value:.0f}"


def format_dashboard(rows: pd.DataFrame, as_of: object) -> str:
    lines = [f"Tech Theme Rotation Monitor | as of {as_of}"]
    header = f"{'主题':<14} {'代理':<7} {'5D相对':>9} {'20D相对':>9} {'趋势':<4} {'量能比':>7} {'广度':>8} {'分数':>5} 状态"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in rows.iterrows():
        lines.append(
            f"{row['主题']:<14} "
            f"{row['代理']:<7} "
            f"{_fmt_pct(row['5D相对%']):>9} "
            f"{_fmt_pct(row['20D相对%']):>9} "
            f"{row['趋势']:<4} "
            f"{(_fmt_num(row['量能比']) + 'x'):>7} "
            f"{_fmt_pct(row['广度%']):>8} "
            f"{_fmt_score(row['分数']):>5} "
            f"{row['状态']}"
        )
    return "\n".join(lines)


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tech theme rotation monitor via IB Gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=393)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--market-data-type", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--calc-bars", type=int, default=300)
    parser.add_argument("--duration", default=None, help="IB durationStr override, e.g. '2 Y'")
    parser.add_argument("--use-rth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symbol-sleep", type=float, default=0.25)
    parser.add_argument("--trend-len", type=int, default=20)
    parser.add_argument("--slow-len", type=int, default=50)
    parser.add_argument("--vol-len", type=int, default=20)
    parser.add_argument("--vol-confirm", type=float, default=1.30)
    parser.add_argument("--crowd-rel-60", type=float, default=18.0)
    parser.add_argument("--crowd-dist-50", type=float, default=8.0)
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def run_once(args: argparse.Namespace) -> pd.DataFrame:
    params = MonitorParams(
        trend_len=args.trend_len,
        slow_len=args.slow_len,
        vol_len=args.vol_len,
        vol_confirm=args.vol_confirm,
        crowd_rel_60=args.crowd_rel_60,
        crowd_dist_50=args.crowd_dist_50,
    )
    symbols = _unique_symbols(args.benchmark, DEFAULT_THEMES)

    ib = _connect(args)
    try:
        bars = fetch_daily_bars(
            ib,
            symbols,
            calc_bars=args.calc_bars,
            duration=args.duration,
            use_rth=args.use_rth,
            symbol_sleep=args.symbol_sleep,
        )
    finally:
        ib.disconnect()

    close_df = align_field(bars, "close")
    volume_df = align_field(bars, "volume")
    rows, _histories = compute_theme_rotation(
        close_df,
        volume_df,
        benchmark=args.benchmark,
        params=params,
    )

    as_of = close_df.index.max() if not close_df.empty else "na"
    print(format_dashboard(rows, as_of))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(args.save_csv, index=False)
        log.info("saved CSV: %s", args.save_csv)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "as_of": as_of,
            "benchmark": _normalize_symbol(args.benchmark),
            "rows": rows.to_dict(orient="records"),
        }
        args.save_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        log.info("saved JSON: %s", args.save_json)

    return rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
