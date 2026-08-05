"""Daily replay and robust parameter optimization for Universal v2.1.

The optimizer uses only completed IB one-minute bars.  Signals observed at a
bar close are executed at the next bar's open, preventing look-ahead.  A
coordinate search is fitted on older days and must beat the v2.1 baseline on
the most recent holdout days before a recommendation is emitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .multi_asset_momentum import MomentumConfig, calculate_momentum_signals
from .time_utils import ET, et_now, market_session_today

log = logging.getLogger("momentum_daily_optimizer")

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT_DIR / "logs"
DEFAULT_BAR_DIR = DEFAULT_LOG_DIR / "momentum_0dte_bars"
DEFAULT_OUTPUT_DIR = DEFAULT_LOG_DIR / "momentum_optimization"

AUTO_KALMAN = {"QQQ": 0.100, "SPY": 0.075}
AUTO_ATR = {"QQQ": 0.70, "SPY": 0.50}


@dataclass(frozen=True)
class ReplayMetrics:
    trades: int
    net_points: float
    avg_points: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    positive_day_rate: float
    score: float


@dataclass(frozen=True)
class ReplayTrade:
    session: str
    direction: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    pnl_points: float


@dataclass(frozen=True)
class ShadowAlignmentMetrics:
    eligible_trades: int
    aligned_trades: int
    realized_events: int
    coverage: float
    bias_agreement_rate: float
    realized_direction_accuracy: float
    reversal_conflict_rate: float
    score: float


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else fallback


def load_symbol_bars(
    data_dir: Path,
    symbol: str,
    lookback_days: int,
    *,
    official_data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the most recent daily journals and normalize their timestamps."""

    paths = sorted(data_dir.glob(f"{symbol.upper()}_????????.csv"))[-lookback_days:]
    frames: list[pd.DataFrame] = []
    if official_data_dir is not None:
        official_paths = sorted(
            official_data_dir.glob(f"official_ohlc_{symbol.upper()}_????????.parquet")
        )[-lookback_days:]
        for path in official_paths:
            try:
                frame = pd.read_parquet(path)
                if "ts" not in frame.columns:
                    continue
                frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("ts"), utc=True))
                required = ["open", "high", "low", "close", "volume"]
                if any(column not in frame.columns for column in required):
                    continue
                frames.append(frame[required])
            except Exception:
                log.exception("Could not load official replay bars: %s", path)
    for path in paths:
        try:
            frame = pd.read_csv(path)
            if "timestamp" not in frame.columns:
                continue
            frame.index = pd.DatetimeIndex(
                pd.to_datetime(frame.pop("timestamp"), utc=True), name="timestamp"
            )
            required = ["open", "high", "low", "close", "volume"]
            if any(column not in frame.columns for column in required):
                continue
            frames.append(frame[required])
        except Exception:
            log.exception("Could not load replay bars: %s", path)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = pd.concat(frames).sort_index()
    # Journals are appended after official files and therefore win when both
    # sources contain the same timestamp.
    return result[~result.index.duplicated(keep="last")]


def load_shadow_events(shadow_data_dir: Path, symbol: str, lookback_days: int) -> pd.DataFrame:
    paths = sorted(shadow_data_dir.glob(f"turning_point_shadow_{symbol.upper()}_????????.parquet"))[
        -lookback_days:
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path)
            if "ts" not in frame.columns:
                continue
            frame = frame.copy()
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
            frames.append(frame)
        except Exception:
            log.exception("Could not load turning-point shadow data: %s", path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("ts")


def session_dates(frame: pd.DataFrame) -> pd.Index:
    if frame.empty:
        return pd.Index([])
    return pd.Index(frame.index.tz_convert(ET).date).unique().sort_values()


def replay_signals(
    bars: pd.DataFrame,
    symbol: str,
    config: MomentumConfig,
    *,
    round_trip_cost_points: float = 0.04,
) -> tuple[ReplayMetrics, list[ReplayTrade]]:
    """Replay full-arrow entries/exits using next-bar-open execution."""

    if bars.empty:
        return _metrics([], {}), []
    result = calculate_momentum_signals(bars, symbol=symbol, config=config)
    et_dates = np.asarray(result.index.tz_convert(ET).date)
    trades: list[ReplayTrade] = []
    daily_pnl: dict[str, float] = {}

    for session_date in pd.Index(et_dates).unique():
        day = result.loc[et_dates == session_date]
        if len(day) < 2:
            continue
        direction: str | None = None
        entry_price = 0.0
        entry_ts: pd.Timestamp | None = None

        def close_position(exit_ts: pd.Timestamp, exit_price: float) -> None:
            nonlocal direction, entry_price, entry_ts
            if direction is None or entry_ts is None:
                return
            multiplier = 1.0 if direction == "long" else -1.0
            pnl = multiplier * (exit_price - entry_price) - round_trip_cost_points
            key = str(session_date)
            daily_pnl[key] = daily_pnl.get(key, 0.0) + pnl
            trades.append(
                ReplayTrade(
                    session=key,
                    direction=direction,
                    entry_ts=entry_ts.isoformat(),
                    exit_ts=exit_ts.isoformat(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_points=pnl,
                )
            )
            direction = None
            entry_price = 0.0
            entry_ts = None

        for i in range(1, len(day)):
            previous = day.iloc[i - 1]
            current = day.iloc[i]
            current_ts = day.index[i]
            current_open = float(current["open"])
            if direction == "long" and not bool(previous["long_signal"]):
                close_position(current_ts, current_open)
            elif direction == "short" and not bool(previous["short_signal"]):
                close_position(current_ts, current_open)

            if direction is None:
                if bool(previous["long_entry"]):
                    direction = "long"
                    entry_price = current_open
                    entry_ts = current_ts
                elif bool(previous["short_entry"]):
                    direction = "short"
                    entry_price = current_open
                    entry_ts = current_ts

        if direction is not None:
            close_position(day.index[-1], float(day.iloc[-1]["close"]))

    return _metrics([trade.pnl_points for trade in trades], daily_pnl), trades


def _metrics(pnls: list[float], daily_pnl: dict[str, float]) -> ReplayMetrics:
    if not pnls:
        return ReplayMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -999.0)
    values = np.asarray(pnls, dtype=float)
    profits = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    profit_factor = profits / losses if losses > 0 else 5.0 if profits > 0 else 0.0
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdowns = peaks[1:] - equity
    max_drawdown = float(drawdowns.max(initial=0.0))
    daily_values = np.asarray(list(daily_pnl.values()), dtype=float)
    daily_mean = float(daily_values.mean()) if len(daily_values) else 0.0
    daily_std = float(daily_values.std(ddof=0)) if len(daily_values) else 0.0
    risk_adjusted = daily_mean / (daily_std + 0.20) * math.sqrt(max(1, len(daily_values)))
    risk_adjusted = float(np.clip(risk_adjusted, -5.0, 5.0))
    win_rate = float((values > 0).mean())
    positive_day_rate = float((daily_values > 0).mean()) if len(daily_values) else 0.0
    score = (
        risk_adjusted
        + 0.25 * min(profit_factor, 3.0)
        + 0.50 * win_rate
        + 0.35 * positive_day_rate
        - 0.10 * max_drawdown
    )
    return ReplayMetrics(
        trades=len(values),
        net_points=float(values.sum()),
        avg_points=float(values.mean()),
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        positive_day_rate=positive_day_rate,
        score=score,
    )


def align_trades_with_shadow(
    trades: list[ReplayTrade],
    shadow_events: pd.DataFrame,
    *,
    tolerance_minutes: int = 15,
) -> ShadowAlignmentMetrics:
    """Score momentum trades against the forward-only turning-point shadow."""

    if not trades or shadow_events.empty:
        return ShadowAlignmentMetrics(len(trades), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    events = shadow_events.copy()
    events["ts"] = pd.to_datetime(events["ts"], utc=True)
    events = events.sort_values("ts")
    agreements: list[bool] = []
    realized_matches: list[bool] = []
    conflicts: list[bool] = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for trade in trades:
        entry_ts = pd.Timestamp(trade.entry_ts)
        entry_ts = (
            entry_ts.tz_convert("UTC") if entry_ts.tz is not None else entry_ts.tz_localize("UTC")
        )
        prior = events[(events["ts"] <= entry_ts) & (events["ts"] >= entry_ts - tolerance)]
        if prior.empty:
            continue
        event = prior.iloc[-1]
        bias = str(event.get("score_bias", "neutral"))
        setup = str(event.get("setup_direction", ""))
        if bias not in ("continuation", "reversal") or setup not in ("after_up", "after_down"):
            continue
        impulse_direction = "long" if setup == "after_up" else "short"
        expected = impulse_direction
        if bias == "reversal":
            expected = "short" if impulse_direction == "long" else "long"
        agreement = trade.direction == expected
        agreements.append(agreement)
        conflicts.append(not agreement)

        outcome = str(event.get("realized_outcome", ""))
        realized_direction = None
        if outcome.endswith("_up"):
            realized_direction = "long"
        elif outcome.endswith("_down"):
            realized_direction = "short"
        if realized_direction is not None:
            realized_matches.append(trade.direction == realized_direction)

    aligned = len(agreements)
    coverage = aligned / len(trades) if trades else 0.0
    agreement_rate = float(np.mean(agreements)) if agreements else 0.0
    accuracy = float(np.mean(realized_matches)) if realized_matches else 0.0
    conflict_rate = float(np.mean(conflicts)) if conflicts else 0.0
    score = 0.35 * coverage + 0.35 * agreement_rate + 0.50 * accuracy - 0.35 * conflict_rate
    return ShadowAlignmentMetrics(
        eligible_trades=len(trades),
        aligned_trades=aligned,
        realized_events=len(realized_matches),
        coverage=coverage,
        bias_agreement_rate=agreement_rate,
        realized_direction_accuracy=accuracy,
        reversal_conflict_rate=conflict_rate,
        score=score,
    )


def _baseline(symbol: str) -> MomentumConfig:
    return MomentumConfig(
        asset=symbol,  # type: ignore[arg-type]
        kalman_thresh=AUTO_KALMAN[symbol],
        atr_min=AUTO_ATR[symbol],
    )


def _search_space(symbol: str) -> dict[str, list[Any]]:
    kalman = AUTO_KALMAN[symbol]
    atr = AUTO_ATR[symbol]
    return {
        "fast_n": [6, 8, 10, 12],
        "slow_n": [28, 34, 40, 48],
        "z_lo": [0.4, 0.6, 0.8, 1.0],
        "z_hi": [2.0, 2.5, 3.0],
        "z_std_win": [15, 20, 30],
        "kalman_q_price": [0.00005, 0.0001, 0.0002],
        "kalman_q_vel": [0.000005, 0.00001, 0.00002],
        "kalman_r": [0.005, 0.01, 0.02],
        "kalman_thresh": [round(kalman * scale, 6) for scale in (0.80, 1.0, 1.20, 1.40)],
        "atr_len": [10, 14, 20],
        "atr_min": [round(atr * scale, 4) for scale in (0.75, 1.0, 1.25, 1.50)],
        "vote_thresh": [2, 3],
        "no_trade_open": [0, 3, 5],
        "no_trade_close": [10, 15, 20, 30],
    }


def _params(config: MomentumConfig) -> dict[str, Any]:
    keys = _search_space(str(config.asset)).keys()
    return {key: getattr(config, key) for key in keys}


def _frame_for_dates(frame: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    allowed = set(dates)
    mask = [value in allowed for value in frame.index.tz_convert(ET).date]
    return frame.loc[mask]


def coordinate_search(
    train: pd.DataFrame, symbol: str, *, rounds: int = 2
) -> tuple[MomentumConfig, ReplayMetrics, int]:
    """Small, deterministic search that limits multiple-testing pressure."""

    current = _baseline(symbol)
    current_metrics, _ = replay_signals(train, symbol, current)
    evaluations = 1
    cache: dict[tuple[tuple[str, Any], ...], ReplayMetrics] = {
        tuple(sorted(_params(current).items())): current_metrics
    }
    for _ in range(rounds):
        improved = False
        for key, options in _search_space(symbol).items():
            best_config = current
            best_metrics = current_metrics
            for value in options:
                candidate = replace(current, **{key: value})
                if candidate.fast_n >= candidate.slow_n:
                    continue
                cache_key = tuple(sorted(_params(candidate).items()))
                metrics = cache.get(cache_key)
                if metrics is None:
                    metrics, _ = replay_signals(train, symbol, candidate)
                    cache[cache_key] = metrics
                    evaluations += 1
                minimum_trades = max(5, len(session_dates(train)) // 2)
                if metrics.trades >= minimum_trades and metrics.score > best_metrics.score + 1e-9:
                    best_config, best_metrics = candidate, metrics
            if best_config != current:
                current, current_metrics = best_config, best_metrics
                improved = True
        if not improved:
            break
    return current, current_metrics, evaluations


def optimize_symbol(
    bars: pd.DataFrame,
    symbol: str,
    *,
    shadow_events: pd.DataFrame | None = None,
    minimum_days: int = 5,
    holdout_days: int = 3,
) -> dict[str, Any]:
    dates = session_dates(bars)
    if len(dates) < minimum_days:
        return {
            "status": "insufficient_data",
            "days": int(len(dates)),
            "required_days": minimum_days,
            "params": _params(_baseline(symbol)),
        }
    validation_count = min(max(2, holdout_days), max(2, len(dates) // 3))
    train_dates = dates[:-validation_count]
    validation_dates = dates[-validation_count:]
    train = _frame_for_dates(bars, train_dates)
    validation = _frame_for_dates(bars, validation_dates)
    baseline = _baseline(symbol)
    candidate, train_candidate, evaluations = coordinate_search(train, symbol)
    train_baseline, _ = replay_signals(train, symbol, baseline)
    validation_baseline, baseline_trades = replay_signals(validation, symbol, baseline)
    validation_candidate, candidate_trades = replay_signals(validation, symbol, candidate)
    shadow_frame = shadow_events if shadow_events is not None else pd.DataFrame()
    shadow_baseline = align_trades_with_shadow(baseline_trades, shadow_frame)
    shadow_candidate = align_trades_with_shadow(candidate_trades, shadow_frame)
    joint_baseline_score = validation_baseline.score + 0.75 * shadow_baseline.score
    joint_candidate_score = validation_candidate.score + 0.75 * shadow_candidate.score
    minimum_validation_trades = max(6, validation_count * 2)
    participation_floor = max(1, math.ceil(validation_baseline.trades * 0.40))
    enough_validation = validation_candidate.trades >= max(
        minimum_validation_trades, participation_floor
    )
    shadow_available = not shadow_frame.empty
    shadow_gate = not shadow_available or (
        shadow_candidate.aligned_trades >= 2
        and shadow_candidate.realized_events >= 2
        and joint_candidate_score > joint_baseline_score + 0.03
        and shadow_candidate.reversal_conflict_rate <= shadow_baseline.reversal_conflict_rate + 0.10
    )
    validation_wins = shadow_gate and (
        joint_candidate_score > joint_baseline_score + 0.05
        and validation_candidate.net_points > validation_baseline.net_points
    )
    status = "recommended" if enough_validation and validation_wins else "keep_baseline"
    selected = candidate if status == "recommended" else baseline
    return {
        "status": status,
        "days": int(len(dates)),
        "train_days": [str(value) for value in train_dates],
        "validation_days": [str(value) for value in validation_dates],
        "evaluations": evaluations,
        "minimum_validation_trades": minimum_validation_trades,
        "participation_floor": participation_floor,
        "params": _params(selected),
        "candidate_params": _params(candidate),
        "train_baseline": asdict(train_baseline),
        "train_candidate": asdict(train_candidate),
        "validation_baseline": asdict(validation_baseline),
        "validation_candidate": asdict(validation_candidate),
        "shadow_available": shadow_available,
        "shadow_baseline": asdict(shadow_baseline),
        "shadow_candidate": asdict(shadow_candidate),
        "joint_baseline_score": joint_baseline_score,
        "joint_candidate_score": joint_candidate_score,
    }


def _actual_trade_summary(trade_log_dir: Path, date_text: str) -> dict[str, Any]:
    paths = list(trade_log_dir.glob(f"momentum_0dte_paper_trades_{date_text}.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            log.exception("Could not read trade log %s", path)
    if not frames:
        return {"exit_fills": 0, "pnl_usd": 0.0, "winners": 0, "losers": 0}
    data = pd.concat(frames, ignore_index=True)
    if not {"event", "filled_qty"}.issubset(data.columns):
        return {"exit_fills": 0, "pnl_usd": 0.0, "winners": 0, "losers": 0}
    exits = data[
        (data["event"] == "EXIT") & pd.to_numeric(data["filled_qty"], errors="coerce").gt(0)
    ]
    pnl_source = (
        exits["pnl_usd"] if "pnl_usd" in exits.columns else pd.Series(0.0, index=exits.index)
    )
    pnl = pd.to_numeric(pnl_source, errors="coerce").fillna(0.0)
    return {
        "exit_fills": int(len(exits)),
        "pnl_usd": float(pnl.sum()),
        "winners": int((pnl >= 0).sum()),
        "losers": int((pnl < 0).sum()),
    }


def _metric_row(label: str, values: dict[str, Any] | None) -> str:
    if not values:
        return f"| {label} | — | — | — | — | — |"
    return (
        f"| {label} | {values['trades']} | {values['net_points']:.2f} | "
        f"{values['win_rate']:.1%} | {values['profit_factor']:.2f} | "
        f"{values['max_drawdown']:.2f} |"
    )


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# Universal v2.1 每日复盘与优化 — {payload['as_of_date']}",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## Paper 实际成交",
        "",
        f"- 平仓成交腿数：{payload['actual_trades']['exit_fills']}",
        f"- 已实现损益：${payload['actual_trades']['pnl_usd']:.2f}",
        (
            f"- 盈利/亏损腿：{payload['actual_trades']['winners']} / "
            f"{payload['actual_trades']['losers']}"
        ),
        "",
    ]
    for symbol, result in payload["recommendations"].items():
        lines.extend([f"## {symbol}", ""])
        if result["status"] == "insufficient_data":
            lines.extend(
                [
                    (
                        f"数据不足：现有 {result['days']} 个交易日，"
                        f"需要至少 {result['required_days']} 日。"
                    ),
                    "",
                ]
            )
            continue
        status_text = (
            "建议下个交易日采用候选参数"
            if result["status"] == "recommended"
            else "候选未通过样本外验证，保持 v2.1 基线"
        )
        lines.extend(
            [
                f"结论：{status_text}。",
                f"训练日：{', '.join(result['train_days'])}",
                f"验证日：{', '.join(result['validation_days'])}",
                f"搜索评估次数：{result['evaluations']}",
                "",
                "| 区间 | 交易数 | 净点数 | 胜率 | Profit Factor | 最大回撤 |",
                "|---|---:|---:|---:|---:|---:|",
                _metric_row("训练基线", result["train_baseline"]),
                _metric_row("训练候选", result["train_candidate"]),
                _metric_row("验证基线", result["validation_baseline"]),
                _metric_row("验证候选", result["validation_candidate"]),
                "",
                "影子评分联合验证：",
                "",
                (
                    f"- 可用：{result['shadow_available']}；基线/候选对齐交易数："
                    f"{result['shadow_baseline']['aligned_trades']} / "
                    f"{result['shadow_candidate']['aligned_trades']}"
                ),
                (
                    f"- 候选验证交易门槛：至少 "
                    f"{max(result['minimum_validation_trades'], result['participation_floor'])} 笔"
                ),
                (
                    f"- 基线/候选方向正确率："
                    f"{result['shadow_baseline']['realized_direction_accuracy']:.1%} / "
                    f"{result['shadow_candidate']['realized_direction_accuracy']:.1%}"
                ),
                (
                    f"- 基线/候选反转冲突率："
                    f"{result['shadow_baseline']['reversal_conflict_rate']:.1%} / "
                    f"{result['shadow_candidate']['reversal_conflict_rate']:.1%}"
                ),
                (
                    f"- 联合分数（行情回放 + 影子）："
                    f"{result['joint_baseline_score']:.3f} / "
                    f"{result['joint_candidate_score']:.3f}"
                ),
                "",
                "最终参数：",
                "",
                "```json",
                json.dumps(result["params"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 方法说明",
            "",
            "- 信号在 K 线收盘确认，统一按下一根 K 线开盘成交，避免未来数据泄漏。",
            "- 每笔回放扣除 0.04 个标的点的往返摩擦成本。",
            "- 参数只在较早日期训练，并必须在最近日期验证集同时改善综合分数和净点数。",
            "- 若有转折点影子数据，候选还必须通过延续/反转偏向和事后方向的联合门槛。",
            "- 这是标的方向代理回放；真实期权损益另在 Paper 成交区统计。",
            "",
        ]
    )
    return "\n".join(lines)


def run_daily_optimization(
    *,
    bar_data_dir: Path = DEFAULT_BAR_DIR,
    official_data_dir: Path | None = None,
    shadow_data_dir: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    trade_log_dir: Path = DEFAULT_LOG_DIR,
    lookback_days: int = 20,
    minimum_days: int = 5,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    now = as_of or et_now()
    date_text = now.strftime("%Y%m%d")
    recommendations = {
        symbol: optimize_symbol(
            load_symbol_bars(
                bar_data_dir,
                symbol,
                lookback_days,
                official_data_dir=official_data_dir,
            ),
            symbol,
            shadow_events=(
                load_shadow_events(shadow_data_dir, symbol, lookback_days)
                if shadow_data_dir is not None
                else None
            ),
            minimum_days=minimum_days,
        )
        for symbol in ("QQQ", "SPY")
    }
    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "as_of_date": now.date().isoformat(),
        "lookback_days": lookback_days,
        "recommendations": recommendations,
        "actual_trades": _actual_trade_summary(trade_log_dir, date_text),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    report_text = render_report(payload)
    for path, content in (
        (output_dir / f"recommendations_{date_text}.json", json_text),
        (output_dir / "latest_recommendations.json", json_text),
        (output_dir / f"review_{date_text}.md", report_text),
        (output_dir / "latest_review.md", report_text),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return payload


def run_daemon(args: argparse.Namespace) -> int:
    run_at = dtime.fromisoformat(args.run_time)
    marker_path = args.output_dir / ".last_optimizer_run"
    while True:
        now = et_now()
        session = market_session_today(now)
        marker = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""
        today = now.date().isoformat()
        if session is not None and now.time() >= run_at and marker != today:
            try:
                run_daily_optimization(
                    bar_data_dir=args.bar_data_dir,
                    official_data_dir=args.official_data_dir,
                    shadow_data_dir=args.shadow_data_dir,
                    output_dir=args.output_dir,
                    trade_log_dir=args.trade_log_dir,
                    lookback_days=args.lookback_days,
                    minimum_days=args.minimum_days,
                    as_of=now,
                )
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text(today, encoding="utf-8")
                log.info("Daily momentum review completed for %s", today)
            except Exception:
                log.exception("Daily momentum optimization failed")
        time.sleep(args.poll_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal v2.1 daily replay optimizer")
    parser.add_argument("--bar-data-dir", type=Path, default=DEFAULT_BAR_DIR)
    parser.add_argument("--official-data-dir", type=Path)
    parser.add_argument("--shadow-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trade-log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--minimum-days", type=int, default=5)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--run-time", default="16:10", help="daily ET time")
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.daemon:
        return run_daemon(args)
    payload = run_daily_optimization(
        bar_data_dir=args.bar_data_dir,
        official_data_dir=args.official_data_dir,
        shadow_data_dir=args.shadow_data_dir,
        output_dir=args.output_dir,
        trade_log_dir=args.trade_log_dir,
        lookback_days=args.lookback_days,
        minimum_days=args.minimum_days,
    )
    statuses = {symbol: result["status"] for symbol, result in payload["recommendations"].items()}
    log.info("Optimization completed: %s", statuses)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
