"""
Realtime GEX + skew alert monitor.

This tails the live ``gex_SYMBOL_YYYYMMDD.parquet`` file produced by the main
collector and applies the SkewTracker ΔRR25 logic without touching the IB data
collection path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .email_notifier import EmailConfig, EmailNotifier
from .skew import SkewSnapshot, SkewTracker
from .time_utils import trading_date_str


@dataclass
class GexSkewAlert:
    ts: str
    symbol: str
    spot: float | None
    total_gex: float | None
    positive_gamma: bool
    call_wall: float | None
    put_wall: float | None
    max_pain: float | None
    rr_25: float | None
    drr_25: float | None
    drr_25_zscore: float | None
    alert_level: str | None
    alert_score: float | None
    alert_note: str | None
    near_put_wall: bool
    wall_distance_pct: float | None


def _num(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _start_timestamp(date_str: str, start_jst: str | None) -> pd.Timestamp | None:
    if not start_jst:
        return None
    day = pd.to_datetime(date_str, format="%Y%m%d").date()
    # A US trading date's Japan midnight after the open is the next JST date.
    jst_day = pd.Timestamp(day) + pd.Timedelta(days=1)
    return pd.Timestamp(f"{jst_day:%Y-%m-%d} {start_jst}", tz="Asia/Tokyo")


def load_live_gex(path: Path, start_jst: str | None = None, date_str: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    if start_jst:
        start = _start_timestamp(date_str or trading_date_str(), start_jst)
        df = df[df["ts"].dt.tz_convert("Asia/Tokyo") >= start]
    return df


def build_alert_frame(
    df: pd.DataFrame,
    symbol: str,
    *,
    resample_rule: str = "1min",
    tracker_window: int = 120,
    drr_window_s: float = 60.0,
    drr_min_samples: int = 20,
    z_hi: float = 2.0,
    z_lo: float = -2.0,
    put_wall_buffer_pct: float = 0.003,
) -> pd.DataFrame:
    """Return one row per resampled bar with SkewTracker alert fields."""
    if df.empty or "rr_25" not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["rr_25"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["ts"] = pd.to_datetime(work["ts"])
    work = work.set_index("ts").resample(resample_rule).last().dropna(subset=["rr_25"]).reset_index()

    tracker = SkewTracker(
        window=tracker_window,
        drr_window_s=drr_window_s,
        drr_min_samples=drr_min_samples,
        z_hi=z_hi,
        z_lo=z_lo,
    )
    rows: list[dict] = []
    for _, r in work.iterrows():
        atm_iv_pct = _num(r.get("atm_iv_pct"))
        snap = SkewSnapshot(
            atm_iv=atm_iv_pct / 100.0 if atm_iv_pct is not None else None,
            rr_25=float(r["rr_25"]),
            skew_slope=_num(r.get("skew_slope")),
            rr_25_zscore=None,
            signal=None,
            ts=pd.Timestamp(r["ts"]).timestamp(),
            spot=_num(r.get("spot")),
        )
        positive_gamma = bool(r.get("positive_gamma", False))
        enriched = tracker.update(snap, positive_gamma=positive_gamma)

        spot = _num(r.get("spot"))
        put_wall = _num(r.get("put_wall"))
        wall_distance_pct = None
        near_put_wall = False
        if spot is not None and put_wall is not None and spot > 0:
            wall_distance_pct = (spot - put_wall) / spot
            near_put_wall = wall_distance_pct <= put_wall_buffer_pct

        rows.append(asdict(GexSkewAlert(
            ts=pd.Timestamp(r["ts"]).isoformat(),
            symbol=symbol,
            spot=spot,
            total_gex=_num(r.get("total_gex")),
            positive_gamma=positive_gamma,
            call_wall=_num(r.get("call_wall")),
            put_wall=put_wall,
            max_pain=_num(r.get("max_pain")),
            rr_25=_num(r.get("rr_25")),
            drr_25=enriched.drr_25,
            drr_25_zscore=enriched.drr_25_zscore,
            alert_level=enriched.alert_level,
            alert_score=enriched.alert_score,
            alert_note=enriched.alert_note,
            near_put_wall=near_put_wall,
            wall_distance_pct=wall_distance_pct,
        )))
    return pd.DataFrame(rows)


def actionable_alerts(
    alerts: pd.DataFrame,
    *,
    require_near_put_wall: bool = False,
    min_negative_gex_b: float = 0.0,
    levels: tuple[str, ...] = ("HIGH",),
) -> pd.DataFrame:
    if alerts.empty:
        return alerts
    out = alerts[alerts["alert_level"].isin(levels)].copy()
    out = out[out["positive_gamma"] == False]  # noqa: E712
    if require_near_put_wall:
        out = out[out["near_put_wall"] == True]  # noqa: E712
    if min_negative_gex_b > 0:
        out = out[out["total_gex"] <= -abs(min_negative_gex_b) * 1e9]
    return out


def risk_reduction_alerts(
    alerts: pd.DataFrame,
    *,
    min_negative_gex_b: float = 3.0,
    put_wall_buffer_pct: float = 0.004,
    call_wall_gap_pct: float = 0.003,
    max_pain_gap_pct: float = 0.006,
) -> pd.DataFrame:
    """
    Earlier risk-reduction trigger than HIGH.

    This is meant to catch the regime-break window where skew has not yet
    accelerated, but dealer gamma is already negative and spot is pressing the
    put wall while call wall / max pain sit overhead.
    """
    if alerts.empty:
        return alerts
    out = alerts.copy()
    out = out[out["positive_gamma"] == False]  # noqa: E712
    out = out[out["total_gex"] <= -abs(min_negative_gex_b) * 1e9]

    out = out[
        (out["put_wall"].notna())
        & (out["spot"].notna())
        & (((out["spot"] - out["put_wall"]) / out["spot"]) <= put_wall_buffer_pct)
    ]
    out = out[
        (out["call_wall"].notna())
        & (((out["call_wall"] - out["spot"]) / out["spot"]) >= call_wall_gap_pct)
    ]
    out = out[
        (out["max_pain"].notna())
        & (((out["max_pain"] - out["spot"]) / out["spot"]) >= max_pain_gap_pct)
    ]
    if out.empty:
        return out
    out = out.copy()
    out["alert_level"] = "RISK_REDUCE"
    out["alert_score"] = 1.5
    out["alert_note"] = (
        "负 gamma 已成立 + spot 贴近 put wall + call/max pain 在上方；"
        "适合减风险/买保护，不必等待 skew HIGH"
    )
    return out


def apply_cooldown(alerts: pd.DataFrame, cooldown_min: float) -> pd.DataFrame:
    """Keep the first alert in a cluster per alert level."""
    if alerts.empty or cooldown_min <= 0:
        return alerts
    work = alerts.copy()
    work["_ts"] = pd.to_datetime(work["ts"])
    kept = []
    last_by_level = {}
    for _, row in work.sort_values("_ts").iterrows():
        ts = row["_ts"]
        level = row.get("alert_level") or "ALERT"
        last_ts = last_by_level.get(level)
        if last_ts is None or (ts - last_ts).total_seconds() >= cooldown_min * 60:
            kept.append(row.drop(labels=["_ts"]).to_dict())
            last_by_level[level] = ts
    return pd.DataFrame(kept)


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _notify(title: str, message: str) -> None:
    subprocess.run(
        ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
        check=False,
    )


def _ensure_password_env_from_crontab(env_name: str) -> None:
    """Mirror old cron wrappers: fill password env from crontab when absent."""
    if os.environ.get(env_name):
        return
    try:
        text = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return
    prefix = f"{env_name}="
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            if value:
                os.environ[env_name] = value
            return


def _email_notifier(args: argparse.Namespace) -> EmailNotifier:
    recipients = [x.strip() for x in args.email_recipients.split(",") if x.strip()]
    return EmailNotifier(EmailConfig(
        enabled=args.email,
        sender=args.email_sender,
        password_env=args.email_password_env,
        recipients=recipients,
        smtp_host=args.email_smtp_host,
        smtp_port=args.email_smtp_port,
        only_strong=False,
        cooldown_sec=int(args.cooldown_min * 60),
        subject_prefix=args.email_subject_prefix,
    ))


def _format_alert_email(args: argparse.Namespace, row: dict) -> tuple[str, str]:
    drr_z = row.get("drr_25_zscore")
    drr_z_txt = "NA" if drr_z is None or pd.isna(drr_z) else f"{float(drr_z):+.2f}"
    gex_b = row.get("total_gex")
    gex_txt = "NA" if gex_b is None or pd.isna(gex_b) else f"{float(gex_b)/1e9:+.1f}B"
    wall_dist = row.get("wall_distance_pct")
    wall_txt = "NA" if wall_dist is None or pd.isna(wall_dist) else f"{float(wall_dist)*100:+.2f}%"
    ts = pd.Timestamp(row["ts"])
    ts_jst = ts.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M:%S JST")
    ts_et = ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S ET")
    level = row.get("alert_level") or "ALERT"
    symbol = row.get("symbol") or args.symbol
    subject = f"{symbol} {level} spot={float(row['spot']):.2f} GEX={gex_txt} dRRz={drr_z_txt}"

    if level == "RISK_REDUCE":
        action = (
            "建议动作：进入账户级减风险检查。可考虑减仓、买 put/put spread、暂停加多；"
            "这是早期风控信号，不必等待 skew HIGH。"
        )
    elif level == "HIGH":
        action = (
            "建议动作：尾部风险已确认。避免硬抄底；已有保护仓优先持有，"
            "反弹更偏减风险；0DTE 短多需要非常短。"
        )
    else:
        action = "建议动作：检查 GEX/skew 风险状态。"

    body = f"""GEX + Skew 风险提醒

标的:        {symbol}
等级:        {level}
时间:        {ts_jst}
美东参考:    {ts_et}

spot:        {float(row['spot']):.2f}
total GEX:   {gex_txt}
positive γ:  {row.get('positive_gamma')}
call wall:   {row.get('call_wall')}
put wall:    {row.get('put_wall')}
max pain:    {row.get('max_pain')}
距 put wall: {wall_txt}

RR25:        {row.get('rr_25')}
dRR25:       {row.get('drr_25')}
dRR z:       {drr_z_txt}

说明:
{row.get('alert_note') or ''}

{action}

仅为风险监控提醒，不是交易指令。
"""
    return subject, body


def _send_email(args: argparse.Namespace, row: dict) -> bool:
    if not args.email:
        return False
    _ensure_password_env_from_crontab(args.email_password_env)
    subject, body = _format_alert_email(args, row)
    return _email_notifier(args).send_alert(subject, body, force=False)


def run_once(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_str = args.date or trading_date_str()
    path = Path(args.data_dir) / f"gex_{args.symbol}_{date_str}.parquet"
    df = load_live_gex(path, start_jst=args.start_jst, date_str=date_str)
    alerts = build_alert_frame(
        df,
        args.symbol,
        resample_rule=args.resample,
        tracker_window=args.tracker_window,
        drr_window_s=args.drr_window_s,
        drr_min_samples=args.drr_min_samples,
        z_hi=args.z_hi,
        z_lo=args.z_lo,
        put_wall_buffer_pct=args.put_wall_buffer_pct,
    )
    high_alerts = actionable_alerts(
        alerts,
        require_near_put_wall=args.require_near_put_wall,
        min_negative_gex_b=args.min_negative_gex_b,
        levels=tuple(args.levels.split(",")),
    )
    risk_alerts = risk_reduction_alerts(
        alerts,
        min_negative_gex_b=args.risk_gex_b,
        put_wall_buffer_pct=args.risk_put_wall_buffer_pct,
        call_wall_gap_pct=args.risk_call_wall_gap_pct,
        max_pain_gap_pct=args.risk_max_pain_gap_pct,
    )
    if args.mode == "high":
        actionable = high_alerts
    elif args.mode == "risk-reduce":
        actionable = risk_alerts
    else:
        actionable = pd.concat([risk_alerts, high_alerts], ignore_index=True)
        if not actionable.empty:
            actionable = actionable.sort_values("ts").drop_duplicates(
                subset=["ts", "symbol", "alert_level"], keep="last"
            )
    actionable = apply_cooldown(actionable, args.cooldown_min)

    if actionable.empty:
        return alerts, actionable

    latest = actionable.iloc[-1].to_dict()
    alert_key = f"{latest['symbol']}|{latest['ts']}|{latest['alert_level']}"
    state_path = Path(args.state_file)
    state = _load_state(state_path)
    if state.get("last_alert_key") == alert_key:
        return alerts, actionable

    _append_jsonl(Path(args.out_jsonl), latest)
    state["last_alert_key"] = alert_key
    _save_state(state_path, state)

    drr_z = latest.get("drr_25_zscore")
    drr_z_txt = "NA" if drr_z is None or pd.isna(drr_z) else f"{drr_z:.2f}"
    msg = (
        f"{args.symbol} {latest['alert_level']} "
        f"spot={latest['spot']:.2f} dRRz={drr_z_txt} "
        f"GEX={latest['total_gex'] / 1e9:.1f}B"
    )
    print(msg)
    if args.notify:
        _notify(f"GEX Skew Alert {args.symbol}", msg)
    if args.email:
        _send_email(args, latest)
    return alerts, actionable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor negative-gamma left-tail skew bursts from live GEX parquet.")
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--date", help="YYYYMMDD, defaults to current ET trading date")
    p.add_argument("--data-dir", default="src/data")
    p.add_argument("--start-jst", default="00:00", help="Only analyze after this JST time on the day after trading date")
    p.add_argument("--resample", default="1min")
    p.add_argument("--interval-sec", type=float, default=20)
    p.add_argument("--once", action="store_true")
    p.add_argument("--notify", action="store_true")
    p.add_argument("--require-near-put-wall", action="store_true")
    p.add_argument("--put-wall-buffer-pct", type=float, default=0.003)
    p.add_argument("--min-negative-gex-b", type=float, default=5.0)
    p.add_argument("--levels", default="HIGH")
    p.add_argument("--mode", choices=["high", "risk-reduce", "both"], default="high")
    p.add_argument("--risk-gex-b", type=float, default=3.0)
    p.add_argument("--risk-put-wall-buffer-pct", type=float, default=0.004)
    p.add_argument("--risk-call-wall-gap-pct", type=float, default=0.003)
    p.add_argument("--risk-max-pain-gap-pct", type=float, default=0.006)
    p.add_argument("--cooldown-min", type=float, default=10.0)
    p.add_argument("--tracker-window", type=int, default=120)
    p.add_argument("--drr-window-s", type=float, default=60.0)
    p.add_argument("--drr-min-samples", type=int, default=20)
    p.add_argument("--z-hi", type=float, default=2.0)
    p.add_argument("--z-lo", type=float, default=-2.0)
    p.add_argument("--out-jsonl", default="logs/gex_skew_alerts.jsonl")
    p.add_argument("--state-file", default="logs/gex_skew_alert_state.json")
    p.add_argument("--email", action="store_true")
    p.add_argument("--email-sender", default=os.environ.get("EMAIL_SENDER", "fzhouxu615@gmail.com"))
    p.add_argument("--email-password-env", default=os.environ.get("EMAIL_PASSWORD_ENV", "GMAIL_APP_PASSWORD"))
    p.add_argument(
        "--email-recipients",
        default=os.environ.get("EMAIL_RECIPIENTS", "wenyi.hann@gmail.com"),
    )
    p.add_argument("--email-smtp-host", default=os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com"))
    p.add_argument("--email-smtp-port", type=int, default=int(os.environ.get("EMAIL_SMTP_PORT", "465")))
    p.add_argument("--email-subject-prefix", default=os.environ.get("EMAIL_SUBJECT_PREFIX", "[GEX-SKEW]"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        alerts, actionable = run_once(args)
        if args.once:
            print(f"rows={len(alerts)} actionable={len(actionable)}")
            if not alerts.empty:
                latest = alerts.iloc[-1]
                print(
                    f"latest {latest['ts']} spot={latest['spot']} "
                    f"GEX={latest['total_gex']} dRRz={latest['drr_25_zscore']} "
                    f"level={latest['alert_level']}"
                )
            if not actionable.empty:
                cols = ["ts", "spot", "total_gex", "rr_25", "drr_25", "drr_25_zscore", "alert_level", "near_put_wall"]
                print(actionable[cols].tail(10).to_string(index=False))
            return
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
