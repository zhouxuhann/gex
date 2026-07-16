import pandas as pd

from gex_monitor.gex_skew_alert_monitor import (
    actionable_alerts,
    apply_cooldown,
    build_alert_frame,
    _format_alert_email,
    parse_args,
    risk_reduction_alerts,
)


def _sample_rows():
    ts = pd.date_range("2026-06-05 11:00", periods=30, freq="1min", tz="America/New_York")
    rr = [0.01 + i * 0.0001 for i in range(24)] + [0.012, 0.014, 0.016, 0.018, 0.035, 0.04]
    return pd.DataFrame({
        "ts": ts,
        "spot": [720.0 - i * 0.2 for i in range(30)],
        "total_gex": [-10e9] * 30,
        "positive_gamma": [False] * 30,
        "call_wall": [730.0] * 30,
        "put_wall": [715.0] * 30,
        "max_pain": [725.0] * 30,
        "atm_iv_pct": [20.0] * 30,
        "rr_25": rr,
        "skew_slope": [0.2] * 30,
    })


def test_build_alert_frame_detects_left_tail_burst():
    alerts = build_alert_frame(
        _sample_rows(),
        "QQQ",
        drr_min_samples=5,
        z_hi=2.0,
    )
    actionable = actionable_alerts(alerts, min_negative_gex_b=5.0)
    assert not actionable.empty
    assert actionable.iloc[-1]["alert_level"] == "HIGH"


def test_require_near_put_wall_filters_far_spot():
    alerts = build_alert_frame(
        _sample_rows().assign(put_wall=690.0),
        "QQQ",
        drr_min_samples=5,
        z_hi=2.0,
        put_wall_buffer_pct=0.001,
    )
    actionable = actionable_alerts(
        alerts,
        require_near_put_wall=True,
        min_negative_gex_b=5.0,
    )
    assert actionable.empty


def test_risk_reduction_alert_triggers_before_high():
    rows = _sample_rows().assign(
        spot=722.6,
        total_gex=-3.7e9,
        positive_gamma=False,
        call_wall=730.0,
        put_wall=720.0,
        max_pain=735.0,
        rr_25=-0.004,
    )
    alerts = build_alert_frame(rows, "QQQ", drr_min_samples=50)
    risk = risk_reduction_alerts(alerts)
    assert not risk.empty
    assert set(risk["alert_level"]) == {"RISK_REDUCE"}


def test_format_alert_email():
    class Args:
        symbol = "QQQ"

    row = {
        "ts": "2026-06-05T11:22:00-04:00",
        "symbol": "QQQ",
        "alert_level": "RISK_REDUCE",
        "spot": 721.96,
        "total_gex": -5.4e9,
        "positive_gamma": False,
        "call_wall": 730.0,
        "put_wall": 720.0,
        "max_pain": 735.0,
        "wall_distance_pct": 0.0027,
        "rr_25": -0.005,
        "drr_25": -0.0002,
        "drr_25_zscore": -0.12,
        "alert_note": "test note",
    }
    subject, body = _format_alert_email(Args(), row)
    assert "QQQ RISK_REDUCE" in subject
    assert "2026-06-06 00:22:00 JST" in body
    assert "减风险" in body


def test_cooldown_is_per_alert_level():
    rows = pd.DataFrame([
        {"ts": "2026-06-05T11:00:00-04:00", "alert_level": "RISK_REDUCE"},
        {"ts": "2026-06-05T11:05:00-04:00", "alert_level": "HIGH"},
        {"ts": "2026-06-05T11:06:00-04:00", "alert_level": "HIGH"},
    ])
    out = apply_cooldown(rows, cooldown_min=10)
    assert list(out["alert_level"]) == ["RISK_REDUCE", "HIGH"]


def test_default_email_recipient_is_single_user(monkeypatch):
    monkeypatch.delenv("EMAIL_RECIPIENTS", raising=False)
    monkeypatch.setattr("sys.argv", ["prog"])
    args = parse_args()
    assert args.email_recipients == "wenyi.hann@gmail.com"
