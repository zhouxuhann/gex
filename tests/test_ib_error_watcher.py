"""Tests for IB critical error watcher."""

from unittest.mock import MagicMock

from gex_monitor.ib_error_watcher import IBErrorWatcher


class DummyContract:
    symbol = "QQQ"


def test_critical_10197_sends_email_once_with_cooldown():
    email = MagicMock()
    watcher = IBErrorWatcher(email_notifier=email, cooldown_sec=600, also_log_console=False)

    watcher._on_error(
        5,
        10197,
        "No market data during competing live session",
        DummyContract(),
    )
    watcher._on_error(
        6,
        10197,
        "No market data during competing live session",
        DummyContract(),
    )

    # 邮件在后台线程发送（SMTP 不能阻塞 ib_insync 事件线程），join 后再断言
    assert watcher._last_send_thread is not None
    watcher._last_send_thread.join(timeout=5)
    email.send_alert.assert_called_once()
    subject, body = email.send_alert.call_args.args
    assert "10197" in subject
    assert "另一端登录" in body
    assert email.send_alert.call_args.kwargs["force"] is True


def test_non_critical_error_is_ignored():
    email = MagicMock()
    watcher = IBErrorWatcher(email_notifier=email, also_log_console=False)

    watcher._on_error(1, 2104, "Market data farm connection is OK", None)

    email.send_alert.assert_not_called()


def test_1102_data_maintained_is_not_alerted():
    """1102 表示订阅已保留，不应报警或触发重连。"""
    email = MagicMock()
    watcher = IBErrorWatcher(email_notifier=email, also_log_console=False)

    watcher._on_error(1, 1102, "Connectivity restored - data maintained", None)

    email.send_alert.assert_not_called()
