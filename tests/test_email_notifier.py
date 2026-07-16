"""测试 EmailNotifier 的节流 / 配置门禁 / SMTP 调用（mock）"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gex_monitor.email_notifier import EmailConfig, EmailNotifier


@pytest.fixture
def base_cfg():
    return EmailConfig(
        enabled=True,
        recipients=['test@example.com'],
        only_strong=True,
        cooldown_sec=900,
    )


def _call(notifier, direction='+', strength='strong'):
    """简化重复调用"""
    return notifier.send_ddput_alert(
        symbol='QQQ', direction=direction, strength=strength,
        z_score=2.8, ddput=1.2, put_gex_B=-50.0, spot=640.0,
        ts_et='2026-04-20 11:30:00 ET',
    )


class TestGating:
    def test_disabled_config_no_send(self, base_cfg):
        base_cfg.enabled = False
        n = EmailNotifier(base_cfg)
        assert _call(n) is False

    def test_no_recipients_no_send(self, base_cfg):
        base_cfg.recipients = []
        n = EmailNotifier(base_cfg)
        assert _call(n) is False

    def test_only_strong_blocks_mild(self, base_cfg):
        base_cfg.only_strong = True
        n = EmailNotifier(base_cfg)
        with patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            assert _call(n, strength='mild') is False

    def test_only_strong_false_allows_mild(self, base_cfg):
        base_cfg.only_strong = False
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            mock_smtp.return_value.__enter__.return_value = MagicMock()
            assert _call(n, strength='mild') is True

    def test_missing_password_no_send(self, base_cfg):
        n = EmailNotifier(base_cfg)
        # 确保环境变量不存在
        with patch.dict('os.environ', {}, clear=True):
            assert _call(n) is False


class TestCooldown:
    def test_second_send_same_direction_throttled(self, base_cfg):
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            mock_smtp.return_value.__enter__.return_value = MagicMock()
            assert _call(n, direction='+') is True
            assert _call(n, direction='+') is False   # 节流

    def test_opposite_direction_not_throttled(self, base_cfg):
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            mock_smtp.return_value.__enter__.return_value = MagicMock()
            assert _call(n, direction='+') is True
            assert _call(n, direction='-') is True    # 不同方向独立

    def test_cooldown_expires(self, base_cfg):
        base_cfg.cooldown_sec = 1
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            mock_smtp.return_value.__enter__.return_value = MagicMock()
            assert _call(n, direction='+') is True
            import time
            time.sleep(1.1)
            assert _call(n, direction='+') is True


class TestSmtpInvocation:
    def test_sends_correct_params(self, base_cfg):
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'sekret'}):
            server_mock = MagicMock()
            mock_smtp.return_value.__enter__.return_value = server_mock

            _call(n)

            mock_smtp.assert_called_once_with(
                'smtp.gmail.com', 465, timeout=15
            )
            server_mock.login.assert_called_once_with('fzhouxu615@gmail.com', 'sekret')
            assert server_mock.send_message.call_count == 1
            msg = server_mock.send_message.call_args[0][0]
            assert 'QQQ' in msg['Subject']
            assert 'z=+2.80' in msg['Subject']
            assert 'test@example.com' in msg['To']

    def test_smtp_exception_returns_false(self, base_cfg):
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL', side_effect=Exception('network')), \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            assert _call(n) is False
            # cooldown 应该没被更新（失败的不算）
            assert n._last_sent == {}

    def test_multi_recipients_joined(self, base_cfg):
        base_cfg.recipients = ['a@x.com', 'b@x.com', 'c@x.com']
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            server_mock = MagicMock()
            mock_smtp.return_value.__enter__.return_value = server_mock
            _call(n)
            msg = server_mock.send_message.call_args[0][0]
            assert msg['To'] == 'a@x.com, b@x.com, c@x.com'


class TestBody:
    def test_body_contains_disclaimer(self, base_cfg):
        """邮件正文必须包含 in-sample 免责声明"""
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            server_mock = MagicMock()
            mock_smtp.return_value.__enter__.return_value = server_mock
            _call(n)
            msg = server_mock.send_message.call_args[0][0]
            body = msg.get_payload(decode=True).decode('utf-8')
            assert '不是交易指令' in body or '观察' in body
            assert 'in-sample' in body

    def test_subject_has_prefix_and_direction(self, base_cfg):
        n = EmailNotifier(base_cfg)
        with patch('smtplib.SMTP_SSL') as mock_smtp, \
             patch.dict('os.environ', {'GMAIL_APP_PASSWORD': 'xxx'}):
            server_mock = MagicMock()
            mock_smtp.return_value.__enter__.return_value = server_mock
            _call(n, direction='-', strength='strong')
            msg = server_mock.send_message.call_args[0][0]
            assert msg['Subject'].startswith('[GEX]')
            assert '-strong' in msg['Subject']
