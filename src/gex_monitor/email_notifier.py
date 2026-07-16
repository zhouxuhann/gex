"""邮件通知 —— ddput 信号触发时发邮件

复用 ~/.scripts/monitor_x_user.sh 的 Gmail SMTP 模式：
  - SMTP SSL (smtp.gmail.com:465)
  - App password 从环境变量读取
  - 多收件人逗号分隔

**不是 trading alert**。邮件内容永远带 "in-sample 观察信号" 免责声明。
默认只对 strong（z>=2.5）发邮件，mild 不发，避免收件箱爆炸。
"""
from __future__ import annotations

import logging
import os
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.utils import formatdate
from threading import Lock

log = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    enabled: bool = False
    sender: str = 'fzhouxu615@gmail.com'
    password_env: str = 'GMAIL_APP_PASSWORD'  # 环境变量名
    recipients: list[str] = field(default_factory=list)
    smtp_host: str = 'smtp.gmail.com'
    smtp_port: int = 465
    only_strong: bool = True       # True: 仅 strong 发邮件, False: strong+mild 都发
    cooldown_sec: int = 900        # 每 symbol+direction 最小间隔 15 min
    subject_prefix: str = '[GEX]'  # 主题前缀方便邮箱过滤


class EmailNotifier:
    """线程安全的邮件发送器（有 per-symbol-direction 节流）"""

    def __init__(self, config: EmailConfig):
        self.config = config
        self._last_sent: dict[tuple[str, str], float] = {}  # (symbol, direction) -> epoch
        self._lock = Lock()
        self._warned_no_password = False

    def _password(self) -> str | None:
        pwd = os.environ.get(self.config.password_env, '')
        if not pwd:
            # warning 级且只报一次：debug 级会让"邮件静默不发"几小时不可见
            if not self._warned_no_password:
                log.warning(f'email: env var {self.config.password_env} 未设，'
                            f'所有邮件将不会发送（此警告只出现一次）')
                self._warned_no_password = True
            return None
        return pwd

    def _should_send(self, symbol: str, direction: str, strength: str) -> tuple[bool, str]:
        """判断是否应该发邮件。返回 (应发, 原因描述用于 log)"""
        if not self.config.enabled:
            return False, 'disabled'
        if not self.config.recipients:
            return False, 'no recipients'
        if self.config.only_strong and strength != 'strong':
            return False, f'only_strong=True, skip {strength}'
        # 节流
        key = (symbol, direction)
        with self._lock:
            last = self._last_sent.get(key, 0)
            now = time.time()
            if now - last < self.config.cooldown_sec:
                return False, f'cooldown {int(now - last)}s / {self.config.cooldown_sec}s'
        return True, 'ok'

    def send_ddput_alert(
        self,
        symbol: str,
        direction: str,       # '+' or '-'
        strength: str,        # 'mild' or 'strong'
        z_score: float,
        ddput: float,
        put_gex_B: float,
        spot: float | None,
        ts_et: str,           # 已格式化的 ET 时间字符串
    ) -> bool:
        """发送 ddput alert 邮件。返回 True = 已发送；False = 被节流/未配置/失败"""
        ok, reason = self._should_send(symbol, direction, strength)
        if not ok:
            log.debug(f'email: {symbol} {direction}{strength} 不发: {reason}')
            return False

        pwd = self._password()
        if pwd is None:
            return False

        arrow = '↑ 看多观察' if direction == '+' else '↓ 看空观察'
        subject = (f'{self.config.subject_prefix} {symbol} ddput {direction}{strength} '
                   f'z={z_score:+.2f}')
        spot_str = f'{spot:.2f}' if spot is not None else '?'
        body = (
            f'ddput 信号触发\n\n'
            f'标的:         {symbol}\n'
            f'时间:         {ts_et}\n'
            f'方向/强度:   {arrow} ({strength})\n'
            f'Z-score:     {z_score:+.3f}\n'
            f'ddput:       {ddput:+.3f}  B$/min²\n'
            f'put_gex:     {put_gex_B:+.2f} B$\n'
            f'spot:        {spot_str}\n'
            f'\n'
            f'历史 in-sample 观察（QQQ 4 天全涨日 + SPX 1 天）:\n'
            f'  Q5 z>2.5 → fwd_5m +5 bps / fwd_15m +5-6 bps\n'
            f'  Q1 z<-2.5 → fwd_5m -2 bps / fwd_15m 接近 0（偏弱）\n'
            f'\n'
            f'⚠ 仅为观察提示，不是交易指令。\n'
            f'  - 样本 in-sample，下跌日未验证\n'
            f'  - 差分 edge 仅 ~5 bps，扣手续费/滑点后 net 可能不足\n'
            f'  - 见 memory/project_ddput_signal_hypothesis.md'
        )

        try:
            msg = MIMEText(body, 'plain', 'utf-8')
            msg['Subject'] = subject
            msg['From'] = self.config.sender
            msg['To'] = ', '.join(self.config.recipients)
            msg['Date'] = formatdate(localtime=True)

            with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port,
                                   timeout=15) as s:
                s.login(self.config.sender, pwd)
                s.send_message(msg)

            with self._lock:
                self._last_sent[(symbol, direction)] = time.time()
            log.info(f'email sent: {symbol} {direction}{strength} z={z_score:+.2f} '
                     f'to {len(self.config.recipients)} recipients')
            return True
        except Exception as e:
            log.error(f'email send failed: {e}')
            return False

    def send_alert(self, subject: str, body: str, force: bool = False) -> bool:
        """通用 alert 邮件 (例: IB error, system warning).

        独立于 ddput 的 should_send 逻辑, 但仍走 enabled + recipients + 密码检查.
        force=True 跳过 enabled gate (用于关键系统告警).
        """
        if not force and not self.config.enabled:
            return False
        if not self.config.recipients:
            return False
        pwd = self._password()
        if pwd is None:
            return False
        try:
            msg = MIMEText(body, 'plain', 'utf-8')
            msg['Subject'] = f'{self.config.subject_prefix} {subject}'
            msg['From'] = self.config.sender
            msg['To'] = ', '.join(self.config.recipients)
            msg['Date'] = formatdate(localtime=True)
            with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port,
                                   timeout=15) as s:
                s.login(self.config.sender, pwd)
                s.send_message(msg)
            log.info(f'alert email sent: "{subject}" → {len(self.config.recipients)} recipients')
            return True
        except Exception as e:
            log.error(f'alert email send failed: {e}')
            return False
