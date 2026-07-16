"""IB error watcher — 钩入 ib_insync.errorEvent，对关键 error 发邮件 + log warning.

最重要的目标 error:
  10197 — "关联的真实账户登录期间无市场数据"
          典型场景: 你在手机/网页/另一台机登录了同账户, 把当前 Gateway 的市场数据踢下线
  10089 — "需要市场数据订阅"
  10168 — "请求的市场数据未被订阅"

cooldown: 错误会对每个 contract 各报一次 (200+ 次), 必须节流.
默认每 error_code 10 分钟内只报第一次.

用法:
    watcher = IBErrorWatcher(email_notifier=shared_email)
    for ib_client in clients:
        watcher.attach(ib_client.ib)   # 在 _connect() 后调用
"""
from __future__ import annotations

import logging
import time
from threading import Lock, Thread

log = logging.getLogger(__name__)


# 关键错误码: 这些是 system 级别问题, 用户必须知道
CRITICAL_CODES = {
    10197: "关联的真实账户登录期间无市场数据 (可能手机/另一端登录了同账户)",
    10089: "需要市场数据订阅",
    10168: "请求的市场数据未被订阅",
    1100:  "IB Gateway 与 TWS 连接断开",
    1101:  "IB Gateway 重连后市场数据 reset",
}


class IBErrorWatcher:
    """单例, 共享 cooldown state. 多个 IB client 共用一个 watcher.

    Args:
        email_notifier: 可选, 用于发邮件 (用 send_alert 通用方法)
        cooldown_sec:    每个 error_code 内最小发送间隔
        also_log_console: 是否同步 log.warning (避免和 ib_insync 自己的 ERROR log 重复)
    """

    def __init__(
        self,
        email_notifier=None,
        cooldown_sec: int = 600,
        also_log_console: bool = True,
    ):
        self.email = email_notifier
        self.cooldown_sec = cooldown_sec
        self.also_log_console = also_log_console
        self._last_fired: dict[int, float] = {}  # error_code -> last alert ts
        self._count_in_window: dict[int, int] = {}  # error_code -> count since last alert
        self._lock = Lock()
        self._last_send_thread: Thread | None = None  # 供测试 join / 优雅关闭

    def attach(self, ib) -> None:
        """钩入 ib_insync IB 实例的 errorEvent."""
        ib.errorEvent += self._on_error

    def detach(self, ib) -> None:
        """解钩 (重连前需要先 detach 再 attach)."""
        try:
            ib.errorEvent -= self._on_error
        except Exception:
            pass

    def _on_error(self, reqId, errorCode, errorString, contract):
        """ib_insync errorEvent handler: signature (reqId, errorCode, errorString, contract)."""
        if errorCode not in CRITICAL_CODES:
            return  # 非关键 error: ib_insync 自己 log, 不重复处理

        with self._lock:
            now = time.time()
            self._count_in_window[errorCode] = self._count_in_window.get(errorCode, 0) + 1
            last = self._last_fired.get(errorCode, 0.0)
            if now - last < self.cooldown_sec:
                # cooldown 中, 仅累加计数, 不发邮件
                return
            count = self._count_in_window[errorCode]
            self._count_in_window[errorCode] = 0
            self._last_fired[errorCode] = now

        meaning = CRITICAL_CODES.get(errorCode, '(未分类关键错误)')
        contract_str = ''
        if contract is not None:
            sym = getattr(contract, 'symbol', '?')
            contract_str = f' contract={sym}'

        summary = f'IB Error {errorCode}: {meaning}'
        details = (
            f'触发数: {count} 次 (本次 cooldown 窗口内)\n'
            f'最近一次 reqId: {reqId}{contract_str}\n'
            f'IB 原文: {errorString}\n'
        )

        if self.also_log_console:
            log.warning(f'⚠⚠ {summary} | {details.replace(chr(10), " ")}')

        if self.email is not None:
            subject = summary
            body = self._build_body(errorCode, meaning, count, reqId, errorString, contract_str)
            # errorEvent handler 跑在 ib_insync 事件线程上，SMTP 阻塞 I/O
            # （最长 15s 超时）会停摆该 worker 的行情处理，必须丢后台线程
            t = Thread(
                target=self.email.send_alert,
                args=(subject, body),
                kwargs={'force': True},
                daemon=True,
                name=f'ib-error-mail-{errorCode}',
            )
            self._last_send_thread = t
            t.start()

    def _build_body(self, code, meaning, count, reqId, errorString, contract_str) -> str:
        guidance = ''
        if code == 10197:
            guidance = (
                '常见原因:\n'
                '  - 同 IBKR 账户在另一端登录 (手机 IBKR app / 网页 / 另一台机器 TWS)\n'
                '  - IB 政策: 同一时刻只能一端获取 market data\n\n'
                '处理:\n'
                '  1. 检查并退出其他端的登录\n'
                '  2. 数据通常在 30s-2min 内自动恢复\n'
                '  3. 如果不恢复, 重启 IB Gateway\n'
            )
        elif code in (10089, 10168):
            guidance = (
                '可能原因:\n'
                '  - 该 symbol 的市场数据订阅过期或未启用\n'
                '  - 检查 IBKR 账户后台 → Market Data Subscriptions\n'
            )
        elif code in (1100, 1101):
            guidance = (
                'IB Gateway 连接事件:\n'
                '  - 1100: 与 TWS/Gateway 连接断开\n'
                '  - 1101: 重连后数据需要重新订阅 (gex_monitor 会自动 reqMktData)\n'
            )
        return (
            f'GEX Monitor — IB Error 检测\n\n'
            f'错误码: {code}\n'
            f'含义:   {meaning}\n'
            f'触发次数: {count} 次 (cooldown 窗口内)\n'
            f'最近 reqId: {reqId}{contract_str}\n\n'
            f'IB 原文:\n  {errorString}\n\n'
            f'{guidance}\n'
            f'本邮件由 ib_error_watcher 发出 (cooldown 后再触发会再次提醒)\n'
        )
