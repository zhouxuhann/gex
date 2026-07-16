"""
GEX Monitor 入口

Usage:
    python -m gex_monitor.main --config config/config.yaml
"""
import argparse
import atexit
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import AppConfig
from .db_storage import GEXDBStorage
from .email_notifier import EmailConfig, EmailNotifier
from .ib_client import IBWorker
from .ib_error_watcher import IBErrorWatcher
from .state import registry
from .storage import SegmentStorage, StorageManager, find_split_data_dirs
from .ui import create_app

# 日志目录
LOG_DIR = Path(__file__).parent.parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 日志文件名：logs/gex_20260409.log
log_file = LOG_DIR / f"gex_{datetime.now().strftime('%Y%m%d')}.log"

# 配置日志（带 rotation）
_file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding='utf-8',
)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # 终端输出
        _file_handler,            # 文件输出（带 rotation）
    ]
)
log = logging.getLogger(__name__)


def _build_ib_error_watcher(config: AppConfig) -> IBErrorWatcher:
    email_cfg = config.alerts.ib_errors.email
    notifier = None
    if email_cfg.enabled:
        # cooldown_sec 排除：EmailNotifier 的 cooldown 控制 ddput signal 节流，
        # 与 IB error 无关。IB error 的节流由 IBErrorWatcher 自己管理。
        notifier = EmailNotifier(EmailConfig(
            **email_cfg.model_dump(exclude={'cooldown_sec'})
        ))
        if not os.environ.get(email_cfg.password_env, ''):
            log.warning(
                "IB error email alerts enabled but env %s is NOT set — "
                "alert emails will silently fail until it is exported",
                email_cfg.password_env,
            )
        log.info(
            "IB error email alerts enabled: recipients=%s cooldown=%ss",
            len(email_cfg.recipients),
            email_cfg.cooldown_sec,
        )
    else:
        log.info("IB error email alerts disabled")
    return IBErrorWatcher(email_notifier=notifier, cooldown_sec=email_cfg.cooldown_sec)


def main():
    parser = argparse.ArgumentParser(description='GEX Monitor')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--host', type=str, default=None,
                        help='覆盖服务器 host')
    parser.add_argument('--port', type=int, default=None,
                        help='覆盖服务器 port')
    parser.add_argument('--no-hedge', action='store_true',
                        help='禁用自动对冲（默认开启）')
    parser.add_argument('--hedge-dry-run', action='store_true',
                        help='对冲信号预览模式（不实际下单）')
    parser.add_argument('--hedge-qty', type=int, default=1,
                        help='每次对冲合约数 (默认 1)')
    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = AppConfig.from_yaml(args.config)
        log.info(f"从 {args.config} 加载配置")
    else:
        config = AppConfig.default()
        log.info("使用默认配置 (QQQ)")

    # 命令行覆盖
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    # 初始化存储
    data_dir = Path(config.storage.data_dir).expanduser().resolve()
    log.info("GEX data directory: %s", data_dir)
    split_dirs = find_split_data_dirs(data_dir)
    if split_dirs:
        log.warning(
            "检测到历史 GEX 数据分散在其他目录（不会自动合并）: %s",
            [str(p) for p in split_dirs],
        )
    quality_options = {
        'min_rth_coverage': config.monitoring.quality_min_rth_coverage,
        'max_gap_seconds': config.monitoring.quality_max_gap_seconds,
        'max_derived_null_ratio': config.monitoring.quality_max_derived_null_ratio,
        'min_contracts': config.monitoring.quality_min_contracts,
    }
    storage = StorageManager(data_dir, quality_options=quality_options)
    segments = SegmentStorage(config.storage.data_dir)
    ib_error_watcher = _build_ib_error_watcher(config)

    # 初始化 DB 存储
    db_storage = None
    if config.database.enabled:
        db_storage = GEXDBStorage(config.database)
        if db_storage.is_available:
            log.info("DB storage enabled (PostgreSQL)")
        else:
            log.warning("DB storage configured but unavailable — parquet only")
            db_storage = None

    # 获取启用的标的
    enabled_symbols = config.get_enabled_symbols()
    if not enabled_symbols:
        log.error("没有启用的标的，退出")
        sys.exit(1)

    log.info(f"启用标的: {[s.name for s in enabled_symbols]}")

    if not args.no_hedge:
        mode = "DRY RUN" if args.hedge_dry_run else f"LIVE (qty={args.hedge_qty})"
        log.info(f"对冲自动执行已启用 [{mode}] — 15:30 ET 自动采集+下单")

    # 创建 workers
    workers: list[IBWorker] = []
    threads: list[threading.Thread] = []

    for i, sym_config in enumerate(enabled_symbols):
        # 注册状态管理器
        state = registry.register(sym_config.name, config.storage.max_history)

        # 创建 worker
        worker = IBWorker(
            symbol=sym_config.name,
            trading_class=sym_config.trading_class,
            state=state,
            storage=storage,
            ib_host=config.ib.host,
            ib_port=config.ib.port,
            client_id=config.ib.client_id_base + i,
            strike_range=sym_config.strike_range,
            spot_sanity_pct=config.monitoring.spot_sanity_pct,
            sec_type=sym_config.sec_type,
            connect_timeout=config.ib.connect_timeout,
            max_retries=config.ib.max_retries,
            timing=config.timing,
            market_data_stale_sec=config.monitoring.reconnect_stale_seconds,
            quality_min_contracts=config.monitoring.quality_min_contracts,
            quality_max_missing_ratio=config.monitoring.quality_max_missing_ratio,
            db_storage=db_storage,
            hedge_enabled=not args.no_hedge,
            hedge_dry_run=args.hedge_dry_run,
            hedge_qty=args.hedge_qty,
            ib_error_watcher=ib_error_watcher,
            extended_hours=sym_config.extended_hours,
            intraday_vrp_config=config.intraday_vrp,
        )
        workers.append(worker)

        # 启动线程
        t = threading.Thread(target=worker.run, daemon=True, name=f"worker-{sym_config.name}")
        threads.append(t)
        t.start()
        log.info(f"启动 {sym_config.name} worker (client_id={config.ib.client_id_base + i})")

    # 创建 Dash 应用
    app = create_app(
        registry=registry,
        storage=storage,
        segments=segments,
        symbols=[s.name for s in enabled_symbols],
        db_storage=db_storage,
        extended_symbols={s.name for s in enabled_symbols if s.extended_hours},
    )

    # 优雅关闭
    shutdown_flag = threading.Event()

    def graceful_shutdown(*_):
        if shutdown_flag.is_set():
            return
        shutdown_flag.set()
        log.info("Shutting down...")

        # 停止 workers
        for w in workers:
            w.stop()

        # 等待持久化完成
        storage.shutdown()

        # 关闭 DB
        if db_storage is not None:
            db_storage.shutdown()

        log.info("Shutdown complete")

    def signal_handler(signum, frame):
        """信号处理器 - 先清理再退出"""
        graceful_shutdown()
        # 使用 raise SystemExit 而不是 sys.exit()
        # 这允许 finally 块和 with 语句正常清理
        raise SystemExit(0)

    atexit.register(graceful_shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, signal_handler)
        except Exception as e:
            log.warning(f"Failed to register signal handler: {e}")

    # 启动服务器
    log.info(f"启动 Dash 服务器: http://{config.server.host}:{config.server.port}")
    app.run(
        debug=False,
        host=config.server.host,
        port=config.server.port,
    )


if __name__ == '__main__':
    main()
