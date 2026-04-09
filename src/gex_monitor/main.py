"""
GEX Monitor 入口

Usage:
    python -m gex_monitor.main --config config/config.yaml
"""
import argparse
import atexit
import logging
import signal
import sys
import threading
from pathlib import Path

from .config import AppConfig
from .ib_client import IBWorker
from .state import registry
from .storage import StorageManager, SegmentStorage
from .ui import create_app

from datetime import datetime

# 日志目录
LOG_DIR = Path(__file__).parent.parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 日志文件名：logs/gex_20260409.log
log_file = LOG_DIR / f"gex_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # 终端输出
        logging.FileHandler(log_file, encoding='utf-8'),  # 文件输出
    ]
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='GEX Monitor')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--host', type=str, default=None,
                        help='覆盖服务器 host')
    parser.add_argument('--port', type=int, default=None,
                        help='覆盖服务器 port')
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
    storage = StorageManager(config.storage.data_dir)
    segments = SegmentStorage(config.storage.data_dir)

    # 获取启用的标的
    enabled_symbols = config.get_enabled_symbols()
    if not enabled_symbols:
        log.error("没有启用的标的，退出")
        sys.exit(1)

    log.info(f"启用标的: {[s.name for s in enabled_symbols]}")

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
