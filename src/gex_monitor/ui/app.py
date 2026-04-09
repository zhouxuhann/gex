"""Dash App 工厂"""
import dash
from flask import Flask, jsonify

from ..state import StateRegistry
from ..storage import StorageManager, SegmentStorage
from .layout import create_layout
from .callbacks import register_callbacks


def create_app(
    registry: StateRegistry,
    storage: StorageManager,
    segments: SegmentStorage,
    symbols: list[str],
) -> dash.Dash:
    """
    创建 Dash 应用

    Args:
        registry: 状态注册表
        storage: 存储管理器
        segments: 分段标注存储
        symbols: 已启用的标的列表

    Returns:
        配置好的 Dash 应用
    """
    server = Flask(__name__)

    # 健康检查端点
    @server.route('/health')
    def health():
        status = {
            'status': 'healthy',
            'symbols': {},
        }
        for sym in symbols:
            state = registry.get(sym)
            if state:
                snapshot = state.get_snapshot()
                status['symbols'][sym] = {
                    'connected': snapshot.get('connected', False),
                    'market_open': snapshot.get('market_open', False),
                    'last_update': snapshot.get('updated', 'unknown'),
                }
        return jsonify(status)

    app = dash.Dash(
        __name__,
        server=server,
        title="GEX Monitor",
    )

    # 设置布局
    app.layout = create_layout(symbols)

    # 注册回调
    register_callbacks(app, registry, storage, segments)

    return app
