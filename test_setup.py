#!/usr/bin/env python3
"""
GEX Monitor 安装验证脚本
运行: python test_setup.py
"""
import sys
import shutil
from pathlib import Path

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
SKIP = "\033[93m⏭️ SKIP\033[0m"

results = []

def test(name, func):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print('='*60)
    try:
        func()
        print(f"{PASS} {name}")
        results.append((name, "PASS", None))
        return True
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"   错误: {e}")
        results.append((name, "FAIL", str(e)))
        return False


# ==================== 测试 1: 基础 import ====================
def test_imports():
    print("尝试 import 各模块...")

    from gex_monitor import time_utils
    print("  - time_utils ✓")

    from gex_monitor import config
    print("  - config ✓")

    from gex_monitor import storage
    print("  - storage ✓")

    from gex_monitor import state
    print("  - state ✓")

    from gex_monitor import gex_calc
    print("  - gex_calc ✓")

    from gex_monitor import ib_client
    print("  - ib_client ✓")

    from gex_monitor.ui import layout, callbacks, app
    print("  - ui.layout ✓")
    print("  - ui.callbacks ✓")
    print("  - ui.app ✓")


# ==================== 测试 2: 配置加载 ====================
def test_config():
    from gex_monitor.config import AppConfig

    # 默认配置
    cfg = AppConfig.default()
    assert len(cfg.symbols) > 0, "默认配置应该有标的"
    print(f"  默认配置: {[s.name for s in cfg.symbols]}")

    # YAML 配置
    yaml_path = Path(__file__).parent / "config" / "config.yaml"
    if yaml_path.exists():
        cfg = AppConfig.from_yaml(yaml_path)
        print(f"  YAML 配置: {[s.name for s in cfg.symbols]}")
        print(f"  启用标的: {[s.name for s in cfg.get_enabled_symbols()]}")
    else:
        print(f"  (跳过 YAML 测试，文件不存在: {yaml_path})")


# ==================== 测试 3: 时间工具 ====================
def test_time_utils():
    from gex_monitor.time_utils import (
        et_now, trading_date_str, is_market_open,
        market_session_today, ET
    )

    now = et_now()
    print(f"  当前 ET 时间: {now}")

    date_str = trading_date_str()
    print(f"  交易日期: {date_str}")
    assert len(date_str) == 8, "日期格式应为 YYYYMMDD"

    is_open = is_market_open()
    print(f"  市场开盘: {is_open}")

    session = market_session_today()
    if session:
        print(f"  今日时段: {session[0].strftime('%H:%M')} - {session[1].strftime('%H:%M')}")
    else:
        print(f"  今日非交易日")


# ==================== 测试 4: 状态管理 ====================
def test_state():
    from gex_monitor.state import StateManager, StateRegistry, registry
    import pandas as pd

    # 创建状态管理器
    sm = StateManager("TEST", max_history=100)
    print(f"  创建 StateManager: {sm.symbol}")

    # 模拟更新
    df = pd.DataFrame([
        {'strike': 480, 'right': 'C', 'gex': 1e6, 'gamma': 0.05, 'oi': 1000, 'iv': 0.15},
        {'strike': 480, 'right': 'P', 'gex': -5e5, 'gamma': 0.04, 'oi': 800, 'iv': 0.16},
    ])
    sm.update(
        spot=480.5, total_gex=5e5, gamma_flip=480,
        call_gex=1e6, put_gex=-5e5, atm_iv_pct=15.5,
        expiry="20240115", is_true_0dte=True, df=df
    )
    print(f"  更新状态成功")

    # 获取快照
    snapshot = sm.get_snapshot()
    assert snapshot['spot'] == 480.5, "spot 应该是 480.5"
    print(f"  快照: spot={snapshot['spot']}, gex={snapshot['total_gex']}")

    # 测试 registry
    reg = StateRegistry()
    reg.register("QQQ")
    reg.register("SPY")
    symbols = reg.list_symbols()
    assert "QQQ" in symbols and "SPY" in symbols
    print(f"  Registry 标的: {symbols}")


# ==================== 测试 5: 存储模块 ====================
def test_storage():
    from gex_monitor.storage import StorageManager, SegmentStorage
    import pandas as pd

    test_dir = Path(__file__).parent / "data_test"
    test_dir.mkdir(exist_ok=True)

    try:
        storage = StorageManager(test_dir)
        print(f"  创建 StorageManager: {storage.data_dir}")

        # 写入测试数据
        now = pd.Timestamp.now(tz='America/New_York')
        test_hist = [{
            'ts': now, 'spot': 480.0, 'total_gex': 1e6,
            'flip': 480, 'call_gex': 5e5, 'put_gex': 5e5, 'atm_iv_pct': 15.0
        }]
        test_ohlc = [{
            'ts': now, 'open': 480, 'high': 481, 'low': 479, 'close': 480.5
        }]
        test_strikes = [{
            'ts': now, 'strike': 480, 'right': 'C',
            'gex': 1e6, 'gamma': 0.05, 'oi': 1000, 'iv': 0.15
        }]

        storage.persist_sync('TEST', test_hist, test_ohlc, test_strikes)
        print(f"  写入测试数据成功")

        # 读取验证
        dates = storage.list_available_dates('TEST')
        print(f"  可用日期: {dates}")
        assert len(dates) > 0, "应该有可用日期"

        ohlc = storage.load_day_ohlc('TEST', dates[0])
        assert ohlc is not None and len(ohlc) > 0
        print(f"  读取 OHLC 成功: {len(ohlc)} 条")

        # 测试分段存储
        segments = SegmentStorage(test_dir)
        segments.save_segment(dates[0], now, now, 'TEST', 'chop', 'test note')
        segs = segments.load_segments()
        assert len(segs) > 0
        print(f"  分段标注: {len(segs)} 条")

    finally:
        # 清理
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"  清理测试目录")


# ==================== 测试 6: GEX 计算 ====================
def test_gex_calc():
    from gex_monitor.gex_calc import calculate_gex, pick_expiry, GEXResult
    from unittest.mock import MagicMock

    # 模拟 ticker
    def make_ticker(strike, right, gamma, oi, iv):
        t = MagicMock()
        t.contract.strike = strike
        t.contract.right = right
        t.contract.multiplier = '100'
        t.modelGreeks = MagicMock()
        t.modelGreeks.gamma = gamma
        t.modelGreeks.impliedVol = iv
        if right == 'C':
            t.callOpenInterest = oi
            t.putOpenInterest = 0
        else:
            t.callOpenInterest = 0
            t.putOpenInterest = oi
        return t

    tickers = [
        make_ticker(480, 'C', 0.05, 1000, 0.15),
        make_ticker(480, 'P', 0.04, 800, 0.16),
        make_ticker(481, 'C', 0.03, 500, 0.14),
    ]

    result = calculate_gex(tickers, spot=480.5)
    assert result is not None, "应该返回结果"
    assert isinstance(result, GEXResult)
    print(f"  Total GEX: ${result.total_gex/1e6:.2f}M")
    print(f"  Gamma Flip: {result.gamma_flip}")
    print(f"  ATM IV: {result.atm_iv_pct:.1f}%")

    # 测试 pick_expiry
    chain = MagicMock()
    chain.expirations = ['20240114', '20240115', '20240116']
    expiry, is_0dte = pick_expiry(chain, '20240115')
    assert expiry == '20240115'
    assert is_0dte == True
    print(f"  pick_expiry: {expiry}, is_0dte={is_0dte}")


# ==================== 测试 7: UI 创建 ====================
def test_ui_create():
    from gex_monitor.state import StateRegistry
    from gex_monitor.storage import StorageManager, SegmentStorage
    from gex_monitor.ui import create_app

    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)

    registry = StateRegistry()
    registry.register('TEST')

    storage = StorageManager(test_dir)
    segments = SegmentStorage(test_dir)

    app = create_app(registry, storage, segments, ['TEST'])

    assert app is not None
    assert app.title == "GEX Monitor"
    print(f"  Dash app 创建成功")
    print(f"  Title: {app.title}")

    # 测试 /health 端点
    with app.server.test_client() as client:
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'status' in data
        print(f"  /health 端点: {data}")


# ==================== 测试 8: IB 连接 (可选) ====================
def test_ib_connection():
    """这个测试需要 IB Gateway 运行"""
    from ib_insync import IB

    ib = IB()
    try:
        ib.connect('127.0.0.1', 7496, clientId=99, timeout=5)
        print(f"  连接成功!")
        print(f"  服务器版本: {ib.client.serverVersion()}")
        ib.disconnect()
    except Exception as e:
        raise Exception(f"无法连接 IB Gateway (127.0.0.1:7496): {e}")


# ==================== 主程序 ====================
def main():
    print("\n" + "="*60)
    print("GEX Monitor 安装验证")
    print("="*60)

    # 必要测试
    test("1. 模块 Import", test_imports)
    test("2. 配置加载", test_config)
    test("3. 时间工具", test_time_utils)
    test("4. 状态管理", test_state)
    test("5. 存储模块", test_storage)
    test("6. GEX 计算", test_gex_calc)
    test("7. UI 创建", test_ui_create)

    # 可选测试 (需要 IB)
    print(f"\n{'='*60}")
    print("可选测试: IB 连接 (需要 IB Gateway 运行在 127.0.0.1:7496)")
    print('='*60)
    try:
        test_ib_connection()
        print(f"{PASS} IB 连接")
        results.append(("8. IB 连接", "PASS", None))
    except Exception as e:
        print(f"{SKIP} IB 连接 (Gateway 未运行或无法连接)")
        print(f"   原因: {e}")
        results.append(("8. IB 连接", "SKIP", str(e)))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    skipped = sum(1 for _, status, _ in results if status == "SKIP")

    for name, status, error in results:
        if status == "PASS":
            print(f"  {PASS} {name}")
        elif status == "FAIL":
            print(f"  {FAIL} {name}: {error}")
        else:
            print(f"  {SKIP} {name}")

    print()
    print(f"通过: {passed}  失败: {failed}  跳过: {skipped}")

    if failed > 0:
        print("\n⚠️  有测试失败，请检查上面的错误信息")
        sys.exit(1)
    else:
        print("\n🎉 核心功能验证通过!")
        if skipped > 0:
            print("   (IB 连接测试跳过，需要启动 IB Gateway 后手动测试)")
        sys.exit(0)


if __name__ == '__main__':
    main()
