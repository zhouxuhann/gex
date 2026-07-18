"""
对冲信号 CLI 入口

用法:
    # 采集 + 生成信号
    python -m gex_monitor.hedge

    # 生成信号 + 自动下单 (Paper 账户)
    python -m gex_monitor.hedge --execute

    # 预览下单但不实际执行
    python -m gex_monitor.hedge --execute --dry-run

    # 平仓所有对冲头寸
    python -m gex_monitor.hedge --reduce

    # 查看历史信号
    python -m gex_monitor.hedge --history

    # 查看交易记录和胜率
    python -m gex_monitor.hedge --trades

    # 指定标的和仓位
    python -m gex_monitor.hedge --execute --symbol QQQ --qty 2
"""
import argparse
import logging
import sys
from pathlib import Path

from .config import DatabaseConfig
from .hedge_signal import generate_hedge_signal, format_recommendation
from .hedge_executor import HedgeExecutor, format_trade_result
from .macro import fetch_macro_snapshot
from .skew_surface import collect_skew_surface
from .storage import StorageManager, SkewSurfaceStorage

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path('./data')
DEFAULT_SYMBOLS = ['QQQ', 'SPY']


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def _get_db_config() -> DatabaseConfig:
    return DatabaseConfig()


def _get_gex_regime(gex_storage: StorageManager, symbol: str) -> str:
    """从今日 GEX 数据推断当前 regime"""
    from .time_utils import et_now
    today = et_now().strftime('%Y%m%d')
    gex_df = gex_storage.load_day_gex(symbol, today)

    if gex_df is None or gex_df.empty:
        log.info(f"[{symbol}] 今日无 GEX 数据，使用 neutral regime")
        return 'neutral'

    latest = gex_df.sort_values('ts').iloc[-1]
    total_gex = latest.get('total_gex', 0)

    if total_gex > 0:
        return 'positive'
    elif total_gex < 0:
        return 'negative'
    return 'neutral'


def _collect_and_signal(
    symbol: str,
    data_dir: Path,
    ib_host: str,
    ib_port: int,
    client_id: int,
    execute: bool = False,
    dry_run: bool = False,
    qty: int = 1,
) -> None:
    """采集 skew surface → 生成信号 → 可选执行交易"""
    from ib_insync import IB

    skew_store = SkewSurfaceStorage(data_dir)
    gex_store = StorageManager(data_dir)
    db_config = _get_db_config()

    # 连接 IB
    log.info(f"[{symbol}] 连接 IB ({ib_host}:{ib_port}, clientId={client_id})...")
    ib = IB()
    try:
        ib.connect(ib_host, ib_port, clientId=client_id, timeout=20)
    except Exception as e:
        log.error(f"IB 连接失败: {e}")
        sys.exit(1)

    executor = None
    try:
        # 采集 skew surface
        log.info(f"[{symbol}] 采集 skew surface...")
        surface = collect_skew_surface(ib, symbol)

        if surface is None:
            log.error(f"[{symbol}] 采集失败")
            return

        # 存储
        skew_store.save_surface(surface.to_records())

        # 加载历史
        history_df = skew_store.load_surface_history(symbol, n_days=90)

        # GEX regime
        gex_regime = _get_gex_regime(gex_store, symbol)

        # 上次信号
        last_signal = skew_store.get_last_signal(symbol)

        # 宏观快照
        try:
            macro = fetch_macro_snapshot(ib)
        except Exception as e:
            log.warning(f"Macro snapshot failed: {e}")
            macro = None

        # 生成信号
        signal = generate_hedge_signal(
            surface=surface,
            history_df=history_df,
            gex_regime=gex_regime,
            last_signal=last_signal,
            macro=macro,
        )

        # 存储信号
        skew_store.save_hedge_signal(signal.to_dict())

        # 输出信号
        print(format_recommendation(signal))

        # 打印 surface 详情
        print(f"\n  Surface Detail:")
        for t in surface.tenors:
            rr = f"{t.rr_25*100:.1f}%" if t.rr_25 is not None else "N/A"
            iv = f"{t.atm_iv*100:.1f}%" if t.atm_iv is not None else "N/A"
            slope = f"{t.skew_slope:.2f}" if t.skew_slope is not None else "N/A"
            print(f"    {t.dte:>2}DTE (exp={t.expiry}): "
                  f"ATM_IV={iv}  RR25={rr}  Slope={slope}  "
                  f"({t.n_contracts} contracts)")
        print()

        # 执行交易
        if execute and signal.action in ('HEDGE_NOW', 'HEDGE_SPREAD'):
            executor = HedgeExecutor(
                ib=ib, db_config=db_config,
                qty=qty, dry_run=dry_run,
            )
            prefix = "[DRY RUN] " if dry_run else ""
            print(f"\n  {prefix}Executing trade...")

            record = executor.execute(signal, surface.spot)
            if record:
                print(format_trade_result(record))
            else:
                print(f"  Trade not executed (check logs)")

        elif execute and signal.action == 'REDUCE':
            executor = HedgeExecutor(
                ib=ib, db_config=db_config,
                qty=qty, dry_run=dry_run,
            )
            prefix = "[DRY RUN] " if dry_run else ""
            print(f"\n  {prefix}Reducing positions...")
            results = executor.execute_reduce(symbol)
            if results:
                for r in results:
                    print(f"  Closed trade #{r['trade_id']} ({r['legs_closed']} legs)")
            else:
                print("  No positions to reduce")

        elif execute:
            print(f"\n  Signal is {signal.action} — no trade needed")

    finally:
        if executor:
            executor.shutdown()
        ib.disconnect()
        log.info("IB disconnected")


def _reduce_positions(
    symbol: str,
    ib_host: str,
    ib_port: int,
    client_id: int,
    dry_run: bool = False,
    qty: int = 1,
) -> None:
    """平仓所有对冲头寸"""
    from ib_insync import IB

    db_config = _get_db_config()

    ib = IB()
    try:
        ib.connect(ib_host, ib_port, clientId=client_id, timeout=20)
    except Exception as e:
        log.error(f"IB 连接失败: {e}")
        sys.exit(1)

    executor = HedgeExecutor(ib=ib, db_config=db_config, qty=qty, dry_run=dry_run)
    try:
        prefix = "[DRY RUN] " if dry_run else ""
        print(f"\n  {prefix}Reducing all {symbol} hedge positions...")
        results = executor.execute_reduce(symbol)
        if results:
            for r in results:
                print(f"  Closed trade #{r['trade_id']} ({r['legs_closed']} legs)")
        else:
            print(f"  [{symbol}] No open positions")
    finally:
        executor.shutdown()
        ib.disconnect()


def _show_history(symbol: str, data_dir: Path) -> None:
    """显示历史信号"""
    skew_store = SkewSurfaceStorage(data_dir)
    df = skew_store.load_hedge_signals(symbol)

    if df is None or df.empty:
        print(f"[{symbol}] 无历史信号")
        return

    df = df.sort_values('ts')
    print(f"\n{'='*70}")
    print(f"  {symbol} Hedge Signal History ({len(df)} records)")
    print(f"{'='*70}")

    for _, row in df.tail(10).iterrows():
        ts = row['ts']
        if hasattr(ts, 'strftime'):
            ts_str = ts.strftime('%m-%d %H:%M')
        else:
            ts_str = str(ts)[:16]
        action = row['action']
        urgency = row.get('urgency', 0)
        cheap = row.get('skew_cheapness', 50)
        regime = row.get('gex_regime', '?')
        struct = row.get('recommended_structure', '?')

        icon = {'HEDGE_NOW': '🛡️', 'HEDGE_SPREAD': '🛡️',
                'MONITOR': '👀', 'SKIP': '✓', 'REDUCE': '📉'}.get(action, '?')

        print(f"  {ts_str}  {icon} {action:<15} "
              f"urgency={urgency:.0%}  "
              f"skew={cheap:.0f}pct  "
              f"gex={regime:<8}  "
              f"struct={struct}")

    print(f"{'='*70}\n")


def _show_trades(symbol: str) -> None:
    """显示交易记录和胜率"""
    db_config = _get_db_config()
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=db_config.host, port=db_config.port,
            dbname=db_config.dbname, user=db_config.user,
            password=db_config.password,
        )
    except Exception as e:
        print(f"DB connect failed: {e}")
        return

    try:
        import pandas as pd

        # 交易记录
        trades = pd.read_sql_query(
            "SELECT * FROM hedge_signal_validation WHERE symbol = %s",
            conn, params=[symbol]
        )

        if trades.empty:
            print(f"[{symbol}] 无交易记录")
            return

        print(f"\n{'='*80}")
        print(f"  {symbol} Trade History ({len(trades)} trades)")
        print(f"{'='*80}")

        for _, t in trades.iterrows():
            ts = t['signal_ts']
            ts_str = ts.strftime('%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)[:16]
            status = t['status']
            action = t['action']
            cost = t.get('entry_cost', 0) or 0
            pnl = t.get('realized_pnl')
            pnl_txt = f"${pnl:.2f}" if pnl is not None else "—"
            ret = t.get('return_pct')
            ret_txt = f"{ret:+.1f}%" if ret is not None else "—"
            days = t.get('holding_days')
            days_txt = f"{days:.1f}d" if days is not None else "—"

            status_icon = {'OPEN': '🟢', 'CLOSED': '⚪', 'EXPIRED': '⚫',
                           'DRY_RUN': '🔵'}.get(status, '?')

            print(f"  {ts_str}  {status_icon} {action:<15} "
                  f"cost=${cost:.2f}  pnl={pnl_txt}  "
                  f"ret={ret_txt}  hold={days_txt}")

        # 胜率统计
        stats = pd.read_sql_query(
            "SELECT * FROM hedge_signal_stats WHERE symbol = %s",
            conn, params=[symbol]
        )

        if not stats.empty:
            print(f"\n  {'─'*60}")
            print(f"  Stats:")
            for _, s in stats.iterrows():
                print(f"    {s['action']}: "
                      f"{s['total_trades']} trades, "
                      f"win={s.get('win_rate_pct', 0):.0f}%, "
                      f"avg_pnl=${s.get('avg_pnl', 0):.2f}, "
                      f"total=${s.get('total_pnl', 0):.2f}")

        print(f"{'='*80}\n")
    finally:
        conn.close()


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(description='GEX Hedge Signal Generator')
    parser.add_argument('--symbol', '-s', action='append', default=None,
                        help='标的代码 (可多次指定，默认 QQQ SPY)')
    parser.add_argument('--data-dir', '-d', type=str, default=str(DEFAULT_DATA_DIR),
                        help='数据目录')
    parser.add_argument('--ib-host', type=str, default='127.0.0.1',
                        help='IB Gateway host')
    parser.add_argument('--ib-port', type=int, default=4002,
                        help='IB Gateway port (4001=live, 4002=paper)')
    parser.add_argument('--client-id', type=int, default=50,
                        help='IB client ID')

    # 执行模式
    parser.add_argument('--execute', '-x', action='store_true',
                        help='信号触发时自动下单 (仅 Paper 账户)')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览下单但不实际执行')
    parser.add_argument('--qty', type=int, default=1,
                        help='每次下单合约数 (默认 1)')

    # 查看模式
    parser.add_argument('--history', action='store_true',
                        help='显示历史信号')
    parser.add_argument('--trades', action='store_true',
                        help='显示交易记录和胜率')
    parser.add_argument('--reduce', action='store_true',
                        help='平仓所有对冲头寸')
    args = parser.parse_args()

    symbols = args.symbol or DEFAULT_SYMBOLS
    data_dir = Path(args.data_dir)

    if args.history:
        for sym in symbols:
            _show_history(sym, data_dir)
        return

    if args.trades:
        for sym in symbols:
            _show_trades(sym)
        return

    if args.reduce:
        for i, sym in enumerate(symbols):
            _reduce_positions(
                symbol=sym,
                ib_host=args.ib_host,
                ib_port=args.ib_port,
                client_id=args.client_id + i,
                dry_run=args.dry_run,
                qty=args.qty,
            )
        return

    for i, sym in enumerate(symbols):
        _collect_and_signal(
            symbol=sym,
            data_dir=data_dir,
            ib_host=args.ib_host,
            ib_port=args.ib_port,
            client_id=args.client_id + i,
            execute=args.execute,
            dry_run=args.dry_run,
            qty=args.qty,
        )


if __name__ == '__main__':
    main()
