"""
QQQ 0DTE Gamma Exposure Monitor — v2
==========================================================
本版在前一版 FIX 1-16 的基础上，处理了 code review 中提出的
17 个问题。所有新改动以 `# REV N:` 标注，方便对照。

GEX 约定 (REV 3)
----------------
本程序计算的是 *dealer* gamma exposure，符号约定:
    gex = sign * gamma * OI * multiplier * spot^2 * 0.01
其中 sign = +1 (call), -1 (put)。
隐含假设: "dealers are short puts and long calls" (经典 dealer
positioning 假设)。单位: 美元 per 1% spot 变动。

注意 (REV 5): IB 返回的 OI 是前一交易日收盘数字，盘中不会变。
因此盘中 GEX 的波动 100% 来自 gamma 和 spot^2 ——
本实现严格说是 "基于前日 OI 的理论 dealer GEX"。
"""
from ib_insync import *
import ib_insync.util as ibutil
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import asyncio
import signal
import atexit
import time
import os
import uuid
import logging
import dash
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 美股交易日历
try:
    import exchange_calendars as xcals
    XNYS = xcals.get_calendar("XNYS")
    HAS_CALENDAR = True
except ImportError:
    XNYS = None
    HAS_CALENDAR = False

# ==================== 配置 ====================
SYMBOL = 'QQQ'
TRADING_CLASS = 'QQQ'          # REV 10: 做成配置项；指数期权需改
STRIKE_RANGE = 0.04
MAX_HISTORY = 8000              # REV 15: 覆盖完整交易日 + 余量
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

IB_HOST = '127.0.0.1'
IB_PORT = 4002
IB_CLIENT_ID = 11

ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

STALE_SECONDS = 15
SPOT_SANITY_PCT = 0.01          # REV 18: 调整为 1%，5% 对 QQQ 太宽松
MAX_LOGS = 30

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('gex')

# ==================== 时间辅助 ====================
def et_now():
    return datetime.now(ET)

def trading_date_str():
    return et_now().strftime('%Y%m%d')

def market_session_today(now=None):
    now = now or et_now()
    today = now.date()
    if HAS_CALENDAR:
        ts = pd.Timestamp(today)
        if not XNYS.is_session(ts):
            return None
        o = XNYS.session_open(ts).tz_convert(ET).to_pydatetime()
        c = XNYS.session_close(ts).tz_convert(ET).to_pydatetime()
        return o, c
    if now.weekday() >= 5:
        return None
    o = datetime.combine(today, MARKET_OPEN, tzinfo=ET)
    c = datetime.combine(today, MARKET_CLOSE, tzinfo=ET)
    return o, c

def is_market_open(now=None):
    now = now or et_now()
    sess = market_session_today(now)
    if sess is None:
        return False
    o, c = sess
    return o <= now <= c

# REV 8: 失败时不再静默兜底，改为抛错由调用方处理
def seconds_until_next_open(now=None):
    now = now or et_now()
    sess = market_session_today(now)
    if sess is not None and now < sess[0]:
        return (sess[0] - now).total_seconds()

    d = now.date() + timedelta(days=1)
    for _ in range(10):
        probe = datetime.combine(d, dtime(0, 1), tzinfo=ET)
        sess = market_session_today(probe)
        if sess is not None:
            return (sess[0] - now).total_seconds()
        d += timedelta(days=1)
    raise RuntimeError("10 天内找不到下一个交易日，日历可能损坏")

# ==================== 全局状态 ====================
# REV 18: 用 deque 替代 list，自动截断，避免重复创建新列表
state = {
    'spot': 0,
    'total_gex': 0,
    'gamma_flip': 0,
    'atm_iv_pct': None,         # REV 13: 重命名 synth_vix -> atm_iv_pct
    'expiry': None,
    'is_true_0dte': False,      # REV 4: 标记当前 expiry 是否为当日
    'df': pd.DataFrame(),
    'history': deque(maxlen=MAX_HISTORY),
    'ohlc_minute': deque(maxlen=MAX_HISTORY),
    'last_minute_bar': None,
    'updated': '未连接',
    'last_update_ts': None,
    'logs': deque(maxlen=MAX_LOGS),
    'market_open': False,
    'connected': False,
    'history_version': 0,       # REV 18: 用于 resample cache 失效检测
}
lock = threading.Lock()

def log_event(level, msg):
    # REV 11: 先进 lock 存 state，再写 logging handler
    # REV 18: deque 自动截断，不再需要手动 slice
    with lock:
        state['logs'].append((level, et_now(), str(msg)))
    getattr(log, level)(msg)

def log_info(msg):  log_event('info', msg)
def log_warn(msg):  log_event('warning', msg)
def log_err(msg):   log_event('error', msg)

# ==================== 文件 I/O ====================
_io_lock = threading.Lock()

# REV 7: persist 异步化，主循环只提交快照
_persist_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='persist')
_persist_future = None

# REV 18: 抽取时区处理为工具函数
def _normalize_ts_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """将 df['ts'] 转换为 UTC 时区（用于存储前）"""
    if 'ts' not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        return df
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize(ET).dt.tz_convert(UTC)
    else:
        df['ts'] = df['ts'].dt.tz_convert(UTC)
    return df

def _normalize_ts_to_et(df: pd.DataFrame) -> pd.DataFrame:
    """将 df['ts'] 转换为 ET 时区（用于读取后）"""
    if 'ts' not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        return df
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize(UTC).dt.tz_convert(ET)
    else:
        df['ts'] = df['ts'].dt.tz_convert(ET)
    return df

def _atomic_write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_parquet(tmp)
    tmp.replace(path)

def _merge_and_write(path: Path, new_df: pd.DataFrame, key_cols):
    if new_df.empty:
        return
    with _io_lock:
        if path.exists():
            try:
                old = pd.read_parquet(path)
                combined = pd.concat([old, new_df], ignore_index=True)
            except Exception as e:
                log_warn(f"读取旧 parquet 失败，覆盖: {e}")
                combined = new_df
        else:
            combined = new_df
        combined = combined.drop_duplicates(subset=key_cols, keep='last')
        combined = _normalize_ts_to_utc(combined)
        if 'ts' in combined.columns:
            combined = combined.sort_values('ts').reset_index(drop=True)
        _atomic_write_parquet(combined, path)

def _read_parquet_et(path: Path) -> pd.DataFrame:
    with _io_lock:
        df = pd.read_parquet(path)
    return _normalize_ts_to_et(df)

def _persist_sync(symbol, hist, ohlc):
    """实际执行落盘的同步函数，在后台线程运行"""
    date_str = et_now().strftime('%Y%m%d')
    try:
        if hist:
            _merge_and_write(
                DATA_DIR / f'gex_{symbol}_{date_str}.parquet',
                pd.DataFrame(hist),
                key_cols=['ts'],
            )
        if ohlc:
            _merge_and_write(
                DATA_DIR / f'ohlc_{symbol}_{date_str}.parquet',
                pd.DataFrame(ohlc),
                key_cols=['ts'],
            )
    except Exception as e:
        log_err(f"persist 失败: {e}")

def persist_async(symbol):
    """抓快照并异步落盘，不阻塞主循环"""
    global _persist_future
    with lock:
        hist = [dict(h) for h in state['history']]
        ohlc = [dict(b) for b in state['ohlc_minute']]
        if state['last_minute_bar'] is not None:
            ohlc.append(dict(state['last_minute_bar']))

    # 如果上一次还没写完，跳过这次，避免队列堆积
    if _persist_future is not None and not _persist_future.done():
        return
    _persist_future = _persist_executor.submit(_persist_sync, symbol, hist, ohlc)

def list_available_dates(symbol):
    files = sorted(DATA_DIR.glob(f'ohlc_{symbol}_*.parquet'))
    return [f.stem.split('_')[-1] for f in files]

def load_day_ohlc(symbol, date_str):
    p = DATA_DIR / f'ohlc_{symbol}_{date_str}.parquet'
    if not p.exists():
        return None
    return _read_parquet_et(p)

def resample_5min(ohlc_df):
    if ohlc_df is None or ohlc_df.empty:
        return None
    df = ohlc_df.set_index('ts')
    agg = df.resample('5min').agg({
        'open': 'first', 'high': 'max',
        'low':  'min',   'close': 'last'
    }).dropna().reset_index()
    return agg

# REV 6: resample cache 加独立锁，所有读写都在锁内
# REV 18: 用 version 替代 len，避免长度不变但内容变化时用到陈旧数据
_cache_lock = threading.Lock()
_resample_cache = {'version': -1, 'df': None}

def _history_df_locked(history, version):
    """调用方必须已持有 _cache_lock"""
    if not history:
        return pd.DataFrame()
    if _resample_cache['version'] == version and _resample_cache['df'] is not None:
        return _resample_cache['df']
    df = pd.DataFrame(list(history)).set_index('ts')
    _resample_cache['version'] = version
    _resample_cache['df'] = df
    return df

def resample_history(history, version, rule):
    """version 参数用于缓存失效检测"""
    if len(history) < 2:
        return pd.DataFrame()
    with _cache_lock:
        h = _history_df_locked(history, version)
        return h.resample(rule).last().dropna(subset=['total_gex'])

# ==================== 分段标注 I/O ====================
SEGMENTS_FILE = DATA_DIR / 'segments.parquet'
SEGMENT_COLUMNS = ['id', 'date', 'start_ts', 'end_ts', 'symbol',
                   'label', 'note', 'labeled_at']

def load_segments():
    if SEGMENTS_FILE.exists():
        df = _read_parquet_et(SEGMENTS_FILE)
        if 'id' not in df.columns:
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
            df = df[SEGMENT_COLUMNS]
            with _io_lock:
                _atomic_write_parquet(df, SEGMENTS_FILE)
        return df
    return pd.DataFrame(columns=SEGMENT_COLUMNS)

def save_segment(date_str, start_ts, end_ts, symbol, label, note):
    df = load_segments()
    new_row = pd.DataFrame([{
        'id': str(uuid.uuid4()),
        'date': date_str,
        'start_ts': pd.Timestamp(start_ts),
        'end_ts': pd.Timestamp(end_ts),
        'symbol': symbol,
        'label': label,
        'note': note or '',
        'labeled_at': et_now().isoformat(timespec='seconds'),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    with _io_lock:
        _atomic_write_parquet(df, SEGMENTS_FILE)
    return df

def delete_segments_by_ids(ids):
    if not ids:
        return
    df = load_segments()
    df = df[~df['id'].isin(ids)]
    with _io_lock:
        _atomic_write_parquet(df, SEGMENTS_FILE)

# ==================== Expiry 选取 ====================
# REV 4: 同时返回 "是否为真 0DTE" 标记
def pick_expiry(chain):
    today_str = trading_date_str()
    future = sorted(e for e in chain.expirations if e >= today_str)
    if not future:
        return None, False
    chosen = future[0]
    return chosen, (chosen == today_str)

# ==================== IB 连接 ====================
def connect_ib():
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    return ib

def setup_underlying(ib):
    underlying = Stock(SYMBOL, 'SMART', 'USD')
    ib.qualifyContracts(underlying)
    chains = ib.reqSecDefOptParams(
        underlying.symbol, '', underlying.secType, underlying.conId)
    # REV 18: 添加 default=None 避免 StopIteration
    chain = next((c for c in chains if c.exchange == 'SMART'), None)
    if chain is None:
        raise RuntimeError(f"未找到 {SYMBOL} 的 SMART 期权链，"
                           f"可用交易所: {[c.exchange for c in chains]}")
    return underlying, chain

GENERIC_TICKS = '100,101,104,106'

def subscribe_market_data(ib, contracts):
    for c in contracts:
        ib.reqMktData(c, genericTickList=GENERIC_TICKS, snapshot=False)

def unsubscribe_market_data(ib, contracts):
    for c in contracts:
        try:
            ib.cancelMktData(c)
        except Exception:
            pass

# ==================== IB 数据线程 ====================
_ib_holder = {'ib': None}

def ib_worker():
    asyncio.set_event_loop(asyncio.new_event_loop())

    ib = None
    underlying = None
    chain = None
    current_key = None
    current_contracts = []
    last_persist = time.time()
    last_expiry_seen = None
    last_good_spot = None       # REV 17: 用于 sanity check

    def _sleep(sec):
        """REV 1: 有连接时用 ib.sleep 推进 event loop；否则退回 time.sleep"""
        if ib is not None and ib.isConnected():
            ib.sleep(sec)
        else:
            time.sleep(sec)

    while True:
        # ---- 非交易时段 ----
        if not is_market_open():
            with lock:
                state['market_open'] = False
                state['updated'] = f"非交易时段 ({et_now().strftime('%H:%M ET')})"

            if ib is not None and ib.isConnected():
                try:
                    persist_async(SYMBOL)
                except Exception as e:
                    log_err(f"盘后 persist 失败: {e}")
                try:
                    unsubscribe_market_data(ib, current_contracts)
                    ib.disconnect()
                except Exception:
                    pass
                ib = None
                _ib_holder['ib'] = None
                current_key = None
                current_contracts = []
                last_good_spot = None

            try:
                sleep_sec = max(seconds_until_next_open() - 60, 30)
                log_info(f"休市中，下次检查 {sleep_sec:.0f}s 后")
                time.sleep(min(sleep_sec, 1800))
            except RuntimeError as e:
                log_err(f"{e}; 5 分钟后重试")
                time.sleep(300)
            continue

        # ---- 确保连接 ----
        if ib is None or not ib.isConnected():
            try:
                if ib is not None:
                    try:
                        ib.disconnect()
                    except Exception:
                        pass
                ib = connect_ib()
                _ib_holder['ib'] = ib
                underlying, chain = setup_underlying(ib)
                # REV 2: underlying 也用 streaming 订阅
                ib.reqMktData(underlying, genericTickList='', snapshot=False)
                current_key = None
                current_contracts = []
                last_good_spot = None
                with lock:
                    state['connected'] = True
                    state['market_open'] = True
                log_info("IB 已连接")
                ib.sleep(1)  # 等 underlying 首个 tick
            except Exception as e:
                with lock:
                    state['connected'] = False
                log_err(f"IB 连接失败: {e}")
                time.sleep(10)
                continue

        # ---- 主循环 ----
        try:
            # REV 2: 从 streaming ticker 读缓存，不再 reqTickers
            u_ticker = ib.ticker(underlying)
            spot = u_ticker.marketPrice() if u_ticker else None
            if not spot or np.isnan(spot) or spot <= 0:
                _sleep(2)
                continue

            # REV 17: spot sanity check
            if last_good_spot is not None:
                drift = abs(spot - last_good_spot) / last_good_spot
                if drift > SPOT_SANITY_PCT:
                    log_warn(f"丢弃异常 spot={spot:.2f} "
                             f"(上次={last_good_spot:.2f}, 漂移 {drift:.1%})")
                    _sleep(2)
                    continue
            last_good_spot = spot

            # REV 4: expiry + 是否为真 0DTE
            expiry, is_true_0dte = pick_expiry(chain)
            if expiry is None:
                log_err('无可用 expiry')
                _sleep(10)
                continue

            strikes = sorted(s for s in chain.strikes
                             if (1 - STRIKE_RANGE) * spot < s < (1 + STRIKE_RANGE) * spot)
            key = (expiry, tuple(strikes))

            if key != current_key:
                if current_contracts:
                    unsubscribe_market_data(ib, current_contracts)
                raw = [Option(SYMBOL, expiry, s, r, 'SMART',
                              tradingClass=TRADING_CLASS)
                       for s in strikes for r in ['C', 'P']]
                current_contracts = ib.qualifyContracts(*raw)
                subscribe_market_data(ib, current_contracts)
                current_key = key
                log_info(f"订阅 {len(current_contracts)} 个合约 "
                         f"expiry={expiry} strikes={len(strikes)}")
                ib.sleep(2)  # 等初始 tick

            # REV 4: 非真 0DTE 时显眼告警
            if expiry != last_expiry_seen:
                last_expiry_seen = expiry
                if is_true_0dte:
                    log_info(f"当前 expiry: {expiry} (真 0DTE)")
                else:
                    log_warn(f"⚠️ 今日无 0DTE 合约，回退到 {expiry} — "
                             f"GEX 语义与 0DTE 不同")

            tickers = [ib.ticker(c) for c in current_contracts]

            rows = []
            missing_oi = 0
            missing_greeks = 0
            for t in tickers:
                if t is None:
                    continue
                g = t.modelGreeks
                if not g or g.gamma is None:
                    missing_greeks += 1
                    continue
                c = t.contract
                oi = t.callOpenInterest if c.right == 'C' else t.putOpenInterest
                if not oi or (isinstance(oi, float) and np.isnan(oi)):
                    missing_oi += 1
                    continue
                # REV 3: dealer 约定见文件头 docstring
                sign = 1 if c.right == 'C' else -1
                multiplier = int(c.multiplier) if c.multiplier else 100
                gex = sign * g.gamma * oi * multiplier * spot ** 2 * 0.01
                rows.append({
                    'strike': c.strike, 'right': c.right,
                    'gamma': g.gamma, 'oi': oi, 'gex': gex,
                    'iv': g.impliedVol,
                })

            if not rows:
                log_warn(f'无有效数据: missing_greeks={missing_greeks} '
                         f'missing_oi={missing_oi} — 检查市场数据订阅')
                _sleep(5)
                continue

            df = pd.DataFrame(rows)

            by_strike = df.groupby('strike')['gex'].sum().sort_index()
            flip = by_strike.cumsum().abs().idxmin()
            total_gex = df['gex'].sum()
            call_gex = df[df.right == 'C']['gex'].sum()
            put_gex = df[df.right == 'P']['gex'].sum()

            atm_strike = min(df['strike'].unique(), key=lambda s: abs(s - spot))
            atm_rows = df[(df['strike'] == atm_strike) & df['iv'].notna()]
            if not atm_rows.empty:
                call_iv = atm_rows[atm_rows.right == 'C']['iv'].mean()
                put_iv = atm_rows[atm_rows.right == 'P']['iv'].mean()
                ivs = [x for x in (call_iv, put_iv) if pd.notna(x)]
                atm_iv = np.mean(ivs) if ivs else None
            else:
                atm_iv = None
            # REV 13: 重命名，避免与 VIX 混淆
            atm_iv_pct = float(atm_iv * 100) if atm_iv is not None else None

            now = et_now()
            minute = now.replace(second=0, microsecond=0)

            with lock:
                state['spot'] = spot
                state['total_gex'] = total_gex
                state['gamma_flip'] = flip
                state['atm_iv_pct'] = atm_iv_pct
                state['expiry'] = expiry
                state['is_true_0dte'] = is_true_0dte
                state['df'] = df
                state['updated'] = now.strftime('%H:%M:%S ET')
                state['last_update_ts'] = now
                state['market_open'] = True
                state['connected'] = True

                state['history'].append({
                    'ts': now,
                    'spot': spot,
                    'total_gex': total_gex,
                    'flip': flip,
                    'call_gex': call_gex,
                    'put_gex': put_gex,
                    'atm_iv_pct': atm_iv_pct,
                })
                # REV 18: deque 自动截断，不再需要手动 slice

                lb = state['last_minute_bar']
                if lb is None or lb['ts'] != minute:
                    if lb is not None:
                        state['ohlc_minute'].append(lb)
                    state['last_minute_bar'] = {
                        'ts': minute, 'open': spot,
                        'high': spot, 'low': spot, 'close': spot
                    }
                else:
                    lb['high'] = max(lb['high'], spot)
                    lb['low'] = min(lb['low'], spot)
                    lb['close'] = spot

                # REV 18: 递增 version 使 resample cache 失效
                state['history_version'] += 1

            if time.time() - last_persist > 60:
                persist_async(SYMBOL)    # REV 7: 异步
                last_persist = time.time()

        except Exception as e:
            log_err(f"主循环异常: {e}")
            if ib is not None and not ib.isConnected():
                with lock:
                    state['connected'] = False

        _sleep(3)  # REV 1: 推进 IB event loop

def _graceful_shutdown(*_):
    log_info("关闭中...")
    ib = _ib_holder.get('ib')
    if ib is not None and ib.isConnected():
        try:
            ib.disconnect()
        except Exception:
            pass
    try:
        persist_async(SYMBOL)
        # 等待最后一次落盘完成
        if _persist_future is not None:
            _persist_future.result(timeout=5)
    except Exception:
        pass
    _persist_executor.shutdown(wait=False)

atexit.register(_graceful_shutdown)

# REV 18: 用 sys.exit 替代 os._exit，确保 atexit 和 finally 能执行
_shutdown_flag = threading.Event()

def _signal_handler(signum, frame):
    if _shutdown_flag.is_set():
        return  # 避免重复触发
    _shutdown_flag.set()
    _graceful_shutdown()
    sys.exit(0)

for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, _signal_handler)
    except Exception as e:
        # REV 9: 非主线程启动时会失败，显式告警
        log_warn(f"signal 注册失败 (非主线程?): {e}")

# ==================== Dash UI ====================
app = dash.Dash(__name__)
app.title = f"{SYMBOL} GEX + Labeling"

LABEL_OPTIONS = [
    {'label': '📈 Trend Up',   'value': 'trend_up'},
    {'label': '📉 Trend Down', 'value': 'trend_down'},
    {'label': '〰️ Chop',       'value': 'chop'},
    {'label': '🔀 Mixed',      'value': 'mixed'},
]

LABEL_COLORS = {
    'trend_up':   'rgba(0, 255, 136, 0.22)',
    'trend_down': 'rgba(255, 68, 68, 0.22)',
    'chop':       'rgba(255, 170, 0, 0.22)',
    'mixed':      'rgba(255, 102, 204, 0.22)',
}

LEVEL_COLORS = {
    'info':    '#888',
    'warning': '#ffaa00',
    'error':   '#ff6666',
}

app.layout = html.Div(style={
    'backgroundColor': '#0e1117', 'color': '#fafafa',
    'fontFamily': 'monospace', 'padding': '20px'
}, children=[

    html.H1(f"{SYMBOL} 0DTE Gamma Exposure Monitor", style={'textAlign': 'center'}),
    html.Div(id='stats', style={'textAlign': 'center', 'fontSize': '18px', 'margin': '20px'}),

    html.Details([
        html.Summary("🔔 日志（最近 30 条）",
                     style={'cursor': 'pointer', 'color': '#888'}),
        html.Div(id='error-log', style={
            'backgroundColor': '#1a1f2e', 'padding': '10px',
            'fontSize': '12px',
            'maxHeight': '220px', 'overflowY': 'auto',
            'fontFamily': 'monospace',
        }),
    ], style={'maxWidth': '1100px', 'margin': '0 auto 20px'}),

    dcc.Graph(id='gex-chart'),

    html.H2("Intraday Evolution", style={'textAlign': 'center', 'marginTop': '30px'}),
    dcc.Graph(id='history-chart'),

    html.Hr(style={'marginTop': '40px', 'borderColor': '#333'}),
    html.H2("📝 Segment Labeling (5min view)", style={'textAlign': 'center'}),

    html.Div(style={
        'maxWidth': '1100px', 'margin': 'auto',
        'backgroundColor': '#1a1f2e', 'padding': '20px', 'borderRadius': '8px'
    }, children=[

        html.Div(style={'marginBottom': '15px'}, children=[
            html.Label("选择日期:"),
            dcc.Dropdown(id='date-dropdown', style={'color': 'black', 'width': '300px'}),
        ]),

        html.Div("👉 在图上按住鼠标左键拖动框选一段区间",
                 style={'color': '#00d4ff', 'fontSize': '13px', 'marginBottom': '8px'}),

        dcc.Graph(
            id='day-chart',
            config={'modeBarButtonsToAdd': ['select2d'], 'displaylogo': False},
            style={'height': '450px'}
        ),

        html.Div(id='selection-info',
                 style={'margin': '10px 0', 'fontSize': '14px', 'color': '#ffaa00'}),

        html.Div(style={
            'display': 'flex', 'gap': '10px',
            'alignItems': 'center', 'flexWrap': 'wrap'
        }, children=[
            html.Label("Regime:"),
            dcc.RadioItems(id='label-radio', options=LABEL_OPTIONS, value='chop',
                           labelStyle={'display': 'inline-block', 'marginRight': '12px'}),
            dcc.Input(id='label-note', type='text', placeholder='备注(可选)',
                      style={'width': '250px', 'backgroundColor': '#0e1117',
                             'color': 'white', 'border': '1px solid #333', 'padding': '6px'}),
            html.Button('➕ 添加分段', id='add-btn', n_clicks=0,
                        style={'padding': '8px 16px', 'backgroundColor': '#00ff88',
                               'color': 'black', 'border': 'none', 'borderRadius': '4px',
                               'cursor': 'pointer'}),
            html.Button('🗑 删除所选行', id='delete-btn', n_clicks=0,
                        style={'padding': '8px 16px', 'backgroundColor': '#ff4444',
                               'color': 'white', 'border': 'none', 'borderRadius': '4px',
                               'cursor': 'pointer'}),
        ]),

        html.Div(id='save-status',
                 style={'marginTop': '10px', 'color': '#00ff88', 'minHeight': '20px'}),

        html.H3("当日已标注分段", style={'marginTop': '25px'}),
        dash_table.DataTable(
            id='segments-table',
            columns=[
                {'name': 'Start', 'id': 'start_str'},
                {'name': 'End',   'id': 'end_str'},
                {'name': 'Label', 'id': 'label'},
                {'name': 'Note',  'id': 'note'},
            ],
            row_selectable='multi',
            selected_rows=[],
            style_cell={'backgroundColor': '#0e1117', 'color': 'white',
                        'fontFamily': 'monospace', 'fontSize': '12px', 'padding': '6px'},
            style_header={'backgroundColor': '#1a1f2e', 'fontWeight': 'bold'},
        ),

        html.H3("历史标注总览", style={'marginTop': '25px'}),
        html.Div(id='segments-summary', style={'color': '#aaa', 'fontSize': '13px'}),
    ]),

    dcc.Store(id='selected-range'),
    dcc.Store(id='refresh-trigger', data=0),
    # REV 12: 和 IB 采集 3s 错开相位
    dcc.Interval(id='interval',      interval=4000,  n_intervals=0),
    dcc.Interval(id='slow-interval', interval=30000, n_intervals=0),
])

# ==================== 实时图回调 ====================
@app.callback(
    [Output('stats', 'children'),
     Output('gex-chart', 'figure'),
     Output('history-chart', 'figure'),
     Output('error-log', 'children')],
    Input('interval', 'n_intervals'))
def update_live(_):
    with lock:
        s = {k: v for k, v in state.items()
             if k not in ('history', 'ohlc_minute', 'df', 'logs')}
        df = state['df'].copy()
        # REV 18: 传递 deque 引用和 version，在锁外做 resample 时用于缓存
        history = state['history']
        history_version = state['history_version']
        logs = list(state['logs'])

    log_children = [
        html.Div(f"[{ts.strftime('%H:%M:%S')}] {msg}",
                 style={'color': LEVEL_COLORS.get(level, '#aaa')})
        for level, ts, msg in reversed(logs)
    ] or [html.Div("(无)", style={'color': '#888'})]

    stale_warning = None
    if s.get('last_update_ts') is not None:
        age = (et_now() - s['last_update_ts']).total_seconds()
        if age > STALE_SECONDS:
            stale_warning = f"⚠️ 数据 {age:.0f}s 未更新"

    if not s.get('market_open'):
        return f"😴 {s['updated']}", go.Figure(), go.Figure(), log_children

    if df.empty:
        return "等待数据...", go.Figure(), go.Figure(), log_children

    iv_txt = f"{s['atm_iv_pct']:.1f}" if s['atm_iv_pct'] else "—"
    exp_txt = s.get('expiry') or "—"
    if not s.get('is_true_0dte'):
        exp_txt += " ⚠️非0DTE"
    updated_color = '#ff4444' if stale_warning else '#888'

    stats_children = [
        html.Span(f"Spot: {s['spot']:.2f}  |  ", style={'color': '#00d4ff'}),
        html.Span(f"Total GEX: ${s['total_gex']/1e6:.1f}M  |  ",
                  style={'color': '#00ff88' if s['total_gex'] > 0 else '#ff4444'}),
        html.Span(f"Flip: {s['gamma_flip']:.0f}  |  ", style={'color': '#ffaa00'}),
        html.Span(f"ATM IV: {iv_txt}%  |  ", style={'color': '#ff66cc'}),
        html.Span(f"Exp: {exp_txt}  |  ",
                  style={'color': '#ff4444' if not s.get('is_true_0dte') else '#aaaaaa'}),
        html.Span(f"Updated: {s['updated']}", style={'color': updated_color}),
    ]
    if stale_warning:
        stats_children.append(
            html.Span(f"  {stale_warning}",
                      style={'color': '#ff4444', 'fontWeight': 'bold'})
        )
    stats = html.Div(stats_children)

    # ---- GEX by strike ----
    by_strike = df.groupby('strike')['gex'].sum().sort_index() / 1e6
    calls = df[df.right == 'C'].groupby('strike')['gex'].sum() / 1e6
    puts  = df[df.right == 'P'].groupby('strike')['gex'].sum() / 1e6

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=('Net GEX by Strike', 'Calls vs Puts'),
                         vertical_spacing=0.12)
    colors = ['#00ff88' if v > 0 else '#ff4444' for v in by_strike.values]
    fig1.add_trace(go.Bar(x=by_strike.index, y=by_strike.values,
                          marker_color=colors, name='Net'), row=1, col=1)
    fig1.add_trace(go.Bar(x=calls.index, y=calls.values,
                          marker_color='#00d4ff', name='Calls'), row=2, col=1)
    fig1.add_trace(go.Bar(x=puts.index, y=puts.values,
                          marker_color='#ff66cc', name='Puts'), row=2, col=1)
    for r in [1, 2]:
        fig1.add_vline(x=s['spot'], line=dict(color='white', dash='dash'), row=r, col=1)
        fig1.add_vline(x=s['gamma_flip'], line=dict(color='#ffaa00', dash='dot'), row=r, col=1)
    fig1.update_layout(template='plotly_dark', height=650, barmode='relative',
                       paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig1.update_yaxes(title_text='GEX ($M per 1%)')
    fig1.update_xaxes(title_text='Strike', row=2, col=1)

    # ---- 历史演化 ----
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=('Total GEX ($M)', 'Spot vs Flip', 'ATM IV (%)'),
                         vertical_spacing=0.08,
                         row_heights=[0.4, 0.35, 0.25])

    grids = [('30s', '30s', '#00d4ff'),
             ('1min', '1m', '#00ff88'),
             ('3min', '3m', '#ffaa00'),
             ('5min', '5m', '#ff66cc')]
    for rule, lbl, color in grids:
        r = resample_history(history, history_version, rule)
        if r.empty:
            continue
        fig2.add_trace(go.Scatter(x=r.index, y=r['total_gex'] / 1e6,
                                  mode='lines', name=f'GEX {lbl}',
                                  line=dict(color=color, width=1.5)),
                       row=1, col=1)

    r1 = resample_history(history, history_version, '1min')
    if not r1.empty:
        fig2.add_trace(go.Scatter(x=r1.index, y=r1['spot'], mode='lines',
                                  name='Spot', line=dict(color='white', width=2)),
                       row=2, col=1)
        fig2.add_trace(go.Scatter(x=r1.index, y=r1['flip'], mode='lines',
                                  name='Flip',
                                  line=dict(color='#ffaa00', width=2, dash='dot')),
                       row=2, col=1)
        if 'atm_iv_pct' in r1.columns:
            fig2.add_trace(go.Scatter(x=r1.index, y=r1['atm_iv_pct'], mode='lines',
                                      name='ATM IV',
                                      line=dict(color='#ff66cc', width=2)),
                           row=3, col=1)

    fig2.add_hline(y=0, line=dict(color='gray', dash='dash'), row=1, col=1)
    fig2.update_layout(template='plotly_dark', height=750,
                       paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig2.update_yaxes(title_text='$M', row=1, col=1)
    fig2.update_yaxes(title_text='Price', row=2, col=1)
    fig2.update_yaxes(title_text='IV%', row=3, col=1)

    return stats, fig1, fig2, log_children

# ==================== 日期下拉刷新 ====================
@app.callback(
    Output('date-dropdown', 'options'),
    Output('date-dropdown', 'value'),
    Input('slow-interval', 'n_intervals'),
    State('date-dropdown', 'value'))
def refresh_dates(_, current):
    dates = list_available_dates(SYMBOL)
    opts = [{'label': d, 'value': d} for d in dates]
    val = current if current in dates else (dates[-1] if dates else None)
    return opts, val

# ==================== 当日 K 线 + 已标分段 ====================
@app.callback(
    Output('day-chart', 'figure'),
    Output('segments-table', 'data'),
    Output('segments-summary', 'children'),
    Input('date-dropdown', 'value'),
    Input('refresh-trigger', 'data'))
def render_day(date_str, _trigger):
    fig = go.Figure()
    table_data = []
    summary = ""

    if date_str:
        ohlc = load_day_ohlc(SYMBOL, date_str)
        bars = resample_5min(ohlc)

        if bars is not None and not bars.empty:
            fig.add_trace(go.Candlestick(
                x=bars['ts'], open=bars['open'], high=bars['high'],
                low=bars['low'], close=bars['close'],
                increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
                name='5m'
            ))

            segs = load_segments()
            if not segs.empty:
                day_segs = segs[(segs['date'] == date_str) & (segs['symbol'] == SYMBOL)]
            else:
                day_segs = pd.DataFrame()

            for _, seg in day_segs.iterrows():
                fig.add_vrect(
                    x0=seg['start_ts'], x1=seg['end_ts'],
                    fillcolor=LABEL_COLORS.get(seg['label'], 'rgba(128,128,128,0.2)'),
                    line_width=0,
                    annotation_text=seg['label'],
                    annotation_position='top left',
                    annotation=dict(font_size=10, font_color='white'),
                )
                table_data.append({
                    'id': seg['id'],
                    'start_str': pd.Timestamp(seg['start_ts']).strftime('%H:%M'),
                    'end_str':   pd.Timestamp(seg['end_ts']).strftime('%H:%M'),
                    'label': seg['label'],
                    'note': seg['note'],
                })

    fig.update_layout(
        template='plotly_dark', height=450,
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False,
        dragmode='select',
        margin=dict(t=20, b=30, l=50, r=20),
    )

    all_segs = load_segments()
    if not all_segs.empty:
        total = len(all_segs)
        days = all_segs['date'].nunique()
        dist = all_segs['label'].value_counts().to_dict()
        dist_str = "  ".join(f"{k}: {v}" for k, v in dist.items())
        summary = f"总分段数: {total}  |  覆盖天数: {days}  |  分布: {dist_str}"

    return fig, table_data, summary

# ==================== 框选事件 ====================
@app.callback(
    Output('selected-range', 'data'),
    Output('selection-info', 'children'),
    Input('day-chart', 'selectedData'))
def on_select(selected):
    if not selected or 'range' not in selected or 'x' not in selected['range']:
        return None, "尚未选中区间 — 在图上拖动鼠标框选"
    x0, x1 = selected['range']['x']
    try:
        t0 = pd.Timestamp(x0).strftime('%H:%M')
        t1 = pd.Timestamp(x1).strftime('%H:%M')
    except Exception:
        return None, "选中范围解析失败"
    return {'x0': x0, 'x1': x1}, f"✂️ 已选中: {t0} → {t1}"

# ==================== 添加 / 删除分段 ====================
@app.callback(
    Output('save-status', 'children'),
    Output('refresh-trigger', 'data'),
    Input('add-btn', 'n_clicks'),
    Input('delete-btn', 'n_clicks'),
    State('date-dropdown', 'value'),
    State('selected-range', 'data'),
    State('label-radio', 'value'),
    State('label-note', 'value'),
    State('segments-table', 'data'),
    State('segments-table', 'selected_rows'),
    State('refresh-trigger', 'data'),
    prevent_initial_call=True)
def modify_segments(_add, _del, date_str, sel_range, label, note,
                    table_data, selected_rows, trigger):
    trigger = (trigger or 0)
    triggered = ctx.triggered_id

    if triggered == 'add-btn':
        if not date_str:
            return "⚠️ 请先选日期", trigger
        if not sel_range:
            return "⚠️ 请先在图上框选一段区间", trigger
        if not label:
            return "⚠️ 请选择 regime 标签", trigger
        save_segment(date_str, sel_range['x0'], sel_range['x1'], SYMBOL, label, note)
        return (f"✅ 已添加: {label} ({et_now().strftime('%H:%M:%S ET')})",
                trigger + 1)

    if triggered == 'delete-btn':
        if not selected_rows or not table_data:
            return "⚠️ 请先在表格勾选要删除的行", trigger
        ids = [table_data[i]['id'] for i in selected_rows
               if i < len(table_data) and 'id' in table_data[i]]
        if not ids:
            return "⚠️ 选中行没有 id", trigger
        delete_segments_by_ids(ids)
        return (f"🗑 已删除 {len(ids)} 条 ({et_now().strftime('%H:%M:%S ET')})",
                trigger + 1)

    return dash.no_update, dash.no_update

# ==================== 启动 ====================
# REV 18: debug=False 时 Werkzeug 不会 fork，无需检测 WERKZEUG_RUN_MAIN
if __name__ == '__main__':
    threading.Thread(target=ib_worker, daemon=True).start()
    app.run(debug=False, host='127.0.0.1', port=8050)
