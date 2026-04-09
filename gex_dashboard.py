"""
QQQ 0DTE Gamma Exposure Monitor — 修订版
主要修复点见代码内 `# FIX N:` 注释，与回复中的编号对应。
"""
from ib_insync import *
import ib_insync.util as ibutil
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import threading
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

# 美股交易日历（节假日 + 半日市）
try:
    import exchange_calendars as xcals
    XNYS = xcals.get_calendar("XNYS")
    HAS_CALENDAR = True
except ImportError:
    XNYS = None
    HAS_CALENDAR = False

# ==================== 配置 ====================
SYMBOL = 'QQQ'
STRIKE_RANGE = 0.04
MAX_HISTORY = 7200
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

# FIX 8: 分离 info / error 日志
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

# FIX 9: 用 exchange_calendars 处理节假日 + 半日市
def market_session_today(now=None):
    """返回 (open_dt, close_dt) 或 None（非交易日）"""
    now = now or et_now()
    today = now.date()
    if HAS_CALENDAR:
        ts = pd.Timestamp(today)
        if not XNYS.is_session(ts):
            return None
        o = XNYS.session_open(ts).tz_convert(ET).to_pydatetime()
        c = XNYS.session_close(ts).tz_convert(ET).to_pydatetime()
        return o, c
    # Fallback: 仅过滤周末
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

# FIX 10: 非交易时段直接 sleep 到下次开盘前 60s
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
    return 3600  # 兜底

# ==================== 全局状态 ====================
state = {
    'spot': 0,
    'total_gex': 0,
    'gamma_flip': 0,
    'synth_vix': None,
    'expiry': None,
    'df': pd.DataFrame(),
    'history': [],
    'ohlc_minute': [],
    'last_minute_bar': None,
    'updated': '未连接',
    'last_update_ts': None,
    'logs': [],                 # FIX 8: 统一 (level, ts, msg)
    'market_open': False,
    'connected': False,
}
lock = threading.Lock()
MAX_LOGS = 30

def log_event(level, msg):
    getattr(log, level)(msg)
    with lock:
        state['logs'].append((level, et_now(), str(msg)))
        state['logs'] = state['logs'][-MAX_LOGS:]

def log_info(msg):  log_event('info', msg)
def log_warn(msg):  log_event('warning', msg)
def log_err(msg):   log_event('error', msg)

# ==================== 文件 I/O ====================
# FIX 4: 原子写 parquet + 进程内写锁
_io_lock = threading.Lock()

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
        if 'ts' in combined.columns:
            # FIX 14: 统一存 UTC，读时转 ET
            if pd.api.types.is_datetime64_any_dtype(combined['ts']):
                if combined['ts'].dt.tz is None:
                    combined['ts'] = combined['ts'].dt.tz_localize(ET).dt.tz_convert(UTC)
                else:
                    combined['ts'] = combined['ts'].dt.tz_convert(UTC)
            combined = combined.sort_values('ts').reset_index(drop=True)
        _atomic_write_parquet(combined, path)

def _read_parquet_et(path: Path) -> pd.DataFrame:
    """读 parquet 并把 ts 转回 ET tz-aware"""
    with _io_lock:
        df = pd.read_parquet(path)
    if 'ts' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ts']):
        if df['ts'].dt.tz is None:
            df['ts'] = df['ts'].dt.tz_localize(UTC).dt.tz_convert(ET)
        else:
            df['ts'] = df['ts'].dt.tz_convert(ET)
    return df

def persist(symbol):
    date_str = trading_date_str()
    with lock:
        # FIX 5: 深拷贝每一条 dict，防止读写冲突
        hist = [dict(h) for h in state['history']]
        ohlc = [dict(b) for b in state['ohlc_minute']]
        if state['last_minute_bar'] is not None:
            ohlc.append(dict(state['last_minute_bar']))

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

# FIX 12: 历史 resample 缓存；只对截断窗口算一次
_resample_cache = {'len': -1, 'df': None}

def _history_df(history):
    if not history:
        return pd.DataFrame()
    if _resample_cache['len'] == len(history) and _resample_cache['df'] is not None:
        return _resample_cache['df']
    df = pd.DataFrame(history).set_index('ts')
    _resample_cache['len'] = len(history)
    _resample_cache['df'] = df
    return df

def resample_history(history, rule):
    if len(history) < 2:
        return pd.DataFrame()
    h = _history_df(history)
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
def pick_expiry(chain):
    today_str = trading_date_str()
    future = sorted(e for e in chain.expirations if e >= today_str)
    return future[0] if future else None

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
    chain = next(c for c in chains if c.exchange == 'SMART')
    return underlying, chain

# FIX 3: 必须订阅 genericTickList='101' 才能拿 OI
GENERIC_TICKS = '100,101,104,106'

def subscribe_market_data(ib, contracts):
    """订阅带 OI 的 streaming 数据"""
    for c in contracts:
        ib.reqMktData(c, genericTickList=GENERIC_TICKS, snapshot=False)

def unsubscribe_market_data(ib, contracts):
    for c in contracts:
        try:
            ib.cancelMktData(c)
        except Exception:
            pass

# ==================== IB 数据线程 ====================
_ib_holder = {'ib': None}  # 供 atexit 关闭使用

def ib_worker():
    # FIX 1: 在线程内建立独立 event loop
    asyncio.set_event_loop(asyncio.new_event_loop())

    ib = None
    underlying = None
    chain = None
    # FIX 6: 缓存当前订阅的 (expiry, strikes_tuple)，切换时清理
    current_key = None
    current_contracts = []
    last_persist = time.time()
    last_expiry_seen = None

    while True:
        # ---- 非交易时段 ----
        if not is_market_open():
            with lock:
                state['market_open'] = False
                state['updated'] = f"非交易时段 ({et_now().strftime('%H:%M ET')})"

            if ib is not None and ib.isConnected():
                try:
                    persist(SYMBOL)
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

            sleep_sec = max(seconds_until_next_open() - 60, 30)
            log_info(f"休市中，下次检查 {sleep_sec:.0f}s 后")
            time.sleep(min(sleep_sec, 1800))  # 最长睡 30 分钟，便于手动重启
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
                current_key = None
                current_contracts = []
                with lock:
                    state['connected'] = True
                    state['market_open'] = True
                log_info("IB 已连接")
            except Exception as e:
                with lock:
                    state['connected'] = False
                log_err(f"IB 连接失败: {e}")
                time.sleep(10)
                continue

        # ---- 主循环 ----
        try:
            [u_ticker] = ib.reqTickers(underlying)
            spot = u_ticker.marketPrice()
            if not spot or np.isnan(spot):
                time.sleep(2)
                continue

            expiry = pick_expiry(chain)
            if expiry is None:
                log_err('无可用 expiry')
                time.sleep(10)
                continue

            # FIX 6: 检测到新 expiry（跨日）或 spot 漂移导致 strikes 变化时，
            # 先取消旧订阅再重新订阅，避免合约/缓存泄漏
            strikes = sorted(s for s in chain.strikes
                             if (1 - STRIKE_RANGE) * spot < s < (1 + STRIKE_RANGE) * spot)
            key = (expiry, tuple(strikes))

            if key != current_key:
                if current_contracts:
                    unsubscribe_market_data(ib, current_contracts)
                # FIX 15: 加 tradingClass 消除合约歧义
                raw = [Option(SYMBOL, expiry, s, r, 'SMART', tradingClass=SYMBOL)
                       for s in strikes for r in ['C', 'P']]
                current_contracts = ib.qualifyContracts(*raw)
                subscribe_market_data(ib, current_contracts)
                current_key = key
                log_info(f"订阅 {len(current_contracts)} 个合约 "
                         f"expiry={expiry} strikes={len(strikes)}")
                # 给 IB 一点时间回推初始 tick
                ib.sleep(2)

            if expiry != last_expiry_seen:
                last_expiry_seen = expiry
                log_info(f"当前 expiry: {expiry}")

            # 从已订阅的 ticker 里读最新快照
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
                time.sleep(5)
                continue

            df = pd.DataFrame(rows)

            by_strike = df.groupby('strike')['gex'].sum().sort_index()
            flip = by_strike.cumsum().abs().idxmin()
            total_gex = df['gex'].sum()
            call_gex = df[df.right == 'C']['gex'].sum()
            put_gex = df[df.right == 'P']['gex'].sum()

            # FIX 7: ATM IV 用 call/put 平均（更接近 ATM implied）
            atm_strike = min(df['strike'].unique(), key=lambda s: abs(s - spot))
            atm_rows = df[(df['strike'] == atm_strike) & df['iv'].notna()]
            if not atm_rows.empty:
                call_iv = atm_rows[atm_rows.right == 'C']['iv'].mean()
                put_iv = atm_rows[atm_rows.right == 'P']['iv'].mean()
                ivs = [x for x in (call_iv, put_iv) if pd.notna(x)]
                atm_iv = np.mean(ivs) if ivs else None
            else:
                atm_iv = None
            synth_vix = float(atm_iv * 100) if atm_iv is not None else None

            now = et_now()
            minute = now.replace(second=0, microsecond=0)

            with lock:
                state['spot'] = spot
                state['total_gex'] = total_gex
                state['gamma_flip'] = flip
                state['synth_vix'] = synth_vix
                state['expiry'] = expiry
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
                    'synth_vix': synth_vix,
                })
                if len(state['history']) > MAX_HISTORY:
                    state['history'] = state['history'][-MAX_HISTORY:]
                _resample_cache['len'] = -1  # 失效缓存

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

            if time.time() - last_persist > 60:
                persist(SYMBOL)
                last_persist = time.time()

        except Exception as e:
            log_err(f"主循环异常: {e}")
            if ib is not None and not ib.isConnected():
                with lock:
                    state['connected'] = False

        time.sleep(3)

# FIX 16: 优雅关闭
def _graceful_shutdown(*_):
    log_info("关闭中...")
    ib = _ib_holder.get('ib')
    if ib is not None and ib.isConnected():
        try:
            ib.disconnect()
        except Exception:
            pass
    try:
        persist(SYMBOL)
    except Exception:
        pass

atexit.register(_graceful_shutdown)
for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, lambda *a: (_graceful_shutdown(), os._exit(0)))
    except Exception:
        pass

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
            # FIX 13: 独立删除按钮，不再靠 data_previous 推断
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
            row_selectable='multi',   # FIX 13
            selected_rows=[],
            style_cell={'backgroundColor': '#0e1117', 'color': 'white',
                        'fontFamily': 'monospace', 'fontSize': '12px', 'padding': '6px'},
            style_header={'backgroundColor': '#1a1f2e', 'fontWeight': 'bold'},
        ),

        html.H3("历史标注总览", style={'marginTop': '25px'}),
        html.Div(id='segments-summary', style={'color': '#aaa', 'fontSize': '13px'}),
    ]),

    dcc.Store(id='selected-range'),
    dcc.Store(id='refresh-trigger', data=0),   # FIX 13: 表格刷新信号
    dcc.Interval(id='interval',      interval=3000,  n_intervals=0),
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
        # FIX 5: 深拷贝 dict 列表
        history = [dict(h) for h in state['history']]
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

    vix_txt = f"{s['synth_vix']:.1f}" if s['synth_vix'] else "—"
    exp_txt = s.get('expiry') or "—"
    updated_color = '#ff4444' if stale_warning else '#888'

    stats_children = [
        html.Span(f"Spot: {s['spot']:.2f}  |  ", style={'color': '#00d4ff'}),
        html.Span(f"Total GEX: ${s['total_gex']/1e6:.1f}M  |  ",
                  style={'color': '#00ff88' if s['total_gex'] > 0 else '#ff4444'}),
        html.Span(f"Flip: {s['gamma_flip']:.0f}  |  ", style={'color': '#ffaa00'}),
        html.Span(f"SynthVIX: {vix_txt}  |  ", style={'color': '#ff66cc'}),
        html.Span(f"Exp: {exp_txt}  |  ", style={'color': '#aaaaaa'}),
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
                         subplot_titles=('Total GEX ($M)', 'Spot vs Flip', 'Synthetic VIX'),
                         vertical_spacing=0.08,
                         row_heights=[0.4, 0.35, 0.25])

    # FIX 11: 用大写 S 的频率别名
    grids = [('30S', '30s', '#00d4ff'),
             ('1min', '1m', '#00ff88'),
             ('3min', '3m', '#ffaa00'),
             ('5min', '5m', '#ff66cc')]
    for rule, lbl, color in grids:
        r = resample_history(history, rule)
        if r.empty:
            continue
        fig2.add_trace(go.Scatter(x=r.index, y=r['total_gex'] / 1e6,
                                  mode='lines', name=f'GEX {lbl}',
                                  line=dict(color=color, width=1.5)),
                       row=1, col=1)

    r1 = resample_history(history, '1min')
    if not r1.empty:
        fig2.add_trace(go.Scatter(x=r1.index, y=r1['spot'], mode='lines',
                                  name='Spot', line=dict(color='white', width=2)),
                       row=2, col=1)
        fig2.add_trace(go.Scatter(x=r1.index, y=r1['flip'], mode='lines',
                                  name='Flip',
                                  line=dict(color='#ffaa00', width=2, dash='dot')),
                       row=2, col=1)
        if 'synth_vix' in r1.columns:
            fig2.add_trace(go.Scatter(x=r1.index, y=r1['synth_vix'], mode='lines',
                                      name='SynthVIX',
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
# FIX 13: 改为仅渲染，不再做"从 data_previous 推断删除"的危险逻辑
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
# FIX 13: 统一出口写 refresh-trigger，让 render_day 重绘
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
def _should_start_worker():
    return (not os.environ.get('WERKZEUG_RUN_MAIN')
            or os.environ.get('WERKZEUG_RUN_MAIN') == 'true')

if __name__ == '__main__':
    if _should_start_worker():
        threading.Thread(target=ib_worker, daemon=True).start()
    app.run(debug=False, host='127.0.0.1', port=8050)