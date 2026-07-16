"""
日内板块轮动检测 — 从 DB 读 5min bar 计算 RS 排名

窗口:
  短期: 6 bars = 30min（捕捉快速切换）
  中期: 24 bars = 2h（确认趋势）
  参考: 78 bars = 1天

用法:
  from .rotation import compute_intraday_rotation
  result = compute_intraday_rotation(db_config)
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DatabaseConfig

log = logging.getLogger(__name__)

# 板块 ETF 定义 (同 rotation_detector.py)
SUB_SECTOR_ETFS = {
    "SMH":  "半导体",
    "IGV":  "软件",
    "SKYY": "云计算",
    "HACK": "网络安全",
    "BOTZ": "AI/机器人",
    "XLK":  "科技大盘",
    "XLE":  "能源",
    "XLF":  "金融",
    "XLV":  "医疗",
    "XLI":  "工业",
    "XLY":  "可选消费",
    "XLP":  "必需消费",
    "XLU":  "公用事业",
    "GDX":  "黄金矿业",
    "XBI":  "生物科技",
    "IYT":  "交通运输",
}

BENCHMARK = "SPY"
ALL_SYMBOLS = list(SUB_SECTOR_ETFS.keys()) + [BENCHMARK]

# 窗口 (5min bar 数)
WIN_30M = 6         # 30min
WIN_1H = 12         # 1h
WIN_2H = 24         # 2h
WIN_HALF = 39       # 半天 (~3.25h)
LONG_WINDOW = 78    # 1 day (用于数据读取量)

# 排名跳升阈值
JUMP_THRESHOLD = 3


@dataclass
class RotationResult:
    """轮动分析结果"""
    rows: list = field(default_factory=list)       # 排名表行
    alerts: list = field(default_factory=list)      # 轮动提醒
    tech_leader: str = ""                           # 科技内部领先
    last_bar_time: str = ""                         # 最新 bar 时间
    error: str = ""                                 # 错误信息


def _read_5min_bars(db_config: DatabaseConfig, n_bars: int = 100) -> dict[str, pd.Series]:
    """从 market_data_bars 读取最近 n_bars 根 5min bar 的 close 价格"""
    try:
        import psycopg2
    except ImportError:
        return {}

    try:
        conn = psycopg2.connect(
            host=db_config.host, port=db_config.port,
            dbname=db_config.dbname, user=db_config.user,
            password=db_config.password,
        )
    except Exception as e:
        log.warning(f"[Rotation] DB connect failed: {e}")
        return {}

    try:
        placeholders = ','.join(['%s'] * len(ALL_SYMBOLS))
        query = f"""
            SELECT symbol, datetime, close
            FROM market_data_bars
            WHERE bar_size = '5 mins'
              AND symbol IN ({placeholders})
            ORDER BY datetime DESC
            LIMIT %s
        """
        df = pd.read_sql_query(
            query, conn,
            params=ALL_SYMBOLS + [n_bars * len(ALL_SYMBOLS)],
        )
    except Exception as e:
        log.warning(f"[Rotation] DB query failed: {e}")
        return {}
    finally:
        conn.close()

    if df.empty:
        return {}

    df['datetime'] = pd.to_datetime(df['datetime'])
    result = {}
    for sym in ALL_SYMBOLS:
        sym_df = df[df['symbol'] == sym].sort_values('datetime')
        if not sym_df.empty:
            series = sym_df.set_index('datetime')['close'].astype(float)
            result[sym] = series

    return result


def compute_intraday_rotation(db_config: DatabaseConfig) -> RotationResult:
    """
    计算日内板块轮动

    Returns:
        RotationResult with ranking table, alerts, and tech leader
    """
    result = RotationResult()

    # 1. 读数据
    price_dict = _read_5min_bars(db_config, n_bars=LONG_WINDOW + 10)

    if BENCHMARK not in price_dict:
        result.error = "无 SPY 数据"
        return result

    bench = price_dict.pop(BENCHMARK)

    # 对齐时间索引
    common_idx = bench.index
    for sym, prices in price_dict.items():
        common_idx = common_idx.intersection(prices.index)
    common_idx = common_idx.sort_values()

    if len(common_idx) < WIN_30M + 1:
        result.error = f"数据不足 ({len(common_idx)} bars)"
        return result

    result.last_bar_time = common_idx[-1].strftime('%Y-%m-%d %H:%M')

    # 2. 构建 RS 矩阵
    rs_data = {}
    for sym, prices in price_dict.items():
        aligned = prices.reindex(common_idx)
        bench_aligned = bench.reindex(common_idx)
        rs_data[sym] = aligned / bench_aligned
    rs_matrix = pd.DataFrame(rs_data)

    # 3. 四个窗口的 RS 变化率 + 排名
    n = len(rs_matrix)
    windows = {
        '30m': min(WIN_30M, n - 1),
        '1h':  min(WIN_1H, n - 1),
        '2h':  min(WIN_2H, n - 1),
        'half': min(WIN_HALF, n - 1),
    }

    rs_chg = {}
    ranks = {}
    for label, win in windows.items():
        chg = rs_matrix.pct_change(win)
        rs_chg[label] = chg
        ranks[label] = chg.rank(axis=1, ascending=False, method='min')

    # 4. 汇总
    rows = []
    for sym in rs_matrix.columns:
        name = SUB_SECTOR_ETFS.get(sym, sym)

        row = {'sym': sym, 'name': name}

        # 每个窗口的 RS 变化率和排名
        for label, win in windows.items():
            chg_val = (rs_matrix[sym].iloc[-1] / rs_matrix[sym].iloc[-win] - 1) * 100
            rank_val = int(ranks[label].iloc[-1][sym]) if not pd.isna(ranks[label].iloc[-1][sym]) else 99
            row[f'rs_{label}'] = round(chg_val, 2)
            row[f'rank_{label}'] = rank_val

        # Delta: 半天排名 vs 30min 排名 (大时间框架 vs 小时间框架)
        rank_delta = row['rank_half'] - row['rank_30m']
        row['delta'] = rank_delta

        # RS 速度/加速度 (基于 2h 窗口的 EMA)
        rs = rs_matrix[sym]
        rs_smooth = rs.ewm(span=5, adjust=False).mean()
        rs_d1 = rs_smooth.diff().ewm(span=5, adjust=False).mean()
        rs_d2 = rs_d1.diff().ewm(span=5, adjust=False).mean()
        vel = rs_d1.iloc[-1]
        acc = rs_d2.iloc[-1]

        if vel > 0 and acc > 0:
            row['signal'] = "加速跑赢"
            row['signal_color'] = "#00ff88"
        elif vel > 0:
            row['signal'] = "跑赢减速"
            row['signal_color'] = "#ffaa00"
        elif acc > 0:
            row['signal'] = "可能拐头"
            row['signal_color'] = "#4488ff"
        else:
            row['signal'] = "加速跑输"
            row['signal_color'] = "#ff4444"

        rows.append(row)

    # 按半天排名排序（更稳定的视角）
    rows.sort(key=lambda x: x['rank_half'])
    result.rows = rows

    # 5. 轮动提醒: 30min 排名 vs 半天排名跳升 ≥3
    for row in rows:
        d = row['delta']
        if d >= JUMP_THRESHOLD:
            result.alerts.append(
                f"⬆️ {row['sym']} ({row['name']}) 30m排名冲到 {row['rank_30m']} (半天 {row['rank_half']}, +{d})"
            )
        elif d <= -JUMP_THRESHOLD:
            result.alerts.append(
                f"⬇️ {row['sym']} ({row['name']}) 30m排名跌到 {row['rank_30m']} (半天 {row['rank_half']}, {d})"
            )

    # 6. 科技内部领先 (用半天排名)
    tech_subs = ["SMH", "IGV", "SKYY", "HACK", "BOTZ"]
    tech_rows = [r for r in rows if r['sym'] in tech_subs]
    if tech_rows:
        leader = min(tech_rows, key=lambda x: x['rank_half'])
        result.tech_leader = f"{leader['sym']} ({leader['name']})"

    return result
