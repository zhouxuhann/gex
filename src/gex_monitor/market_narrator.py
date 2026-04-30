"""
Market Narrator — 把 GEX 状态翻译成人话

读取 GEX snapshots，生成一句话盘面解读。
可独立运行，也可集成到 Dash UI。

用法：
  python -m gex_monitor.market_narrator          # 实时刷新
  python -m gex_monitor.market_narrator --once    # 只看一次
"""

import sys
import os
import time
import argparse
from datetime import datetime

import psycopg2
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')

DB = {
    'host': '127.0.0.1', 'port': 5433,
    'dbname': 'ibkr_market_data', 'user': 'ibkr_user',
    'password': os.environ.get('POSTGRES_PASSWORD', 'ibkr_secure_password_2026'),
}

# ── 颜色 ──
G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; C = '\033[96m'
B = '\033[1m'; D = '\033[2m'; X = '\033[0m'


def get_latest():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute('''SELECT spot, total_gex, gamma_flip, call_wall, put_wall,
        max_pain, atm_iv_pct, positive_gamma, regime_code,
        rr_25, rr_25_zscore, skew_signal, datetime
        FROM gex_snapshots ORDER BY datetime DESC LIMIT 1''')
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'spot': float(row[0]), 'gex': float(row[1]), 'flip': float(row[2]),
        'cw': float(row[3]) if row[3] else None,
        'pw': float(row[4]) if row[4] else None,
        'mp': float(row[5]) if row[5] else None,
        'iv': float(row[6]) if row[6] else None,
        'pos_gamma': bool(row[7]), 'regime': row[8],
        'rr25': float(row[9]) if row[9] else 0,
        'rr25_z': float(row[10]) if row[10] else 0,
        'skew_sig': row[11], 'ts': row[12],
    }


def narrate(d) -> str:
    """生成一段人话解读。"""
    lines = []
    spot = d['spot']

    # ── 1. Gamma 环境 ──
    if d['pos_gamma']:
        lines.append(f"{G}正Gamma{X} — Dealer 在稳定市场，波动被压制")
    else:
        lines.append(f"{R}负Gamma{X} — Dealer 追涨杀跌，波动放大")

    # ── 2. 价格位置 ──
    flip = d['flip']
    dist_flip = spot - flip
    if abs(dist_flip) < 0.5:
        lines.append(f"  价格 ${spot:.2f} 在 Flip ${flip:.1f} {Y}附近（翻转边缘）{X}")
    elif dist_flip > 0:
        lines.append(f"  价格 ${spot:.2f} 在 Flip ${flip:.1f} {G}上方 ${dist_flip:.1f}{X}（稳定区）")
    else:
        lines.append(f"  价格 ${spot:.2f} 在 Flip ${flip:.1f} {R}下方 ${abs(dist_flip):.1f}{X}（危险区）")

    # ── 3. Wall 夹击 ──
    cw, pw = d['cw'], d['pw']
    if cw and pw:
        range_width = cw - pw
        to_cw = cw - spot
        to_pw = spot - pw
        lines.append(f"  区间 {B}${pw:.0f}-${cw:.0f}{X}（宽 ${range_width:.0f}）"
                     f"  ↑阻力 ${to_cw:.1f}  ↓支撑 ${to_pw:.1f}")

        # 靠近哪个 wall
        if to_cw < 0.5:
            lines.append(f"  {Y}⚠ 紧贴 Call Wall！{X}突破则 gamma squeeze 加速上涨")
        elif to_pw < 0.5:
            lines.append(f"  {Y}⚠ 紧贴 Put Wall！{X}跌破则支撑消失加速下跌")
        elif to_cw < to_pw * 0.5:
            lines.append(f"  偏向区间上沿，上方压力大")
        elif to_pw < to_cw * 0.5:
            lines.append(f"  偏向区间下沿，下方支撑近")

    # ── 4. Max Pain 引力 ──
    mp = d['mp']
    if mp:
        mp_dist = spot - mp
        if abs(mp_dist) > 5:
            lines.append(f"  Max Pain ${mp:.0f}（偏离 ${mp_dist:+.1f}，到期日有回归压力）")

    # ── 5. IV ──
    iv = d['iv']
    if iv:
        if iv > 30:
            lines.append(f"  IV {iv:.1f}% {R}偏高{X} — 期权贵，波动预期大")
        elif iv < 15:
            lines.append(f"  IV {iv:.1f}% {G}偏低{X} — 市场平静")
        else:
            lines.append(f"  IV {iv:.1f}%")

    # ── 6. Skew / RR25 ──
    rr25_z = d['rr25_z']
    if rr25_z > 2.0:
        lines.append(f"  Skew z={rr25_z:.1f} {R}恐慌{X} — put 保护需求极高，市场怕跌")
    elif rr25_z > 1.0:
        lines.append(f"  Skew z={rr25_z:.1f} {Y}偏高{X} — put 需求升温")
    elif rr25_z < -1.5:
        lines.append(f"  Skew z={rr25_z:.1f} {G}偏低{X} — 没人买保护，风险偏好高")
    elif rr25_z < -0.5:
        lines.append(f"  Skew z={rr25_z:.1f} — put 需求平淡")

    # ── 7. Skew 信号 ──
    sig = d['skew_sig']
    if sig == 'PIN_BREAK_RISK':
        lines.append(f"  {Y}⚡ PIN突破风险{X} — 区间钉住要松了，准备突破")
    elif sig == 'HEDGE_NOW':
        lines.append(f"  {R}🔴 对冲信号{X} — skew+GEX 双重警告，考虑买 put")
    elif sig == 'MONITOR':
        lines.append(f"  {C}👀 监控中{X}")

    # ── 8. 交易建议 ──
    lines.append('')
    if d['pos_gamma'] and cw and pw:
        if to_cw < 1.0:
            lines.append(f"  {B}建议{X}：靠近 Call Wall，{Y}不追多{X}。突破 ${cw:.0f} 后再跟")
        elif to_pw < 1.0:
            lines.append(f"  {B}建议{X}：靠近 Put Wall，{G}可以试多{X}（有支撑）。破 ${pw:.0f} 则止损")
        else:
            lines.append(f"  {B}建议{X}：正 Gamma 区间内，{C}高抛低吸{X}。区间 ${pw:.0f}-${cw:.0f}")
    elif not d['pos_gamma']:
        if dist_flip > 0:
            lines.append(f"  {B}建议{X}：负 Gamma + spot > flip，{G}顺势做多{X}，趋势可能延续")
        else:
            lines.append(f"  {B}建议{X}：负 Gamma + spot < flip，{R}顺势做空{X}或持有 put")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Market Narrator')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=30, help='刷新间隔秒')
    args = parser.parse_args()

    while True:
        d = get_latest()
        if d:
            ts = d['ts'].strftime('%H:%M:%S') if d['ts'] else '?'
            os.system('clear')
            print(f"{C}{'═'*60}{X}")
            print(f"{C}  Market Narrator  |  QQQ  |  {ts} ET{X}")
            print(f"{C}{'═'*60}{X}")
            print()
            print(narrate(d))
            print()
            print(f"{D}{'─'*60}{X}")
            print(f"{D}  {args.interval}秒后刷新  Ctrl+C 退出{X}")
        else:
            print("无数据")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
