"""
GEX Macro Dashboard — 独立运行的宏观数据面板

用法:
  cd ~/Downloads/gex
  .venv/bin/python -m gex_monitor.macro_app

三大类监测:
  1. 真实利率 (Real Yield): 10Y TIPS, 5Y5Y Breakeven
  2. 美元强度 (Dollar Strength): DXY, 广义美元
  3. 融资条件 (Funding Stress): VIX, MOVE, SOFR-OIS, HY OAS
"""
import logging
import os

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Flask, jsonify

from .macro import (
    fetch_macro_snapshot, interpret_macro,
    TIPS_10Y_ALERT, TIPS_10Y_DANGER,
    VIX_CALM, VIX_ELEVATED,
    MOVE_CALM, MOVE_ELEVATED,
    SOFR_OIS_STRESS_BPS,
    HY_OAS_ELEVATED, HY_OAS_STRESS,
)
from .ui.layout import (
    DARK_BG, PANEL_BG, TEXT_COLOR,
    MACRO_GREEN, MACRO_ORANGE, MACRO_RED, MACRO_GRAY,
    _macro_panel,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ── Flask + Dash ────────────────────────────────────────────

server = Flask(__name__)


@server.route('/api/macro')
def api_macro():
    """REST endpoint: 返回宏观快照 JSON"""
    try:
        snap = fetch_macro_snapshot()
        return jsonify({
            'regime': snap.regime_label,
            'urgency_adjustment': snap.urgency_adjustment,
            'vix': snap.vix, 'move': snap.move,
            'tips_10y': snap.tips_10y,
            'breakeven_5y5y': snap.breakeven_5y5y,
            'dxy': snap.dxy, 'broad_dollar': snap.broad_dollar,
            'sofr_ois_spread_bps': snap.sofr_ois_spread_bps,
            'hy_oas_bps': snap.hy_oas_bps,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@server.route('/api/interpret')
def api_interpret():
    """REST endpoint: AI 宏观解读"""
    try:
        snap = fetch_macro_snapshot()
        result = interpret_macro(snap)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'result': f'Error: {e}'}), 500


app = dash.Dash(
    __name__,
    server=server,
)

app.layout = html.Div(style={
    'backgroundColor': DARK_BG, 'color': TEXT_COLOR,
    'fontFamily': 'monospace', 'padding': '20px', 'minHeight': '100vh',
}, children=[
    html.H1('GEX Macro Dashboard', style={'textAlign': 'center', 'marginBottom': '24px'}),
    _macro_panel(),
    dcc.Interval(id='macro-interval', interval=60_000, n_intervals=0),
])


# ── Helpers ─────────────────────────────────────────────────

def _val(v, fmt, suffix='', thresholds=None):
    """格式化数值，带条件变色"""
    if v is None:
        return html.Span('N/A', style={'color': MACRO_GRAY})
    text = f'{v:{fmt}}{suffix}'
    color = MACRO_GREEN
    if thresholds:
        lo, hi = thresholds
        if v > hi:
            color = MACRO_RED
        elif v > lo:
            color = MACRO_ORANGE
    return html.Span(text, style={'color': color, 'fontWeight': 'bold'})


def _row(label, value_span):
    """指标行: 左 label 右 value"""
    return html.Div(style={
        'display': 'flex', 'justifyContent': 'space-between',
        'padding': '6px 0', 'borderBottom': '1px solid #1a1f2e',
    }, children=[
        html.Span(label, style={'color': '#888', 'fontSize': '13px'}),
        value_span,
    ])


def _card(title, rows, footer):
    """单个类别卡片"""
    return html.Div(style={
        'flex': '1', 'minWidth': '260px', 'maxWidth': '360px',
        'backgroundColor': '#0e1117', 'borderRadius': '8px',
        'padding': '16px', 'border': '1px solid #2a2f3e',
    }, children=[
        html.Div(title, style={
            'fontSize': '14px', 'fontWeight': 'bold', 'color': '#aaa',
            'marginBottom': '10px', 'borderBottom': '1px solid #333',
            'paddingBottom': '8px',
        }),
        *rows,
        html.Div(style={'marginTop': '10px', 'textAlign': 'right'}, children=[footer]),
    ])


# ── Callback ────────────────────────────────────────────────

@app.callback(
    [Output('macro-cards', 'children'),
     Output('macro-summary', 'children')],
    Input('macro-interval', 'n_intervals'),
)
def update_macro(_n):
    try:
        s = fetch_macro_snapshot()
    except Exception as e:
        err = html.Div(f'Error fetching macro data: {e}', style={'color': MACRO_ORANGE})
        return [err], ''

    # SOFR-OIS 特殊处理
    if s.sofr_ois_spread_bps is not None:
        sofr_color = MACRO_RED if s.sofr_ois_spread_bps > SOFR_OIS_STRESS_BPS else MACRO_GREEN
        sofr_span = html.Span(f'{s.sofr_ois_spread_bps:+.1f}bps',
                              style={'color': sofr_color, 'fontWeight': 'bold'})
    else:
        sofr_span = html.Span('N/A', style={'color': MACRO_GRAY})

    cards = [
        _card('📊 Real Yield (真实利率)', [
            _row('10Y TIPS', _val(s.tips_10y, '.2f', '%', (TIPS_10Y_ALERT, TIPS_10Y_DANGER))),
            _row('5Y5Y BkEven', _val(s.breakeven_5y5y, '.2f', '%')),
        ], html.Span(f'score: {s.real_yield_score:+.1f}',
                     style={'fontSize': '12px', 'color': '#666'})),

        _card('💵 Dollar Strength (美元强度)', [
            _row('DXY', _val(s.dxy, '.1f')),
            _row('Broad USD', _val(s.broad_dollar, '.1f')),
        ], html.Span('display only', style={
            'fontSize': '11px', 'color': '#555', 'fontStyle': 'italic'})),

        _card('🏦 Funding Stress (融资条件)', [
            _row('VIX', _val(s.vix, '.1f', thresholds=(VIX_CALM, VIX_ELEVATED))),
            _row('MOVE', _val(s.move, '.1f', thresholds=(MOVE_CALM, MOVE_ELEVATED))),
            _row('SOFR-OIS', sofr_span),
            _row('HY OAS', _val(s.hy_oas_bps, '.0f', 'bps', (HY_OAS_ELEVATED, HY_OAS_STRESS))),
        ], html.Span(f'score: {s.funding_score:+.1f}',
                     style={'fontSize': '12px', 'color': '#666'})),
    ]

    # Summary bar
    adj = s.urgency_adjustment
    adj_color = MACRO_GREEN if adj <= 0 else (MACRO_ORANGE if adj < 0.3 else MACRO_RED)
    regime_colors = {
        'stressed': MACRO_RED, 'calm': MACRO_GREEN,
        'normal': '#aaa', 'unknown': MACRO_GRAY,
    }
    regime_color = regime_colors.get(s.regime_label, '#aaa')

    summary = html.Div([
        html.Span('Regime: ', style={'color': '#888'}),
        html.Span(s.regime_label.upper(), style={
            'color': regime_color, 'fontWeight': 'bold', 'marginRight': '24px'}),
        html.Span('Urgency Adj: ', style={'color': '#888'}),
        html.Span(f'{adj:+.2f}', style={
            'color': adj_color, 'fontWeight': 'bold', 'fontSize': '18px'}),
    ])

    return cards, summary


@app.callback(
    Output('macro-ai-interpretation', 'children'),
    Input('macro-interval', 'n_intervals'),
)
def update_ai_interpretation(_n):
    try:
        snap = fetch_macro_snapshot()
        text = interpret_macro(snap)
        return text
    except Exception as e:
        return f'Error: {e}'


# ── Entry point ─────────────────────────────────────────────

def main():
    log.info("Starting GEX Macro Dashboard on http://0.0.0.0:8051")
    app.run(host='0.0.0.0', port=8051, debug=False, threaded=True)


if __name__ == '__main__':
    main()
