"""Dash 布局定义"""
from dash import dcc, html, dash_table

# 样式常量
DARK_BG = '#0e1117'
PANEL_BG = '#1a1f2e'
TEXT_COLOR = '#fafafa'

LABEL_OPTIONS = [
    {'label': '📈 Trend Up', 'value': 'trend_up'},
    {'label': '📉 Trend Down', 'value': 'trend_down'},
    {'label': '〰️ Chop', 'value': 'chop'},
    {'label': '🔀 Mixed', 'value': 'mixed'},
]

LABEL_COLORS = {
    'trend_up': 'rgba(0, 255, 136, 0.22)',
    'trend_down': 'rgba(255, 68, 68, 0.22)',
    'chop': 'rgba(255, 170, 0, 0.22)',
    'mixed': 'rgba(255, 102, 204, 0.22)',
}

LEVEL_COLORS = {
    'info': '#888',
    'warning': '#ffaa00',
    'error': '#ff6666',
}


def create_layout(symbols: list[str]) -> html.Div:
    """
    创建 Dash 布局

    Args:
        symbols: 已启用的标的列表

    Returns:
        Dash 布局组件
    """
    # 标的选择器（单标的时隐藏，但仍用 Dropdown 保持回调兼容）
    symbol_selector = []
    if len(symbols) > 1:
        symbol_selector = [
            html.Div(style={'textAlign': 'center', 'marginBottom': '20px'}, children=[
                html.Label("选择标的: ", style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=[{'label': s, 'value': s} for s in symbols],
                    value=symbols[0],
                    style={'width': '200px', 'display': 'inline-block', 'color': 'black'},
                    clearable=False,
                ),
            ])
        ]
    else:
        # 单标的：用隐藏的 Dropdown（不用 Store，因为回调依赖 value 属性）
        symbol_selector = [
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[{'label': s, 'value': s} for s in symbols],
                value=symbols[0] if symbols else None,
                style={'display': 'none'},
                clearable=False,
            )
        ]

    return html.Div(style={
        'backgroundColor': DARK_BG, 'color': TEXT_COLOR,
        'fontFamily': 'monospace', 'padding': '20px', 'minHeight': '100vh'
    }, children=[

        html.H1("0DTE Gamma Exposure Monitor", style={'textAlign': 'center'}),

        *symbol_selector,

        html.Div(id='stats', style={
            'textAlign': 'center', 'fontSize': '18px', 'margin': '20px'
        }),

        # 日志面板
        html.Details([
            html.Summary("🔔 日志（最近 30 条）",
                         style={'cursor': 'pointer', 'color': '#888'}),
            html.Div(id='error-log', style={
                'backgroundColor': PANEL_BG, 'padding': '10px',
                'fontSize': '12px',
                'maxHeight': '220px', 'overflowY': 'auto',
                'fontFamily': 'monospace',
            }),
        ], style={'maxWidth': '1100px', 'margin': '0 auto 20px'}),

        # GEX 柱状图
        dcc.Graph(id='gex-chart'),

        # 历史演化图
        html.H2("Intraday Evolution", style={'textAlign': 'center', 'marginTop': '30px'}),
        dcc.Graph(id='history-chart'),

        # 分段标注区域
        html.Hr(style={'marginTop': '40px', 'borderColor': '#333'}),
        html.H2("📝 Segment Labeling (5min view)", style={'textAlign': 'center'}),

        html.Div(style={
            'maxWidth': '1100px', 'margin': 'auto',
            'backgroundColor': PANEL_BG, 'padding': '20px', 'borderRadius': '8px'
        }, children=[

            html.Div(style={'marginBottom': '15px'}, children=[
                html.Label("选择日期:"),
                dcc.Dropdown(id='date-dropdown',
                             style={'color': 'black', 'width': '300px'}),
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
                          style={'width': '250px', 'backgroundColor': DARK_BG,
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
                    {'name': 'End', 'id': 'end_str'},
                    {'name': 'Label', 'id': 'label'},
                    {'name': 'Note', 'id': 'note'},
                ],
                row_selectable='multi',
                selected_rows=[],
                style_cell={'backgroundColor': DARK_BG, 'color': 'white',
                            'fontFamily': 'monospace', 'fontSize': '12px', 'padding': '6px'},
                style_header={'backgroundColor': PANEL_BG, 'fontWeight': 'bold'},
            ),

            html.H3("历史标注总览", style={'marginTop': '25px'}),
            html.Div(id='segments-summary', style={'color': '#aaa', 'fontSize': '13px'}),
        ]),

        # ==================== 历史回放区域 ====================
        html.Hr(style={'marginTop': '40px', 'borderColor': '#333'}),
        html.H2("🔄 GEX 历史回放", style={'textAlign': 'center'}),

        html.Div(style={
            'maxWidth': '1100px', 'margin': 'auto',
            'backgroundColor': PANEL_BG, 'padding': '20px', 'borderRadius': '8px'
        }, children=[

            html.Div(style={'marginBottom': '15px'}, children=[
                html.Label("选择日期: ", style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='replay-date-dropdown',
                    style={'color': 'black', 'width': '200px', 'display': 'inline-block'},
                ),
            ]),

            html.Div(id='replay-status',
                     style={'color': '#00d4ff', 'marginBottom': '10px', 'fontSize': '14px'}),

            # 时间滑块
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Label("拖动选择时间点:", style={'marginBottom': '10px', 'display': 'block'}),
                dcc.Slider(
                    id='replay-slider',
                    min=0, max=100, step=1, value=0,
                    marks={},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                ),
            ]),

            # 当前时间点信息
            html.Div(id='replay-time-info', style={
                'textAlign': 'center', 'fontSize': '18px', 'marginBottom': '20px',
                'color': '#ffaa00', 'fontWeight': 'bold'
            }),

            # 回放 K 线图（显示当前位置，支持框选标注）
            dcc.Graph(
                id='replay-kline-chart',
                config={'modeBarButtonsToAdd': ['select2d'], 'displaylogo': False},
                style={'height': '300px'}
            ),

            # 复盘时的快速标注
            html.Div(style={
                'display': 'flex', 'gap': '10px', 'alignItems': 'center',
                'flexWrap': 'wrap', 'marginTop': '10px', 'marginBottom': '15px',
                'padding': '12px', 'backgroundColor': DARK_BG, 'borderRadius': '6px'
            }, children=[
                html.Span("📝 快速标注:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='replay-label-radio',
                    options=LABEL_OPTIONS,
                    value='chop',
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                ),
                dcc.Input(
                    id='replay-label-note',
                    type='text',
                    placeholder='备注(可选)',
                    style={'width': '200px', 'backgroundColor': '#1a1a2e',
                           'color': 'white', 'border': '1px solid #333', 'padding': '6px'}
                ),
                html.Button(
                    '➕ 添加', id='replay-add-label-btn', n_clicks=0,
                    style={'padding': '6px 14px', 'backgroundColor': '#00ff88',
                           'color': 'black', 'border': 'none', 'borderRadius': '4px',
                           'cursor': 'pointer'}
                ),
            ]),
            html.Div(id='replay-label-status',
                     style={'color': '#00ff88', 'fontSize': '13px', 'minHeight': '18px'}),

            # 回放 GEX 柱状图
            html.H3("该时刻 GEX 分布", style={'marginTop': '20px', 'textAlign': 'center'}),
            dcc.Graph(id='replay-gex-chart', style={'height': '500px'}),

            # 聚合指标
            html.Div(id='replay-stats', style={
                'textAlign': 'center', 'fontSize': '16px', 'marginTop': '15px',
                'padding': '15px', 'backgroundColor': DARK_BG, 'borderRadius': '8px'
            }),
        ]),

        # Stores
        dcc.Store(id='selected-range'),
        dcc.Store(id='refresh-trigger', data=0),
        dcc.Store(id='replay-timestamps', data=[]),
        dcc.Store(id='replay-label-trigger', data=0),

        # Intervals
        dcc.Interval(id='interval', interval=4000, n_intervals=0),
        dcc.Interval(id='slow-interval', interval=30000, n_intervals=0),
    ])
