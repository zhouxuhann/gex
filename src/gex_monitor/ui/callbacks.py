"""Dash 回调逻辑"""
import dash
from dash import html, ctx
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..state import StateRegistry
from ..storage import StorageManager, SegmentStorage
from ..time_utils import et_now
from .layout import LABEL_COLORS, LEVEL_COLORS

STALE_SECONDS = 15


def register_callbacks(
    app: dash.Dash,
    registry: StateRegistry,
    storage: StorageManager,
    segments: SegmentStorage,
):
    """注册所有回调"""

    # ==================== 实时图回调 ====================
    @app.callback(
        [Output('stats', 'children'),
         Output('gex-chart', 'figure'),
         Output('history-chart', 'figure'),
         Output('error-log', 'children')],
        [Input('interval', 'n_intervals'),
         Input('symbol-dropdown', 'value')])
    def update_live(_, symbol):
        if not symbol:
            return "请选择标的", go.Figure(), go.Figure(), []

        state = registry.get(symbol)
        if state is None:
            return f"标的 {symbol} 未注册", go.Figure(), go.Figure(), []

        s = state.get_snapshot()
        df = state.get_df()
        history, history_version = state.get_history_for_resample()
        logs = state.get_logs()

        # 日志渲染
        log_children = [
            html.Div(f"[{ts.strftime('%H:%M:%S')}] {msg}",
                     style={'color': LEVEL_COLORS.get(level, '#aaa')})
            for level, ts, msg in reversed(logs)
        ] or [html.Div("(无)", style={'color': '#888'})]

        # 数据陈旧检测
        stale_warning = None
        if s.get('last_update_ts') is not None:
            age = (et_now() - s['last_update_ts']).total_seconds()
            if age > STALE_SECONDS:
                stale_warning = f"⚠️ 数据 {age:.0f}s 未更新"

        if not s.get('market_open'):
            return f"😴 {s['updated']}", go.Figure(), go.Figure(), log_children

        if df.empty:
            return "等待数据...", go.Figure(), go.Figure(), log_children

        # 统计面板
        iv_txt = f"{s['atm_iv_pct']:.1f}" if s['atm_iv_pct'] else "—"
        exp_txt = s.get('expiry') or "—"
        if not s.get('is_true_0dte'):
            exp_txt += " ⚠️非0DTE"
        updated_color = '#ff4444' if stale_warning else '#888'

        # Gamma 环境指示
        positive_gamma = s.get('positive_gamma', False)
        gamma_env_txt = "正Gamma" if positive_gamma else "负Gamma"
        gamma_env_color = '#00ff88' if positive_gamma else '#ff4444'

        # Call Wall / Put Wall / Max Pain
        call_wall = s.get('call_wall')
        put_wall = s.get('put_wall')
        max_pain = s.get('max_pain')
        call_wall_txt = f"{call_wall:.0f}" if call_wall else "—"
        put_wall_txt = f"{put_wall:.0f}" if put_wall else "—"
        max_pain_txt = f"{max_pain:.0f}" if max_pain else "—"

        stats_children = [
            html.Span(f"{symbol}  |  ", style={'color': '#ffaa00', 'fontWeight': 'bold'}),
            html.Span(f"Spot: {s['spot']:.2f}  |  ", style={'color': '#00d4ff'}),
            html.Span(f"Total GEX: ${s['total_gex']/1e6:.1f}M  |  ",
                      style={'color': '#00ff88' if s['total_gex'] > 0 else '#ff4444'}),
            html.Span(f"[{gamma_env_txt}]  |  ",
                      style={'color': gamma_env_color, 'fontWeight': 'bold'}),
            html.Span(f"Flip: {s['gamma_flip']:.0f}  |  ", style={'color': '#ffaa00'}),
            html.Span(f"Call Wall: {call_wall_txt}  |  ", style={'color': '#00d4ff'}),
            html.Span(f"Put Wall: {put_wall_txt}  |  ", style={'color': '#ff66cc'}),
            html.Span(f"Max Pain: {max_pain_txt}  |  ", style={'color': '#aaaaaa'}),
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

        # Regime 分类显示
        regime_code = s.get('regime_code')
        regime_tags = s.get('regime_tags') or {}
        if regime_code:
            # 根据 tags 选择颜色和图标
            gamma_sign = regime_tags.get('gamma_sign', 'neutral')
            position = regime_tags.get('position', 'mid_range')
            concentration = regime_tags.get('concentration', 'diffuse')

            # 图标
            gamma_icon = {'long_gamma': '🟢', 'short_gamma': '🔴', 'neutral': '⚪'}.get(gamma_sign, '⚪')
            pos_icon = {'above_flip': '📈', 'below_flip': '📉', 'at_flip': '⚡'}.get(position, '•')
            conc_icon = '🧲' if concentration == 'concentrated' else '🌫️'

            regime_children = [
                html.Span("Regime: ", style={'color': '#888'}),
                html.Span(f"{gamma_icon} ", style={'fontSize': '14px'}),
                html.Span(f"{gamma_sign.replace('_', ' ')}  ",
                          style={'color': '#00ff88' if gamma_sign == 'long_gamma' else '#ff4444' if gamma_sign == 'short_gamma' else '#aaa'}),
                html.Span(f"{pos_icon} ", style={'fontSize': '14px'}),
                html.Span(f"{position.replace('_', ' ')}  ", style={'color': '#ffaa00'}),
                html.Span(f"{conc_icon} ", style={'fontSize': '14px'}),
                html.Span(f"{concentration}", style={'color': '#00d4ff'}),
            ]
            regime_div = html.Div(regime_children, style={'marginTop': '4px', 'fontSize': '13px'})
        else:
            regime_div = html.Div()

        stats = html.Div([html.Div(stats_children), regime_div])

        # GEX 柱状图
        by_strike = df.groupby('strike')['gex'].sum().sort_index() / 1e6
        calls = df[df.right == 'C'].groupby('strike')['gex'].sum() / 1e6
        puts = df[df.right == 'P'].groupby('strike')['gex'].sum() / 1e6

        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=('Net GEX by Strike', 'Calls vs Puts'),
                             vertical_spacing=0.12)
        colors = ['#00ff88' if v > 0 else '#ff4444' for v in by_strike.values]
        # 执行价标签
        strike_labels = [f'{int(s)}' for s in by_strike.index]
        fig1.add_trace(go.Bar(x=by_strike.index, y=by_strike.values,
                              marker_color=colors, name='Net',
                              text=strike_labels, textposition='outside',
                              textfont=dict(size=9, color='#888')), row=1, col=1)
        fig1.add_trace(go.Bar(x=calls.index, y=calls.values,
                              marker_color='#00d4ff', name='Calls'), row=2, col=1)
        fig1.add_trace(go.Bar(x=puts.index, y=puts.values,
                              marker_color='#ff66cc', name='Puts'), row=2, col=1)
        for r in [1, 2]:
            fig1.add_vline(x=s['spot'], line=dict(color='white', dash='dash'), row=r, col=1)
            fig1.add_vline(x=s['gamma_flip'], line=dict(color='#ffaa00', dash='dot'), row=r, col=1)
            # Call Wall (阻力) - 青色
            if call_wall:
                fig1.add_vline(x=call_wall, line=dict(color='#00d4ff', width=2), row=r, col=1)
            # Put Wall (支撑) - 粉色
            if put_wall:
                fig1.add_vline(x=put_wall, line=dict(color='#ff66cc', width=2), row=r, col=1)
            # Max Pain - 灰色虚线
            if max_pain:
                fig1.add_vline(x=max_pain, line=dict(color='#888888', width=2, dash='dot'), row=r, col=1)
        # x 轴自动缩放，确保所有数据可见（不强制 range，避免丢数据）
        fig1.update_layout(template='plotly_dark', height=700, barmode='relative',
                           paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                           margin=dict(t=60))  # 顶部留空给标签
        fig1.update_yaxes(title_text='GEX ($M per 1%)')
        fig1.update_xaxes(title_text='Strike', row=1, col=1)
        fig1.update_xaxes(title_text='Strike', row=2, col=1)

        # 历史演化图
        fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             subplot_titles=('Total GEX ($M)', 'Spot / Flip / Walls', 'ATM IV (%)'),
                             vertical_spacing=0.08,
                             row_heights=[0.4, 0.35, 0.25])

        grids = [('30s', '30s', '#00d4ff'),
                 ('1min', '1m', '#00ff88'),
                 ('3min', '3m', '#ffaa00'),
                 ('5min', '5m', '#ff66cc')]
        for rule, lbl, color in grids:
            r = state.resample_history(rule)
            if r.empty:
                continue
            fig2.add_trace(go.Scatter(x=r.index, y=r['total_gex'] / 1e6,
                                      mode='lines', name=f'GEX {lbl}',
                                      line=dict(color=color, width=1.5)),
                           row=1, col=1)

        r1 = state.resample_history('1min')
        if not r1.empty:
            fig2.add_trace(go.Scatter(x=r1.index, y=r1['spot'], mode='lines',
                                      name='Spot', line=dict(color='white', width=2)),
                           row=2, col=1)
            fig2.add_trace(go.Scatter(x=r1.index, y=r1['flip'], mode='lines',
                                      name='Flip',
                                      line=dict(color='#ffaa00', width=2, dash='dot')),
                           row=2, col=1)
            # Call Wall / Put Wall 历史线
            if 'call_wall' in r1.columns:
                fig2.add_trace(go.Scatter(x=r1.index, y=r1['call_wall'], mode='lines',
                                          name='Call Wall',
                                          line=dict(color='#00d4ff', width=1.5, dash='dash')),
                               row=2, col=1)
            if 'put_wall' in r1.columns:
                fig2.add_trace(go.Scatter(x=r1.index, y=r1['put_wall'], mode='lines',
                                          name='Put Wall',
                                          line=dict(color='#ff66cc', width=1.5, dash='dash')),
                               row=2, col=1)
            if 'max_pain' in r1.columns:
                fig2.add_trace(go.Scatter(x=r1.index, y=r1['max_pain'], mode='lines',
                                          name='Max Pain',
                                          line=dict(color='#888888', width=1.5, dash='dot')),
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
        [Output('date-dropdown', 'options'),
         Output('date-dropdown', 'value')],
        [Input('slow-interval', 'n_intervals'),
         Input('symbol-dropdown', 'value')],
        State('date-dropdown', 'value'))
    def refresh_dates(_, symbol, current):
        if not symbol:
            return [], None
        dates = storage.list_available_dates(symbol)
        opts = [{'label': d, 'value': d} for d in dates]
        val = current if current in dates else (dates[-1] if dates else None)
        return opts, val

    # ==================== 当日 K 线 + 已标分段 ====================
    @app.callback(
        [Output('day-chart', 'figure'),
         Output('segments-table', 'data'),
         Output('segments-summary', 'children')],
        [Input('date-dropdown', 'value'),
         Input('refresh-trigger', 'data'),
         Input('symbol-dropdown', 'value')])
    def render_day(date_str, _trigger, symbol):
        fig = go.Figure()
        table_data = []
        summary = ""

        if date_str and symbol:
            ohlc = storage.load_day_ohlc(symbol, date_str)
            bars = storage.resample_5min(ohlc)

            if bars is not None and not bars.empty:
                fig.add_trace(go.Candlestick(
                    x=bars['ts'], open=bars['open'], high=bars['high'],
                    low=bars['low'], close=bars['close'],
                    increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
                    name='5m'
                ))

                segs = segments.load_segments()
                if not segs.empty:
                    day_segs = segs[(segs['date'] == date_str) & (segs['symbol'] == symbol)]
                else:
                    day_segs = pd.DataFrame()

                for _, seg in day_segs.iterrows():
                    fig.add_vrect(
                        x0=str(seg['start_ts']), x1=str(seg['end_ts']),
                        fillcolor=LABEL_COLORS.get(seg['label'], 'rgba(128,128,128,0.2)'),
                        line_width=0,
                        annotation_text=seg['label'],
                        annotation_position='top left',
                        annotation=dict(font_size=10, font_color='white'),
                    )
                    table_data.append({
                        'id': seg['id'],
                        'start_str': pd.Timestamp(seg['start_ts']).strftime('%H:%M'),
                        'end_str': pd.Timestamp(seg['end_ts']).strftime('%H:%M'),
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

        all_segs = segments.load_segments()
        if not all_segs.empty:
            total = len(all_segs)
            days = all_segs['date'].nunique()
            dist = all_segs['label'].value_counts().to_dict()
            dist_str = "  ".join(f"{k}: {v}" for k, v in dist.items())
            summary = f"总分段数: {total}  |  覆盖天数: {days}  |  分布: {dist_str}"

        return fig, table_data, summary

    # ==================== 框选事件 ====================
    @app.callback(
        [Output('selected-range', 'data'),
         Output('selection-info', 'children')],
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
        [Output('save-status', 'children'),
         Output('refresh-trigger', 'data')],
        [Input('add-btn', 'n_clicks'),
         Input('delete-btn', 'n_clicks')],
        [State('date-dropdown', 'value'),
         State('selected-range', 'data'),
         State('label-radio', 'value'),
         State('label-note', 'value'),
         State('segments-table', 'data'),
         State('segments-table', 'selected_rows'),
         State('refresh-trigger', 'data'),
         State('symbol-dropdown', 'value')],
        prevent_initial_call=True)
    def modify_segments(_add, _del, date_str, sel_range, label, note,
                        table_data, selected_rows, trigger, symbol):
        trigger = (trigger or 0)
        triggered = ctx.triggered_id

        if triggered == 'add-btn':
            if not symbol:
                return "⚠️ 请先选标的", trigger
            if not date_str:
                return "⚠️ 请先选日期", trigger
            if not sel_range:
                return "⚠️ 请先在图上框选一段区间", trigger
            if not label:
                return "⚠️ 请选择 regime 标签", trigger
            segments.save_segment(date_str, sel_range['x0'], sel_range['x1'],
                                  symbol, label, note)
            return (f"✅ 已添加: {label} ({et_now().strftime('%H:%M:%S ET')})",
                    trigger + 1)

        if triggered == 'delete-btn':
            if not selected_rows or not table_data:
                return "⚠️ 请先在表格勾选要删除的行", trigger
            ids = [table_data[i]['id'] for i in selected_rows
                   if i < len(table_data) and 'id' in table_data[i]]
            if not ids:
                return "⚠️ 选中行没有 id", trigger
            segments.delete_segments_by_ids(ids)
            return (f"🗑 已删除 {len(ids)} 条 ({et_now().strftime('%H:%M:%S ET')})",
                    trigger + 1)

        return dash.no_update, dash.no_update

    # ==================== 回放：日期下拉刷新 ====================
    @app.callback(
        [Output('replay-date-dropdown', 'options'),
         Output('replay-date-dropdown', 'value')],
        [Input('slow-interval', 'n_intervals'),
         Input('symbol-dropdown', 'value')],
        State('replay-date-dropdown', 'value'))
    def refresh_replay_dates(_, symbol, current):
        if not symbol:
            return [], None
        # 使用优化后的方法直接列出有 strikes 数据的日期
        dates = storage.list_available_strikes_dates(symbol)
        opts = [{'label': d, 'value': d} for d in dates]
        val = current if current in dates else (dates[-1] if dates else None)
        return opts, val

    # ==================== 回放：加载时间戳 ====================
    @app.callback(
        [Output('replay-timestamps', 'data'),
         Output('replay-slider', 'max'),
         Output('replay-slider', 'marks'),
         Output('replay-slider', 'value'),
         Output('replay-status', 'children')],
        [Input('replay-date-dropdown', 'value'),
         Input('symbol-dropdown', 'value')])
    def load_replay_timestamps(date_str, symbol):
        if not date_str or not symbol:
            return [], 0, {}, 0, "请选择日期"

        timestamps = storage.get_replay_timestamps(symbol, date_str)
        if not timestamps:
            return [], 0, {}, 0, f"⚠️ {date_str} 没有 strike-level 数据"

        # 转为字符串存储
        ts_strings = [str(t) for t in timestamps]
        max_idx = len(timestamps) - 1

        # 创建 marks（只显示部分以免太密集）
        marks = {}
        step = max(1, len(timestamps) // 10)
        for i in range(0, len(timestamps), step):
            t = pd.Timestamp(timestamps[i])
            marks[i] = t.strftime('%H:%M')
        if max_idx not in marks:
            marks[max_idx] = pd.Timestamp(timestamps[-1]).strftime('%H:%M')

        return ts_strings, max_idx, marks, 0, f"📊 共 {len(timestamps)} 个时间点可回放"

    # ==================== 回放：渲染图表 ====================
    @app.callback(
        [Output('replay-time-info', 'children'),
         Output('replay-kline-chart', 'figure'),
         Output('replay-gex-chart', 'figure'),
         Output('replay-stats', 'children')],
        [Input('replay-slider', 'value'),
         Input('replay-timestamps', 'data'),
         Input('replay-date-dropdown', 'value'),
         Input('symbol-dropdown', 'value'),
         Input('replay-label-trigger', 'data')])  # 标注后刷新
    def render_replay(slider_idx, ts_strings, date_str, symbol, _label_trigger):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117'
        )

        if not ts_strings or not date_str or not symbol:
            return "请先选择日期", empty_fig, empty_fig, ""

        if slider_idx >= len(ts_strings):
            slider_idx = len(ts_strings) - 1

        target_ts = pd.Timestamp(ts_strings[slider_idx])
        time_info = f"⏱️ {target_ts.strftime('%Y-%m-%d %H:%M:%S ET')}"

        # 获取该时间点的 strike 数据
        strikes_df = storage.get_strikes_at_time(symbol, date_str, target_ts)

        # K 线图 + 当前位置
        ohlc = storage.load_day_ohlc(symbol, date_str)
        bars = storage.resample_5min(ohlc)

        fig_kline = go.Figure()
        if bars is not None and not bars.empty:
            fig_kline.add_trace(go.Candlestick(
                x=bars['ts'], open=bars['open'], high=bars['high'],
                low=bars['low'], close=bars['close'],
                increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
                name='5m'
            ))
            # 当前时间点的竖线（用 add_shape 避免 Timestamp 兼容问题）
            fig_kline.add_shape(
                type='line',
                x0=str(target_ts), x1=str(target_ts),
                y0=0, y1=1, yref='paper',
                line=dict(color='#00d4ff', width=2, dash='dash'),
            )
            fig_kline.add_annotation(
                x=str(target_ts), y=1.05, yref='paper',
                text=target_ts.strftime('%H:%M'),
                showarrow=False, font=dict(color='#00d4ff', size=12)
            )

            # 显示已有的标注区域
            all_segs = segments.load_segments()
            day_segs = all_segs[(all_segs['date'] == date_str) & (all_segs['symbol'] == symbol)]
            for _, seg in day_segs.iterrows():
                fig_kline.add_vrect(
                    x0=str(seg['start_ts']), x1=str(seg['end_ts']),
                    fillcolor=LABEL_COLORS.get(seg['label'], 'rgba(128,128,128,0.2)'),
                    line_width=0,
                    annotation_text=seg['label'],
                    annotation_position='top left',
                    annotation=dict(font_size=9, font_color='white'),
                )

        fig_kline.update_layout(
            template='plotly_dark', height=300,
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            xaxis_rangeslider_visible=False,
            margin=dict(t=30, b=30, l=50, r=20),
            showlegend=False,
            dragmode='select',  # 默认启用框选模式
        )

        # GEX 柱状图
        fig_gex = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=('Net GEX by Strike', 'Calls vs Puts'),
                                vertical_spacing=0.12)

        stats_text = ""
        if strikes_df is not None and not strikes_df.empty:
            by_strike = strikes_df.groupby('strike')['gex'].sum().sort_index() / 1e6
            calls = strikes_df[strikes_df.right == 'C'].groupby('strike')['gex'].sum() / 1e6
            puts = strikes_df[strikes_df.right == 'P'].groupby('strike')['gex'].sum() / 1e6

            colors = ['#00ff88' if v > 0 else '#ff4444' for v in by_strike.values]
            fig_gex.add_trace(go.Bar(x=by_strike.index, y=by_strike.values,
                                     marker_color=colors, name='Net'), row=1, col=1)
            fig_gex.add_trace(go.Bar(x=calls.index, y=calls.values,
                                     marker_color='#00d4ff', name='Calls'), row=2, col=1)
            fig_gex.add_trace(go.Bar(x=puts.index, y=puts.values,
                                     marker_color='#ff66cc', name='Puts'), row=2, col=1)

            # 计算聚合指标
            total_gex = strikes_df['gex'].sum()
            call_gex = strikes_df[strikes_df.right == 'C']['gex'].sum()
            put_gex = strikes_df[strikes_df.right == 'P']['gex'].sum()
            gamma_flip = by_strike.cumsum().abs().idxmin() if not by_strike.empty else 0

            # 加载对应时间的 spot
            gex_df = storage.load_day_gex(symbol, date_str)
            spot = None
            if gex_df is not None and not gex_df.empty:
                time_diff = (gex_df['ts'] - target_ts).abs()
                closest_idx = time_diff.idxmin()
                # 只有当最近记录在 5 分钟内才使用
                max_deviation_sec = 300
                if time_diff.loc[closest_idx].total_seconds() <= max_deviation_sec:
                    spot = gex_df.loc[closest_idx, 'spot']

            if spot:
                for r in [1, 2]:
                    fig_gex.add_vline(x=spot, line=dict(color='white', dash='dash'),
                                      row=r, col=1)
                    fig_gex.add_vline(x=gamma_flip, line=dict(color='#ffaa00', dash='dot'),
                                      row=r, col=1)

            gex_color = '#00ff88' if total_gex > 0 else '#ff4444'
            stats_text = (
                f"Total GEX: ${total_gex/1e6:.2f}M  |  "
                f"Call GEX: ${call_gex/1e6:.2f}M  |  "
                f"Put GEX: ${put_gex/1e6:.2f}M  |  "
                f"Gamma Flip: {gamma_flip:.0f}"
            )
            if spot:
                stats_text = f"Spot: {spot:.2f}  |  " + stats_text
        else:
            stats_text = "⚠️ 该时间点没有数据"

        fig_gex.update_layout(
            template='plotly_dark', height=500, barmode='relative',
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117'
        )
        fig_gex.update_yaxes(title_text='GEX ($M per 1%)')
        fig_gex.update_xaxes(title_text='Strike', row=2, col=1)

        return time_info, fig_kline, fig_gex, stats_text

    # ==================== 复盘时快速标注 ====================
    @app.callback(
        [Output('replay-label-status', 'children'),
         Output('replay-label-trigger', 'data')],
        Input('replay-add-label-btn', 'n_clicks'),
        [State('replay-kline-chart', 'selectedData'),
         State('replay-date-dropdown', 'value'),
         State('replay-label-radio', 'value'),
         State('replay-label-note', 'value'),
         State('symbol-dropdown', 'value'),
         State('replay-label-trigger', 'data')],
        prevent_initial_call=True)
    def add_replay_label(n_clicks, selected_data, date_str, label, note, symbol, trigger):
        trigger = (trigger or 0) + 1
        if not n_clicks:
            return "", trigger
        if not date_str or not symbol:
            return "⚠️ 请先选择日期和标的", trigger
        if not label:
            return "⚠️ 请选择标签类型", trigger
        if not selected_data or 'range' not in selected_data:
            return "⚠️ 请先在 K 线图上框选时间段（按住鼠标拖动）", trigger

        sel_range = selected_data['range']
        if 'x' not in sel_range or len(sel_range['x']) < 2:
            return "⚠️ 框选区域无效", trigger

        start_ts = sel_range['x'][0]
        end_ts = sel_range['x'][1]

        segments.save_segment(date_str, start_ts, end_ts, symbol, label, note or '')
        return f"✅ 已标注: {label} ({start_ts[:16]} ~ {end_ts[:16]})", trigger
