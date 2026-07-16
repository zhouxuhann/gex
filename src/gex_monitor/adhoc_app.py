"""Standalone Dash app for on-demand single-symbol GEX snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from flask import Flask, jsonify

from .adhoc_gex import fetch_adhoc_gex
from .config import AppConfig

DARK_BG = "#0e1117"
PANEL_BG = "#1a1f2e"
TEXT = "#fafafa"
MUTED = "#9aa4b2"
GREEN = "#00ff88"
RED = "#ff5577"
ORANGE = "#ffaa00"
BLUE = "#45b8ff"


def _fmt_big(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) / 1e6:,.1f}M"


def _fmt_num(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.2f}"


def _quality_color(quality: str) -> str:
    return {"ok": GREEN, "partial": ORANGE, "bad": RED}.get(quality, MUTED)


def _snapshot_to_payload(snapshot) -> dict:
    q = snapshot.quality
    payload = {
        "symbol": snapshot.symbol,
        "spot": snapshot.spot,
        "expiry": snapshot.expiry,
        "is_true_0dte": snapshot.is_true_0dte,
        "ts": snapshot.ts.isoformat(),
        "error": snapshot.error,
        "quality": q.quality,
        "requested_contracts": q.requested_contracts,
        "qualified_contracts": q.qualified_contracts,
        "greeks_count": q.greeks_count,
        "oi_count": q.oi_count,
        "bidask_count": q.bidask_count,
        "result": None,
        "rows": [],
    }
    if snapshot.result is not None:
        r = snapshot.result
        payload["result"] = {
            "total_gex": r.total_gex,
            "call_gex": r.call_gex,
            "put_gex": r.put_gex,
            "gamma_flip": r.gamma_flip,
            "call_wall": r.call_wall,
            "put_wall": r.put_wall,
            "max_pain": r.max_pain,
            "atm_iv_pct": r.atm_iv_pct,
            "missing_greeks": r.missing_greeks,
            "missing_oi": r.missing_oi,
        }
        rows = r.df.copy()
        rows["gex_m"] = rows["gex"] / 1e6
        payload["rows"] = rows.to_dict("records")
    return payload


def _empty_figure(message: str = "No snapshot") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(
        template="plotly_dark",
        height=520,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    return fig


def _figure_from_rows(rows: list[dict], result: dict | None) -> go.Figure:
    if not rows:
        return _empty_figure("No valid Greeks returned")
    df = pd.DataFrame(rows)
    calls = df[df["right"] == "C"].groupby("strike")["gex"].sum().sort_index() / 1e6
    puts = df[df["right"] == "P"].groupby("strike")["gex"].sum().sort_index() / 1e6

    fig = go.Figure()
    fig.add_bar(x=calls.index, y=calls.values, name="Call GEX", marker_color=GREEN)
    fig.add_bar(x=puts.index, y=puts.values, name="Put GEX", marker_color=RED)

    if result:
        flip = result.get("gamma_flip")
        if flip is not None and not pd.isna(flip):
            fig.add_vline(x=flip, line=dict(color=ORANGE, dash="dot"), annotation_text="Flip")
        call_wall = result.get("call_wall")
        if call_wall is not None and not pd.isna(call_wall):
            fig.add_vline(x=call_wall, line=dict(color=GREEN, dash="dash"), annotation_text="Call Wall")
        put_wall = result.get("put_wall")
        if put_wall is not None and not pd.isna(put_wall):
            fig.add_vline(x=put_wall, line=dict(color=RED, dash="dash"), annotation_text="Put Wall")

    fig.update_layout(
        template="plotly_dark",
        height=520,
        barmode="relative",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis_title="Strike",
        yaxis_title="GEX ($M)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _stat_card(label: str, value: str, color: str = TEXT) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "12px", "color": MUTED}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": color}),
        ],
        style={
            "backgroundColor": PANEL_BG,
            "border": "1px solid #2a3246",
            "borderRadius": "6px",
            "padding": "12px",
            "minWidth": "130px",
        },
    )


def _stats_from_payload(payload: dict | None):
    if not payload:
        return html.Div("Ready", style={"color": MUTED})
    if payload.get("error"):
        return html.Div(payload["error"], style={"color": RED, "fontSize": "16px"})

    result = payload.get("result") or {}
    q = payload["quality"]
    qc = _quality_color(q)
    qualified = payload["qualified_contracts"]
    quality_line = (
        f"Greeks {payload['greeks_count']}/{qualified} | "
        f"OI {payload['oi_count']}/{qualified} | "
        f"Bid/Ask {payload['bidask_count']}/{qualified}"
    )
    return html.Div(
        [
            html.Div(
                [
                    _stat_card("Symbol", payload["symbol"], BLUE),
                    _stat_card("Spot", _fmt_num(payload["spot"])),
                    _stat_card("Expiry", payload["expiry"] or "N/A", ORANGE),
                    _stat_card("Quality", q.upper(), qc),
                    _stat_card("Total GEX", _fmt_big(result.get("total_gex"))),
                    _stat_card("Call GEX", _fmt_big(result.get("call_gex")), GREEN),
                    _stat_card("Put GEX", _fmt_big(result.get("put_gex")), RED),
                    _stat_card("Flip", _fmt_num(result.get("gamma_flip")), ORANGE),
                    _stat_card("Max Pain", _fmt_num(result.get("max_pain"))),
                ],
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "justifyContent": "center"},
            ),
            html.Div(
                quality_line,
                style={"textAlign": "center", "color": qc, "marginTop": "12px", "fontSize": "14px"},
            ),
        ]
    )


def create_app(config: AppConfig) -> Dash:
    server = Flask(__name__)

    @server.route("/health")
    def health():
        return jsonify({"status": "healthy", "app": "adhoc-gex"})

    app = Dash(__name__, server=server, title="Ad-hoc GEX")
    app.layout = html.Div(
        style={
            "backgroundColor": DARK_BG,
            "color": TEXT,
            "fontFamily": "monospace",
            "minHeight": "100vh",
            "padding": "18px",
        },
        children=[
            dcc.Store(id="snapshot-store"),
            html.H2("Ad-hoc GEX", style={"textAlign": "center", "margin": "0 0 16px"}),
            html.Div(
                [
                    dcc.Input(
                        id="symbol-input",
                        type="text",
                        value="TSLA",
                        debounce=True,
                        placeholder="Symbol",
                        style={
                            "width": "120px",
                            "backgroundColor": "#101522",
                            "color": TEXT,
                            "border": "1px solid #2a3246",
                            "padding": "9px",
                            "fontSize": "16px",
                        },
                    ),
                    dcc.Input(
                        id="strikes-input",
                        type="number",
                        value=10,
                        min=3,
                        max=40,
                        step=1,
                        style={
                            "width": "90px",
                            "backgroundColor": "#101522",
                            "color": TEXT,
                            "border": "1px solid #2a3246",
                            "padding": "9px",
                            "fontSize": "16px",
                        },
                    ),
                    dcc.Input(
                        id="wait-input",
                        type="number",
                        value=8,
                        min=3,
                        max=20,
                        step=1,
                        style={
                            "width": "80px",
                            "backgroundColor": "#101522",
                            "color": TEXT,
                            "border": "1px solid #2a3246",
                            "padding": "9px",
                            "fontSize": "16px",
                        },
                    ),
                    html.Button(
                        "Fetch",
                        id="fetch-btn",
                        n_clicks=0,
                        style={
                            "padding": "10px 18px",
                            "backgroundColor": BLUE,
                            "color": "black",
                            "border": "none",
                            "borderRadius": "6px",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "Save",
                        id="save-btn",
                        n_clicks=0,
                        style={
                            "padding": "10px 18px",
                            "backgroundColor": GREEN,
                            "color": "black",
                            "border": "none",
                            "borderRadius": "6px",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "10px",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "flexWrap": "wrap",
                    "marginBottom": "14px",
                },
            ),
            dcc.Loading(html.Div(id="stats-panel", style={"minHeight": "88px"}), type="dot"),
            html.Div(id="save-status", style={"textAlign": "center", "minHeight": "22px", "color": MUTED}),
            dcc.Graph(id="gex-chart", figure=_empty_figure()),
            dash_table.DataTable(
                id="details-table",
                columns=[
                    {"name": "Strike", "id": "strike", "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "Right", "id": "right"},
                    {"name": "GEX $M", "id": "gex_m", "type": "numeric", "format": {"specifier": ",.2f"}},
                    {"name": "Gamma", "id": "gamma", "type": "numeric", "format": {"specifier": ".6f"}},
                    {"name": "OI", "id": "oi", "type": "numeric", "format": {"specifier": ",.0f"}},
                    {"name": "Volume", "id": "volume", "type": "numeric", "format": {"specifier": ",.0f"}},
                    {"name": "IV", "id": "iv", "type": "numeric", "format": {"specifier": ".4f"}},
                ],
                data=[],
                sort_action="native",
                page_size=30,
                style_cell={
                    "backgroundColor": DARK_BG,
                    "color": TEXT,
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                    "padding": "6px",
                },
                style_header={"backgroundColor": PANEL_BG, "fontWeight": "bold"},
                style_table={"overflowX": "auto"},
            ),
        ],
    )

    @app.callback(
        Output("snapshot-store", "data"),
        Output("stats-panel", "children"),
        Output("gex-chart", "figure"),
        Output("details-table", "data"),
        Output("save-status", "children"),
        Input("fetch-btn", "n_clicks"),
        State("symbol-input", "value"),
        State("strikes-input", "value"),
        State("wait-input", "value"),
        prevent_initial_call=True,
    )
    def fetch_snapshot(_, symbol, strikes_each_side, wait_sec):
        snapshot = fetch_adhoc_gex(
            symbol or "",
            host=config.ib.host,
            port=config.ib.port,
            client_id=config.ib.client_id_base + 80,
            strikes_each_side=int(strikes_each_side or 10),
            wait_sec=float(wait_sec or 8),
        )
        payload = _snapshot_to_payload(snapshot)
        return (
            payload,
            _stats_from_payload(payload),
            _figure_from_rows(payload["rows"], payload.get("result")),
            payload["rows"],
            "",
        )

    @app.callback(
        Output("save-status", "children", allow_duplicate=True),
        Input("save-btn", "n_clicks"),
        State("snapshot-store", "data"),
        prevent_initial_call=True,
    )
    def save_snapshot(_, payload):
        if not payload:
            return "No snapshot to save"
        data_dir = Path(config.storage.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        stamp = payload["ts"].replace(":", "").replace("-", "").split(".")[0]
        base = f"{payload['symbol']}_{stamp}"
        summary_path = data_dir / f"adhoc_gex_{base}.parquet"
        rows_path = data_dir / f"adhoc_strikes_{base}.parquet"

        summary = {k: v for k, v in payload.items() if k not in {"rows", "result"}}
        if payload.get("result"):
            summary.update(payload["result"])
        pd.DataFrame([summary]).to_parquet(summary_path, index=False)
        if payload.get("rows"):
            rows = pd.DataFrame(payload["rows"])
            rows.insert(0, "symbol", payload["symbol"])
            rows.insert(1, "expiry", payload["expiry"])
            rows.insert(2, "ts", payload["ts"])
            rows.to_parquet(rows_path, index=False)
            return f"Saved {summary_path.name}, {rows_path.name}"
        return f"Saved {summary_path.name}"

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Ad-hoc GEX web app")
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config) if args.config else AppConfig.default()
    if args.host:
        config.server.host = args.host
    config.server.port = args.port
    app = create_app(config)
    app.run(debug=False, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
