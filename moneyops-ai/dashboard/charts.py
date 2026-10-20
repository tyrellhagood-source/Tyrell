"""Plotly chart builders. Palette: validated reference categorical slots
1-3 (blue, orange, aqua) - max three series per chart - plus the reserved
status colours for good/critical. Transparent backgrounds follow the
Streamlit light/dark theme; one y-axis per chart; recessive grid."""
from __future__ import annotations

import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#8a8984"


def _layout(fig: go.Figure, title: str, x: str = "", y: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
    fig.update_xaxes(title=x, showgrid=False, zeroline=False)
    fig.update_yaxes(title=y, gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def calibration_chart(table: list[dict], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=NEUTRAL, width=1, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[r["mean_predicted"] for r in table], y=[r["observed_rate"] for r in table], mode="lines+markers",
        name="Model", line=dict(color=SERIES[0], width=2), marker=dict(size=[max(8, min(22, r["n"] ** 0.5 * 2))
                                                                             for r in table]),
        customdata=[r["n"] for r in table],
        hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>n=%{customdata}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _layout(fig, title, "Mean predicted probability", "Observed frequency")


def model_vs_market(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=[r["market"] for r in rows], y=[r["model"] for r in rows], mode="markers",
        marker=dict(size=9, color=SERIES[0], line=dict(color="rgba(255,255,255,0.8)", width=2)),
        text=[r["label"] for r in rows], showlegend=False,
        hovertemplate="%{text}<br>Market (no-vig) %{x:.1%}<br>Model %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, width=1, dash="dot"),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _layout(fig, "Model probability vs market no-vig probability", "Market (no-vig)", "Model")


def interval_chart(rows: list[dict]) -> go.Figure:
    """rows: label, p, lo, hi, be (break-even)."""
    rows = rows[:20]
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["p"] for r in rows], y=labels, mode="markers", name="Estimated probability",
        marker=dict(size=10, color=SERIES[0]),
        error_x=dict(type="data", symmetric=False, array=[r["hi"] - r["p"] for r in rows],
                     arrayminus=[r["p"] - r["lo"] for r in rows], color=SERIES[0], thickness=2, width=0),
        hovertemplate="%{y}<br>p=%{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["be"] for r in rows], y=labels, mode="markers", name="Break-even",
                             marker=dict(size=10, symbol="line-ns-open", color=SERIES[1], line=dict(width=2)),
                             hovertemplate="%{y}<br>break-even %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%")
    return _layout(fig, "Probability interval (5-95%) vs break-even", "Probability", "",
                   height=max(260, 34 * len(rows) + 90))


def histogram(values: list[float], title: str, x: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=20, marker=dict(color=SERIES[0], line=dict(color="white", width=2)),
                                 hovertemplate="%{x}<br>count %{y}<extra></extra>"))
    return _layout(fig, title, x, "Count")


def line_chart(x, y, title: str, xlab: str, ylab: str, color: str = SERIES[0]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2),
                               hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    return _layout(fig, title, xlab, ylab)


def bar_chart(labels, values, title: str, ylab: str, signed: bool = False) -> go.Figure:
    colors = [(GOOD if v >= 0 else CRITICAL) if signed else SERIES[0] for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
                           hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    return _layout(fig, title, "", ylab)
