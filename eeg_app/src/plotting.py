"""Plotly figures used by the EEG dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .channels import channel_type, channel_unit
from .signal_utils import min_max_downsample


COLORS = {"EEG": "#125b78", "EMG": "#158a8a", "ECG": "#d97745", "SLI": "#6b5ca5", "Other": "#667085"}


def signal_figure(frame: pd.DataFrame, channels: list[str], seizures: pd.DataFrame, mode: str, max_points: int, theme: str = "plotly_white") -> go.Figure:
    """Create an overlaid or clinical-style stacked signal plot."""
    visible = min_max_downsample(frame[["Time", *channels]], "Time", max_points)
    figure = go.Figure()
    span = max((visible[channel].max() - visible[channel].min() for channel in channels), default=1) or 1
    for index, channel in enumerate(channels):
        values = visible[channel]
        offset = index * span * 1.25 if mode == "stacked" else 0
        figure.add_trace(go.Scattergl(x=visible["Time"], y=values + offset, mode="lines", name=channel, line={"width": 0.8, "color": COLORS[channel_type(channel)]}, legendgroup=channel_type(channel), hovertemplate=f"%{{x:.3f}} s | {channel}: %{{y:.3f}}<extra></extra>"))
    for row in seizures.itertuples(index=False):
        figure.add_vrect(x0=row[1], x1=row[2], fillcolor="#d64550", opacity=0.16, line_width=0, layer="below")
    ylabel = "Amplitude (offsets)" if mode == "stacked" else "Amplitude"
    figure.update_layout(template=theme, height=560, margin={"l": 50, "r": 20, "t": 30, "b": 45}, hovermode="x unified", xaxis={"title": "Temps (s)", "rangeslider": {"visible": True}}, yaxis_title=ylabel, legend_title="Canaux")
    return figure


def seizure_timeline(seizures: pd.DataFrame) -> go.Figure:
    """Create a compact Gantt-style seizure timeline."""
    figure = go.Figure()
    for row in seizures.itertuples(index=False):
        figure.add_trace(go.Bar(x=[row[3]], y=[row[0]], base=[row[1]], orientation="h", marker_color="#d64550", hovertemplate=f"{row[0]}<br>%{{base:.2f}} - %{{x:.2f}} s<extra></extra>"))
    figure.update_layout(template="plotly_white", height=max(180, 50 * len(seizures)), showlegend=False, xaxis_title="Temps (s)", yaxis_title="Crise")
    return figure


def duration_figures(seizures: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """Return duration bar and histogram figures."""
    bar = go.Figure(go.Bar(x=seizures["Crise"], y=seizures["Durée (s)"], marker_color="#0e7490"))
    histogram = go.Figure(go.Histogram(x=seizures["Durée (s)"], nbinsx=min(20, max(5, len(seizures))), marker_color="#158a8a"))
    for figure in (bar, histogram):
        figure.update_layout(template="plotly_white", height=280, margin={"t": 20})
    bar.update_layout(xaxis_title="Crise", yaxis_title="Durée (s)")
    histogram.update_layout(xaxis_title="Durée (s)", yaxis_title="Nombre")
    return bar, histogram


def target_figure(frame: pd.DataFrame, channel: str) -> go.Figure:
    """Plot a channel and the binary target in aligned panels."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
    figure.add_trace(go.Scattergl(x=frame["Time"], y=frame[channel], name=channel, line={"color": COLORS[channel_type(channel)], "width": 0.8}), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame["Time"], y=frame["target"], name="Cible", line={"color": "#d64550", "shape": "hv"}, fill="tozeroy"), row=2, col=1)
    figure.update_layout(template="plotly_white", height=560, hovermode="x unified", margin={"t": 25})
    figure.update_yaxes(title_text="Signal", row=1, col=1)
    figure.update_yaxes(title_text="Cible", range=[-0.05, 1.05], row=2, col=1)
    figure.update_xaxes(title_text="Temps (s)", row=2, col=1)
    return figure