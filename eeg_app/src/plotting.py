"""Plotly figures used by the EEG dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .channels import channel_type
from .signal_utils import min_max_downsample


COLORS = {"EEG": "#125b78", "EMG": "#158a8a", "ECG": "#d97745", "SLI": "#6b5ca5", "Other": "#667085"}
SEIZURE_COLOR = "#d64550"


def add_seizure_overlays(figure: go.Figure, seizures: pd.DataFrame, start: float | None = None, end: float | None = None, rows: tuple[int, ...] | None = None) -> go.Figure:
    """Add clipped translucent red regions for seizures intersecting a view."""
    if seizures.empty:
        return figure
    for index, row in enumerate(seizures.itertuples(index=False), start=1):
        seizure_start = float(row[1])
        seizure_end = float(row[2])
        if start is not None and seizure_end < start:
            continue
        if end is not None and seizure_start > end:
            continue
        left = max(seizure_start, start) if start is not None else seizure_start
        right = min(seizure_end, end) if end is not None else seizure_end
        for row_number in rows or (None,):
            figure.add_vrect(x0=left, x1=right, fillcolor=SEIZURE_COLOR, opacity=0.28, line_color=SEIZURE_COLOR, line_width=2, layer="below", annotation_text=f"Crise {index}" if row_number is None else None, annotation_position="top left", row=row_number, col=1)
    if rows is None:
        figure.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": SEIZURE_COLOR, "width": 8}, name="Crise - zone rouge", hoverinfo="skip"))
    return figure


def signal_figure(frame: pd.DataFrame, channels: list[str], seizures: pd.DataFrame, mode: str, max_points: int, theme: str = "plotly_white") -> go.Figure:
    """Create an overlaid or clinical-style stacked signal plot."""
    visible = min_max_downsample(frame[["Time", *channels]], "Time", max_points)
    figure = go.Figure()
    span = max((visible[channel].max() - visible[channel].min() for channel in channels), default=1) or 1
    for index, channel in enumerate(channels):
        values = visible[channel]
        offset = index * span * 1.25 if mode == "stacked" else 0
        figure.add_trace(go.Scattergl(x=visible["Time"], y=values + offset, mode="lines", name=channel, line={"width": 0.8, "color": COLORS[channel_type(channel)]}, legendgroup=channel_type(channel), hovertemplate=f"%{{x:.3f}} s | {channel}: %{{y:.3f}}<extra></extra>"))
    if len(visible):
        add_seizure_overlays(figure, seizures, float(visible["Time"].min()), float(visible["Time"].max()))
    ylabel = "Amplitude (offsets)" if mode == "stacked" else "Amplitude"
    figure.update_layout(template=theme, height=560, margin={"l": 50, "r": 20, "t": 45, "b": 45}, hovermode="x unified", xaxis={"title": "Temps (s)", "rangeslider": {"visible": True}}, yaxis_title=ylabel, legend_title="Canaux")
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


def target_figure(frame: pd.DataFrame, channel: str, seizures: pd.DataFrame | None = None) -> go.Figure:
    """Plot a channel and the binary target in aligned panels."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
    figure.add_trace(go.Scattergl(x=frame["Time"], y=frame[channel], name=channel, line={"color": COLORS[channel_type(channel)], "width": 0.8}), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame["Time"], y=frame["Class"], name="Classe", line={"color": SEIZURE_COLOR, "shape": "hv"}, fill="tozeroy", fillcolor="rgba(214,69,80,0.18)"), row=2, col=1)
    if len(frame) and seizures is not None:
        add_seizure_overlays(figure, seizures, float(frame["Time"].min()), float(frame["Time"].max()), rows=(1, 2))
    figure.update_layout(template="plotly_white", height=560, hovermode="x unified", margin={"t": 25})
    figure.update_yaxes(title_text="Signal", row=1, col=1)
    figure.update_yaxes(title_text="Classe", range=[-0.05, 1.05], row=2, col=1)
    figure.update_xaxes(title_text="Temps (s)", row=2, col=1)
    return figure


def window_class_figure(windows: pd.DataFrame) -> go.Figure:
    """Plot classified windows over time with red class-1 bars."""
    figure = go.Figure()
    for class_value, color, label in ((0, "#0e7490", "Hors crise"), (1, SEIZURE_COLOR, "Crise")):
        subset = windows.loc[windows["Class"].eq(class_value)]
        if subset.empty:
            continue
        figure.add_trace(go.Bar(x=subset["Duration"], y=[label] * len(subset), base=subset["Start_Time"], orientation="h", name=label, marker_color=color, customdata=subset[["Window", "Seizure_Ratio"]], hovertemplate="Fenetre %{customdata[0]}<br>Debut: %{base:.2f} s<br>Ratio crise: %{customdata[1]:.1%}<extra></extra>"))
    figure.update_layout(template="plotly_white", height=260, barmode="overlay", xaxis_title="Temps (s)", yaxis_title="Classe")
    return figure