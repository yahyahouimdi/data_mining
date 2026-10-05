"""Small Streamlit presentation helpers and the clinical dashboard theme."""

from __future__ import annotations

import streamlit as st


PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def inject_css(dark: bool = False) -> None:
    """Apply the dashboard visual system."""
    background = "#101b26" if dark else "#f5f8fa"
    surface = "#172635" if dark else "#ffffff"
    text = "#ecf4f7" if dark else "#173042"
    st.markdown(f"""<style>
    .stApp {{ background: {background}; color: {text}; }}
    [data-testid="stMetric"] {{ background: {surface}; border: 1px solid #d9e5ea; border-radius: 12px; padding: 14px; }}
    .app-header {{ padding: 1rem 0 .5rem; border-bottom: 1px solid #d9e5ea; margin-bottom: 1rem; }}
    .app-header h1 {{ color: #0e7490; margin: 0; letter-spacing: 0; }}
    .status {{ color: #667085; font-size: .9rem; }}
    .warning-badge {{ display: inline-block; padding: .25rem .6rem; border-radius: 999px; background: #fff0c2; color: #805b00; font-size: .8rem; font-weight: 600; }}
    .pipeline {{ background: {surface}; border-left: 4px solid #0e7490; border-radius: 10px; padding: .8rem 1rem; margin: .5rem 0 1rem; color: {text}; line-height: 1.8; }}
    </style>""", unsafe_allow_html=True)


def metric_row(metrics: list[tuple[str, str]]) -> None:
    """Render a row of metric cards."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)