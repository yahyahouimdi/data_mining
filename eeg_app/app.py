"""Streamlit entry point for EEG Seizure Explorer."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .src.annotations import duration_statistics, normalize_seizure_table, pair_seizures
from .src.channels import channel_type, group_channels, layout_name
from .src.io_utils import annotation_source_warning, discover_recordings, read_annotations, read_csv_with_fallback, read_signal_window, signal_columns, signal_metadata
from .src.plotting import duration_figures, seizure_timeline, signal_figure, target_figure
from .src.signal_utils import normalize_series, spectrogram, welch_psd
from .src.target import build_target
from .src.ui import inject_css, metric_row


st.set_page_config(page_title="EEG Seizure Explorer", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def cached_recordings(folder: str):
    """Cache recording discovery."""
    return discover_recordings(folder)


@st.cache_data(show_spinner=False)
def cached_metadata(path: str, cache_dir: str):
    """Cache signal metadata."""
    return signal_metadata(path, cache_dir)


@st.cache_data(show_spinner=False)
def cached_window(path: str, channels: tuple[str, ...], start: float, end: float, cache_dir: str):
    """Cache one selected signal window."""
    return read_signal_window(path, channels, start, end, cache_dir)


def _format_duration(seconds: float) -> str:
    """Format seconds as mm:ss."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _load_uploaded_xlsx(uploaded: object) -> pd.DataFrame | None:
    """Read the first worksheet from an optional annotation workbook."""
    if uploaded is None:
        return None
    try:
        return pd.read_excel(uploaded)
    except (ValueError, OSError):
        st.error("Le fichier Excel n'a pas pu etre lu.")
        return None


def _sidebar(records):
    """Render sidebar controls and return the selected configuration."""
    st.sidebar.header("Configuration")
    folder = st.sidebar.text_input("Dossier des donnees", value="data")
    if st.sidebar.button("Recharger les fichiers", use_container_width=True):
        cached_recordings.clear()
        st.rerun()
    if not records:
        st.sidebar.error("Aucun couple *_0000d.csv / *_0000h.csv trouve.")
        return None
    selected_id = st.sidebar.selectbox("Enregistrement", [record.recording_id for record in records])
    selected = next(record for record in records if record.recording_id == selected_id)
    badges = ("d disponible" if selected.signal_path else "d manquant") + " | " + ("h disponible" if selected.annotation_path else "h manquant")
    st.sidebar.caption(badges)
    if selected.signal_path is None:
        return None
    cache_dir = str(Path(folder) / ".cache")
    with st.spinner("Lecture des metadonnees..."):
        metadata = cached_metadata(str(selected.signal_path), cache_dir)
    channels = [value for value in metadata["columns"] if value != "Time"]
    groups = group_channels(channels)
    selected_channels: list[str] = []
    presets = {"Tous les EEG": groups.get("EEG", []), "Frontal": [c for c in groups.get("EEG", []) if any(token in c.upper() for token in ("FP", "F7", "F8", "F3", "F4", "FZ"))], "Temporal": [c for c in groups.get("EEG", []) if any(token in c.upper() for token in ("T3", "T4", "T5", "T6"))], "Occipital": [c for c in groups.get("EEG", []) if "O" in c.upper()]}
    preset = st.sidebar.selectbox("Preset canaux", ["Personnalise", *presets])
    for kind, values in groups.items():
        selected_channels.extend(st.sidebar.multiselect(kind, values, default=presets[preset] if preset != "Personnalise" and kind == "EEG" else []))
    if not selected_channels:
        selected_channels = channels[:1]
    duration = float(metadata["duration"])
    time_range = st.sidebar.slider("Plage temporelle (s)", 0.0, max(duration, 1.0), (0.0, max(duration, 1.0)), step=max(duration / 1000, 0.1))
    window = st.sidebar.selectbox("Fenetre", [5, 10, 30, 60, "full"], format_func=lambda value: "Pleine duree" if value == "full" else f"{value} s")
    normalization = st.sidebar.selectbox("Normalisation", ["none", "z-score", "min-max"], format_func=lambda value: {"none": "Aucune", "z-score": "Z-score", "min-max": "Min-max"}[value])
    mode = st.sidebar.radio("Affichage", ["stacked", "overlay"], format_func=lambda value: "Canaux empiles" if value == "stacked" else "Canaux superposes")
    downsample = st.sidebar.select_slider("Densite du tracage", options=[1000, 2500, 5000, 10000, 20000], value=5000)
    dark = st.sidebar.toggle("Theme sombre")
    overlays = st.sidebar.toggle("Afficher les crises", True)
    return {"folder": folder, "record": selected, "metadata": metadata, "cache_dir": cache_dir, "channels": selected_channels, "time_range": time_range, "window": window, "normalization": normalization, "mode": mode, "downsample": downsample, "dark": dark, "overlays": overlays}


def _get_seizures(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load persistent edited seizures and raw h annotations."""
    record_id = config["record"].recording_id
    if st.session_state.get("seizure_record") != record_id:
        st.session_state.seizure_record = record_id
        st.session_state.seizures = pd.DataFrame(columns=["Crise", "Début (s)", "Fin (s)", "Durée (s)"])
    raw = pd.DataFrame(columns=["Time", "Annotations", "Duration"])
    path = config["record"].annotation_path
    if path:
        try:
            raw = read_annotations(path)
        except (OSError, ValueError) as error:
            st.error(f"Annotation illisible: {error}")
    uploaded = st.session_state.get("xlsx_data")
    if uploaded is not None and st.session_state.get("xlsx_record") != record_id:
        st.session_state.seizures = normalize_seizure_table(pair_seizures(uploaded, labels=["absence"]))
        st.session_state.xlsx_record = record_id
    return raw, normalize_seizure_table(st.session_state.seizures)


def overview_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Render Part I."""
    metadata = config["metadata"]
    missing_total = sum(metadata["missing"].values())
    metric_row([("Lignes", f"{metadata['rows']:,}"), ("Variables", str(len(metadata["columns"]) - 1)), ("Frequence", f"{metadata['frequency']:.2f} Hz"), ("Duree", f"{_format_duration(metadata['duration'])} ({metadata['duration']:.1f} s)"), ("Valeurs manquantes", f"{missing_total:,}")])
    row_count = st.selectbox("Nombre de lignes a afficher", [10, 20], index=0)
    with st.spinner("Lecture de la fenetre d'aperçu..."):
        preview = read_csv_with_fallback(config["record"].signal_path, nrows=row_count)
    st.dataframe(preview, use_container_width=True, hide_index=True)
    stats = config["metadata"]["frame"].describe().T.reset_index().rename(columns={"index": "Canal"})
    stats = stats[["Canal", "mean", "std", "min", "max"]].round(3)
    st.subheader("Statistiques par canal")
    st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True, hide_index=True)
    frame = cached_window(str(config["record"].signal_path), tuple(config["channels"]), *config["time_range"], config["cache_dir"])
    for channel in config["channels"]:
        frame[channel] = normalize_series(frame[channel], config["normalization"])
    st.plotly_chart(signal_figure(frame, config["channels"], seizures if config["overlays"] else seizures.iloc[0:0], config["mode"], config["downsample"], "plotly_dark" if config["dark"] else "plotly_white"), use_container_width=True)
    st.subheader("Vue multi-enregistrements")
    rows = [{"Enregistrement": item.recording_id, "d": item.signal_path is not None, "h": item.annotation_path is not None} for item in cached_recordings(config["folder"])]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def annotations_tab(config: dict, raw: pd.DataFrame, seizures: pd.DataFrame) -> pd.DataFrame:
    """Render Part II and return the edited seizure table."""
    st.warning(annotation_source_warning())
    uploaded = st.file_uploader("Importer Annotations.xlsx (optionnel)", type=["xlsx"], key="xlsx_upload")
    if uploaded is not None:
        st.session_state.xlsx_data = _load_uploaded_xlsx(uploaded)
        st.session_state.xlsx_record = None
        st.rerun()
    label_filter = st.text_input("Filtrer les labels", placeholder="absence, ARTEFACT, HPN...")
    display_raw = raw if not label_filter else raw[raw["Annotations"].str.contains(label_filter, case=False, na=False)]
    st.dataframe(display_raw, use_container_width=True, hide_index=True)
    st.subheader("Intervalles valides")
    edited = st.data_editor(seizures, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"editor_{config['record'].recording_id}")
    edited = normalize_seizure_table(edited)
    st.session_state.seizures = edited
    stats = duration_statistics(edited)
    metric_row([("Crises", str(int(stats["count"]))), ("Duree moyenne", f"{stats['mean']:.2f} s"), ("Min / max", f"{stats['min']:.2f} / {stats['max']:.2f} s"), ("Intervalle moyen", f"{stats['mean_interval']:.2f} s")])
    if not edited.empty:
        st.plotly_chart(seizure_timeline(edited), use_container_width=True)
        bar, histogram = duration_figures(edited)
        left, right = st.columns(2)
        left.plotly_chart(bar, use_container_width=True)
        right.plotly_chart(histogram, use_container_width=True)
        conclusion = "similaires" if stats["cv"] < 0.25 else "variables"
        st.info(f"Coefficient de variation: {stats['cv']:.3f}. Les durees sont {conclusion} (seuil CV < 0.25).")
        st.download_button("Exporter les crises CSV", edited.to_csv(index=False).encode("utf-8"), f"{config['record'].recording_id}_seizures.csv", "text/csv")
    return edited


def target_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Render Part III and labeled-data export."""
    frame = cached_window(str(config["record"].signal_path), tuple(config["channels"]), *config["time_range"], config["cache_dir"])
    frame["target"] = build_target(frame["Time"], seizures)
    counts = frame["target"].value_counts().reindex([0, 1], fill_value=0)
    proportion = float(counts[1] / max(len(frame), 1))
    metric_row([("Hors crise", f"{1 - proportion:.1%}"), ("Pendant crise", f"{proportion:.1%}"), ("Temps de crise", f"{frame.loc[frame.target.eq(1), 'Time'].max() - frame.loc[frame.target.eq(1), 'Time'].min():.1f} s" if counts[1] else "0 s")])
    if proportion < 0.1:
        st.warning("Des classes tres desequilibrees peuvent biaiser un futur classifieur.")
    channel = st.selectbox("Canal cible", config["channels"])
    st.plotly_chart(target_figure(frame, channel), use_container_width=True)
    st.download_button("Exporter CSV annote", frame.to_csv(index=False).encode("utf-8"), f"{config['record'].recording_id}_labeled.csv", "text/csv")
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    st.download_button("Exporter Parquet annote", buffer.getvalue(), f"{config['record'].recording_id}_labeled.parquet", "application/octet-stream")


def comparison_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Render the optional signal comparison and spectral analysis."""
    channel = st.selectbox("Canal a comparer", config["channels"], key="compare_channel")
    frame = cached_window(str(config["record"].signal_path), (channel,), *config["time_range"], config["cache_dir"])
    fs = config["metadata"]["frequency"]
    frequencies, power = welch_psd(frame[channel].to_numpy(), fs)
    if len(frequencies):
        st.plotly_chart(px.line(x=frequencies, y=power, labels={"x": "Frequence (Hz)", "y": "Puissance"}, title="Welch PSD"), use_container_width=True)
    frequencies, times, values = spectrogram(frame[channel].to_numpy(), fs)
    if values.size:
        st.plotly_chart(px.imshow(10 * np.log10(values + 1e-12), x=times, y=frequencies, aspect="auto", labels={"x": "Temps (s)", "y": "Frequence (Hz)"}, title="Spectrogramme"), use_container_width=True)
    st.caption("La bande autour de 3 Hz est souvent informative pour les pointes-ondes des absences; interpretez-la avec les annotations valides.")


def help_tab() -> None:
    """Render concise lab guidance."""
    st.markdown("""### Aide\n\n- **Apercu / Overview**: explorez les signaux, les valeurs manquantes et les statistiques.\n- **Annotations et crises**: les fichiers `d` contiennent les signaux, les fichiers `h` les labels bruts. Les temps des `h` sont des index et doivent etre valides manuellement.\n- **Variable cible**: construisez la cible binaire sample par sample, prete pour les fenetres et les features.\n- **Comparaison**: observez la puissance spectrale et le spectrogramme, notamment autour de 3 Hz.\n\nLes montages et les canaux sont decouverts depuis les en-tetes. Les signaux sont limites a la fenetre visible et reduits avant tracage pour garder l'interface reactive.\n""")


def main() -> None:
    """Run the Streamlit application."""
    folder = st.session_state.get("folder", "data")
    inject_css(st.session_state.get("dark", False))
    records = cached_recordings(folder)
    config = _sidebar(records)
    if config is None:
        st.title("EEG Seizure Explorer")
        st.info("Selectionnez un dossier contenant les fichiers CSV de signaux.")
        return
    st.session_state.folder = config["folder"]
    inject_css(config["dark"])
    raw, seizures = _get_seizures(config)
    st.markdown(f"<div class='app-header'><h1>EEG Seizure Explorer</h1><div class='status'><b>{config['record'].recording_id}</b> · {layout_name(config['channels'])} · {len(config['channels'])} canal(aux)</div></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Apercu / Overview", "Annotations et crises", "Variable cible", "Comparaison", "Aide"])
    with tabs[0]:
        overview_tab(config, seizures)
    with tabs[1]:
        seizures = annotations_tab(config, raw, seizures)
    with tabs[2]:
        target_tab(config, seizures)
    with tabs[3]:
        comparison_tab(config, seizures)
    with tabs[4]:
        help_tab()