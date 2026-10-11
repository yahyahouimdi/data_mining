"""Streamlit entry point for EEG Seizure Explorer."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .src.annotations import duration_statistics, excel_seizures_for_recording, normalize_recording_id, normalize_seizure_table, pair_seizures
from .src.channels import channel_type, group_channels, layout_name
from .src.dataset import DATASET_METADATA, balance_dataset, build_recording_dataset, detect_signal_channels, feature_columns, group_split, patient_id_for_recording, split_xy
from .src.features import FEATURE_NAMES, build_feature_dataset
from .src.io_utils import annotation_source_warning, discover_recordings, read_annotations, read_csv_with_fallback, read_signal_window, signal_columns, signal_metadata
from .src.plotting import duration_figures, seizure_timeline, signal_figure, target_figure, window_class_figure
from .src.signal_utils import normalize_series, spectrogram, welch_psd
from .src.target import build_target, class_distribution, segment_signal, validate_seizures
from .src.ui import PLOTLY_CONFIG, inject_css, metric_row


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


@st.cache_data(show_spinner=False)
def cached_labeled_window(path: str, channels: tuple[str, ...], start: float, end: float, cache_dir: str, seizure_intervals: tuple[tuple[float, float], ...]):
    """Load raw selected samples and attach the validated sample-level Class."""
    frame = cached_window(path, channels, start, end, cache_dir).copy()
    seizure_frame = pd.DataFrame(seizure_intervals, columns=["Début (s)", "Fin (s)"])
    frame["Class"] = build_target(frame["Time"], seizure_frame)
    return frame


@st.cache_data(show_spinner=False)
def cached_feature_dataset(path: str, channels: tuple[str, ...], start: float, end: float, cache_dir: str, sampling_rate: float, window_duration: float, overlap: float, seizure_intervals: tuple[tuple[float, float], ...]):
    """Cache raw-sample feature extraction by recording and segmentation parameters."""
    frame = cached_labeled_window(path, channels, start, end, cache_dir, seizure_intervals)
    return build_feature_dataset(frame, list(channels), sampling_rate, window_duration, overlap)


@st.cache_data(show_spinner=False)
def cached_excel(path: str) -> pd.DataFrame:
    """Cache the small workbook used as the authoritative seizure source."""
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def cached_dataset_record(path: str, source_mtime: float, patient_id: str, recording_id: str, channels: tuple[str, ...], seizure_signature: tuple[tuple[float, float], ...], window_duration: float, overlap: float, threshold: float) -> pd.DataFrame:
    """Cache one raw recording's full-resolution feature extraction."""
    del source_mtime
    signal = read_csv_with_fallback(path, usecols=["Time", *channels])
    seizures = pd.DataFrame(seizure_signature, columns=["Début (s)", "Fin (s)"])
    return build_recording_dataset(signal, patient_id, recording_id, channels, seizures, window_duration=window_duration, overlap=overlap, label_threshold=threshold)


def _format_duration(seconds: float) -> str:
    """Format seconds as mm:ss."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _seizure_signature(seizures: pd.DataFrame, duration: float) -> tuple[tuple[float, float], ...]:
    """Return validated intervals in a hashable form for Streamlit caching."""
    valid = validate_seizures(seizures, duration)
    return tuple((float(row["Début (s)"]), float(row["Fin (s)"])) for _, row in valid.iterrows())


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
    st.sidebar.caption("Exemple de temps: 2.54 ms. Les calculs internes utilisent les secondes du fichier source.")
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
    st.sidebar.caption("Zoom graphique: molette, boutons +/-, glisser pour zoomer, double-clic pour réinitialiser.")
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
        st.session_state.seizures_initialized = False
        st.session_state.timing_validation = pd.DataFrame()
        st.session_state.annotation_source = "Fichiers h (provisoire)"
    raw = pd.DataFrame(columns=["Time", "Annotations", "Duration"])
    path = config["record"].annotation_path
    if path:
        try:
            raw = read_annotations(path)
        except (OSError, ValueError) as error:
            st.error(f"Annotation illisible: {error}")
    workbook = st.session_state.get("xlsx_data")
    workbook_source = "Annotations.xlsx (upload)"
    default_workbook = Path(config["folder"]) / "Annotations.xlsx"
    if workbook is None and default_workbook.exists():
        try:
            workbook = cached_excel(str(default_workbook))
            workbook_source = "Annotations.xlsx"
        except (OSError, ValueError, ImportError) as error:
            st.warning(f"Annotations.xlsx n'a pas pu etre lu: {error}")
    if workbook is not None and st.session_state.get("xlsx_record") != record_id:
        excel_seizures, timing_validation = excel_seizures_for_recording(workbook, record_id, config["metadata"]["duration"])
        id_column = next((column for column in workbook.columns if str(column).strip().casefold() == "unnamed: 0"), None)
        has_excel_record = id_column is not None and workbook[id_column].map(normalize_recording_id).eq(normalize_recording_id(record_id)).any()
        if has_excel_record:
            st.session_state.seizures = excel_seizures
            st.session_state.timing_validation = timing_validation
            st.session_state.annotation_source = workbook_source
            st.session_state.xlsx_record = record_id
            st.session_state.seizures_initialized = True
    if not st.session_state.get("seizures_initialized", False):
        # YO/YF are the event markers used by the supplied h files. Their
        # times remain provisional because the h-file clock is not trustworthy.
        st.session_state.seizures = pair_seizures(raw, labels=["absence", "yo", "yf"])
        st.session_state.seizures_initialized = True
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
    st.plotly_chart(signal_figure(frame, config["channels"], seizures if config["overlays"] else seizures.iloc[0:0], config["mode"], config["downsample"], "plotly_dark" if config["dark"] else "plotly_white"), use_container_width=True, config=PLOTLY_CONFIG)
    st.subheader("Vue multi-enregistrements")
    rows = [{"Enregistrement": item.recording_id, "d": item.signal_path is not None, "h": item.annotation_path is not None} for item in cached_recordings(config["folder"])]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def annotations_tab(config: dict, raw: pd.DataFrame, seizures: pd.DataFrame) -> pd.DataFrame:
    """Render Part II and return the edited seizure table."""
    source = st.session_state.get("annotation_source", "Fichiers h (provisoire)")
    if source.startswith("Annotations.xlsx"):
        st.success(f"Source des crises : {source}")
    else:
        st.warning(annotation_source_warning())
    timing_validation = st.session_state.get("timing_validation", pd.DataFrame())
    if not timing_validation.empty:
        with st.expander("🔍 Vérification de la synchronisation", expanded=True):
            metric_row([("Début du tracé", str(timing_validation.iloc[0]["Début du tracé"])), ("Durée EEG", f"{config['metadata']['duration']:.3f} s"), ("Fréquence", f"{config['metadata']['frequency']:.2f} Hz"), ("Nombre de crises", str(len(timing_validation)))])
            st.dataframe(timing_validation, use_container_width=True, hide_index=True)
            if (timing_validation["Statut"] != "Valide").any():
                st.warning("Un ou plusieurs intervalles Excel dépassent la durée du signal et ne sont pas utilisés pour les overlays ou Class.")
            else:
                st.success(f"Première crise: {timing_validation.iloc[0]['Début (s)']:.3f} s · Dernière crise: {timing_validation.iloc[-1]['Fin (s)']:.3f} s")
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
        st.plotly_chart(seizure_timeline(edited), use_container_width=True, config=PLOTLY_CONFIG)
        bar, histogram = duration_figures(edited)
        left, right = st.columns(2)
        left.plotly_chart(bar, use_container_width=True, config=PLOTLY_CONFIG)
        right.plotly_chart(histogram, use_container_width=True, config=PLOTLY_CONFIG)
        conclusion = "similaires" if stats["cv"] < 0.25 else "variables"
        st.info(f"Coefficient de variation: {stats['cv']:.3f}. Les durees sont {conclusion} (seuil CV < 0.25).")
        st.download_button("Exporter les crises CSV", edited.to_csv(index=False).encode("utf-8"), f"{config['record'].recording_id}_seizures.csv", "text/csv")
    return edited


def target_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Render Part III and labeled-data export."""
    validated = validate_seizures(seizures, config["metadata"]["duration"])
    if validated.empty:
        st.warning("Aucune crise valide: validez les intervalles dans l'onglet des annotations avant de construire Class.")
    signature = _seizure_signature(validated, config["metadata"]["duration"])
    frame = cached_labeled_window(str(config["record"].signal_path), tuple(config["channels"]), *config["time_range"], config["cache_dir"], signature)
    distribution = class_distribution(frame["Class"])
    st.subheader("Construction de la variable cible")
    st.caption("Class = 0 hors crise · Class = 1 pendant une crise. Les intervalles sont fermes: Début <= Time <= Fin.")
    metric_row([("Classe 0 - Hors crise", f"{distribution['class_0']:,} ({distribution['percentage_0']:.1%})"), ("Classe 1 - Crise", f"{distribution['class_1']:,} ({distribution['percentage_1']:.1%})"), ("Nombre d'échantillons", f"{distribution['total']:,}"), ("Ratio de déséquilibre", f"{distribution['imbalance_ratio']:.1f}x" if np.isfinite(distribution["imbalance_ratio"]) else "inf")])
    if distribution["percentage_1"] < 0.20 or distribution["percentage_0"] < 0.20:
        st.warning("Le dataset présente un déséquilibre important entre les deux classes (seuil: classe minoritaire < 20%).")
    range_start, range_end = config["time_range"]
    seizure_time = sum(max(0.0, min(end, range_end) - max(start, range_start)) for start, end in signature)
    recording_time = max(0.0, range_end - range_start)
    metric_row([("Temps total des crises", f"{seizure_time:.2f} s"), ("Durée de l'enregistrement", f"{recording_time:.2f} s"), ("Temps en crise", f"{seizure_time / recording_time:.1%}" if recording_time else "0.0%")])
    donut = px.pie(names=["Hors crise", "Crise"], values=[distribution["class_0"], distribution["class_1"]], hole=0.58, color=["Hors crise", "Crise"], color_discrete_map={"Hors crise": "#0e7490", "Crise": "#d64550"}, title="Distribution de Class")
    st.plotly_chart(donut, use_container_width=True, config=PLOTLY_CONFIG)
    channel = st.selectbox("Canal cible", config["channels"])
    st.plotly_chart(target_figure(frame, channel, validated), use_container_width=True, config=PLOTLY_CONFIG)
    export_columns = ["Time", *config["channels"], "Class"]
    export_frame = frame[export_columns]
    st.download_button("Télécharger le dataset échantillon par échantillon - CSV", export_frame.to_csv(index=False).encode("utf-8"), f"{config['record'].recording_id}_sample_dataset.csv", "text/csv")
    buffer = io.BytesIO()
    export_frame.to_parquet(buffer, index=False)
    st.download_button("Télécharger le dataset échantillon par échantillon - Parquet", buffer.getvalue(), f"{config['record'].recording_id}_sample_dataset.parquet", "application/octet-stream")


def segmentation_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Render Part IV window segmentation and window-level labels."""
    st.subheader("Partie IV - Segmentation du signal")
    st.caption("Une fenêtre est classée Crise si au moins 50% de ses échantillons ont Class = 1.")
    duration_choice = st.selectbox("Durée de fenêtre", [1, 2, 5, 10, 30, "Personnalisée"], index=1, format_func=lambda value: f"{value} s" if value != "Personnalisée" else value, key="segmentation_duration")
    window_duration = float(st.number_input("Durée personnalisée (s)", min_value=0.1, value=2.0, step=0.1, disabled=duration_choice != "Personnalisée", key="custom_window_duration")) if duration_choice == "Personnalisée" else float(duration_choice)
    overlap_label = st.selectbox("Recouvrement", ["0 %", "25 %", "50 %", "75 %"], index=2, key="segmentation_overlap")
    overlap = int(overlap_label.rstrip(" %")) / 100
    st.session_state.segmentation_parameters = (window_duration, overlap)
    validated = validate_seizures(seizures, config["metadata"]["duration"])
    signature = _seizure_signature(validated, config["metadata"]["duration"])
    eeg_channels = [channel for channel in config["channels"] if channel_type(channel) == "EEG"]
    frame_channels = ["Time", *eeg_channels]
    if not eeg_channels:
        st.warning("Sélectionnez au moins un canal EEG pour segmenter le signal.")
        return
    with st.spinner("Construction des fenêtres à partir des échantillons bruts..."):
        frame = cached_labeled_window(str(config["record"].signal_path), tuple(eeg_channels), *config["time_range"], config["cache_dir"], signature)
        windows = segment_signal(frame[["Time", "Class"]], config["metadata"]["frequency"], window_duration, overlap)
    samples_per_window = max(1, round(window_duration * config["metadata"]["frequency"]))
    step_samples = max(1, round(samples_per_window * (1 - overlap)))
    distribution = class_distribution(windows["Class"] if not windows.empty else np.array([], dtype=np.uint8))
    metric_row([("Fréquence d'échantillonnage", f"{config['metadata']['frequency']:.2f} Hz"), ("Échantillons par fenêtre", str(samples_per_window)), ("Pas entre fenêtres", str(step_samples)), ("Nombre de fenêtres", f"{len(windows):,}")])
    metric_row([("Fenêtres classe 0", f"{distribution['class_0']:,} ({distribution['percentage_0']:.1%})"), ("Fenêtres classe 1", f"{distribution['class_1']:,} ({distribution['percentage_1']:.1%})")])
    if windows.empty:
        st.info("La plage sélectionnée est trop courte pour créer une fenêtre complète.")
        return
    st.plotly_chart(window_class_figure(windows), use_container_width=True, config=PLOTLY_CONFIG)
    st.dataframe(windows, use_container_width=True, hide_index=True)
    st.download_button("Télécharger fenêtres CSV", windows.to_csv(index=False).encode("utf-8"), f"{config['record'].recording_id}_windows.csv", "text/csv")
    window_buffer = io.BytesIO()
    windows.to_parquet(window_buffer, index=False)
    st.download_button("Télécharger fenêtres Parquet", window_buffer.getvalue(), f"{config['record'].recording_id}_windows.parquet", "application/octet-stream")


def features_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Render Part V temporal feature extraction and exploratory plots."""
    st.subheader("Partie V - Extraction des caractéristiques")
    st.caption("Les caractéristiques sont calculées sur les échantillons bruts, jamais sur les points réduits pour l'affichage.")
    window_duration, overlap = st.session_state.get("segmentation_parameters", (2.0, 0.50))
    st.info(f"Paramètres actifs: fenêtre {window_duration:g} s · recouvrement {overlap:.0%}")
    validated = validate_seizures(seizures, config["metadata"]["duration"])
    signature = _seizure_signature(validated, config["metadata"]["duration"])
    eeg_channels = [channel for channel in config["channels"] if channel_type(channel) == "EEG"]
    if not eeg_channels:
        st.warning("Sélectionnez au moins un canal EEG pour extraire les caractéristiques.")
        return
    with st.spinner("Extraction des huit caractéristiques temporelles..."):
        features = cached_feature_dataset(str(config["record"].signal_path), tuple(eeg_channels), *config["time_range"], config["cache_dir"], config["metadata"]["frequency"], window_duration, overlap, signature)
    distribution = class_distribution(features["Class"] if not features.empty else np.array([], dtype=np.uint8))
    metric_row([("Nombre total de fenêtres", f"{distribution['total']:,}"), ("Fenêtres classe 0", f"{distribution['class_0']:,}"), ("Fenêtres classe 1", f"{distribution['class_1']:,}"), ("Classe 0 / classe 1", f"{distribution['percentage_0']:.1%} / {distribution['percentage_1']:.1%}")])
    if features.empty:
        st.info("Aucune fenêtre complète disponible pour calculer les caractéristiques.")
        return
    st.dataframe(features.head(100), use_container_width=True, hide_index=True)
    feature_channel = st.selectbox("Canal EEG", eeg_channels, key="feature_channel")
    feature_name = st.selectbox("Caractéristique temporelle", list(FEATURE_NAMES), format_func=lambda value: {"Mean": "Moyenne", "Std": "Écart-type", "Variance": "Variance", "Min": "Minimum", "Max": "Maximum", "Amplitude": "Amplitude", "RMS": "RMS", "Energy": "Énergie"}[value], key="feature_name")
    selected_column = f"{feature_channel}_{feature_name}"
    box = px.box(features, x="Class", y=selected_column, color="Class", points=False, color_discrete_map={0: "#0e7490", 1: "#d64550"}, labels={"Class": "Classe", selected_column: feature_name}, title=f"{feature_name} par classe - {feature_channel}")
    st.plotly_chart(box, use_container_width=True, config=PLOTLY_CONFIG)
    rms_column = f"{feature_channel}_RMS"
    energy_column = f"{feature_channel}_Energy"
    scatter = px.scatter(features, x=rms_column, y=energy_column, color="Class", color_discrete_map={0: "#0e7490", 1: "#d64550"}, labels={rms_column: "RMS", energy_column: "Énergie"}, title=f"RMS vs Énergie - {feature_channel}")
    st.plotly_chart(scatter, use_container_width=True, config=PLOTLY_CONFIG)
    st.download_button("Télécharger caractéristiques CSV", features.to_csv(index=False).encode("utf-8"), f"{config['record'].recording_id}_features.csv", "text/csv")
    feature_buffer = io.BytesIO()
    features.to_parquet(feature_buffer, index=False)
    st.download_button("Télécharger caractéristiques Parquet", feature_buffer.getvalue(), f"{config['record'].recording_id}_features.parquet", "application/octet-stream")


def _dataset_annotations(config: dict, record, workbook: pd.DataFrame | None, allow_provisional: bool) -> tuple[pd.DataFrame, list[str]]:
    """Resolve authoritative Excel intervals, or explicitly opt into h labels."""
    warnings: list[str] = []
    if workbook is not None:
        seizures, validation = excel_seizures_for_recording(workbook, record.recording_id, config["metadata"]["duration"])
        if not validation.empty and (validation["Statut"] != "Valide").any():
            warnings.append(f"{record.recording_id}: certains intervalles Excel sont invalides ou hors durée.")
        if not seizures.empty:
            return validate_seizures(seizures, config["metadata"]["duration"]), warnings
        warnings.append(f"{record.recording_id}: aucune annotation Excel valide.")
    if allow_provisional and record.annotation_path:
        try:
            provisional = pair_seizures(read_annotations(record.annotation_path), labels=["absence", "yo", "yf"])
            if not provisional.empty:
                warnings.append(f"{record.recording_id}: labels h provisoires utilisés explicitement.")
                return validate_seizures(provisional, config["metadata"]["duration"]), warnings
        except (OSError, ValueError) as error:
            warnings.append(f"{record.recording_id}: annotation h illisible ({error}).")
    return pd.DataFrame(columns=["Crise", "Début (s)", "Fin (s)", "Durée (s)" ]), warnings


def _download_dataset(frame: pd.DataFrame, stem: str) -> None:
    """Render consistent CSV and optional Parquet downloads."""
    st.download_button("Exporter CSV", frame.to_csv(index=False).encode("utf-8"), f"{stem}.csv", "text/csv", key=f"csv_{stem}")
    try:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        st.download_button("Exporter Parquet", buffer.getvalue(), f"{stem}.parquet", "application/octet-stream", key=f"parquet_{stem}")
    except (ImportError, ValueError):
        st.caption("Parquet indisponible: installez pyarrow.")


def window_features_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Generate, balance, inspect, split, and export the supervised ML dataset."""
    st.subheader("Window Features & Dataset")
    st.caption("Fenêtres complètes de 2 s, numérotées à partir de 1, calculées sur les échantillons bruts.")
    records = cached_recordings(config["folder"])
    scope = st.radio("Portée", ["Enregistrement sélectionné", "Tous les enregistrements"], horizontal=True, key="dataset_scope")
    source_records = [config["record"]] if scope == "Enregistrement sélectionné" else records
    all_channels = detect_signal_channels(config["metadata"]["columns"])
    eeg_defaults = [channel for channel in all_channels if channel_type(channel) == "EEG"]
    channels = st.multiselect("Canaux", all_channels, default=eeg_defaults, key="dataset_channels")
    left, middle, right = st.columns(3)
    window_duration = left.number_input("Durée fenêtre (s)", min_value=0.1, value=2.0, step=0.1, key="dataset_duration")
    overlap = middle.slider("Recouvrement", 0.0, 0.95, 0.50, 0.05, key="dataset_overlap")
    threshold = right.slider("Seuil de recouvrement crise", 0.0, 1.0, 0.50, 0.05, key="dataset_threshold")
    workbook = st.session_state.get("xlsx_data")
    default_workbook = Path(config["folder"]) / "Annotations.xlsx"
    if workbook is None and default_workbook.exists():
        try:
            workbook = cached_excel(str(default_workbook))
        except (OSError, ValueError, ImportError) as error:
            st.warning(f"Annotations.xlsx n'a pas pu être lu: {error}")
    allow_provisional = st.checkbox("Autoriser les annotations h provisoires si Excel manque", value=False)
    if workbook is None and not allow_provisional:
        st.warning("Aucune annotation Excel validée n'est disponible. Les enregistrements seront exclus; importez Annotations.xlsx ou activez explicitement les labels h provisoires.")
    if not channels:
        st.warning("Sélectionnez au moins un canal.")
        return
    generate = st.button("Générer / actualiser le dataset", type="primary", use_container_width=True)
    if generate:
        frames: list[pd.DataFrame] = []
        warning_messages: list[str] = []
        progress = st.progress(0.0)
        for index, record in enumerate(source_records):
            if record.signal_path is None:
                warning_messages.append(f"{record.recording_id}: signal d manquant.")
                continue
            record_config = {**config, "record": record}
            if record is not config["record"]:
                try:
                    record_config["metadata"] = cached_metadata(str(record.signal_path), config["cache_dir"])
                except (OSError, ValueError) as error:
                    warning_messages.append(f"{record.recording_id}: métadonnées invalides ({error}).")
                    continue
            record_seizures, annotation_warnings = _dataset_annotations(record_config, record, workbook, allow_provisional)
            warning_messages.extend(annotation_warnings)
            if record_seizures.empty:
                warning_messages.append(f"{record.recording_id}: exclu du dataset supervisé faute d'intervalle valide.")
                continue
            selected = [channel for channel in channels if channel in record_config["metadata"]["columns"]]
            try:
                signature = _seizure_signature(record_seizures, float(record_config["metadata"]["duration"]))
                frames.append(cached_dataset_record(str(record.signal_path), record.signal_path.stat().st_mtime, patient_id_for_recording(record.recording_id), record.recording_id, tuple(selected), signature, float(window_duration), float(overlap), float(threshold)))
            except (OSError, ValueError, KeyError) as error:
                warning_messages.append(f"{record.recording_id}: dataset impossible ({error}).")
            progress.progress((index + 1) / max(1, len(source_records)))
        dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*DATASET_METADATA, *feature_columns(channels), "label"])
        st.session_state.dataset_feature_channels = tuple(channels)
        st.session_state.window_dataset = dataset
        st.session_state.window_dataset_warnings = warning_messages
        st.session_state.balanced_dataset = None
    dataset = st.session_state.get("window_dataset")
    if dataset is None:
        st.info("Configurez les paramètres puis lancez la génération.")
        return
    for warning in st.session_state.get("window_dataset_warnings", []):
        st.warning(warning)
    if dataset.empty:
        st.error("Aucune fenêtre supervisée n'a été générée.")
        return
    distribution = class_distribution(dataset["label"])
    metric_row([("Fenêtres", f"{len(dataset):,}"), ("Features", f"{len(dataset.columns) - 6:,}"), ("Classe 0", f"{distribution['class_0']:,} ({distribution['percentage_0']:.1%})"), ("Classe 1", f"{distribution['class_1']:,} ({distribution['percentage_1']:.1%})"), ("Patients / enregistrements", f"{dataset['patient_id'].nunique()} / {dataset['recording_id'].nunique()}")])
    st.dataframe(dataset.head(100), use_container_width=True, hide_index=True)
    st.subheader("Balance Dataset")
    balance_method = st.selectbox("Méthode", ["none", "undersample", "oversample", "hybrid"], format_func=lambda value: {"none": "Aucune", "undersample": "Random undersampling", "oversample": "Random oversampling", "hybrid": "Hybrid sampling"}[value])
    balance_scope = st.selectbox("Portée du balancing", ["Global", "Par patient"])
    seed = int(st.number_input("Seed", min_value=0, value=42, step=1))
    default_target = max(1, min(distribution["class_0"], distribution["class_1"]))
    target_count = int(st.number_input("Fenêtres par classe (hybrid/optionnel)", min_value=1, value=default_target, step=1))
    if st.button("Appliquer le balancing", key="apply_balance"):
        if balance_scope == "Global":
            balanced, report = balance_dataset(dataset, balance_method, target_per_class=target_count, seed=seed)
            st.session_state.balanced_dataset = balanced
            st.session_state.balance_report = report
        else:
            parts: list[pd.DataFrame] = []
            for patient, group in dataset.groupby("patient_id", sort=False):
                if group["label"].nunique() < 2:
                    st.warning(f"Patient {patient}: une seule classe, balancing ignoré.")
                    parts.append(group)
                else:
                    parts.append(balance_dataset(group, balance_method, target_per_class=target_count, seed=seed)[0])
            st.session_state.balanced_dataset = pd.concat(parts, ignore_index=True)
    balanced = st.session_state.get("balanced_dataset")
    if balanced is not None:
        balanced_distribution = class_distribution(balanced["label"])
        report = st.session_state.get("balance_report")
        metric_row([("Final classe 0", str(balanced_distribution["class_0"])), ("Final classe 1", str(balanced_distribution["class_1"])), ("Taille finale", f"{len(balanced):,}"), ("Doublons", str(report.duplicated_windows if report else "par patient"))])
        st.caption("L'oversampling duplique des fenêtres existantes; il ne doit être appliqué qu'au train lors d'une évaluation.")
    st.subheader("Splits anti-fuite")
    split_group = st.selectbox("Grouper par", ["patient", "recording"], format_func=lambda value: "Patient" if value == "patient" else "Enregistrement")
    if st.button("Créer train / validation / test", key="create_split"):
        split_frame, assignments = group_split(dataset, group_by=split_group, seed=seed)
        st.session_state.window_split = split_frame
        st.session_state.window_split_assignments = assignments
    split_frame = st.session_state.get("window_split")
    if split_frame is not None:
        st.info(f"Split par {split_group}: aucun groupe n'est partagé entre train, validation et test. Appliquez le balancing uniquement au train; les fenêtres adjacentes restent corrélées.")
        st.dataframe(split_frame.groupby("split").size().rename("windows").reset_index(), hide_index=True)
    st.subheader("Visualisations et inspection")
    plot_left, plot_right = st.columns(2)
    plot_left.plotly_chart(px.histogram(dataset, x="label", color="label", title="Distribution avant balancing"), use_container_width=True, config=PLOTLY_CONFIG)
    if balanced is not None:
        plot_right.plotly_chart(px.histogram(balanced, x="label", color="label", title="Distribution après balancing"), use_container_width=True, config=PLOTLY_CONFIG)
    feature_names = feature_columns(list(st.session_state.get("dataset_feature_channels", channels)))
    selected_feature = st.selectbox("Feature à comparer", feature_names)
    st.plotly_chart(px.box(dataset, x="label", y=selected_feature, color="label", points=False, title=f"{selected_feature} par classe"), use_container_width=True, config=PLOTLY_CONFIG)
    patient_counts = dataset.groupby(["patient_id", "label"]).size().reset_index(name="windows")
    st.plotly_chart(px.bar(patient_counts, x="patient_id", y="windows", color="label", barmode="group", title="Fenêtres par patient"), use_container_width=True, config=PLOTLY_CONFIG)
    selected_index = st.selectbox("Fenêtre à inspecter", list(dataset.index), format_func=lambda index: f"{dataset.loc[index, 'recording_id']} · fenêtre {dataset.loc[index, 'window_id']} · label {dataset.loc[index, 'label']}")
    selected_row = dataset.loc[selected_index]
    record = next(item for item in records if item.recording_id == selected_row["recording_id"])
    inspection = cached_window(str(record.signal_path), (channels[0],), float(selected_row["start_time"]), float(selected_row["end_time"]), config["cache_dir"])
    inspection_config = config if record.recording_id == config["record"].recording_id else {**config, "record": record, "metadata": cached_metadata(str(record.signal_path), config["cache_dir"])}
    inspection_seizures, _ = _dataset_annotations(inspection_config, record, workbook, allow_provisional)
    inspection_fig = signal_figure(inspection, [channels[0]], inspection_seizures, "stacked", 5000, "plotly_white")
    inspection_fig.add_vrect(x0=float(selected_row["start_time"]), x1=float(selected_row["end_time"]), fillcolor="#d64550", opacity=0.15, line_width=1)
    st.plotly_chart(inspection_fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(px.bar(dataset.groupby("recording_id").size().rename("windows").reset_index(), x="recording_id", y="windows", title="Fenêtres par enregistrement"), use_container_width=True, config=PLOTLY_CONFIG)
    st.subheader("Exports")
    whole_name = "eeg_features_all_patients" if scope == "Tous les enregistrements" else f"eeg_features_{config['record'].recording_id}"
    export_columns = st.columns(2)
    with export_columns[0]:
        st.markdown("**Whole dataset**")
        st.caption("Complete unbalanced feature dataset")
        _download_dataset(dataset, whole_name)
    with export_columns[1]:
        st.markdown("**Balanced dataset**")
        if balanced is None:
            st.caption("Apply balancing above to enable this export.")
        else:
            st.caption("Dataset after the selected balancing method")
            _download_dataset(balanced, "eeg_features_balanced")
    st.markdown("**Train / validation / test datasets**")
    st.caption("Group-isolated files for machine-learning evaluation. Generate the split above first.")
    if split_frame is None:
        st.info("Create a train / validation / test split above to enable these exports.")
    else:
        split_columns = st.columns(3)
        for column, split_name in zip(split_columns, ("train", "validation", "test")):
            split_part = split_frame.loc[split_frame["split"].eq(split_name)]
            with column:
                st.markdown(f"**{split_name.title()} dataset**")
                if split_part.empty:
                    st.caption("No windows in this split.")
                else:
                    _download_dataset(split_part, f"eeg_features_{split_name}")
    st.markdown("**Feature matrix X**")
    st.caption("Only predictive feature columns; identifiers and label are excluded.")
    x, y = split_xy(dataset)
    _download_dataset(x, "eeg_features_X")
    st.markdown("**Target vector y**")
    st.caption("The binary target label for each window.")
    _download_dataset(y.to_frame(), "eeg_features_y")
    st.markdown("**Compact dataset**")
    st.caption("Patient ID, window ID, all features, and label; timing and recording metadata are excluded.")
    compact_columns = ["patient_id", "window_id", *feature_names, "label"]
    _download_dataset(dataset[compact_columns], "eeg_features_compact")
    st.markdown("**Individual recording dataset**")
    export_recording = st.selectbox("Choose recording", sorted(dataset["recording_id"].unique()), key="export_recording")
    recording_dataset = dataset.loc[dataset["recording_id"].eq(export_recording)]
    _download_dataset(recording_dataset, f"eeg_features_recording_{export_recording}")


def comparison_tab(config: dict, seizures: pd.DataFrame) -> None:
    """Render the optional signal comparison and spectral analysis."""
    channel = st.selectbox("Canal a comparer", config["channels"], key="compare_channel")
    frame = cached_window(str(config["record"].signal_path), (channel,), *config["time_range"], config["cache_dir"])
    fs = config["metadata"]["frequency"]
    frequencies, power = welch_psd(frame[channel].to_numpy(), fs)
    if len(frequencies):
        st.plotly_chart(px.line(x=frequencies, y=power, labels={"x": "Frequence (Hz)", "y": "Puissance"}, title="Welch PSD"), use_container_width=True, config=PLOTLY_CONFIG)
    frequencies, times, values = spectrogram(frame[channel].to_numpy(), fs)
    if values.size:
        st.plotly_chart(px.imshow(10 * np.log10(values + 1e-12), x=times, y=frequencies, aspect="auto", labels={"x": "Temps (s)", "y": "Frequence (Hz)"}, title="Spectrogramme"), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("La bande autour de 3 Hz est souvent informative pour les pointes-ondes des absences; interpretez-la avec les annotations valides.")


def help_tab() -> None:
    """Render concise lab guidance."""
    st.markdown("""### Aide

- **Apercu / Overview**: explorez les signaux, les valeurs manquantes et les statistiques.
- **Annotations et crises**: les fichiers `d` contiennent les signaux, les fichiers `h` les labels bruts. Les temps des `h` sont des index et doivent etre valides manuellement.
- **Variable cible**: construisez `Class`, avec 0 hors crise et 1 pendant une crise.
- **Segmentation**: transformez les echantillons bruts en fenetres temporelles chevauchantes.
- **Caracteristiques**: calculez le dataset ML avec Mean, Std, Variance, Min, Max, Amplitude, RMS et Energy.
- **Comparaison**: observez la puissance spectrale et le spectrogramme, notamment autour de 3 Hz.

Les montages et les canaux sont decouverts depuis les en-tetes. Les signaux sont limites a la fenetre visible et reduits avant tracage pour garder l'interface reactive.
""")


def pipeline_view() -> None:
    """Display the lab pipeline at the top of the application."""
    st.markdown("""<div class='pipeline'><b>RAW EEG</b> &darr; <b>Annotations</b> &darr; <b>Class</b> (0 hors crise / 1 crise) &darr; <b>Fenêtres temporelles</b> (2 s, 50%) &darr; <b>Caractéristiques temporelles</b> &darr; <b>Dataset ML</b> &darr; <b>Classification future</b></div>""", unsafe_allow_html=True)


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
    st.markdown(f"<div class='app-header'><h1>EEG Seizure Explorer</h1><div class='status'><b>{config['record'].recording_id}</b> · {layout_name(config['channels'])} · {len(config['channels'])} canal(aux) · <b>{len(seizures)} crise(s)</b> · temps en secondes</div></div>", unsafe_allow_html=True)
    pipeline_view()
    tabs = st.tabs(["Apercu / Overview", "Annotations et crises", "Partie III - Variable cible", "Partie IV - Segmentation", "Partie V - Caracteristiques", "Window Features & Dataset", "Comparaison", "Aide"])
    with tabs[0]:
        overview_tab(config, seizures)
    with tabs[1]:
        seizures = annotations_tab(config, raw, seizures)
    with tabs[2]:
        target_tab(config, seizures)
    with tabs[3]:
        segmentation_tab(config, seizures)
    with tabs[4]:
        features_tab(config, seizures)
    with tabs[5]:
        window_features_tab(config, seizures)
    with tabs[6]:
        comparison_tab(config, seizures)
    with tabs[7]:
        help_tab()