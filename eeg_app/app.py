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
    tabs = st.tabs(["Apercu / Overview", "Annotations et crises", "Partie III - Variable cible", "Partie IV - Segmentation", "Partie V - Caracteristiques", "Comparaison", "Aide"])
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
        comparison_tab(config, seizures)
    with tabs[6]:
        help_tab()