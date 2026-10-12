"""Reusable, leakage-aware experimentation helpers for the EEG notebooks."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from .annotations import excel_seizures_for_recording
from .channels import channel_type
from .dataset import DATASET_METADATA, balance_dataset, build_recording_dataset, dataset_audit, detect_signal_channels, feature_columns, group_split, representative_group_split
from .io_utils import discover_recordings, read_csv_with_fallback, signal_metadata
from .signal_utils import sampling_frequency

DEFAULT_SEED = 42
BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)}
IDENTIFIER_COLUMNS = ["patient_id", "recording_id", "window_id", "start_time", "end_time"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def experiment_dirs(root: str | Path | None = None) -> dict[str, Path]:
    base = Path(root) if root else project_root()
    return {name: base / name for name in ("extracted_data", "models", "results")}


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _recording_signal(root: Path, recording_id: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = root / "data" / f"{recording_id}_0000d.csv"
    if not path.exists():
        path = root / "data" / f"{recording_id.rstrip('_')}_0000d.csv"
    if not path.exists():
        raise FileNotFoundError(f"Signal CSV not found for {recording_id}: {path}")
    cache_path = root / "data" / ".cache" / f"{path.stem}.parquet"
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache_path, columns=list(columns) if columns else None)
    return read_csv_with_fallback(path, usecols=list(columns) if columns else None)


def build_supervised_dataset(root: str | Path | None = None, window_duration: float = 2.0, overlap: float = 0.5, label_threshold: float = 0.5) -> tuple[pd.DataFrame, list[str]]:
    """Build validated temporal features from raw signals and authoritative Excel labels."""
    base = Path(root) if root else project_root()
    annotation_path = base / "data" / "Annotations.xlsx"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Validated annotation workbook is required: {annotation_path}")
    workbook = pd.read_excel(annotation_path)
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for record in discover_recordings(base / "data"):
        if record.signal_path is None:
            warnings.append(f"{record.recording_id}: signal missing")
            continue
        metadata = signal_metadata(record.signal_path, base / "data" / ".cache")
        seizures, validation = excel_seizures_for_recording(workbook, record.recording_id, float(metadata["duration"]))
        if validation.empty or (validation["Statut"] != "Valide").any() or seizures.empty:
            warnings.append(f"{record.recording_id}: excluded because validated Excel intervals are missing or invalid")
            continue
        channels = [column for column in detect_signal_channels(metadata["columns"]) if channel_type(column) == "EEG"]
        if not channels:
            warnings.append(f"{record.recording_id}: no EEG channels")
            continue
        signal = _recording_signal(base, record.recording_id, ["Time", *channels])
        patient_id = record.recording_id
        frames.append(build_recording_dataset(signal, patient_id, record.recording_id, channels, seizures, window_duration=window_duration, overlap=overlap, label_threshold=label_threshold))
    if not frames:
        raise RuntimeError("No supervised rows were produced from validated annotations")
    return pd.concat(frames, ignore_index=True), warnings


def band_powers(windows: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """FFT band powers for a (windows, samples) block of raw samples.

    Each window is mean-centred, its periodogram is computed with the real FFT,
    and the power spectral density is integrated over every band in ``BANDS``.
    """
    windows = np.asarray(windows, dtype=float)
    samples = windows.shape[1]
    centered = np.nan_to_num(windows - np.nanmean(windows, axis=1, keepdims=True))
    frequencies = np.fft.rfftfreq(samples, d=1.0 / fs)
    power = 2 * (np.abs(np.fft.rfft(centered, axis=1)) ** 2) / (fs * samples)  # one-sided PSD
    result: dict[str, np.ndarray] = {}
    for band, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        result[band] = np.trapezoid(power[:, mask], frequencies[mask], axis=1) if mask.sum() >= 2 else power[:, mask].sum(axis=1) * fs / samples
    return result


def add_frequency_features(frame: pd.DataFrame, root: str | Path | None = None) -> pd.DataFrame:
    """Add raw-sample FFT band powers (Delta, Theta, Alpha, Beta) to each window row."""
    result = frame.reset_index(drop=True)
    base = Path(root) if root else project_root()
    all_channels = [column[: -len("_Mean")] for column in result.columns if column.endswith("_Mean")]
    columns = {f"{channel}_{band}_Power": np.full(len(result), np.nan) for channel in all_channels for band in BANDS}
    for recording_id, group in result.groupby("recording_id", sort=False):
        channels = [channel for channel in all_channels if group[f"{channel}_Mean"].notna().any()]
        signal = _recording_signal(base, str(recording_id), ["Time", *channels])
        time = pd.to_numeric(signal["Time"], errors="coerce").to_numpy(float)
        fs = sampling_frequency(time)
        starts = np.searchsorted(time, group["start_time"].to_numpy(float), side="left")
        ends = np.searchsorted(time, group["end_time"].to_numpy(float), side="right")
        samples = int(np.median(ends - starts))
        index = np.minimum(starts[:, None] + np.arange(samples), len(time) - 1)
        for channel in channels:
            values = pd.to_numeric(signal[channel], errors="coerce").to_numpy(float)
            for band, power in band_powers(values[index], fs).items():
                columns[f"{channel}_{band}_Power"][group.index] = power
    result = pd.concat([result.drop(columns="label"), pd.DataFrame(columns), result["label"]], axis=1)
    return result


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in (*IDENTIFIER_COLUMNS, "label", "split")]


def validate_dataset(frame: pd.DataFrame) -> dict[str, object]:
    required = [*IDENTIFIER_COLUMNS, "label"]
    missing_columns = [column for column in required if column not in frame]
    duplicate_keys = int(frame.duplicated(IDENTIFIER_COLUMNS).sum()) if not missing_columns else -1
    return {"rows": int(len(frame)), "columns": list(frame.columns), "missing_required_columns": missing_columns, "missing_values": int(frame.isna().sum().sum()), "duplicate_windows": duplicate_keys, "class_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()} if "label" in frame else {}, "recordings": int(frame["recording_id"].nunique()) if "recording_id" in frame else 0, "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else 0}


def make_shared_split(frame: pd.DataFrame, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Create the one shared split; use patients, recordings, or chronological blocks."""
    if frame["patient_id"].nunique() > 1:
        split, assignments = group_split(frame, group_by="patient", train_size=0.64, validation_size=0.16, seed=seed)
        return split, assignments, "patient"
    if frame["recording_id"].nunique() > 1:
        split, assignments = group_split(frame, group_by="recording", train_size=0.64, validation_size=0.16, seed=seed)
        return split, assignments, "recording"
    ordered = frame.sort_values(["recording_id", "start_time"]).copy()
    n = len(ordered)
    train_end, validation_end = int(n * 0.64), int(n * 0.80)
    ordered["split"] = ["train" if i < train_end else "validation" if i < validation_end else "test" for i in range(n)]
    return ordered.sort_index(), {"single_recording": "chronological_blocks"}, "chronological_blocks"


def save_json(value: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def prepare_data(root: str | Path | None = None, seed: int = DEFAULT_SEED) -> dict[str, object]:
    dirs = experiment_dirs(root)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    dataset, warnings = build_supervised_dataset(root)
    dataset = add_frequency_features(dataset, root)
    split, assignments, split_group = representative_group_split(dataset, group_by="recording", test_size=0.20, validation_size_within_development=0.20, seed=seed)
    extracted = dirs["extracted_data"]
    dataset.to_parquet(extracted / "window_features_unbalanced.parquet", index=False)
    dataset.to_csv(extracted / "window_features_unbalanced.csv", index=False)
    split.to_parquet(extracted / "window_features_with_split.parquet", index=False)
    split.to_csv(extracted / "window_features_with_split.csv", index=False)
    for split_name in ("train", "validation", "test"):
        split_part = split.loc[split["split"].eq(split_name)].drop(columns="split")
        split_part.to_parquet(extracted / f"{split_name}_unbalanced.parquet", index=False)
        split_part.to_csv(extracted / f"{split_name}_unbalanced.csv", index=False)
    audit_rows = [dataset_audit(dataset, "original")]
    for split_name in ("train", "validation", "test"):
        audit_rows.append(dataset_audit(split.loc[split["split"].eq(split_name)], split_name, split_name=split_name))
    for method in ("none", "undersample", "oversample", "hybrid"):
        train = split.loc[split["split"].eq("train")].drop(columns="split")
        balanced, report = balance_dataset(train, method, seed=seed)
        audit_rows.append(dataset_audit(balanced, f"train_{method}", split_name="train", balanced=method != "none"))
        balanced.to_parquet(extracted / f"train_{method}.parquet", index=False)
        balanced.to_csv(extracted / f"train_{method}.csv", index=False)
        save_json(report.__dict__, extracted / f"train_{method}_report.json")
    pd.DataFrame(audit_rows).to_csv(extracted / "balancing_audit_report.csv", index=False)
    save_json({"seed": seed, "split_group": split_group, "split_sizes": split["split"].value_counts().to_dict(), "warnings": warnings, "validation": validate_dataset(dataset), "frequency_method": "FFT periodogram of each mean-centred window; band power is the trapezoidal integral of the PSD over each named band", "bands_hz": BANDS}, extracted / "preparation_report.json")
    save_json(assignments, extracted / "split_assignments.json")
    return {"dataset": dataset, "split": split, "warnings": warnings, "split_group": split_group}


def prepare_balanced_exports(root: str | Path | None = None, seed: int = DEFAULT_SEED, durations: Iterable[float] = (1.0, 2.0), methods: Iterable[str] = ("undersample", "oversample", "hybrid"), include_frequency: bool = True) -> dict[str, object]:
    """Write six duration/balancing folders with whole/train/validation/test CSVs.

    ``whole.csv`` is the balanced training partition plus untouched validation and
    test rows. Validation and test are never sampled or duplicated.
    """
    base = Path(root) if root else project_root()
    extracted = experiment_dirs(base)["extracted_data"]
    extracted.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, object] = {}
    shared_assignments: dict[str, str] | None = None
    all_audit_rows: list[dict[str, object]] = []
    for duration in durations:
        dataset, warnings = build_supervised_dataset(base, window_duration=float(duration))
        if include_frequency:
            dataset = add_frequency_features(dataset, base)
        if shared_assignments is None:
            split, assignments, split_group = representative_group_split(dataset, group_by="recording", test_size=0.20, validation_size_within_development=0.20, seed=seed)
            shared_assignments = assignments
        else:
            split = dataset.copy()
            split["split"] = split["recording_id"].map(lambda value: shared_assignments[str(value)])
            assignments = shared_assignments
            split_group = "recording"
        split.to_parquet(extracted / f"window_{duration:g}s_unbalanced.parquet", index=False)
        validation = split.loc[split["split"].eq("validation")].drop(columns="split")
        test = split.loc[split["split"].eq("test")].drop(columns="split")
        train_source = split.loc[split["split"].eq("train")].drop(columns="split")
        audit_rows = [dataset_audit(dataset, f"window_{duration:g}s_original")]
        for split_name, split_part in (("train", train_source), ("validation", validation), ("test", test)):
            audit_rows.append(dataset_audit(split_part, f"window_{duration:g}s_{split_name}", split_name=split_name))
        for method in methods:
            balanced, report = balance_dataset(train_source, method, seed=seed)
            audit_rows.append(dataset_audit(balanced, f"window_{duration:g}s_train_{method}", split_name="train", balanced=True))
            folder = extracted / f"window_{duration:g}s_{method}"
            folder.mkdir(parents=True, exist_ok=True)
            whole = pd.concat([balanced.assign(split="train"), validation.assign(split="validation"), test.assign(split="test")], ignore_index=True)
            whole.to_csv(folder / "whole.csv", index=False)
            balanced.to_csv(folder / "train.csv", index=False)
            validation.to_csv(folder / "validation.csv", index=False)
            test.to_csv(folder / "test.csv", index=False)
            save_json({"window_duration_seconds": duration, "balancing_method": method, "seed": seed, "split_group": split_group, "warnings": warnings, "balance_report": report.__dict__, "whole_rows": len(whole), "train_rows": len(balanced), "validation_rows": len(validation), "test_rows": len(test), "frequency_features": include_frequency, "unbalanced_source": f"window_{duration:g}s_unbalanced.parquet", "note": "whole.csv combines balanced train with untouched validation and test"}, folder / "manifest.json")
            (folder / "README.md").write_text(f"# {folder.name}\n\n{duration:g}-second windows, 50% overlap, training rows balanced with `{method}` ({report.final_class_0} rows per class).\n\n- `train.csv`: balanced training rows.\n- `validation.csv`, `test.csv`: untouched, unbalanced rows.\n- `whole.csv`: the three files together with a `split` column.\n- `manifest.json`: seed, split group, balancing report, and excluded recordings.\n\nThe unbalanced table and the cross-validation folds are one level up: `window_{duration:g}s_unbalanced.parquet` and `window_{duration:g}s_cv_folds.csv`.\n", encoding="utf-8")
            manifests[f"{duration:g}s_{method}"] = folder
        all_audit_rows.extend(audit_rows)
        pd.DataFrame(audit_rows).to_csv(extracted / f"window_{duration:g}s_balancing_audit_report.csv", index=False)
        save_json(assignments, extracted / f"window_{duration:g}s_split_assignments.json")
    pd.DataFrame(all_audit_rows).to_csv(extracted / "balancing_audit_report.csv", index=False)
    return manifests


def create_stratified_group_cv_reports(root: str | Path | None = None, seed: int = DEFAULT_SEED, requested_folds: int = 5) -> dict[str, object]:
    """Assign development recordings to StratifiedGroupKFold folds for each duration.

    Folds are built on the unbalanced development rows (train and validation);
    test recordings are excluded. Balancing belongs inside each training fold.
    """
    extracted = experiment_dirs(root)["extracted_data"]
    summary: list[dict[str, object]] = []
    for source in sorted(extracted.glob("window_*s_unbalanced.parquet")):
        name = source.name.replace("_unbalanced.parquet", "")
        frame = pd.read_parquet(source, columns=["recording_id", "window_id", "label", "split"])
        development = frame.loc[frame["split"].isin(["train", "validation"])].reset_index(drop=True)
        n_splits = min(requested_folds, development["recording_id"].nunique())
        if n_splits < 2:
            raise ValueError(f"{name}: at least two recording groups are required for CV")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds: dict[str, int] = {}
        for fold, (_, validation_index) in enumerate(splitter.split(development, development["label"], development["recording_id"]), start=1):
            part = development.iloc[validation_index]
            folds.update({str(recording): fold for recording in part["recording_id"].unique()})
            audit = dataset_audit(part, f"{name}_fold_{fold}")
            summary.append({"dataset": name, "fold": fold, "recordings": audit["unique_recordings"], "rows": audit["total_rows"], "label_0_rows": audit["label_0_rows"], "label_1_rows": audit["label_1_rows"], "seizure_rate": audit["label_1_percent"], "seizure_events": audit["seizure_events"]})
        pd.DataFrame({"recording_id": list(folds), "cv_fold": list(folds.values())}).sort_values(["cv_fold", "recording_id"]).to_csv(extracted / f"{name}_cv_folds.csv", index=False)
    save_json({"seed": seed, "grouping": "recording_id", "patient_mapping_available": False, "test_used": False, "balanced_rows_used": False, "folds": summary}, extracted / "stratified_group_cv_summary.json")
    readme = ["# Extracted data", "", "Generated by `prepare_balanced_exports` and `create_stratified_group_cv_reports`; see the project README for the split and balancing policy.", "", "- `window_<d>s_unbalanced.parquet`: every window with its `split` column.", "- `window_<d>s_<method>/`: balanced training rows plus untouched validation and test rows.", "- `window_<d>s_split_assignments.json`: recording to partition.", "- `window_<d>s_cv_folds.csv`: development recording to cross-validation fold.", "- `balancing_audit_report.csv`: class counts for every dataset.", "", "## Cross-validation folds", "", "Development recordings only (train and validation); test recordings are never in a fold. Folds are groups of whole recordings, built on unbalanced rows.", "", "| Dataset | Fold | Recordings | Rows | Seizure rows | Seizure rate | Seizure events |", "|---|---:|---:|---:|---:|---:|---:|"]
    readme.extend(f"| {item['dataset']} | {item['fold']} | {item['recordings']} | {item['rows']} | {item['label_1_rows']} | {item['seizure_rate']:.4f} | {item['seizure_events']} |" for item in summary)
    (extracted / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return {"folds": summary, "summary_path": extracted / "stratified_group_cv_summary.json"}


def load_unbalanced_split(root: str | Path | None = None, folder_name: str | None = None) -> tuple[pd.DataFrame, str]:
    """Return unbalanced windows with their split, and the balancing method of the export folder.

    Sequence models need contiguous windows, so they build sequences from this
    table and balance the training sequences with ``balance_sequences``.
    """
    extracted = experiment_dirs(root)["extracted_data"]
    prepared = extracted / "window_features_with_split.parquet"
    if folder_name is None and prepared.exists():
        return pd.read_parquet(prepared), "none"
    folder = extracted / (folder_name or os.environ.get("EEG_DATASET_FOLDER", "window_1s_hybrid"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    source = extracted / f"window_{float(manifest['window_duration_seconds']):g}s_unbalanced.parquet"
    if not source.exists():
        raise FileNotFoundError(f"{source} is missing; rerun prepare_balanced_exports")
    return pd.read_parquet(source), str(manifest["balancing_method"])


def balance_sequences(values: np.ndarray, labels: np.ndarray, method: str, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Balance training sequences by target label with the shared balancing rules."""
    if method == "none":
        return values, labels
    selected, _ = balance_dataset(pd.DataFrame({"row": np.arange(len(labels)), "label": labels}), method, seed=seed)
    rows = selected["row"].to_numpy()
    return values[rows], labels[rows]


def verify_exports(root: str | Path | None = None) -> list[str]:
    """Check every export folder against its manifest and the unbalanced source."""
    extracted = experiment_dirs(root)["extracted_data"]
    issues: list[str] = []
    columns = ["recording_id", "window_id", "label"]
    for folder in sorted(path for path in extracted.glob("window_*s_*") if path.is_dir()):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        source, method = load_unbalanced_split(root, folder.name)
        parts = {name: pd.read_csv(folder / f"{name}.csv", usecols=columns) for name in ("train", "validation", "test")}
        groups = {name: set(part["recording_id"].astype(str)) for name, part in parts.items()}
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if groups[first] & groups[second]:
                issues.append(f"{folder.name}: recordings shared by {first} and {second}")
        original = source.loc[source["split"].eq("train"), "label"].value_counts()
        minority, majority = int(original.min()), int(original.max())
        target = {"undersample": minority, "oversample": majority, "hybrid": (minority + majority) // 2}[method]
        counts = parts["train"]["label"].value_counts().to_dict()
        if counts != {0: target, 1: target} or manifest["balance_report"]["target_per_class"] != target:
            issues.append(f"{folder.name}: train counts {counts} do not match the {method} target {target}")
        for name in ("validation", "test"):
            expected = source.loc[source["split"].eq(name), columns].reset_index(drop=True)
            if not parts[name].astype({"recording_id": str}).equals(expected.astype({"recording_id": str})):
                issues.append(f"{folder.name}: {name}.csv differs from the unbalanced {name} partition")
    return issues


def load_prepared(root: str | Path | None = None) -> pd.DataFrame:
    path = experiment_dirs(root)["extracted_data"] / "window_features_with_split.parquet"
    if path.exists():
        return pd.read_parquet(path)
    folder_name = os.environ.get("EEG_DATASET_FOLDER", "window_1s_hybrid")
    exported = path.parent / folder_name / "whole.csv"
    if exported.exists():
        return pd.read_csv(exported)
    raise FileNotFoundError(f"No prepared dataset found. Expected {path} or {exported}")


def evaluate_predictions(y_true: Iterable[int], y_pred: Iterable[int], scores: Iterable[float] | None = None) -> dict[str, object]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    metrics: dict[str, object] = {"Accuracy": float(accuracy_score(true, pred)), "Precision": float(precision_score(true, pred, zero_division=0)), "Recall": float(recall_score(true, pred, zero_division=0)), "F1": float(f1_score(true, pred, zero_division=0)), "ConfusionMatrix": confusion_matrix(true, pred, labels=[0, 1]).tolist(), "ClassificationReport": classification_report(true, pred, labels=[0, 1], target_names=["normal", "seizure"], zero_division=0, output_dict=True)}
    if scores is not None and len(np.unique(true)) == 2:
        metrics["PR-AUC"] = float(average_precision_score(true, list(scores)))
    return metrics


def save_model_result(model_name: str, model: object, test_frame: pd.DataFrame, y_pred: Iterable[int], scores: Iterable[float] | None, root: str | Path | None = None) -> dict[str, object]:
    dirs = experiment_dirs(root)
    dirs["models"].mkdir(parents=True, exist_ok=True)
    dirs["results"].mkdir(parents=True, exist_ok=True)
    predictions = test_frame[IDENTIFIER_COLUMNS + ["label"]].copy()
    predictions["prediction"] = np.asarray(list(y_pred), dtype=int)
    if scores is not None:
        predictions["score"] = np.asarray(list(scores), dtype=float)
    predictions.to_csv(dirs["results"] / f"{model_name}_test_predictions.csv", index=False)
    metrics = evaluate_predictions(test_frame["label"], predictions["prediction"], predictions.get("score"))
    save_json(metrics, dirs["results"] / f"{model_name}_test_metrics.json")
    joblib.dump(model, dirs["models"] / f"{model_name}.joblib")
    return metrics


def select_threshold(y_true: Iterable[int], scores: Iterable[float]) -> float:
    """Return the score threshold with the best F1; ties go to the higher recall."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(list(y_true), dtype=int), np.asarray(list(scores), dtype=float))
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = max(range(len(thresholds)), key=lambda index: (f1[index], recall[index]))
    return float(thresholds[best])


def merge_predicted_intervals(predictions: pd.DataFrame, gap_tolerance: float = 1.0, column: str = "prediction") -> pd.DataFrame:
    """Merge positive windows of one recording into intervals, bridging short gaps."""
    positives = predictions.loc[predictions[column].eq(1)].sort_values(["recording_id", "start_time"])
    intervals: list[dict[str, object]] = []
    for recording_id, group in positives.groupby("recording_id", sort=False):
        current: dict[str, object] | None = None
        for row in group.itertuples(index=False):
            if current is None or float(row.start_time) > float(current["end_time"]) + gap_tolerance:
                if current is not None:
                    intervals.append(current)
                current = {"recording_id": recording_id, "start_time": float(row.start_time), "end_time": float(row.end_time)}
            else:
                current["end_time"] = max(float(current["end_time"]), float(row.end_time))
        if current is not None:
            intervals.append(current)
    return pd.DataFrame(intervals, columns=["recording_id", "start_time", "end_time"])


def annotated_intervals(recording_ids: Iterable[str], root: str | Path | None = None) -> pd.DataFrame:
    """Return the validated Excel seizure intervals of the given recordings, in seconds."""
    base = Path(root) if root else project_root()
    workbook = pd.read_excel(base / "data" / "Annotations.xlsx")
    wanted = {str(value) for value in recording_ids}
    rows: list[dict[str, object]] = []
    for record in discover_recordings(base / "data"):
        if record.recording_id not in wanted or record.signal_path is None:
            continue
        duration = float(signal_metadata(record.signal_path, base / "data" / ".cache")["duration"])
        seizures, _ = excel_seizures_for_recording(workbook, record.recording_id, duration)
        rows.extend({"recording_id": record.recording_id, "start_time": float(row["Début (s)"]), "end_time": float(row["Fin (s)"])} for _, row in seizures.iterrows())
    return pd.DataFrame(rows, columns=["recording_id", "start_time", "end_time"])


def event_report(predictions: pd.DataFrame, gap_tolerance: float = 1.0, true_intervals: pd.DataFrame | None = None) -> dict[str, object]:
    """Compare reconstructed seizure intervals with the annotated ones.

    An annotated seizure is detected when a predicted interval of the same
    recording overlaps it; a predicted interval with no overlap is a false alarm.
    Without ``true_intervals`` the annotated seizures are rebuilt from the
    window labels.
    """
    true = true_intervals.copy() if true_intervals is not None else merge_predicted_intervals(predictions, gap_tolerance=0.0, column="label")
    predicted = merge_predicted_intervals(predictions, gap_tolerance=gap_tolerance)

    def overlaps(row: pd.Series, others: pd.DataFrame) -> pd.DataFrame:
        return others.loc[others["recording_id"].eq(row["recording_id"]) & others["start_time"].le(row["end_time"]) & others["end_time"].ge(row["start_time"])]

    true["detected"] = [not overlaps(row, predicted).empty for _, row in true.iterrows()]
    true["onset_delay"] = [overlaps(row, predicted)["start_time"].min() - row["start_time"] if detected else np.nan for (_, row), detected in zip(true.iterrows(), true["detected"])]
    predicted["matches_seizure"] = [not overlaps(row, true).empty for _, row in predicted.iterrows()]
    spans = predictions.groupby("recording_id").agg(start=("start_time", "min"), end=("end_time", "max"))
    hours = float((spans["end"] - spans["start"]).sum() / 3600)
    false_alarms = int((~predicted["matches_seizure"]).sum())
    return {"true_events": int(len(true)), "detected_events": int(true["detected"].sum()), "missed_events": int((~true["detected"]).sum()), "event_recall": float(true["detected"].mean()) if len(true) else 0.0, "predicted_intervals": int(len(predicted)), "false_alarm_intervals": false_alarms, "false_alarms_per_hour": false_alarms / hours if hours else 0.0, "recorded_hours": hours, "mean_onset_delay_seconds": float(true["onset_delay"].mean()) if true["detected"].any() else None, "true_intervals": true, "predicted": predicted}


def make_sequences(frame: pd.DataFrame, length: int, features: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build consecutive same-recording, same-split sequences; target is final label."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    columns = IDENTIFIER_COLUMNS + ["label", "split"]
    for _, group in frame.sort_values(["recording_id", "split", "start_time"]).groupby(["recording_id", "split"], sort=False):
        if len(group) < length:
            continue
        values = group[features].to_numpy(dtype=np.float32)
        xs.append(np.lib.stride_tricks.sliding_window_view(values, length, axis=0).transpose(0, 2, 1))
        ys.append(group["label"].to_numpy(dtype=np.int64)[length - 1:])
        metadata.append(group[columns].iloc[length - 1:])
    if not xs:
        return np.empty((0, length, len(features)), dtype=np.float32), np.empty(0, dtype=np.int64), pd.DataFrame(columns=columns)
    return np.concatenate(xs), np.concatenate(ys), pd.concat(metadata).reset_index(drop=True)


def split_xy_by_assignment(frame: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    selected = features or feature_names(frame)
    parts = [frame.loc[frame["split"].eq(name)] for name in ("train", "validation", "test")]
    return tuple((part[selected], part["label"]) for part in parts)  # type: ignore[return-value]


def classical_pipeline(estimator: object, *, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)