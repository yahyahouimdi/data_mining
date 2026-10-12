"""Reusable window-level feature datasets for EEG seizure detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .annotations import normalize_recording_id
from .features import FEATURE_NAMES, temporal_features
from .signal_utils import sampling_frequency


DATASET_METADATA = ["patient_id", "recording_id", "window_id", "start_time", "end_time"]
BALANCING_METHODS = ("undersample", "oversample", "hybrid")
# Below this many seizure events a partition is flagged as weak evidence.
MIN_SEIZURE_EVENTS = 5


@dataclass(frozen=True)
class DatasetWarning:
    """Reason a recording was excluded or requires user attention."""

    recording_id: str
    message: str
    excluded: bool = True


@dataclass(frozen=True)
class BalanceReport:
    """Reproducibility details for one balancing operation."""

    method: str
    seed: int
    target_per_class: int | None
    original_class_0: int
    original_class_1: int
    final_class_0: int
    final_class_1: int
    removed_windows: int
    duplicated_windows: int


def detect_signal_channels(columns: Iterable[object]) -> list[str]:
    """Return numeric signal candidates, excluding time, labels, and metadata."""
    excluded = {"time", "timestamp", "class", "label", "annotations", "duration"}
    return [str(column) for column in columns if str(column).strip().casefold() not in excluded]


def feature_columns(channels: Iterable[str]) -> list[str]:
    """Return deterministic channel-qualified feature names."""
    return [f"{channel}_{feature}" for channel in channels for feature in FEATURE_NAMES]


def patient_id_for_recording(recording_id: str, patient_map: dict[str, str] | None = None) -> str:
    """Use an explicit mapping when available, otherwise retain the recording id."""
    normalized = normalize_recording_id(recording_id)
    return (patient_map or {}).get(normalized, recording_id)


def _valid_time_and_channels(frame: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Coerce time/signals and retain finite, monotonically increasing samples."""
    result = frame.copy()
    result["Time"] = pd.to_numeric(result["Time"], errors="coerce")
    for channel in channels:
        result[channel] = pd.to_numeric(result[channel], errors="coerce")
    result = result.loc[result["Time"].notna()].reset_index(drop=True)
    return result, [channel for channel in channels if channel in result]


def _sample_labels(time: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Mark samples covered by validated half-open seizure intervals."""
    labels = np.zeros(len(time), dtype=np.uint8)
    for start, end in intervals:
        labels |= ((time >= start) & (time <= end)).astype(np.uint8)
    return labels


def build_recording_dataset(
    frame: pd.DataFrame,
    patient_id: str,
    recording_id: str,
    channels: Iterable[str],
    seizure_intervals: pd.DataFrame,
    *,
    window_duration: float = 2.0,
    overlap: float = 0.50,
    label_threshold: float = 0.50,
    annotation_valid: bool = True,
) -> pd.DataFrame:
    """Build the full feature schema for one recording from raw samples.

    Windows are complete, contiguous sample blocks numbered from 1. End times
    are the final sample time in the window, matching the existing plot logic.
    """
    selected = list(dict.fromkeys(str(channel) for channel in channels))
    if not annotation_valid:
        raise ValueError("Validated seizure annotations are required for supervised labels")
    if not 0 < window_duration or not 0 <= overlap < 1 or not 0 <= label_threshold <= 1:
        raise ValueError("window_duration must be positive; overlap and label_threshold must be in [0, 1)")
    if "Time" not in frame:
        raise ValueError("Signal data must contain a Time column")
    clean, selected = _valid_time_and_channels(frame, selected)
    if not selected:
        raise ValueError("No selected signal channels are present")
    time = clean["Time"].to_numpy(dtype=float)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("Time must contain at least two strictly increasing finite samples")
    frequency = sampling_frequency(time)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Sampling frequency could not be computed from Time")
    samples_per_window = max(1, round(window_duration * frequency))
    step_samples = max(1, round(samples_per_window * (1 - overlap)))
    intervals = seizure_intervals[["Début (s)", "Fin (s)"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    intervals = intervals[(intervals[:, 0] <= intervals[:, 1]) & np.isfinite(intervals).all(axis=1)] if len(intervals) else np.empty((0, 2))
    labels = _sample_labels(time, intervals)
    rows: list[dict[str, object]] = []
    for window_number, start_index in enumerate(range(0, len(clean) - samples_per_window + 1, step_samples), start=1):
        end_index = start_index + samples_per_window
        row: dict[str, object] = {
            "patient_id": patient_id,
            "recording_id": recording_id,
            "window_id": window_number,
            "start_time": float(time[start_index]),
            "end_time": float(time[end_index - 1]),
        }
        for channel in selected:
            values = clean[channel].iloc[start_index:end_index].to_numpy(dtype=float)
            if not np.isfinite(values).any():
                raise ValueError(f"{recording_id}: window {window_number} has no finite samples for {channel}")
            for name, value in temporal_features(values).items():
                row[f"{channel}_{name}"] = value
        row["label"] = int(np.mean(labels[start_index:end_index]) >= label_threshold)
        rows.append(row)
    columns = [*DATASET_METADATA, *feature_columns(selected), "label"]
    return pd.DataFrame(rows, columns=columns)


def build_dataset_from_recordings(
    recordings: Iterable[tuple[str, str, pd.DataFrame, pd.DataFrame]],
    channels: Iterable[str],
    **kwargs: object,
) -> tuple[pd.DataFrame, list[DatasetWarning]]:
    """Combine recording datasets; invalid or unannotated recordings are excluded."""
    frames: list[pd.DataFrame] = []
    warnings: list[DatasetWarning] = []
    selected = list(channels)
    for patient_id, recording_id, signal, seizures in recordings:
        if seizures is None or seizures.empty:
            warnings.append(DatasetWarning(recording_id, "Validated seizure annotations are missing", True))
            continue
        try:
            frames.append(build_recording_dataset(signal, patient_id, recording_id, selected, seizures, **kwargs))
        except (TypeError, ValueError, KeyError) as error:
            warnings.append(DatasetWarning(recording_id, str(error), True))
    if not frames:
        return pd.DataFrame(columns=[*DATASET_METADATA, *feature_columns(selected), "label"]), warnings
    return pd.concat(frames, ignore_index=True), warnings


def balance_dataset(frame: pd.DataFrame, method: str = "none", *, target_per_class: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, BalanceReport]:
    """Balance labels reproducibly without changing feature values or schema."""
    if method not in {"none", "undersample", "oversample", "hybrid"}:
        raise ValueError(f"Unknown balancing method: {method}")
    source = frame.reset_index(drop=True)
    if "label" not in source.columns:
        raise ValueError("Balancing requires a label column")
    labels = pd.to_numeric(source["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError("Balancing requires non-null numeric labels")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Balancing requires both label classes 0 and 1 and no other label")
    counts = source["label"].value_counts().to_dict()
    count_zero, count_one = int(counts.get(0, 0)), int(counts.get(1, 0))
    minority = min(count_zero, count_one)
    majority = max(count_zero, count_one)
    default_targets = {"undersample": minority, "oversample": majority, "hybrid": (minority + majority) // 2}
    target = target_per_class if target_per_class is not None else default_targets.get(method)
    if method == "none":
        result = source.copy()
        target = None
    else:
        if target is None or target < 0:
            raise ValueError("target_per_class must be non-negative")
        if method == "undersample" and target > minority:
            raise ValueError("undersample target cannot exceed the original minority count")
        if method == "oversample" and target < majority:
            raise ValueError("oversample target cannot be below the original majority count")
        rng = np.random.default_rng(seed)
        parts: list[pd.DataFrame] = []
        for label in (0, 1):
            group = source.loc[source["label"].eq(label)]
            indices = rng.choice(len(group), size=target, replace=len(group) < target)
            parts.append(group.iloc[indices])
        result = pd.concat(parts, ignore_index=True) if parts else source.iloc[0:0].copy()
        result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    duplicated = max(0, len(result) - len(result.drop_duplicates()))
    report = BalanceReport(method, seed, target, count_zero, count_one, int(result["label"].eq(0).sum()), int(result["label"].eq(1).sum()), max(0, len(source) - len(result)), duplicated)
    return result, report


def _group_event_count(frame: pd.DataFrame, group_column: str) -> int:
    ordered = frame.sort_values([group_column, "window_id"] if "window_id" in frame else [group_column])
    events = 0
    for _, group in ordered.groupby(group_column, sort=False):
        labels = group["label"].to_numpy(dtype=int)
        events += int(np.sum((labels == 1) & np.r_[True, labels[:-1] == 0]))
    return events


def dataset_audit(frame: pd.DataFrame, dataset_name: str, *, split_name: str | None = None, balanced: bool = False) -> dict[str, object]:
    """Summarize labels, groups, and seizure-event runs for an audit report.

    Seizure events are runs of consecutive seizure windows, so they are only
    counted for unbalanced data; resampled rows no longer form a timeline.
    """
    labels = pd.to_numeric(frame["label"], errors="coerce")
    counts = labels.value_counts().to_dict()
    count_zero, count_one = int(counts.get(0, 0)), int(counts.get(1, 0))
    events = None if balanced or "recording_id" not in frame else _group_event_count(frame, "recording_id")
    warnings = []
    if frame.empty:
        warnings.append("partition is empty")
    elif min(count_zero, count_one) == 0:
        warnings.append("partition lacks one label class")
    if events is not None and events < MIN_SEIZURE_EVENTS:
        warnings.append(f"fewer than {MIN_SEIZURE_EVENTS} seizure events")
    return {"dataset_name": dataset_name, "split": split_name or "all", "total_rows": len(frame), "label_0_rows": count_zero, "label_1_rows": count_one, "label_0_percent": count_zero / len(frame) if len(frame) else 0.0, "label_1_percent": count_one / len(frame) if len(frame) else 0.0, "imbalance_ratio": max(count_zero, count_one) / min(count_zero, count_one) if min(count_zero, count_one) else float("inf"), "unique_recordings": int(frame["recording_id"].nunique()) if "recording_id" in frame else 0, "unique_patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else 0, "seizure_events": events, "warnings": "; ".join(warnings)}


def balancing_audit(split_frame: pd.DataFrame, *, methods: Iterable[str] = BALANCING_METHODS, seed: int = 42, prefix: str = "") -> pd.DataFrame:
    """Audit the original data, each partition, and each balanced training set."""
    unsplit = split_frame.drop(columns="split")
    rows = [dataset_audit(unsplit, f"{prefix}original")]
    for split_name in ("train", "validation", "test"):
        rows.append(dataset_audit(unsplit.loc[split_frame["split"].eq(split_name)], f"{prefix}{split_name}", split_name=split_name))
    train = unsplit.loc[split_frame["split"].eq("train")]
    for method in methods:
        balanced, _ = balance_dataset(train, method, seed=seed)
        rows.append(dataset_audit(balanced, f"{prefix}train_{method}", split_name="train", balanced=True))
    return pd.DataFrame(rows)


def representative_group_split(frame: pd.DataFrame, *, group_by: str = "recording", test_size: float = 0.20, validation_size_within_development: float = 0.20, seed: int = 42) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Create reproducible group-isolated partitions with representative labels.

    A five-fold StratifiedGroupKFold candidate is selected using only group labels
    and prevalence, never model performance. The selected test fold is ~20% of
    groups; development is split into train/validation using the same policy.
    """
    if group_by not in {"patient", "recording"}:
        raise ValueError("group_by must be 'patient' or 'recording'")
    column = "patient_id" if group_by == "patient" else "recording_id"
    if frame[column].nunique() < 5:
        validation_size = (1 - test_size) * validation_size_within_development
        fallback, assignments = group_split(frame, group_by=group_by, train_size=1 - test_size - validation_size, validation_size=validation_size, seed=seed)
        return fallback, assignments, group_by
    overall_rate = float(frame["label"].mean())
    groups = frame[column].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for train_index, test_index in splitter.split(frame, frame["label"], groups):
        test_part = frame.iloc[test_index]
        class_penalty = 0.0 if test_part["label"].nunique() == 2 else 1.0
        score = abs(len(test_part) / len(frame) - test_size) + abs(float(test_part["label"].mean()) - overall_rate) + class_penalty
        candidates.append((score, train_index, test_index))
    _, development_index, test_index = min(candidates, key=lambda item: item[0])
    development = frame.iloc[development_index].copy()
    development_rate = float(development["label"].mean())
    inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    development_groups = development[column].astype(str)
    for train_index, validation_index in inner_splitter.split(development, development["label"], development_groups):
        validation = development.iloc[validation_index]
        score = abs(len(validation) / len(development) - validation_size_within_development) + abs(float(validation["label"].mean()) - development_rate)
        inner_candidates.append((score, train_index, validation_index))
    _, train_index, validation_index = min(inner_candidates, key=lambda item: item[0])
    assignments: dict[str, str] = {}
    assignments.update({str(value): "train" for value in development.iloc[train_index][column].unique()})
    assignments.update({str(value): "validation" for value in development.iloc[validation_index][column].unique()})
    assignments.update({str(value): "test" for value in frame.iloc[test_index][column].unique()})
    result = frame.copy()
    result["split"] = result[column].map(lambda value: assignments[str(value)])
    return result, assignments, group_by


def group_split(frame: pd.DataFrame, *, group_by: str = "patient", train_size: float = 0.70, validation_size: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assign whole patients or recordings to train/validation/test groups."""
    if group_by not in {"patient", "recording"}:
        raise ValueError("group_by must be 'patient' or 'recording'")
    if train_size <= 0 or validation_size < 0 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must leave a positive test split")
    groups = pd.Index(frame["patient_id"] if group_by == "patient" else frame["recording_id"]).drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    # With few groups the held-out test set is filled before validation.
    test_count = max(1, round(len(groups) * (1 - train_size - validation_size))) if len(groups) > 1 else 0
    validation_count = min(round(len(groups) * validation_size), max(0, len(groups) - test_count - 1))
    train_end = len(groups) - test_count - validation_count
    validation_end = train_end + validation_count
    assignments = {str(group): "train" if index < train_end else "validation" if index < validation_end else "test" for index, group in enumerate(groups)}
    result = frame.copy()
    result["split"] = (result["patient_id"] if group_by == "patient" else result["recording_id"]).map(lambda value: assignments[str(value)])
    return result, assignments


def export_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    """Export without a DataFrame index, using CSV or Parquet by suffix."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.casefold() == ".parquet":
        frame.to_parquet(destination, index=False)
    elif destination.suffix.casefold() == ".csv":
        frame.to_csv(destination, index=False)
    else:
        raise ValueError("Dataset exports must use .csv or .parquet")
    return destination


def split_xy(frame: pd.DataFrame, *, include_metadata: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target y, excluding identifiers by default."""
    metadata = set(DATASET_METADATA + ["label", "split"])
    feature_names = [column for column in frame.columns if column not in metadata]
    if include_metadata:
        feature_names = [column for column in frame.columns if column != "label"]
    return frame[feature_names].copy(), frame["label"].copy()
