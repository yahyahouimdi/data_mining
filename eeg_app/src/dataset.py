"""Reusable window-level feature datasets for EEG seizure detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .annotations import normalize_recording_id
from .features import FEATURE_NAMES, temporal_features
from .signal_utils import sampling_frequency


DATASET_METADATA = ["patient_id", "recording_id", "window_id", "start_time", "end_time"]


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
    counts = source["label"].value_counts().to_dict()
    count_zero, count_one = int(counts.get(0, 0)), int(counts.get(1, 0))
    target = target_per_class if target_per_class is not None else min(count_zero, count_one)
    if method == "none":
        result = source.copy()
        target = None
    else:
        if target is None or target < 0:
            raise ValueError("target_per_class must be non-negative")
        rng = np.random.default_rng(seed)
        parts: list[pd.DataFrame] = []
        for label in (0, 1):
            group = source.loc[source["label"].eq(label)]
            if len(group) >= target:
                indices = rng.choice(len(group), size=target, replace=False)
            else:
                if method == "undersample":
                    indices = np.arange(len(group))
                else:
                    indices = rng.choice(len(group), size=target, replace=True) if len(group) else np.array([], dtype=int)
            parts.append(group.iloc[indices])
        result = pd.concat(parts, ignore_index=True) if parts else source.iloc[0:0].copy()
        result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    duplicated = max(0, len(result) - len(result.drop_duplicates()))
    report = BalanceReport(method, seed, target, count_zero, count_one, int(result["label"].eq(0).sum()), int(result["label"].eq(1).sum()), max(0, len(source) - len(result)), duplicated)
    return result, report


def group_split(frame: pd.DataFrame, *, group_by: str = "patient", train_size: float = 0.70, validation_size: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assign whole patients or recordings to train/validation/test groups."""
    if group_by not in {"patient", "recording"}:
        raise ValueError("group_by must be 'patient' or 'recording'")
    if train_size <= 0 or validation_size < 0 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must leave a positive test split")
    groups = pd.Index(frame["patient_id"] if group_by == "patient" else frame["recording_id"]).drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    train_end = max(1, round(len(groups) * train_size))
    validation_end = min(len(groups), train_end + round(len(groups) * validation_size))
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
