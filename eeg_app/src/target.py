"""Sample-level targets and temporal window segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .annotations import normalize_seizure_table


def validate_seizures(seizures: pd.DataFrame, recording_duration: float | None = None) -> pd.DataFrame:
    """Clean, clip, sort, and merge valid closed seizure intervals."""
    intervals = normalize_seizure_table(seizures)
    if intervals.empty:
        return intervals
    if recording_duration is not None and np.isfinite(recording_duration):
        intervals["Début (s)"] = intervals["Début (s)"].clip(lower=0, upper=recording_duration)
        intervals["Fin (s)"] = intervals["Fin (s)"].clip(lower=0, upper=recording_duration)
    intervals = intervals.loc[intervals["Fin (s)"].ge(intervals["Début (s)"])].sort_values("Début (s)").reset_index(drop=True)
    merged: list[list[float]] = []
    for start, end in intervals[["Début (s)", "Fin (s)"]].itertuples(index=False, name=None):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    return pd.DataFrame({"Crise": [f"Crise {index + 1}" for index in range(len(merged))], "Début (s)": [row[0] for row in merged], "Fin (s)": [row[1] for row in merged]}).assign(**{"Durée (s)": lambda frame: frame["Fin (s)"] - frame["Début (s)"]})


def build_target(time: pd.Series | np.ndarray, seizures: pd.DataFrame) -> np.ndarray:
    """Return 1 for samples inside any closed seizure interval, otherwise 0.

    The implementation uses sorted-array search boundaries and a difference
    array, so it does not iterate over individual samples. Endpoints are
    included consistently: ``start <= Time <= end``.
    """
    values = pd.to_numeric(pd.Series(time), errors="coerce").to_numpy(dtype=float)
    if not len(values):
        return np.zeros(0, dtype=np.uint8)
    order = np.argsort(values, kind="stable")
    sorted_time = values[order]
    intervals = validate_seizures(seizures)
    differences = np.zeros(len(values) + 1, dtype=np.int32)
    for start, end in intervals[["Début (s)", "Fin (s)"]].itertuples(index=False, name=None):
        left = int(np.searchsorted(sorted_time, start, side="left"))
        right = int(np.searchsorted(sorted_time, end, side="right"))
        differences[left] += 1
        differences[right] -= 1
    sorted_target = np.cumsum(differences[:-1]) > 0
    target = np.zeros(len(values), dtype=np.uint8)
    target[order] = sorted_target.astype(np.uint8)
    return target


def class_distribution(target: pd.Series | np.ndarray) -> dict[str, float]:
    """Return class counts, percentages, and majority/minority ratio."""
    values = np.asarray(target, dtype=np.uint8)
    class_zero = int(np.count_nonzero(values == 0))
    class_one = int(np.count_nonzero(values == 1))
    total = class_zero + class_one
    minority = min(class_zero, class_one)
    majority = max(class_zero, class_one)
    return {"class_0": class_zero, "class_1": class_one, "total": total, "percentage_0": class_zero / total if total else 0.0, "percentage_1": class_one / total if total else 0.0, "imbalance_ratio": majority / minority if minority else float("inf")}


def segment_signal(frame: pd.DataFrame, sampling_frequency: float, window_duration: float, overlap: float) -> pd.DataFrame:
    """Create one metadata row per overlapping temporal window."""
    if sampling_frequency <= 0 or window_duration <= 0 or not 0 <= overlap < 1:
        return pd.DataFrame(columns=["Window", "Start_Time", "End_Time", "Duration", "Seizure_Ratio", "Class"])
    samples_per_window = max(1, round(window_duration * sampling_frequency))
    step_samples = max(1, round(samples_per_window * (1 - overlap)))
    target = pd.to_numeric(frame["Class"], errors="coerce").fillna(0).to_numpy(dtype=float)
    time = pd.to_numeric(frame["Time"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []
    window_number = 1
    for start_index in range(0, max(0, len(frame) - samples_per_window + 1), step_samples):
        end_index = start_index + samples_per_window
        values = target[start_index:end_index]
        if len(values) < samples_per_window:
            break
        ratio = float(np.mean(values))
        rows.append({"Window": window_number, "Start_Time": float(time[start_index]), "End_Time": float(time[end_index - 1]), "Duration": float(time[end_index - 1] - time[start_index]), "Seizure_Ratio": ratio, "Class": int(ratio >= 0.50)})
        window_number += 1
    return pd.DataFrame(rows, columns=["Window", "Start_Time", "End_Time", "Duration", "Seizure_Ratio", "Class"])