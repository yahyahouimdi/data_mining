"""Window-level temporal feature extraction for EEG signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .target import segment_signal


FEATURE_NAMES = ("Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy")


def temporal_features(values: np.ndarray | pd.Series) -> dict[str, float]:
    """Calculate eight NaN-safe temporal features for one raw signal window."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return {name: 0.0 for name in FEATURE_NAMES}
    minimum = float(np.min(clean))
    maximum = float(np.max(clean))
    return {"Mean": float(np.mean(clean)), "Std": float(np.std(clean)), "Variance": float(np.var(clean)), "Min": minimum, "Max": maximum, "Amplitude": maximum - minimum, "RMS": float(np.sqrt(np.mean(clean ** 2))), "Energy": float(np.sum(clean ** 2))}


def build_feature_dataset(frame: pd.DataFrame, eeg_channels: list[str], sampling_frequency: float, window_duration: float, overlap: float) -> pd.DataFrame:
    """Build one row per raw-sample window with channel-specific features."""
    windows = segment_signal(frame[["Time", "Class"]], sampling_frequency, window_duration, overlap)
    if windows.empty:
        return windows
    rows: list[dict[str, float | int]] = []
    samples_per_window = max(1, round(window_duration * sampling_frequency))
    step_samples = max(1, round(samples_per_window * (1 - overlap)))
    for window_index, start_index in enumerate(range(0, max(0, len(frame) - samples_per_window + 1), step_samples)):
        end_index = start_index + samples_per_window
        if end_index > len(frame):
            break
        row = windows.iloc[window_index].to_dict()
        for channel in eeg_channels:
            for feature_name, value in temporal_features(frame[channel].iloc[start_index:end_index]).items():
                row[f"{channel}_{feature_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(channels: list[str]) -> list[str]:
    """Return the channel-qualified feature column names."""
    return [f"{channel}_{feature}" for channel in channels for feature in FEATURE_NAMES]