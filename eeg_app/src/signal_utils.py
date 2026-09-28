"""Signal processing helpers designed for bounded-memory visualisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def sampling_frequency(time: pd.Series | np.ndarray) -> float:
    """Estimate sampling frequency from the median positive time step."""
    values = pd.to_numeric(pd.Series(time), errors="coerce").dropna().to_numpy()
    steps = np.diff(values)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(1.0 / np.median(steps)) if len(steps) else float("nan")


def normalize_series(values: pd.Series, method: str) -> pd.Series:
    """Normalize one channel while preserving missing values and flat signals."""
    numeric = pd.to_numeric(values, errors="coerce")
    if method == "z-score":
        spread = numeric.std()
        return (numeric - numeric.mean()) / spread if spread and np.isfinite(spread) else numeric * 0
    if method == "min-max":
        span = numeric.max() - numeric.min()
        return (numeric - numeric.min()) / span if span and np.isfinite(span) else numeric * 0
    return numeric


def min_max_downsample(frame: pd.DataFrame, x_column: str, max_points: int) -> pd.DataFrame:
    """Downsample each channel with min/max buckets, retaining sharp EEG spikes."""
    if len(frame) <= max_points:
        return frame.copy()
    bucket_count = max(1, max_points // 2)
    indices = np.linspace(0, len(frame), bucket_count + 1, dtype=int)
    selected: list[int] = []
    numeric_columns = [column for column in frame.columns if column != x_column]
    for start, end in zip(indices[:-1], indices[1:]):
        if end <= start:
            continue
        block = frame.iloc[start:end]
        for column in numeric_columns:
            series = pd.to_numeric(block[column], errors="coerce")
            if series.notna().any():
                selected.extend([int(series.idxmin()), int(series.idxmax())])
    return frame.loc[sorted(set(selected))].sort_values(x_column).reset_index(drop=True)


def welch_psd(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute a robust Welch power spectrum for a finite signal segment."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 4 or not np.isfinite(fs) or fs <= 0:
        return np.array([]), np.array([])
    frequencies, power = signal.welch(clean, fs=fs, nperseg=min(1024, len(clean)))
    return frequencies, power


def spectrogram(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a bounded spectrogram for the comparison tab."""
    clean = np.asarray(values, dtype=float)
    clean = np.nan_to_num(clean)
    if len(clean) < 8 or not np.isfinite(fs) or fs <= 0:
        return np.array([]), np.array([]), np.empty((0, 0))
    frequencies, times, power = signal.spectrogram(clean, fs=fs, nperseg=min(256, len(clean)))
    return frequencies, times, power