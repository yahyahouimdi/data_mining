"""Figures and facts for Parts I to V of the report (questions 1 to 8).

Uses one example recording for the per-file questions and the 18 annotated
recordings for the dataset-level counts. Writes figures to ``reports/figures``
and the numbers quoted in the report to ``reports/exploration_summary.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_app.src.annotations import duration_statistics  # noqa: E402
from eeg_app.src.experiments import _recording_signal, annotated_intervals, save_json  # noqa: E402
from eeg_app.src.signal_utils import sampling_frequency  # noqa: E402
from report_style import MUTED, SEIZURE, SERIES, apply_style  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
EXAMPLE = "211104B_D_"


def overview_figure(signal: pd.DataFrame, seizures: pd.DataFrame, channel: str) -> None:
    figure, axis = plt.subplots(figsize=(11, 3.4))
    step = 8
    axis.plot(signal["Time"].to_numpy()[::step], signal[channel].to_numpy()[::step], color=MUTED, linewidth=0.5)
    for row in seizures.itertuples(index=False):
        axis.axvspan(row.start_time, row.end_time, color=SEIZURE, alpha=0.25, linewidth=0)
    axis.set_title(f"EEG signal of recording {EXAMPLE.rstrip('_')}, channel {channel} (annotated seizures shaded)", loc="left")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("amplitude (recorded units)")
    axis.margins(x=0)
    low, high = np.percentile(signal[channel].to_numpy(), [0.05, 99.95])
    axis.set_ylim(low - 0.2 * (high - low), high + 0.2 * (high - low))
    figure.tight_layout()
    figure.savefig(FIGURES / "signal_overview.png", dpi=160)
    plt.close(figure)


def zoom_figure(signal: pd.DataFrame, seizure: pd.Series, channels: list[str]) -> None:
    start, end = seizure["start_time"] - 6, seizure["end_time"] + 6
    part = signal.loc[signal["Time"].between(start, end)]
    figure, axes = plt.subplots(len(channels), 1, figsize=(11, 1.5 * len(channels) + 0.8), sharex=True)
    for axis, channel in zip(axes, channels):
        axis.plot(part["Time"], part[channel] - part[channel].mean(), color=SERIES[0], linewidth=0.7)
        axis.axvspan(seizure["start_time"], seizure["end_time"], color=SEIZURE, alpha=0.18, linewidth=0)
        axis.set_ylabel(channel.replace("EEG", ""))
        axis.margins(x=0)
    axes[0].set_title(f"Six seconds before and after one seizure of {EXAMPLE.rstrip('_')} (seizure shaded, offset removed)", loc="left")
    axes[-1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(FIGURES / "seizure_zoom.png", dpi=160)
    plt.close(figure)


def duration_figure(durations: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 3.4))
    axis.hist(durations, bins=np.arange(0, 55, 5), color=SERIES[0], edgecolor="#fcfcfb", linewidth=1.5)
    axis.axvline(np.median(durations), color=SERIES[1], linewidth=1.5)
    axis.text(np.median(durations) + 0.6, axis.get_ylim()[1] * 0.92, f"median {np.median(durations):.0f} s", color="#52514e", fontsize=9)
    axis.set_title(f"Duration of the {len(durations)} annotated seizures (18 recordings)", loc="left")
    axis.set_xlabel("duration (s)")
    axis.set_ylabel("number of seizures")
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "seizure_durations.png", dpi=160)
    plt.close(figure)


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    signal = _recording_signal(ROOT, EXAMPLE)
    time = signal["Time"].to_numpy(float)
    fs = sampling_frequency(time)
    eeg = [column for column in signal.columns if column.startswith("EEG")]
    seizures = annotated_intervals([EXAMPLE], ROOT)
    seizures["duration"] = seizures["end_time"] - seizures["start_time"]
    table = pd.DataFrame({"Début (s)": seizures["start_time"], "Fin (s)": seizures["end_time"]})
    statistics = duration_statistics(table)
    labels = np.zeros(len(time), dtype=int)
    for row in seizures.itertuples(index=False):
        labels |= ((time >= row.start_time) & (time <= row.end_time)).astype(int)

    overview_figure(signal, seizures, eeg[0])
    longest = seizures.sort_values("duration").iloc[len(seizures) // 2]
    zoom_figure(signal, longest, eeg[:4])

    windows = {duration: pd.read_parquet(ROOT / "extracted_data" / f"window_{duration}_unbalanced.parquet", columns=["recording_id", "label"]) for duration in ("1s", "2s")}
    all_seizures = annotated_intervals(windows["2s"]["recording_id"].unique(), ROOT)
    all_durations = (all_seizures["end_time"] - all_seizures["start_time"]).to_numpy()
    duration_figure(all_durations)
    example_windows = windows["2s"].loc[windows["2s"]["recording_id"].eq(EXAMPLE), "label"]
    total_seconds = 0.0
    for recording in windows["2s"]["recording_id"].unique():
        recording_time = _recording_signal(ROOT, str(recording), ["Time"])["Time"]
        total_seconds += float(recording_time.iloc[-1] - recording_time.iloc[0])

    save_json({
        "example_recording": EXAMPLE.rstrip("_"),
        "rows": int(len(signal)), "variables": int(signal.shape[1]), "eeg_channels": len(eeg), "other_channels": [column for column in signal.columns if column not in eeg and column != "Time"],
        "sampling_frequency_hz": fs, "duration_seconds": float(time[-1] - time[0]), "missing_values": int(signal.isna().sum().sum()),
        "first_rows": signal.iloc[:5, :5].round(3).to_dict(orient="records"),
        "seizures": seizures.round(1).to_dict(orient="records"),
        "duration_statistics": statistics,
        "sample_class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "sample_class_percent": {"0": float((labels == 0).mean() * 100), "1": float((labels == 1).mean() * 100)},
        "example_windows_2s": {"total": int(len(example_windows)), "class_0": int(example_windows.eq(0).sum()), "class_1": int(example_windows.eq(1).sum())},
        "all_recordings": {"recordings": int(windows["2s"]["recording_id"].nunique()), "seizures": int(len(all_durations)), "duration_mean": float(all_durations.mean()), "duration_median": float(np.median(all_durations)), "duration_min": float(all_durations.min()), "duration_max": float(all_durations.max()), "duration_cv": float(all_durations.std() / all_durations.mean()), "seizure_seconds": float(all_durations.sum()), "recorded_seconds": total_seconds, "seizure_time_percent": float(all_durations.sum() / total_seconds * 100)},
        "windows": {name: {"total": int(len(frame)), "class_0": int(frame["label"].eq(0).sum()), "class_1": int(frame["label"].eq(1).sum())} for name, frame in windows.items()},
    }, ROOT / "reports" / "exploration_summary.json")
    print((ROOT / "reports" / "exploration_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
