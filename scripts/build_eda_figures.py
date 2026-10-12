"""Exploratory figures and tables for the report (questions 9 to 11).

Uses the training partition only, so nothing learned here comes from the
validation or test recordings. Features are averaged over the EEG channels of
each window to give one value per feature family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_app.src.experiments import BANDS, _recording_signal, save_json  # noqa: E402
from eeg_app.src.features import FEATURE_NAMES  # noqa: E402
from eeg_app.src.signal_utils import sampling_frequency  # noqa: E402
from report_style import DIVERGING, INK, MUTED, SERIES, apply_style  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
FAMILIES = [*FEATURE_NAMES, *[f"{band}_Power" for band in BANDS]]
CLASS_LABELS = {0: "no seizure (class 0)", 1: "seizure (class 1)"}


def family_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Average each feature family over the EEG channels of every window."""
    table = pd.DataFrame({family: frame[[column for column in frame.columns if column.endswith(f"_{family}")]].mean(axis=1) for family in FAMILIES})
    table["label"] = frame["label"].to_numpy()
    return table


def spectrum_figure(train: pd.DataFrame) -> None:
    """Time-domain signal and FFT spectrum of one seizure and one non-seizure window."""
    recording = train.loc[train["label"].eq(1), "recording_id"].value_counts().index[0]
    rows = train.loc[train["recording_id"].eq(recording)].sort_values("start_time")
    seizure = rows.loc[rows["label"].eq(1)].iloc[len(rows.loc[rows["label"].eq(1)]) // 2]
    normal = rows.loc[rows["label"].eq(0)].iloc[len(rows.loc[rows["label"].eq(0)]) // 2]
    signal = _recording_signal(ROOT, str(recording))
    channel = next(column for column in signal.columns if column.startswith("EEG"))
    time = signal["Time"].to_numpy(float)
    fs = sampling_frequency(time)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 5.6))
    for column, (row, label, color) in enumerate(((normal, CLASS_LABELS[0], SERIES[0]), (seizure, CLASS_LABELS[1], SERIES[1]))):
        mask = (time >= row["start_time"]) & (time <= row["end_time"])
        values = signal.loc[mask, channel].to_numpy(float)
        axes[0, column].plot(time[mask], values, color=color, linewidth=1.2)
        axes[0, column].set_title(f"Time domain, {label}")
        axes[0, column].set_xlabel("time (s)")
        axes[0, column].set_ylabel("amplitude")
        frequencies = np.fft.rfftfreq(len(values), d=1 / fs)
        power = 2 * np.abs(np.fft.rfft(values - values.mean())) ** 2 / (fs * len(values))
        keep = frequencies <= 30
        axes[1, column].plot(frequencies[keep], power[keep], color=color, linewidth=1.2)
        axes[1, column].set_title(f"FFT power spectrum, {label}")
        axes[1, column].set_xlabel("frequency (Hz)")
        axes[1, column].set_ylabel("power spectral density")
        for band, (low, _) in BANDS.items():
            axes[1, column].axvline(low, color=MUTED, linewidth=0.6, linestyle=":")
            axes[1, column].text(low + 0.2, axes[1, column].get_ylim()[1] * 0.92, band, fontsize=8, color=MUTED)
    figure.suptitle(f"One 2-second window of {str(recording).rstrip('_')}, channel {channel}", fontsize=11, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(FIGURES / "window_time_and_spectrum.png", dpi=160)
    plt.close(figure)


def class_comparison_figure(summary: pd.DataFrame) -> None:
    """Small multiples: class means per feature family, each on its own scale."""
    shown = ["Energy", "Std", "Amplitude", "RMS", "Delta_Power", "Theta_Power", "Alpha_Power", "Beta_Power"]
    figure, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    for axis, family in zip(axes.ravel(), shown):
        means = [summary.loc[family, "mean_class_0"], summary.loc[family, "mean_class_1"]]
        bars = axis.bar(["class 0", "class 1"], means, color=SERIES[:2], width=0.55)
        axis.bar_label(bars, labels=[f"{value:,.3g}" for value in means], fontsize=8, color=INK, padding=2)
        axis.set_title(f"{family.replace('_', ' ')} (x{summary.loc[family, 'ratio_1_over_0']:.1f})")
        axis.grid(axis="x", visible=False)
        axis.set_ylim(0, max(means) * 1.18)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    figure.suptitle("Mean of each feature per class, training windows, averaged over EEG channels (class 0 = no seizure, class 1 = seizure)", fontsize=10.5, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(FIGURES / "class_comparison.png", dpi=160)
    plt.close(figure)


def correlation_figure(correlation: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 6.8))
    image = axis.imshow(correlation.to_numpy(), cmap=LinearSegmentedColormap.from_list("diverging", DIVERGING), vmin=-1, vmax=1)
    labels = [name.replace("_", " ") for name in correlation.columns]
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, f"{correlation.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    axis.grid(False)
    figure.colorbar(image, ax=axis, shrink=0.8, label="Spearman correlation")
    axis.set_title("Correlation between feature families (training windows, channel average)", loc="left")
    figure.tight_layout()
    figure.savefig(FIGURES / "correlation_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(ROOT / "extracted_data" / "window_2s_unbalanced.parquet")
    train = frame.loc[frame["split"].eq("train")]
    table = family_table(train)
    grouped = table.groupby("label")[FAMILIES]
    summary = pd.DataFrame({"mean_class_0": grouped.mean().loc[0], "mean_class_1": grouped.mean().loc[1], "std_class_0": grouped.std().loc[0], "std_class_1": grouped.std().loc[1]})
    summary["ratio_1_over_0"] = summary["mean_class_1"] / summary["mean_class_0"]
    # AUC of each feature used alone as a seizure score: 0.5 is uninformative.
    summary["auc"] = [roc_auc_score(table["label"], table[family]) for family in FAMILIES]
    summary["discrimination"] = (summary["auc"] - 0.5).abs() + 0.5
    summary.index.name = "feature"
    correlation = table[FAMILIES].corr(method="spearman")
    pairs = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1)).stack().sort_values(key=np.abs, ascending=False)
    summary.to_csv(ROOT / "reports" / "eda_class_comparison.csv")
    correlation.to_csv(ROOT / "reports" / "eda_correlation.csv")
    save_json({"source": "training partition of window_2s_unbalanced.parquet", "windows": int(len(table)), "class_counts": {str(key): int(value) for key, value in table["label"].value_counts().sort_index().items()}, "most_discriminative": summary["discrimination"].sort_values(ascending=False).round(3).to_dict(), "strongest_correlations": [{"a": a, "b": b, "spearman": round(float(value), 3)} for (a, b), value in pairs.head(15).items()]}, ROOT / "reports" / "eda_summary.json")
    spectrum_figure(train)
    class_comparison_figure(summary)
    correlation_figure(correlation)
    print(summary.round(3).to_string())
    print(pairs.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
