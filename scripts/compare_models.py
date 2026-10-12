"""Compare the trained models and reconstruct seizures with the selected one.

Reads the per-dataset metrics written by the two training scripts. The model is
selected by validation F1 (seizure recall as tie-breaker); test metrics are
reported, never used for selection. Writes the comparison tables, the
reconstructed intervals, the event-level report and the error analysis to
``results/``, and the confusion-matrix and reconstruction figures to
``reports/figures/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_app.src.experiments import _recording_signal, annotated_intervals, event_report, save_json  # noqa: E402
from report_style import INK, MUTED, SEIZURE, SERIES, apply_style  # noqa: E402

MODEL_LABELS = {"decision_tree": "Decision Tree", "random_forest": "Random Forest", "knn": "KNN", "svm": "SVM", "cnn_1d": "1D-CNN", "lstm": "LSTM"}
METRICS = ("Accuracy", "Precision", "Recall", "F1", "PR-AUC")
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"


def collect() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(RESULTS.glob("*/*_validation_metrics.json")):
        name = path.name.replace("_validation_metrics.json", "")
        test_path = path.with_name(f"{name}_test_metrics.json")
        if name not in MODEL_LABELS or not test_path.exists():
            continue
        validation = json.loads(path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))
        (true_negative, false_positive), (false_negative, true_positive) = test["ConfusionMatrix"]
        rows.append({"dataset": path.parent.name, "model": name, "validation_F1": validation["F1"], "validation_Recall": validation["Recall"], "threshold": validation.get("threshold"), "params": json.dumps(validation.get("params", {})), **{metric: test.get(metric) for metric in METRICS}, "TN": true_negative, "FP": false_positive, "FN": false_negative, "TP": true_positive})
    if not rows:
        raise FileNotFoundError("No metrics found; run the two training scripts first")
    return pd.DataFrame(rows).sort_values(["validation_F1", "validation_Recall"], ascending=False).reset_index(drop=True)


def confusion_figure(best_per_model: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for axis, row in zip(axes.ravel(), best_per_model.itertuples(index=False)):
        matrix = [[row.TN, row.FP], [row.FN, row.TP]]
        axis.imshow([[0, 0], [0, 0]], cmap="Greys", vmin=0, vmax=1)
        for (i, j), value in ((index, matrix[index[0]][index[1]]) for index in ((0, 0), (0, 1), (1, 0), (1, 1))):
            correct = i == j
            axis.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, color=SERIES[0] if correct else SERIES[1], alpha=0.16 if correct else 0.22, linewidth=0))
            axis.text(j, i - 0.08, f"{value:,}", ha="center", va="center", fontsize=13, color=INK)
            axis.text(j, i + 0.22, ("TN", "FP", "FN", "TP")[i * 2 + j], ha="center", va="center", fontsize=9, color=MUTED)
        axis.set_xticks([0, 1], ["predicted 0", "predicted 1"])
        axis.set_yticks([0, 1], ["actual 0", "actual 1"])
        axis.set_title(f"{MODEL_LABELS[row.model]} ({row.dataset.replace('window_', '').replace('_', ', ')})", fontsize=10)
        axis.grid(False)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("Test confusion matrices, best dataset per model (0 = no seizure, 1 = seizure)", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "confusion_matrices.png", dpi=160)
    plt.close(figure)


def reconstruction_figure(predictions: pd.DataFrame, report: dict[str, object], title: str) -> None:
    """Plot one EEG channel per test recording with annotated and reconstructed seizures."""
    recordings = sorted(predictions["recording_id"].astype(str).unique())
    figure, axes = plt.subplots(len(recordings), 1, figsize=(11, 2.5 * len(recordings)), squeeze=False)
    true, predicted = report["true_intervals"], report["predicted"]
    for axis, recording in zip(axes[:, 0], recordings):
        signal = _recording_signal(ROOT, recording)
        channel = next(column for column in signal.columns if column.startswith("EEG"))
        step = 16  # 256 Hz -> 16 Hz is enough to show the envelope on a full recording
        axis.plot(signal["Time"].to_numpy()[::step], signal[channel].to_numpy()[::step], color=MUTED, linewidth=0.5)
        low, high = axis.get_ylim()
        height = high - low
        for row in true.loc[true["recording_id"].astype(str).eq(recording)].itertuples(index=False):
            axis.axvspan(row.start_time, row.end_time, ymin=0.0, ymax=1.0, color=SEIZURE, alpha=0.22, linewidth=0)
        for row in predicted.loc[predicted["recording_id"].astype(str).eq(recording)].itertuples(index=False):
            axis.plot([row.start_time, row.end_time], [high - 0.04 * height] * 2, color=SERIES[0], linewidth=4, solid_capstyle="butt")
        axis.set_title(f"{recording.rstrip('_')} ({channel})", fontsize=10, loc="left")
        axis.set_ylabel("amplitude")
        axis.margins(x=0)
    axes[-1, 0].set_xlabel("time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEIZURE, alpha=0.22), plt.Line2D([0], [0], color=SERIES[0], linewidth=4)]
    figure.legend(handles, ["annotated seizure", "seizure predicted by the model"], loc="lower right", bbox_to_anchor=(0.99, 0.0), ncols=2, frameon=False)
    figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    figure.savefig(FIGURES / "seizure_reconstruction.png", dpi=150)
    plt.close(figure)


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison = collect()
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    best_per_model = comparison.drop_duplicates("model").reset_index(drop=True)
    best_per_model.to_csv(RESULTS / "best_per_model.csv", index=False)
    confusion_figure(best_per_model)

    best = comparison.iloc[0]
    predictions = pd.read_csv(RESULTS / best["dataset"] / f"{best['model']}_test_predictions.csv")
    report = event_report(predictions, gap_tolerance=1.0, true_intervals=annotated_intervals(predictions["recording_id"].unique(), ROOT))
    true, predicted = report.pop("true_intervals"), report.pop("predicted")
    predicted.assign(duration=predicted["end_time"] - predicted["start_time"]).to_csv(RESULTS / "selected_model_predicted_intervals.csv", index=False)
    true.assign(duration=true["end_time"] - true["start_time"]).to_csv(RESULTS / "selected_model_true_intervals.csv", index=False)
    errors = predictions.copy()
    errors["outcome"] = errors.apply(lambda row: ("T" if row["label"] == row["prediction"] else "F") + ("P" if row["prediction"] == 1 else "N"), axis=1)
    per_recording = errors.groupby(["recording_id", "outcome"]).size().unstack(fill_value=0).reindex(columns=["TP", "FP", "FN", "TN"], fill_value=0).reset_index()
    per_recording.to_csv(RESULTS / "selected_model_errors_by_recording.csv", index=False)
    save_json({"selected_model": best["model"], "selected_dataset": best["dataset"], "selection_rule": "highest validation F1, then validation recall", "validation_F1": best["validation_F1"], "validation_Recall": best["validation_Recall"], "threshold": best["threshold"], "test": {metric: best[metric] for metric in METRICS}, "window_confusion": {key: int(best[key]) for key in ("TP", "FP", "FN", "TN")}, "events": report}, RESULTS / "selected_model_report.json")
    reconstruction_figure(predictions, {"true_intervals": true, "predicted": predicted}, f"Annotated and reconstructed seizures on the test recordings: {MODEL_LABELS[best['model']]}, {best['dataset']}")

    print(best_per_model[["model", "dataset", "validation_F1", *METRICS]].round(3).to_string(index=False))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
