"""Grouped cross-validation of each selected classical model.

The fixed validation and test partitions hold only 13 and 12 seizure events.
This script re-estimates each classical model on five recording-level folds of
the development data (59 events), using the dataset and hyperparameters already
selected on validation. It repeats the main protocol in rotation: one fold is
held out for evaluation, the next fold selects the decision threshold, and the
remaining three are balanced with the dataset's method and used for fitting.
Held-out folds stay unbalanced. Test recordings are never used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_app.src.dataset import balance_dataset  # noqa: E402
from eeg_app.src.experiments import evaluate_predictions, feature_names, load_unbalanced_split, select_threshold  # noqa: E402
from train_classical_models import CLASSICAL_MODELS, SEED, build_model, seizure_scores  # noqa: E402


def selected_configuration(name: str) -> tuple[str, dict[str, object]]:
    """Return the dataset folder and validation record with the best validation F1, then recall."""
    records = {path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "results").glob(f"*/{name}_validation_metrics.json"))}
    if not records:
        raise FileNotFoundError(f"No validation metrics for {name}; run train_classical_models.py first")
    dataset = max(records, key=lambda key: (records[key]["F1"], records[key]["Recall"]))
    return dataset, records[dataset]


def main() -> None:
    rows: list[dict[str, object]] = []
    for name in CLASSICAL_MODELS:
        dataset, record = selected_configuration(name)
        frame, method = load_unbalanced_split(ROOT, dataset)
        features = feature_names(frame)
        duration = dataset.split("_")[1]
        folds = pd.read_csv(ROOT / "extracted_data" / f"window_{duration}_cv_folds.csv").astype({"recording_id": str})
        development = frame.loc[frame["split"].isin(["train", "validation"])].drop(columns="split")
        fold_of = development["recording_id"].astype(str).map(folds.set_index("recording_id")["cv_fold"])
        fold_ids = sorted(folds["cv_fold"].unique())
        for position, fold in enumerate(fold_ids):
            threshold_fold = fold_ids[(position + 1) % len(fold_ids)]
            train, _ = balance_dataset(development.loc[~fold_of.isin([fold, threshold_fold])], method, seed=SEED)
            inner_validation = development.loc[fold_of.eq(threshold_fold)]
            held_out = development.loc[fold_of.eq(fold)]
            model = build_model(name, record["params"]).fit(train[features], train["label"])
            threshold = select_threshold(inner_validation["label"], seizure_scores(model, inner_validation[features]))
            scores = seizure_scores(model, held_out[features])
            metrics = evaluate_predictions(held_out["label"], scores >= threshold, scores)
            rows.append({"model": name, "dataset": dataset, "fold": int(fold), "recordings": int(held_out["recording_id"].nunique()), "seizure_windows": int(held_out["label"].sum()), **{key: metrics[key] for key in ("Accuracy", "Precision", "Recall", "F1", "PR-AUC")}})
            print(f"{name} {dataset} fold {fold}: F1={metrics['F1']:.3f} recall={metrics['Recall']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "cross_validation_folds.csv", index=False)
    summary = table.groupby(["model", "dataset"])[["Precision", "Recall", "F1", "PR-AUC"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.reset_index().to_csv(ROOT / "results" / "cross_validation_summary.csv", index=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
