"""Train the four classical models on every exported duration/balancing folder.

Hyperparameters and the decision threshold are selected on the validation
partition (F1 first, seizure recall as tie-breaker). The test partition is
evaluated once with the selected model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_app.src.experiments import classical_pipeline, evaluate_predictions, feature_names, select_threshold, verify_exports  # noqa: E402

SEED = 42
PREDICTION_COLUMNS = ["patient_id", "recording_id", "window_id", "start_time", "end_time", "label"]
# name: (estimator factory, scale features, validation grid)
CLASSICAL_MODELS = {
    "decision_tree": (lambda: DecisionTreeClassifier(random_state=SEED), False, {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [5, 20]}),
    "random_forest": (lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1), False, {"model__max_depth": [12, None], "model__min_samples_leaf": [1, 5]}),
    "knn": (lambda: KNeighborsClassifier(), True, {"model__n_neighbors": [5, 11, 21], "model__weights": ["uniform", "distance"]}),
    "svm": (lambda: SVC(kernel="rbf", cache_size=1000, random_state=SEED), True, {"model__C": [0.1, 1.0, 10.0]}),
}


def build_model(name: str, params: dict[str, object]) -> Pipeline:
    factory, scale, _ = CLASSICAL_MODELS[name]
    return classical_pipeline(factory(), scale=scale).set_params(**params)


def seizure_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Continuous seizure score: class-1 probability, or the SVM margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def select_on_validation(name: str, train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> tuple[Pipeline, dict[str, object]]:
    """Fit every grid candidate on train and keep the best validation F1, then recall."""
    best: tuple[tuple[float, float], Pipeline, dict[str, object]] | None = None
    candidates: list[dict[str, object]] = []
    for params in ParameterGrid(CLASSICAL_MODELS[name][2]):
        model = build_model(name, params).fit(train[features], train["label"])
        scores = seizure_scores(model, validation[features])
        threshold = select_threshold(validation["label"], scores)
        metrics = evaluate_predictions(validation["label"], scores >= threshold, scores)
        candidates.append({"params": params, "threshold": threshold, "F1": metrics["F1"], "Recall": metrics["Recall"], "Precision": metrics["Precision"]})
        key = (float(metrics["F1"]), float(metrics["Recall"]))
        if best is None or key > best[0]:
            best = (key, model, {**metrics, "params": params, "threshold": threshold})
    assert best is not None
    return best[1], {**best[2], "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(CLASSICAL_MODELS), choices=list(CLASSICAL_MODELS))
    parser.add_argument("--datasets", nargs="+", default=None, help="Export folder names; all folders by default")
    arguments = parser.parse_args()
    folders = sorted(path for path in (ROOT / "extracted_data").glob("window_*s_*") if path.is_dir() and (arguments.datasets is None or path.name in arguments.datasets))
    if not folders:
        raise FileNotFoundError("No exported window_* dataset folders found")
    issues = verify_exports(ROOT)
    if issues:
        raise RuntimeError("Export folders are inconsistent; rerun prepare_balanced_exports:\n" + "\n".join(issues))
    for folder in folders:
        train = pd.read_csv(folder / "train.csv")
        validation = pd.read_csv(folder / "validation.csv")
        test = pd.read_csv(folder / "test.csv")
        features = feature_names(train)
        destination = ROOT / "results" / folder.name
        artifact = ROOT / "models" / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        artifact.mkdir(parents=True, exist_ok=True)
        for name in arguments.models:
            model, validation_metrics = select_on_validation(name, train, validation, features)
            threshold = float(validation_metrics["threshold"])
            scores = seizure_scores(model, test[features])
            prediction = (scores >= threshold).astype(int)
            test_metrics = {**evaluate_predictions(test["label"], prediction, scores), "threshold": threshold, "params": validation_metrics["params"]}
            predictions = test[PREDICTION_COLUMNS].copy()
            predictions["prediction"] = prediction
            predictions["score"] = scores
            predictions.to_csv(destination / f"{name}_test_predictions.csv", index=False)
            (destination / f"{name}_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
            (destination / f"{name}_validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
            joblib.dump({"pipeline": model, "threshold": threshold, "features": features, "params": validation_metrics["params"], "dataset": folder.name}, artifact / f"{name}.joblib")
            print(f"{folder.name} {name}: validation F1={validation_metrics['F1']:.3f} test F1={test_metrics['F1']:.3f} recall={test_metrics['Recall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
