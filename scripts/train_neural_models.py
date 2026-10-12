"""Train CNN and LSTM with the same six-folder train/validation/test protocol.

Sequences of consecutive windows are built from the unbalanced rows; only the
training sequences are balanced. Early stopping and the decision threshold use
the validation partition, and the test partition is evaluated once.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_app.src.experiments import balance_sequences, evaluate_predictions, feature_names, load_unbalanced_split, make_sequences, seed_everything, select_threshold, verify_exports  # noqa: E402

SEED = 42
SEQUENCE_LENGTH = 5
PREDICTION_COLUMNS = ["patient_id", "recording_id", "window_id", "start_time", "end_time", "label"]


class CNN1D(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(n_features, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32), nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(16, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(values.transpose(1, 2)).squeeze(-1))


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(32, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.head(self.dropout(output[:, -1]))


NEURAL_MODELS = {"cnn_1d": CNN1D, "lstm": LSTMClassifier}


def seizure_scores(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    """Class-1 probability for every sequence."""
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.tensor(values)), batch_size=512):
            scores.append(torch.softmax(model(batch.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def train_one(model: nn.Module, train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray, device: torch.device) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train with Adam and stop when validation F1 has not improved for five epochs."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    best_state, best_f1, patience, history = None, -1.0, 0, []
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for values, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(values.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(labels)
        validation_f1 = f1_score(validation_y, seizure_scores(model, validation_x, device) >= 0.5, zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(1, len(train_y)), "validation_f1": validation_f1})
        if validation_f1 > best_f1:
            best_f1, best_state, patience = validation_f1, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if patience >= 5:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(NEURAL_MODELS), choices=list(NEURAL_MODELS))
    parser.add_argument("--datasets", nargs="+", default=None, help="Export folder names; all folders by default")
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folders = sorted(path for path in (ROOT / "extracted_data").glob("window_*s_*") if path.is_dir() and (arguments.datasets is None or path.name in arguments.datasets))
    if not folders:
        raise FileNotFoundError("No exported window_* dataset folders found")
    issues = verify_exports(ROOT)
    if issues:
        raise RuntimeError("Export folders are inconsistent; rerun prepare_balanced_exports:\n" + "\n".join(issues))
    for folder in folders:
        # Sequences need contiguous windows, so they are built from the unbalanced
        # rows and only the training sequences are balanced afterwards.
        whole, balance_method = load_unbalanced_split(ROOT, folder.name)
        features = feature_names(whole)
        train_rows = whole.loc[whole["split"].eq("train")]
        imputer = SimpleImputer(strategy="median").fit(train_rows[features])
        scaler = StandardScaler().fit(imputer.transform(train_rows[features]))
        scaled = whole.copy()
        scaled[features] = scaler.transform(imputer.transform(whole[features]))
        values, labels, metadata = make_sequences(scaled, SEQUENCE_LENGTH, features)
        masks = {name: metadata["split"].eq(name).to_numpy() for name in ("train", "validation", "test")}
        train_x, train_y = balance_sequences(values[masks["train"]], labels[masks["train"]], balance_method, seed=SEED)
        output_model = ROOT / "models" / folder.name
        output_result = ROOT / "results" / folder.name
        output_model.mkdir(parents=True, exist_ok=True)
        output_result.mkdir(parents=True, exist_ok=True)
        for name in arguments.models:
            seed_everything(SEED)
            model, history = train_one(NEURAL_MODELS[name](len(features)).to(device), train_x, train_y, values[masks["validation"]], labels[masks["validation"]], device)
            validation_scores = seizure_scores(model, values[masks["validation"]], device)
            threshold = select_threshold(labels[masks["validation"]], validation_scores)
            validation_metrics = {**evaluate_predictions(labels[masks["validation"]], validation_scores >= threshold, validation_scores), "threshold": threshold, "params": {"sequence_length": SEQUENCE_LENGTH, "epochs": len(history)}}
            test_scores = seizure_scores(model, values[masks["test"]], device)
            test_prediction = (test_scores >= threshold).astype(int)
            test_metrics = {**evaluate_predictions(labels[masks["test"]], test_prediction, test_scores), "threshold": threshold}
            torch.save({"state_dict": model.state_dict(), "features": features, "sequence_length": SEQUENCE_LENGTH, "imputer": imputer, "scaler": scaler, "threshold": threshold, "dataset": folder.name, "balancing_method": balance_method}, output_model / f"{name}.pt")
            predictions = metadata.loc[masks["test"], PREDICTION_COLUMNS].copy()
            predictions["prediction"] = test_prediction
            predictions["score"] = test_scores
            predictions.to_csv(output_result / f"{name}_test_predictions.csv", index=False)
            (output_result / f"{name}_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
            (output_result / f"{name}_validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
            pd.DataFrame(history).to_csv(output_result / f"{name}_training_history.csv", index=False)
            print(f"{folder.name} {name}: train sequences={len(train_y)} ({balance_method}) validation F1={validation_metrics['F1']:.3f} test F1={test_metrics['F1']:.3f} recall={test_metrics['Recall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
