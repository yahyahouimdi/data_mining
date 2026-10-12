# Decision Tree

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\train_classical_models.py
```

The launcher trains this model on every `window_*` dataset folder and writes `decision_tree.joblib` under the matching dataset folder in `models/`. Test predictions and metrics are under `results/<dataset>/`.

This model does not require feature scaling; median imputation is fitted on training data.
