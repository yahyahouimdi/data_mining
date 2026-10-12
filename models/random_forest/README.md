# Random Forest

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\train_classical_models.py
```

The launcher writes one `random_forest.joblib` per exported dataset under `models/<dataset>/`, with test predictions and metrics under `results/<dataset>/`.

The forest uses class weighting and no feature scaling. Test rows remain untouched until evaluation.
