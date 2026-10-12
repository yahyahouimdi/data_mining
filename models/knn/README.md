# KNN

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\train_classical_models.py
```

The launcher writes one `knn.joblib` per exported dataset under `models/<dataset>/` and stores test predictions/metrics under `results/<dataset>/`.

KNN uses median imputation and standardization fitted on training data only.
