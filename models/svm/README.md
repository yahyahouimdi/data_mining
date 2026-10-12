# SVM

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\train_classical_models.py
```

The launcher writes one `svm.joblib` per exported dataset under `models/<dataset>/` and stores test predictions/metrics under `results/<dataset>/`.

SVM uses median imputation, standardization, class weighting, and probability scores.
