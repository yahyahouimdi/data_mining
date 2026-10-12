# Notebooks

The notebooks are the interactive, one-dataset-at-a-time view of the pipeline.
The scripts in `scripts/` run the same pipeline over all six datasets and
produce the numbers used in the report; see the project README.

| Notebook | Purpose |
|---|---|
| `00_data_preparation_and_eda.ipynb` | Build the window datasets, the split, and the balanced exports; explore the features |
| `01` to `04` | Decision Tree, Random Forest, KNN, SVM with a validation grid search |
| `05_cnn_1d.ipynb`, `06_lstm.ipynb` | Sequence models over five consecutive windows |
| `07_model_comparison_and_crisis_reconstruction.ipynb` | Compare the notebook runs and merge predicted windows into intervals |
| `08` to `11` | Google Colab launchers for the notebooks above |

## Choosing the dataset

Notebooks `01` to `06` read one export folder, selected with the
`EEG_DATASET_FOLDER` environment variable (default `window_1s_hybrid`):

```powershell
$env:EEG_DATASET_FOLDER = "window_2s_oversample"
jupyter notebook notebooks\02_random_forest.ipynb
```

The classical notebooks train on that folder's balanced training rows. The CNN
and LSTM notebooks build their sequences from the unbalanced rows of the same
window duration and then balance the training sequences with the folder's
method, because resampled rows are no longer consecutive in time.

## Rules that every notebook follows

- Training rows only are used to fit models, imputation, and scaling.
- The validation partition selects hyperparameters; the test partition is
  evaluated once.
- Partitions are groups of whole recordings. `patient_id` is a copy of the
  recording ID, so results are recording-independent, not patient-independent.

## Google Colab

Clone or upload the repository, then:

```python
%cd /content/project
%pip install -r requirements.txt
!python scripts/train_classical_models.py
!python scripts/train_neural_models.py
```

A GPU runtime only helps the CNN and LSTM.
