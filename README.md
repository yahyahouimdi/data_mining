# EEG Absence-Seizure Detection

A complete Data Mining pipeline that goes from raw EEG recordings to a model
that detects absence seizures and reconstructs their start and end times. It
was built for the mini-project *Détection automatique des crises d'absence à
partir de signaux EEG* (`objectif.docx`).

The repository has two parts:

- **An interactive Streamlit application** for exploring recordings, reviewing
  annotations, building the target, segmenting the signal, and extracting
  features (Parts I to V of the assignment).
- **A reproducible machine-learning pipeline** that builds leakage-safe
  datasets, trains six models, compares them, and reconstructs seizures
  (Parts VI to XII).

The written deliverables are in `reports/`:

| File | Content |
|---|---|
| [EEG_Seizure_Detection_Report.docx](reports/EEG_Seizure_Detection_Report.docx) (and `.pdf`) | Full report: introduction, methodology, results for Questions 1 to 14, discussion, conclusion |
| [EEG_Seizure_Detection_Presentation.pptx](reports/EEG_Seizure_Detection_Presentation.pptx) | 16-slide presentation with speaker notes |
| [questions_9_14_report.md](reports/questions_9_14_report.md) | Answers to Questions 9 to 14 in Markdown |

```text
MAT files ──> CSV signals + Annotations.xlsx
                     │
                     ▼
     Exploration and annotation checks        (Streamlit app)
                     │
                     ▼
     Sample-level target: 0 = no seizure, 1 = seizure
                     │
                     ▼
     Windows of 1 s or 2 s, 50% overlap, labelled at the 50% rule
                     │
                     ▼
     Features: 8 temporal + 4 FFT band powers per EEG channel
                     │
                     ▼
     Split by recording: train / validation / test
                     │
                     ▼
     Balance the training rows only (undersample, oversample, hybrid)
                     │
                     ▼
     Decision Tree, Random Forest, KNN, SVM, 1D-CNN, LSTM
                     │
                     ▼
     Selection on validation ──> one evaluation on test
                     │
                     ▼
     Predicted windows merged into seizure intervals
```

## Data

| Item | Value |
|---|---|
| Recordings | 22 pairs in `data/`: `*_0000d.csv` (signal) and `*_0000h.csv` (raw annotation log) |
| Sampling rate | 256 Hz, measured from the `Time` column |
| Total duration | about 9.2 hours, 8.5 million samples |
| Channels | 12 to 23 per recording: EEG plus EMG, ECG, and SLI (photic stimulation) |
| Seizure labels | `data/Annotations.xlsx`: trace start time and `DebCEi` / `FinCEi` clock times per seizure |

`data/converter.py` converts the original MAT files to CSV, and
`data/conversion_summary.md` describes the result. The CSV and MAT files are
not committed (see `.gitignore`).

**Which labels are trusted.** The `h` files list events, but their `Time`
column is only a row index, so they cannot place a seizure in the signal.
`Annotations.xlsx` is therefore the only source of supervised labels. Each
seizure time is converted to seconds from the start of the trace:

```python
relative_seconds = (excel_event_time - DebutTrace) * 86400
```

The interval is then checked against the measured duration of the signal. For
`190304A-E`, a trace starting at 10:13:35 and a seizure from 10:35:06 to
10:35:13 give the interval 1291 s to 1298 s.

**Recordings used.** 18 of the 22 recordings have fully valid intervals and
are used for machine learning; they contain 71 seizures. Four are excluded:

| Recording | Reason |
|---|---|
| `200625A_F`, `210204B_C`, `210208B_G` | not listed in `Annotations.xlsx` |
| `210504B_C` | its first seizure is entered as lasting six hours (12:50:58 to 18:51:06), longer than the 52-minute recording |

`210504B_C` has 11 other valid seizures. It is still excluded, because the
true position of the first one is unknown and its windows would be labelled
"no seizure". The workbook also lists `230406B-A` and `230515B-G` twice with
slightly different times; the first row is used.

## Repository layout

```text
app.py                          Streamlit launcher
objectif.docx                   Assignment statement
requirements.txt

eeg_app/
  app.py                        Dashboard tabs, state, and exports
  src/
    io_utils.py                 Recording discovery, encoding fallback, Parquet cache, window reads
    channels.py                 Channel grouping (EEG, EMG, ECG, SLI) and presets
    annotations.py              Label cleaning, Excel clock conversion, seizure statistics
    target.py                   Sample-level binary target
    features.py                 The eight temporal features
    signal_utils.py             Sampling rate, normalization, decimation, spectra
    plotting.py                 Plotly figures
    ui.py                       CSS and metric cards
    dataset.py                  Window datasets, balancing, group splits, audit report
    experiments.py              Dataset exports, band powers, sequences, thresholds, event metrics

scripts/
  train_classical_models.py     Decision Tree, Random Forest, KNN, SVM on the six datasets
  train_neural_models.py        1D-CNN and LSTM on the six datasets
  cross_validate_models.py      Grouped 5-fold check of the selected classical models
  compare_models.py             Comparison tables, model selection, seizure reconstruction
  build_eda_figures.py          Class comparison, correlation, and spectrum figures
  build_exploration_figures.py  Signal, seizure, and duration figures for Questions 1 to 8
  report_style.py               Shared figure style

notebooks/                      Interactive version of the pipeline (see notebooks/README.md)
tests/                          Unit tests
reports/                        Written answers, figures, and exploratory tables
data/                           Recordings, annotations, MAT converter
extracted_data/                 Generated datasets and audit reports
models/                         Generated model files
results/                        Generated metrics, predictions, and comparison tables
```

`extracted_data/`, `models/`, and `results/` are generated; every file in them
can be rebuilt with the commands below.

## Installation

Python 3.10 or newer, from the repository root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## The Streamlit application

```powershell
streamlit run app.py
```

The app opens at `http://localhost:8501`. The sidebar selects the data folder
(default `data/`), the recording, and the channels. User-facing labels are in
French.

| Tab | What it does | Assignment |
|---|---|---|
| **Aperçu** | Rows, variables, sampling rate, duration, missing values, first rows, per-channel statistics, and an interactive signal viewer with seizure overlays | Q1, Q2 |
| **Annotations et crises** | Raw annotation log, seizure table (start, end, duration), duration statistics, timeline, and the answer to "are durations similar?" from the coefficient of variation | Q3, Q4 |
| **Variable cible** | `Class` column per sample, signal with seizure periods highlighted, class percentages, and an imbalance warning | Q5, Q6 |
| **Segmentation** | Windows of a chosen duration and overlap, labelled 1 when at least 50% of samples are in a seizure | Q7 |
| **Extraction des caractéristiques** | Eight temporal features per window and channel, class counts, box plots | Q8 |
| **Window Features & Dataset** | Multi-recording dataset, group split, training-only balancing, audit report, CSV/Parquet exports | Part VIII |
| **Comparison** | Welch power spectrum and spectrogram of a channel | Part VI |
| **Aide** | Explanations of the tabs and of the data limitations | |

Design choices worth knowing:

- **Large files.** Signals are converted to a Parquet cache in `data/.cache/`
  on first use, read by time window, and decimated with min/max buckets so
  spikes stay visible.
- **Channels.** Names are read from each file, never hard-coded, because the
  recordings use four different layouts.
- **Annotations.** The seizure table is editable and an `Annotations.xlsx` can
  be uploaded. When a recording is missing from the workbook, the app shows
  provisional intervals from the `h` file with a warning; those are never used
  for training.
- **Target.** `Class = 1` when `start <= Time <= end`. Overlapping intervals
  are merged first.
- **Windows.** `samples_per_window = round(duration * fs)` and
  `step = round(samples_per_window * (1 - overlap))`; at 256 Hz, a 2-second
  window with 50% overlap is 512 samples with a step of 256.

## The machine-learning pipeline

Run the whole pipeline from the repository root:

```powershell
# 1. Build the datasets, the split, the balanced training sets, and the CV folds
.\.venv\Scripts\python.exe -c "from eeg_app.src.experiments import prepare_balanced_exports, create_stratified_group_cv_reports, verify_exports; prepare_balanced_exports(); create_stratified_group_cv_reports(); print(verify_exports())"

# 2. Train the six models on the six datasets
.\.venv\Scripts\python.exe scripts\train_classical_models.py
.\.venv\Scripts\python.exe scripts\train_neural_models.py

# 3. Check the selected classical models with grouped cross-validation
.\.venv\Scripts\python.exe scripts\cross_validate_models.py

# 4. Compare, select, and reconstruct seizures; build the report figures
.\.venv\Scripts\python.exe scripts\compare_models.py
.\.venv\Scripts\python.exe scripts\build_eda_figures.py
.\.venv\Scripts\python.exe scripts\build_exploration_figures.py
```

Step 1 prints `[]` when every export folder passes its consistency checks.
Both training scripts accept `--models` and `--datasets` to run a subset, for
example `--models svm --datasets window_1s_hybrid`.

### Features

Each window has 12 features per EEG channel, 228 in total for 19 channels:

- **Temporal (8):** Mean, Std, Variance, Min, Max, Amplitude (`Max - Min`),
  RMS, Energy (sum of squares). Std and Variance use the population
  definition.
- **Frequency (4):** power in the Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha
  (8–13 Hz), and Beta (13–30 Hz) bands. Each window is mean-centred, its FFT
  gives a one-sided power spectral density, and the density is integrated
  over the band.

Features are always computed from the raw samples, never from the decimated
signal used for plotting. Identifier and time columns (`patient_id`,
`recording_id`, `window_id`, `start_time`, `end_time`), `split`, and `label`
are never model inputs.

### Split

- **Grouping.** Windows are grouped by `recording_id`, so every window of a
  recording is in exactly one partition. Windows overlap by 50%, so a random
  split of windows would put near-copies of training windows in the test set.
- **Proportions.** About 20% of recordings form the held-out test set. The
  remaining 80% is development data, divided about 80/20 into train and
  validation: 11 / 3 / 4 recordings.
- **Selection.** `representative_group_split` draws five
  `StratifiedGroupKFold` folds with seed 42 and keeps the fold whose size and
  seizure-window rate are closest to the whole dataset. Only labels and group
  sizes are used; model performance never influences the split.
- **Shared.** The 1-second and 2-second datasets use the same recording
  assignment.

The assignment asks for a stratified 80/20 split of windows. The split here is
stratified at the level of recordings instead, for the leakage reason above,
and adds a validation partition so that the test set is used once.

| Windows | Partition | Recordings | Rows | Label 0 | Label 1 | Seizure windows | Seizure events |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 s | train | 11 | 32,819 | 31,657 | 1,162 | 3.5% | 46 |
| 1 s | validation | 3 | 8,499 | 8,115 | 384 | 4.5% | 13 |
| 1 s | test | 4 | 10,448 | 10,028 | 420 | 4.0% | 12 |
| 2 s | train | 11 | 16,404 | 15,811 | 593 | 3.6% | 46 |
| 2 s | validation | 3 | 4,248 | 4,052 | 196 | 4.6% | 13 |
| 2 s | test | 4 | 5,222 | 5,010 | 212 | 4.1% | 12 |

The audit report flags a partition that is empty, lacks one class, or holds
fewer than five seizure events (`MIN_SEIZURE_EVENTS`). No current partition is
flagged.

### Balancing

`balance_dataset` is applied to training rows only, with a configurable seed:

| Method | Label 0 | Label 1 | 1 s rows per class | 2 s rows per class |
|---|---|---|---:|---:|
| **Undersample** | sampled without replacement down to the minority count | every row kept | 1,162 | 593 |
| **Oversample** | every row kept | sampled with replacement up to the majority count | 31,657 | 15,811 |
| **Hybrid** | sampled without replacement down to the midpoint | sampled with replacement up to the midpoint | 16,409 | 8,202 |

The hybrid target is `(majority + minority) // 2`. All columns are preserved,
each sampled row is an unmodified source row, and the result is shuffled. A
missing class, a null label, or a label other than 0 and 1 raises a
`ValueError`. Two window durations and three methods give the six datasets
`extracted_data/window_<d>s_<method>/`.

`extracted_data/balancing_audit_report.csv` lists, for the original data, each
partition, and each balanced training set: total rows, label 0 and label 1
rows and percentages, imbalance ratio, unique recordings and patients, seizure
events, and warnings. The same table is shown in the app.

### Models and selection

| Model | Input | Selected on validation |
|---|---|---|
| Decision Tree | one window | `max_depth`, `min_samples_leaf` |
| Random Forest (300 trees) | one window | `max_depth`, `min_samples_leaf` |
| KNN (standardized) | one window | `n_neighbors`, `weights` |
| SVM, RBF kernel (standardized) | one window | `C` |
| 1D-CNN | five consecutive windows | stopping epoch |
| LSTM | five consecutive windows | stopping epoch |

For every model and dataset:

1. Missing values are imputed with the training median and, where needed,
   features are standardized with training statistics.
2. Each candidate is fitted on the balanced training rows.
3. The decision threshold on the model's seizure score is chosen on the
   validation partition to maximize F1, ties going to the higher recall.
4. The candidate with the best validation F1, then recall, is kept.
5. The test partition is scored once with that model and threshold.

The final model is the one with the best validation F1 across all models and
datasets. Test metrics are reported, never used to choose.

**Sequence models.** The CNN and LSTM classify a window from the five windows
ending at it. Resampled rows are no longer consecutive in time, so sequences
are built from the unbalanced rows of one recording and one partition, and the
training sequences are then balanced with the same three rules
(`balance_sequences`).

### Leakage safeguards

- No recording is shared between train, validation, and test, or between
  cross-validation folds.
- Validation and test rows are never sampled, duplicated, or used to fit
  imputation or scaling.
- Balancing happens after the split, and inside each training fold during
  cross-validation.
- Identifier, time, `split`, and `label` columns are excluded from every
  feature matrix.
- `verify_exports` checks each export folder before either training script
  runs: no shared recording, training counts equal to the method's target, and
  validation/test files identical to the unbalanced partitions.

## Results

All numbers are on the four held-out test recordings (12 seizures, 1.45 hours),
with the training set of each model chosen on validation F1. Precision,
recall, and F1 are for the seizure class.

| Model | Training set | Validation F1 | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 2 s, undersample | 0.765 | 0.951 | 0.449 | 0.896 | 0.598 | 0.419 |
| Random Forest | 2 s, undersample | 0.797 | 0.970 | 0.582 | 0.939 | 0.718 | 0.732 |
| KNN | 2 s, undersample | 0.782 | 0.960 | 0.504 | 0.925 | 0.652 | 0.493 |
| SVM | 2 s, oversample | 0.830 | 0.973 | 0.712 | 0.571 | 0.634 | 0.630 |
| 1D-CNN | 2 s, oversample | 0.822 | 0.956 | 0.481 | 0.887 | 0.624 | 0.606 |
| **LSTM** (selected) | 1 s, undersample | **0.836** | 0.975 | 0.644 | 0.838 | **0.728** | 0.719 |

Seizure reconstruction with the selected LSTM:

| Event-level result | Value |
|---|---:|
| Seizures detected | 12 of 12 |
| Mean start error | +0.4 s |
| False-alarm intervals | 21 (14.5 per hour), 15 of them in one recording |

![Annotated and reconstructed seizures on the test recordings](reports/figures/seizure_reconstruction.png)

How far to trust these numbers:

- A model that always answers "no seizure" already has 96% accuracy, so
  accuracy is not used to compare models.
- Grouped 5-fold cross-validation of the classical models on the development
  recordings gives an F1 of 0.61 ± 0.08 for the Random Forest, 0.54 ± 0.10 for
  KNN, 0.39 ± 0.22 for the Decision Tree, and 0.30 ± 0.24 for the SVM. The
  spread between folds is as large as the differences between models.
- The LSTM's lead over the Random Forest (0.728 against 0.718) is within that
  spread, and the neural networks were not cross-validated.

Where the outputs are:

| File | Content |
|---|---|
| `results/model_comparison.csv` | all 36 runs (6 models on 6 training sets), sorted by validation F1 |
| `results/best_per_model.csv` | the table above, with confusion-matrix counts |
| `results/cross_validation_summary.csv`, `cross_validation_folds.csv` | grouped cross-validation |
| `results/selected_model_report.json` | selected model, test metrics, event-level report |
| `results/selected_model_predicted_intervals.csv`, `selected_model_true_intervals.csv` | reconstructed and annotated seizures |
| `results/selected_model_errors_by_recording.csv` | TP, FP, FN, TN per test recording |
| `results/<dataset>/` | metrics, predictions, and training history of each run |
| `models/<dataset>/` | fitted pipelines (`.joblib`) and network checkpoints (`.pt`) |
| `reports/figures/` | figures used in the report |

The full discussion, error analysis, and proposed improvements are in
[reports/questions_9_14_report.md](reports/questions_9_14_report.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Last run: `40 passed`.

| File | Covers |
|---|---|
| `tests/test_core.py` | sampling rate, seizure pairing, duration statistics, target construction, Excel clock conversion |
| `tests/test_dataset.py` | window schema and labels, exact balancing counts (900/100 gives 100/100, 900/900, 500/500), preserved columns and rows, reproducibility, errors for a missing class or null label, untouched validation and test data, group isolation, audit report, sequence balancing |
| `tests/test_experiments.py` | band powers, sequence construction, threshold selection, interval merging, event-level report |

## Limitations

- **Recording-independent, not patient-independent.** The data has no patient
  identifier: `Annotations.xlsx` gives only sex and age, so `patient_id` is a
  copy of the recording ID. If one patient has several recordings, they may
  sit in different partitions. A patient-to-recording mapping from the data
  provider is the only fix.
- **Few seizures.** The test set holds 12 seizures and the validation set 13,
  so every metric has wide uncertainty; the grouped cross-validation shows how
  much results move between recordings.
- **One excluded recording has usable seizures.** `210504B_C` could be
  recovered if the wrong end time of its first seizure were corrected in the
  workbook.
- **Small search.** Hyperparameter grids are small and the neural networks use
  one architecture each on CPU.
- **Window-level labels.** A window is "seizure" from 50% coverage, so seizure
  edges and seizures shorter than about half a window are hard to detect by
  construction.
