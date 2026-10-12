# Final report outline: Questions 9-14

## Q9. Data and labels
- State the exact row/column counts from `extracted_data/preparation_report.json`.
- Confirm that labels came from validated `Annotations.xlsx` intervals and list excluded recordings.
- Report class counts and missing/duplicate windows.

## Q10. Feature engineering
- Describe the eight raw-sample temporal features per EEG channel.
- Describe Welch PSD band-power integration for Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), and Beta (13-30 Hz).
- Discuss correlated features and any feature-selection decision.

## Q11. Leakage control and split
- Report the actual 64/16/20 approximate split sizes and whether grouping used patients, recordings, or chronological blocks.
- Explain why overlapping windows are never randomly mixed across partitions.
- Confirm that imputation/scaling and balancing are training-only.

## Q12. Classical models
- Fill the comparison table from `results/model_comparison.csv`.
- Discuss Accuracy, Precision, Recall, F1, PR-AUC, confusion matrices, false alarms, and missed seizure windows.
- State that no score is reported for a notebook that did not execute successfully.

## Q13. Deep models
- Report CNN/LSTM sequence length, architecture, early-stopping epoch, and training/validation curves.
- Compare measured results against classical baselines without claiming neural superiority.
- State small-data limitations, group count, and whether the neural result is reliable enough for inference.

## Q14. Crisis reconstruction and next experiments
- Describe the selected model policy: validation F1 first, seizure recall tie-breaker.
- Report merged predicted intervals, event matches, window TP/FP/TN/FN, missed seizures, and false alarms.
- Improvement experiment 1: nested group cross-validation with patient-level calibration and threshold selection for high recall.
- Improvement experiment 2: raw-EEG sequence models with subject-normalized channels and an evaluation split by held-out patient.
- Optional improvement: ablate each frequency band and quantify whether it adds value beyond temporal features.
