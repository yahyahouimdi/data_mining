# window_2s_oversample

2-second windows, 50% overlap, training rows balanced with `oversample` (15811 rows per class).

- `train.csv`: balanced training rows.
- `validation.csv`, `test.csv`: untouched, unbalanced rows.
- `whole.csv`: the three files together with a `split` column.
- `manifest.json`: seed, split group, balancing report, and excluded recordings.

The unbalanced table and the cross-validation folds are one level up: `window_2s_unbalanced.parquet` and `window_2s_cv_folds.csv`.
