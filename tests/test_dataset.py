"""Tests for the supervised window-level dataset pipeline."""

import numpy as np
import pandas as pd
import pytest

from eeg_app.src.dataset import balance_dataset, balancing_audit, build_dataset_from_recordings, build_recording_dataset, export_dataset, group_split, representative_group_split, split_xy
from eeg_app.src.experiments import balance_sequences, feature_names


def synthetic_signal() -> pd.DataFrame:
    return pd.DataFrame({"Time": np.arange(16, dtype=float), "EEG1": np.arange(16, dtype=float), "EEG2": np.ones(16), "EMG1": np.arange(16, dtype=float) * 2})


def intervals() -> pd.DataFrame:
    return pd.DataFrame({"Début (s)": [2.0], "Fin (s)": [5.0]})


def test_dataset_schema_and_threshold():
    result = build_recording_dataset(synthetic_signal(), "patient-a", "recording-a", ["EEG1", "EEG2"], intervals(), window_duration=4, overlap=0)
    assert list(result.columns) == ["patient_id", "recording_id", "window_id", "start_time", "end_time", *[f"EEG{channel}_{name}" for channel in ("1", "2") for name in ("Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy")], "label"]
    assert result["window_id"].tolist() == [1, 2, 3, 4]
    assert result["label"].tolist() == [1, 1, 0, 0]


def test_seventeen_channels_produce_136_features():
    channels = [f"EEG{index}" for index in range(17)]
    frame = pd.DataFrame({"Time": np.arange(4, dtype=float), **{channel: np.ones(4) for channel in channels}})
    result = build_recording_dataset(frame, "p", "r", channels, pd.DataFrame(columns=["Début (s)", "Fin (s)"]), window_duration=2, overlap=0)
    assert len([column for column in result.columns if column not in {"patient_id", "recording_id", "window_id", "start_time", "end_time", "label"}]) == 136
    assert len(result.columns) == 142


def test_invalid_annotation_recording_is_excluded():
    data, warnings = build_dataset_from_recordings([("p", "r", synthetic_signal(), pd.DataFrame())], ["EEG1"])
    assert data.empty
    assert warnings and warnings[0].excluded


def test_all_invalid_channel_window_is_rejected():
    frame = synthetic_signal()
    frame["EEG1"] = np.nan
    try:
        build_recording_dataset(frame, "p", "r", ["EEG1"], intervals(), window_duration=4, overlap=0)
    except ValueError as error:
        assert "no finite samples" in str(error)
    else:
        raise AssertionError("An all-invalid channel window must not become a silent feature row")


def test_balance_methods_and_reproducibility():
    frame = pd.DataFrame({"label": [0, 0, 0, 1], "value": [1, 2, 3, 4]})
    undersampled, report = balance_dataset(frame, "undersample", seed=7)
    assert undersampled["label"].value_counts().to_dict() == {0: 1, 1: 1}
    assert report.final_class_0 == report.final_class_1
    oversampled, _ = balance_dataset(frame, "oversample", target_per_class=3, seed=7)
    repeat, _ = balance_dataset(frame, "oversample", target_per_class=3, seed=7)
    pd.testing.assert_frame_equal(oversampled, repeat)
    assert oversampled["label"].value_counts().to_dict() == {0: 3, 1: 3}


def synthetic_balancing_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": [f"p-{index % 10}" for index in range(1000)],
        "recording_id": [f"r-{index % 20}" for index in range(1000)],
        "window_id": np.arange(1000),
        "start_time": np.arange(1000, dtype=float),
        "end_time": np.arange(1000, dtype=float) + 1,
        "feature": np.arange(1000, dtype=float) * 0.5,
        "label": [0] * 900 + [1] * 100,
    })


EXPECTED_BALANCED_COUNTS = {"undersample": (100, 100, 200), "oversample": (900, 900, 1800), "hybrid": (500, 500, 1000)}


def test_synthetic_original_counts():
    frame = synthetic_balancing_frame()
    assert frame["label"].value_counts().to_dict() == {0: 900, 1: 100}
    assert len(frame) == 1000


@pytest.mark.parametrize("method", list(EXPECTED_BALANCED_COUNTS))
def test_balancing_exact_counts_schema_and_reproducibility(method):
    frame = synthetic_balancing_frame()
    label_zero, label_one, total = EXPECTED_BALANCED_COUNTS[method]
    balanced, report = balance_dataset(frame, method, seed=11)
    counts = balanced["label"].value_counts()
    assert set(balanced["label"].unique()) == {0, 1}
    assert counts[0] == counts[1]
    assert counts.to_dict() == {0: label_zero, 1: label_one}
    assert len(balanced) == total
    assert list(balanced.columns) == list(frame.columns)
    assert balanced["label"].notna().all()
    assert (report.final_class_0, report.final_class_1, report.target_per_class) == (label_zero, label_one, label_zero)
    repeated, _ = balance_dataset(frame, method, seed=11)
    pd.testing.assert_frame_equal(balanced, repeated)
    # Row alignment: every sampled row is an unmodified source row.
    source = frame.set_index("window_id")
    pd.testing.assert_frame_equal(balanced.set_index("window_id"), source.loc[balanced["window_id"]], check_names=False)


def test_balancing_methods_are_genuinely_different():
    frame = synthetic_balancing_frame()
    undersampled, _ = balance_dataset(frame, "undersample", seed=3)
    oversampled, _ = balance_dataset(frame, "oversample", seed=3)
    hybrid, _ = balance_dataset(frame, "hybrid", seed=3)
    minority, majority = set(range(900, 1000)), set(range(900))
    # Undersampling keeps every minority row once and drops majority rows.
    assert set(undersampled.loc[undersampled["label"].eq(1), "window_id"]) == minority
    assert not undersampled["window_id"].duplicated().any()
    # Oversampling keeps every majority row once and duplicates minority rows.
    assert set(oversampled.loc[oversampled["label"].eq(0), "window_id"]) == majority
    assert not oversampled.loc[oversampled["label"].eq(0), "window_id"].duplicated().any()
    assert oversampled.loc[oversampled["label"].eq(1), "window_id"].duplicated().any()
    # Hybrid drops majority rows and duplicates minority rows.
    assert not hybrid.loc[hybrid["label"].eq(0), "window_id"].duplicated().any()
    assert hybrid.loc[hybrid["label"].eq(1), "window_id"].duplicated().any()
    assert len({len(undersampled), len(oversampled)}) == 2
    assert not hybrid["window_id"].sort_values().reset_index(drop=True).equals(frame["window_id"])


def test_balancing_seed_changes_selection():
    frame = synthetic_balancing_frame()
    first, _ = balance_dataset(frame, "undersample", seed=1)
    second, _ = balance_dataset(frame, "undersample", seed=2)
    assert not first.equals(second)


def test_balancing_rejects_missing_class():
    with pytest.raises(ValueError, match="both label classes"):
        balance_dataset(pd.DataFrame({"label": [0, 0], "feature": [1, 2]}), "oversample")
    with pytest.raises(ValueError, match="both label classes"):
        balance_dataset(pd.DataFrame({"label": [1, 1], "feature": [1, 2]}), "undersample")


def test_balancing_rejects_null_labels():
    with pytest.raises(ValueError, match="non-null"):
        balance_dataset(pd.DataFrame({"label": [0, 1, None], "feature": [1, 2, 3]}), "hybrid")


def test_identifiers_and_label_are_not_features():
    frame = synthetic_balancing_frame().assign(split="train")
    assert feature_names(frame) == ["feature"]
    x, _ = split_xy(frame)
    assert list(x.columns) == ["feature"]


def test_balancing_does_not_change_validation_or_test():
    frame = synthetic_balancing_frame()
    frame["label"] = [0] * 700 + [1] * 100 + [0] * 90 + [1] * 10 + [0] * 90 + [1] * 10
    frame["split"] = ["train"] * 800 + ["validation"] * 100 + ["test"] * 100
    validation_before = frame.loc[frame["split"].eq("validation")].copy(deep=True)
    test_before = frame.loc[frame["split"].eq("test")].copy(deep=True)
    for method, total in (("undersample", 200), ("oversample", 1400), ("hybrid", 800)):
        training, _ = balance_dataset(frame.loc[frame["split"].eq("train")].drop(columns="split"), method, seed=4)
        assert len(training) == total
        pd.testing.assert_frame_equal(frame.loc[frame["split"].eq("validation")], validation_before)
        pd.testing.assert_frame_equal(frame.loc[frame["split"].eq("test")], test_before)
        # No validation or test window leaks into the balanced training rows.
        assert set(training["window_id"]).isdisjoint(set(validation_before["window_id"]) | set(test_before["window_id"]))


def grouped_frame(recordings: int = 10) -> pd.DataFrame:
    rows = []
    for recording in range(recordings):
        rows.extend({"patient_id": f"p-{recording}", "recording_id": f"r-{recording}", "window_id": index, "start_time": index, "label": int(index < 20 and recording % 2 == 0)} for index in range(40))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("group_by, column", [("recording", "recording_id"), ("patient", "patient_id")])
def test_split_keeps_groups_in_exactly_one_partition(group_by, column):
    split, assignments, _ = representative_group_split(grouped_frame(), group_by=group_by, seed=42)
    groups = {name: set(split.loc[split["split"].eq(name), column]) for name in ("train", "validation", "test")}
    assert all(groups.values())
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert set().union(*groups.values()) == set(split[column])
    assert set(assignments) == set(split[column])
    # About 20% of the groups are held out for the final test.
    assert len(groups["test"]) == 2


def test_split_is_reproducible():
    first, _, _ = representative_group_split(grouped_frame(), seed=42)
    second, _, _ = representative_group_split(grouped_frame(), seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_small_group_fallback_split_stays_isolated():
    split, _, _ = representative_group_split(grouped_frame(4), group_by="recording", seed=42)
    assert all(split.groupby("recording_id")["split"].nunique() == 1)
    assert {"train", "test"} <= set(split["split"])
    assert split.loc[split["split"].eq("train"), "recording_id"].nunique() >= 2


def test_balancing_audit_report_rows_and_warnings():
    split, _, _ = representative_group_split(grouped_frame(), seed=42)
    audit = balancing_audit(split, seed=42)
    assert audit["dataset_name"].tolist() == ["original", "train", "validation", "test", "train_undersample", "train_oversample", "train_hybrid"]
    assert {"total_rows", "label_0_rows", "label_1_rows", "label_0_percent", "label_1_percent", "imbalance_ratio", "unique_recordings", "unique_patients"} <= set(audit.columns)
    original = audit.iloc[0]
    assert (original["total_rows"], original["label_0_rows"], original["label_1_rows"]) == (400, 300, 100)
    assert original["imbalance_ratio"] == 3.0
    assert original["unique_recordings"] == 10
    balanced = audit.loc[audit["dataset_name"].str.startswith("train_")]
    assert (balanced["label_0_rows"] == balanced["label_1_rows"]).all()
    assert (balanced["imbalance_ratio"] == 1.0).all()
    assert balanced["total_rows"].nunique() == 3
    # Validation and test hold a single seizure event each: flagged as weak evidence.
    assert "seizure events" in audit.loc[audit["dataset_name"].eq("test"), "warnings"].iloc[0]


def test_balance_sequences_uses_shared_targets():
    values = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
    labels = np.array([0] * 900 + [1] * 100)
    for method, (label_zero, label_one, total) in EXPECTED_BALANCED_COUNTS.items():
        balanced_values, balanced_labels = balance_sequences(values, labels, method, seed=5)
        assert len(balanced_values) == total
        assert np.bincount(balanced_labels).tolist() == [label_zero, label_one]
        # Each sequence keeps its own label.
        assert np.array_equal(balanced_labels, labels[balanced_values[:, 0, 0].astype(int)])


def test_group_split_isolation_and_xy():
    frame = pd.DataFrame({"patient_id": ["a", "a", "b", "c"], "recording_id": ["r1", "r1", "r2", "r3"], "EEG1_Mean": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
    split, assignments = group_split(frame, group_by="patient", seed=3)
    assert all(split.groupby("patient_id")["split"].nunique() == 1)
    assert set(assignments) == {"a", "b", "c"}
    x, y = split_xy(frame)
    assert list(x.columns) == ["EEG1_Mean"]
    assert y.tolist() == [0, 1, 0, 1]


def test_export_has_no_index(tmp_path):
    frame = pd.DataFrame({"patient_id": ["p"], "label": [1], "EEG1_Mean": [2.0]})
    path = export_dataset(frame, tmp_path / "features.csv")
    exported = pd.read_csv(path)
    assert list(exported.columns) == list(frame.columns)
    assert len(exported) == 1
