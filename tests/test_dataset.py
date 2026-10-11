"""Tests for the supervised window-level dataset pipeline."""

import numpy as np
import pandas as pd

from eeg_app.src.dataset import balance_dataset, build_dataset_from_recordings, build_recording_dataset, export_dataset, group_split, split_xy


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
