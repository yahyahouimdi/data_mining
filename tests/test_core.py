"""Focused tests for the pure EEG lab helpers."""

import numpy as np
import pandas as pd

from eeg_app.src.annotations import annotation_time_to_signal_seconds, duration_statistics, excel_seizures_for_recording, pair_seizures
from eeg_app.src.features import build_feature_dataset, temporal_features
from eeg_app.src.signal_utils import sampling_frequency
from eeg_app.src.target import build_target, segment_signal


def test_sampling_frequency():
    assert sampling_frequency(pd.Series([0, 1 / 256, 2 / 256])) == 256


def test_pair_seizures():
    annotations = pd.DataFrame({"Time": [2, 5], "Annotations": ["absence Début", "absence Fin"]})
    result = pair_seizures(annotations)
    assert result.loc[0, "Durée (s)"] == 3


def test_pair_real_world_yo_yf_labels():
    annotations = pd.DataFrame({"Time": [10, 12, 20, 22], "Annotations": ["YO", "YF", "YO", "YF"]})
    result = pair_seizures(annotations, labels=["absence", "yo", "yf"])
    assert result[["Début (s)", "Fin (s)"]].values.tolist() == [[10.0, 12.0], [20.0, 22.0]]


def test_excel_clock_time_conversion():
    trace_start = 0.426099537037037
    seizure_start = 0.441041666666667
    seizure_end = 0.441122685185185
    assert np.isclose(annotation_time_to_signal_seconds(seizure_start, trace_start), 1291.0, atol=0.01)
    assert np.isclose(annotation_time_to_signal_seconds(seizure_end, trace_start), 1298.0, atol=0.01)


def test_excel_recording_specific_intervals():
    workbook = pd.DataFrame({"Unnamed: 0": ["190304A-E", "230105B-H"], "DebutTrace": [0.426099537037037, 0.5], "DebCE1": [0.441041666666667, 0.51], "FinCE1": [0.441122685185185, 0.52]})
    seizures, _ = excel_seizures_for_recording(workbook, "190304A_E", 1787.996)
    assert np.isclose(seizures.loc[0, "Début (s)"], 1291.0, atol=0.01)
    assert np.isclose(seizures.loc[0, "Fin (s)"], 1298.0, atol=0.01)


def test_duration_statistics():
    seizures = pd.DataFrame({"Crise": ["a", "b"], "Début (s)": [1, 10], "Fin (s)": [3, 13]})
    stats = duration_statistics(seizures)
    assert stats["count"] == 2 and stats["mean_interval"] == 7


def test_build_target():
    seizures = pd.DataFrame({"Début (s)": [2], "Fin (s)": [4]})
    np.testing.assert_array_equal(build_target(np.arange(6), seizures), [0, 0, 1, 1, 1, 0])


def test_window_classification_threshold():
    frame = pd.DataFrame({"Time": np.arange(8, dtype=float), "Class": [0, 0, 1, 1, 1, 0, 0, 0]})
    windows = segment_signal(frame, 1, 4, 0)
    assert windows["Class"].tolist() == [1, 0]
    assert windows["Seizure_Ratio"].tolist() == [0.5, 0.25]


def test_window_count_at_256_hz():
    frame = pd.DataFrame({"Time": np.arange(1024) / 256, "Class": np.zeros(1024, dtype=np.uint8)})
    windows = segment_signal(frame, 256, 2, 0.5)
    assert len(windows) == 3


def test_temporal_features_known_array():
    features = temporal_features(np.array([1.0, 2.0, 3.0]))
    assert features["Mean"] == 2
    assert np.isclose(features["Std"], np.sqrt(2 / 3))
    assert features["Variance"] == 2 / 3
    assert features["Min"] == 1
    assert features["Max"] == 3
    assert features["Amplitude"] == 2
    assert np.isclose(features["RMS"], np.sqrt(14 / 3))
    assert features["Energy"] == 14


def test_feature_dataset_schema_and_values():
    frame = pd.DataFrame({"Time": np.arange(8, dtype=float), "Class": [0, 0, 1, 1, 1, 0, 0, 0], "EEGFp1": np.arange(8, dtype=float)})
    features = build_feature_dataset(frame, ["EEGFp1"], 1, 4, 0)
    expected = {"Window", "Start_Time", "End_Time", "Duration", "Seizure_Ratio", "Class", "EEGFp1_Mean", "EEGFp1_Std", "EEGFp1_Variance", "EEGFp1_Min", "EEGFp1_Max", "EEGFp1_Amplitude", "EEGFp1_RMS", "EEGFp1_Energy"}
    assert len(features) == 2
    assert expected.issubset(features.columns)
    assert not features[list(expected)].isna().any().any()