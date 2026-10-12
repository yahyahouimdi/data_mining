"""Tests for the experiment helpers: band powers, sequences, thresholds, and events."""

import numpy as np
import pandas as pd

from eeg_app.src.experiments import BANDS, band_powers, event_report, feature_names, make_sequences, merge_predicted_intervals, select_threshold


def sine_windows(frequencies, fs=256, seconds=2):
    time = np.arange(int(fs * seconds)) / fs
    return np.vstack([np.sin(2 * np.pi * frequency * time) for frequency in frequencies])


def test_band_powers_put_each_sine_in_its_band():
    powers = band_powers(sine_windows([3, 6, 10, 20]), fs=256)
    assert list(powers) == list(BANDS)
    for row, band in enumerate(BANDS):
        # A unit sine has power 0.5, all of it inside its own band.
        assert np.isclose(powers[band][row], 0.5, atol=1e-6)
        assert all(np.isclose(powers[other][row], 0.0, atol=1e-6) for other in BANDS if other != band)


def test_band_powers_ignore_offset_and_missing_samples():
    windows = sine_windows([3]) + 100.0
    windows[0, 5] = np.nan
    powers = band_powers(windows, fs=256)
    assert np.isfinite(powers["Delta"][0])
    assert np.isclose(powers["Delta"][0], 0.5, atol=0.01)


def sequence_frame() -> pd.DataFrame:
    rows = []
    for recording, split, count in (("r1", "train", 6), ("r2", "test", 4), ("r3", "validation", 2)):
        for index in range(count):
            rows.append({"patient_id": recording, "recording_id": recording, "window_id": index + 1, "start_time": float(index), "end_time": float(index + 2), "feature": float(index), "label": int(index >= count - 1), "split": split})
    return pd.DataFrame(rows)


def test_make_sequences_stay_inside_one_recording_and_use_last_label():
    frame = sequence_frame()
    values, labels, metadata = make_sequences(frame, 3, feature_names(frame))
    # 6 -> 4 sequences, 4 -> 2 sequences, 2 windows are too few for a sequence.
    assert values.shape == (6, 3, 1)
    assert metadata["recording_id"].tolist() == ["r1"] * 4 + ["r2"] * 2
    assert values[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert values[4, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert labels.tolist() == metadata["label"].tolist() == [0, 0, 0, 1, 0, 1]
    assert metadata["window_id"].tolist() == [3, 4, 5, 6, 3, 4]


def test_select_threshold_maximises_f1_and_prefers_recall_on_ties():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.6, 0.5, 0.8, 0.9]
    # 0.5 keeps all three seizures with one false alarm (F1 0.857); 0.8 gives F1 0.8.
    assert select_threshold(labels, scores) == 0.5
    assert select_threshold([0, 1], [0.3, 0.7]) == 0.7


def prediction_frame() -> pd.DataFrame:
    frame = pd.DataFrame({"recording_id": "r1", "start_time": np.arange(0, 100, dtype=float), "end_time": np.arange(2, 102, dtype=float), "label": 0, "prediction": 0})
    frame.loc[10:14, "label"] = 1   # seizure A, detected two windows late
    frame.loc[12:15, "prediction"] = 1
    frame.loc[40:43, "label"] = 1   # seizure B, missed
    frame.loc[70:71, "prediction"] = 1  # false alarm
    return frame


def test_merge_predicted_intervals_bridges_only_short_gaps():
    frame = prediction_frame()
    frame.loc[74:75, "prediction"] = 1  # starts 1 s after the previous interval ends
    merged = merge_predicted_intervals(frame, gap_tolerance=1.0)
    assert merged[["start_time", "end_time"]].to_numpy().tolist() == [[12.0, 17.0], [70.0, 77.0]]
    strict = merge_predicted_intervals(frame, gap_tolerance=0.0)
    assert len(strict) == 3


def test_event_report_counts_detected_missed_and_false_alarms():
    report = event_report(prediction_frame(), gap_tolerance=1.0)
    assert report["true_events"] == 2
    assert report["detected_events"] == 1
    assert report["missed_events"] == 1
    assert report["event_recall"] == 0.5
    assert report["predicted_intervals"] == 2
    assert report["false_alarm_intervals"] == 1
    assert report["mean_onset_delay_seconds"] == 2.0
    assert np.isclose(report["false_alarms_per_hour"], 1 / (101 / 3600))
