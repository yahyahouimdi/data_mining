"""Focused tests for the pure EEG lab helpers."""

import numpy as np
import pandas as pd

from eeg_app.src.annotations import duration_statistics, pair_seizures
from eeg_app.src.signal_utils import sampling_frequency
from eeg_app.src.target import build_target


def test_sampling_frequency():
    assert sampling_frequency(pd.Series([0, 1 / 256, 2 / 256])) == 256


def test_pair_seizures():
    annotations = pd.DataFrame({"Time": [2, 5], "Annotations": ["absence Début", "absence Fin"]})
    result = pair_seizures(annotations)
    assert result.loc[0, "Durée (s)"] == 3


def test_duration_statistics():
    seizures = pd.DataFrame({"Crise": ["a", "b"], "Début (s)": [1, 10], "Fin (s)": [3, 13]})
    stats = duration_statistics(seizures)
    assert stats["count"] == 2 and stats["mean_interval"] == 7


def test_build_target():
    seizures = pd.DataFrame({"Début (s)": [1], "Fin (s)": [2]})
    np.testing.assert_array_equal(build_target(np.arange(5), seizures), [0, 1, 1, 0, 0])