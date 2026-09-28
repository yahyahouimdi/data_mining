"""Sample-level binary target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .annotations import normalize_seizure_table


def build_target(time: pd.Series | np.ndarray, seizures: pd.DataFrame) -> np.ndarray:
    """Return 1 for samples inside any validated interval, otherwise 0."""
    values = pd.to_numeric(pd.Series(time), errors="coerce").to_numpy(dtype=float)
    target = np.zeros(len(values), dtype=np.uint8)
    for row in normalize_seizure_table(seizures).itertuples(index=False):
        target[(values >= row[1]) & (values <= row[2])] = 1
    return target