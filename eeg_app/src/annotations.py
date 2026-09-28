"""Annotation cleaning, seizure pairing, editing, and statistics."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd


def clean_label(value: object) -> str:
    """Repair common mojibake and whitespace in annotation labels."""
    text = "" if pd.isna(value) else str(value).strip()
    for _ in range(2):
        try:
            repaired = text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == text:
            break
        text = repaired
    return unicodedata.normalize("NFC", text)


def clean_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return annotations with standard columns and numeric times."""
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    for column in ("Time", "Duration"):
        if column not in result:
            result[column] = np.nan
    if "Annotations" not in result:
        result["Annotations"] = ""
    result["Annotations"] = result["Annotations"].map(clean_label)
    result["Time"] = pd.to_numeric(result["Time"], errors="coerce")
    result["Duration"] = pd.to_numeric(result["Duration"], errors="coerce")
    return result[["Time", "Annotations", "Duration"]]


def pair_seizures(annotations: pd.DataFrame, labels: Iterable[str] | None = None) -> pd.DataFrame:
    """Pair start/end annotation events into a normalized seizure table."""
    frame = clean_annotations(annotations).sort_values("Time", na_position="last")
    terms = [str(label).casefold() for label in (labels or ["absence"])]
    starts: list[float] = []
    rows: list[dict[str, float | str]] = []
    for row in frame.itertuples(index=False):
        label = str(row.Annotations)
        lower = label.casefold()
        is_match = any(term in lower for term in terms)
        if not is_match or pd.isna(row.Time):
            continue
        if re.search(r"\b(debut|début|start|begin)\b", lower) or "yo" in lower:
            starts.append(float(row.Time))
        elif re.search(r"\b(fin|end|stop)\b", lower) or "yf" in lower:
            if starts:
                start = starts.pop(0)
                if float(row.Time) >= start:
                    rows.append({"Crise": f"Crise {len(rows) + 1}", "Début (s)": start, "Fin (s)": float(row.Time)})
    return normalize_seizure_table(pd.DataFrame(rows))


def normalize_seizure_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize manually edited seizure intervals and compute their duration."""
    result = frame.copy()
    rename = {"Start": "Début (s)", "End": "Fin (s)", "Duration": "Durée (s)"}
    result = result.rename(columns=rename)
    for column, default in (("Crise", ""), ("Début (s)", np.nan), ("Fin (s)", np.nan)):
        if column not in result:
            result[column] = default
    result["Début (s)"] = pd.to_numeric(result["Début (s)"], errors="coerce")
    result["Fin (s)"] = pd.to_numeric(result["Fin (s)"], errors="coerce")
    result["Durée (s)"] = result["Fin (s)"] - result["Début (s)"]
    result = result.loc[result["Début (s)"].notna() & result["Fin (s)"].notna()]
    result = result.loc[result["Durée (s)"].gt(0)].copy()
    result["Crise"] = [value or f"Crise {index + 1}" for index, value in enumerate(result["Crise"].astype(str))]
    return result[["Crise", "Début (s)", "Fin (s)", "Durée (s)"]].reset_index(drop=True)


def duration_statistics(seizures: pd.DataFrame) -> dict[str, float]:
    """Calculate duration and between-seizure interval summary statistics."""
    values = normalize_seizure_table(seizures)
    durations = values["Durée (s)"].to_numpy(dtype=float)
    ordered = values.sort_values("Début (s)")
    intervals = (ordered["Début (s)"].iloc[1:].to_numpy() - ordered["Fin (s)"].iloc[:-1].to_numpy()) if len(ordered) > 1 else np.array([])
    return {"count": float(len(durations)), "mean": float(np.mean(durations)) if len(durations) else 0.0, "min": float(np.min(durations)) if len(durations) else 0.0, "max": float(np.max(durations)) if len(durations) else 0.0, "mean_interval": float(np.mean(intervals)) if len(intervals) else 0.0, "cv": float(np.std(durations) / np.mean(durations)) if len(durations) and np.mean(durations) else 0.0}