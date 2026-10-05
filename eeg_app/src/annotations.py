"""Annotation cleaning, seizure pairing, editing, and statistics."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, time
from typing import Iterable

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0


def excel_time_to_seconds(value: object) -> float:
    """Convert an Excel fractional-day or time value to seconds since midnight."""
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, pd.Timestamp):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    if isinstance(value, datetime):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)
    numeric = float(value)
    return (numeric * SECONDS_PER_DAY) % SECONDS_PER_DAY


def annotation_time_to_signal_seconds(annotation_time: object, trace_start: object) -> float:
    """Convert an absolute Excel clock time to elapsed seconds from trace start."""
    annotation_seconds = excel_time_to_seconds(annotation_time)
    trace_seconds = excel_time_to_seconds(trace_start)
    if not np.isfinite(annotation_seconds) or not np.isfinite(trace_seconds):
        return float("nan")
    return (annotation_seconds - trace_seconds) % SECONDS_PER_DAY


def normalize_recording_id(value: object) -> str:
    """Normalize recording ids such as ``190304A-E`` and ``190304A_E``."""
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def format_excel_clock(value: object) -> str:
    """Format an Excel clock value as a readable HH:MM:SS string."""
    seconds = excel_time_to_seconds(value)
    if not np.isfinite(seconds):
        return ""
    whole_seconds = int(round(seconds)) % int(SECONDS_PER_DAY)
    return f"{whole_seconds // 3600:02d}:{whole_seconds % 3600 // 60:02d}:{whole_seconds % 60:02d}"


def excel_seizures_for_recording(excel: pd.DataFrame, recording_id: str, recording_duration: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one recording's Excel seizure intervals in CSV elapsed seconds.

    The returned validation table retains the original clock examples and a
    status column. Intervals outside the signal duration are reported but are
    excluded from the intervals used for targets and red overlays.
    """
    columns = {str(column).strip().casefold(): column for column in excel.columns}
    id_column = columns.get("unnamed: 0")
    trace_column = columns.get("debuttrace")
    empty = pd.DataFrame(columns=["Crise", "Début (s)", "Fin (s)", "Durée (s)"])
    validation_columns = ["Crise", "Début du tracé", "Excel début", "Excel fin", "Début (s)", "Fin (s)", "Durée (s)", "Statut"]
    if id_column is None or trace_column is None:
        return empty, pd.DataFrame(columns=validation_columns)
    matches = excel.loc[excel[id_column].map(normalize_recording_id).eq(normalize_recording_id(recording_id))]
    if matches.empty:
        return empty, pd.DataFrame(columns=validation_columns)
    row = matches.iloc[0]
    pairs: list[tuple[int, object, object]] = []
    for column in excel.columns:
        match = re.match(r"^(?:deb|debut|debt)ce(\d+)$", str(column).strip(), re.IGNORECASE)
        if not match:
            continue
        number = int(match.group(1))
        end_column = next((candidate for candidate in excel.columns if str(candidate).strip().casefold() == f"fince{number}"), None)
        if end_column is not None:
            pairs.append((number, row[column], row[end_column]))
    intervals: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    for number, start_value, end_value in sorted(pairs):
        if pd.isna(start_value) or pd.isna(end_value):
            continue
        start = annotation_time_to_signal_seconds(start_value, row[trace_column])
        end = annotation_time_to_signal_seconds(end_value, row[trace_column])
        duration = end - start
        if duration < 0:
            end += SECONDS_PER_DAY
            duration = end - start
        status = "Valide"
        valid = np.isfinite(start) and np.isfinite(end) and start >= 0 and end >= start
        if recording_duration is not None and valid and (start > recording_duration or end > recording_duration):
            status = "Hors durée du signal"
            valid = False
        debug_rows.append({"Crise": f"Crise {number}", "Début du tracé": format_excel_clock(row[trace_column]), "Excel début": format_excel_clock(start_value), "Excel fin": format_excel_clock(end_value), "Début (s)": start, "Fin (s)": end, "Durée (s)": duration, "Statut": status})
        if valid:
            intervals.append({"Crise": f"Crise {number}", "Début (s)": start, "Fin (s)": end, "Durée (s)": duration})
    return normalize_seizure_table(pd.DataFrame(intervals)), pd.DataFrame(debug_rows, columns=validation_columns)


def clean_label(value: object) -> str:
    """Repair common mojibake and whitespace in annotation labels."""
    text = "" if pd.isna(value) else str(value).strip()
    text = text.replace("ï¿½ï¿½", "é").replace("ï¿½", "é").replace("��", "é").replace("�", "é")
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