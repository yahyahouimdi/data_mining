"""File discovery and bounded CSV/Parquet loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .annotations import clean_annotations
from .signal_utils import sampling_frequency


@dataclass(frozen=True)
class RecordingFiles:
    """Paths and availability metadata for one recording id."""

    recording_id: str
    signal_path: Path | None
    annotation_path: Path | None


def discover_recordings(folder: str | Path) -> list[RecordingFiles]:
    """Discover recordings from suffixes rather than hard-coded recording ids."""
    root = Path(folder).expanduser()
    signals = {path.name[:-9]: path for path in root.glob("*_0000d.csv")}
    annotations = {path.name[:-9]: path for path in root.glob("*_0000h.csv")}
    ids = sorted(set(signals) | set(annotations))
    return [RecordingFiles(recording_id, signals.get(recording_id), annotations.get(recording_id)) for recording_id in ids]


def read_csv_with_fallback(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV with UTF-8 BOM and Latin-1 fallback for legacy labels."""
    source = Path(path)
    try:
        return pd.read_csv(source, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(source, encoding="latin-1", **kwargs)


def signal_columns(path: str | Path) -> list[str]:
    """Read only the signal header and return its columns."""
    return list(pd.read_csv(path, nrows=0).columns)


def _cache_path(path: Path, cache_dir: Path) -> Path:
    """Build a collision-resistant cache path from the source filename."""
    return cache_dir / f"{path.stem}.parquet"


def ensure_parquet(path: str | Path, cache_dir: str | Path) -> Path | None:
    """Convert a CSV to Parquet on demand, returning None if conversion is unavailable."""
    source = Path(path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = _cache_path(source, cache)
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    try:
        read_csv_with_fallback(source).to_parquet(target, index=False)
    except (ImportError, OSError, ValueError):
        return None
    return target


def read_signal_window(path: str | Path, channels: Iterable[str], start: float, end: float, cache_dir: str | Path) -> pd.DataFrame:
    """Read only selected columns and the requested time window."""
    source = Path(path)
    selected = ["Time", *dict.fromkeys(channels)]
    parquet = ensure_parquet(source, cache_dir)
    if parquet:
        frame = pd.read_parquet(parquet, columns=selected, filters=[[('Time', '>=', start), ('Time', '<=', end)]])
    else:
        chunks = []
        for chunk in read_csv_with_fallback(source, usecols=selected, chunksize=100_000):
            values = chunk.loc[chunk["Time"].between(start, end)]
            if not values.empty:
                chunks.append(values)
        frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=selected)
    return frame


def signal_metadata(path: str | Path, cache_dir: str | Path) -> dict[str, object]:
    """Return header, row count, sampling frequency, duration, and missing counts."""
    source = Path(path)
    columns = signal_columns(source)
    parquet = ensure_parquet(source, cache_dir)
    if parquet:
        frame = pd.read_parquet(parquet, columns=columns)
    else:
        frame = read_csv_with_fallback(source, usecols=columns)
    frequency = sampling_frequency(frame["Time"])
    missing = frame.isna().sum().to_dict()
    return {"rows": int(len(frame)), "columns": columns, "frequency": frequency, "duration": float(frame["Time"].max()) if len(frame) else 0.0, "missing": missing, "frame": frame}


def read_annotations(path: str | Path) -> pd.DataFrame:
    """Read and clean an annotation file, preserving the warning-worthy source semantics."""
    return clean_annotations(read_csv_with_fallback(path))


def annotation_source_warning() -> str:
    """Explain why h-file event times must be validated by the user."""
    return "Les fichiers h contiennent un index temporel et des durées NaT : leurs temps d'evenement ne sont pas fiables. Validez les crises manuellement ou importez Annotations.xlsx."