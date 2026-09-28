"""Channel classification and recording layout helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


CHANNEL_TYPES = ("EEG", "EMG", "ECG", "SLI", "Other")


def channel_type(name: str) -> str:
    """Classify a channel from its prefix without relying on a fixed montage."""
    upper = name.upper().strip()
    if upper.startswith("EEG"):
        return "EEG"
    if upper.startswith("EMG"):
        return "EMG"
    if upper.startswith("ECG"):
        return "ECG"
    if upper.startswith("SLI"):
        return "SLI"
    return "Other"


def group_channels(channels: Iterable[str]) -> dict[str, list[str]]:
    """Return channels grouped in stable display order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for channel in channels:
        groups[channel_type(channel)].append(channel)
    return {kind: groups[kind] for kind in CHANNEL_TYPES if groups[kind]}


def layout_name(channels: Iterable[str]) -> str:
    """Describe a discovered montage using its channel count and categories."""
    values = list(channels)
    eeg_count = sum(channel_type(value) == "EEG" for value in values)
    return f"{len(values)} colonnes ({eeg_count} EEG)"


def channel_unit(name: str) -> str:
    """Return the display unit associated with a channel category."""
    return {"EEG": "uV", "ECG": "mV", "SLI": "Hz"}.get(channel_type(name), "")