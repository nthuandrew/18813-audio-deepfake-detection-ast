"""Parse the official ASVspoof 2019 LA countermeasure protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LABEL_TO_TARGET = {"bonafide": 1, "spoof": 0}

PROTOCOL_FILENAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

EXPECTED_COUNTS = {
    "train": {"bonafide": 2580, "spoof": 22800},
    "dev": {"bonafide": 2548, "spoof": 22296},
    "eval": {"bonafide": 7355, "spoof": 63882},
}


@dataclass(frozen=True)
class ProtocolEntry:
    speaker_id: str
    utterance_id: str
    attack_id: str
    label: str
    target: int
    audio_path: Path


def protocol_path(la_root: str | Path, split: str) -> Path:
    """Return the official protocol file for a canonical split."""
    _check_split(split)
    return (
        Path(la_root)
        / "ASVspoof2019_LA_cm_protocols"
        / PROTOCOL_FILENAMES[split]
    )


def audio_directory(la_root: str | Path, split: str) -> Path:
    """Return the FLAC directory for a canonical split."""
    _check_split(split)
    return Path(la_root) / f"ASVspoof2019_LA_{split}" / "flac"


def load_protocol(la_root: str | Path, split: str) -> list[ProtocolEntry]:
    """Load labels and paths without requiring pandas."""
    source = protocol_path(la_root, split)
    audio_dir = audio_directory(la_root, split)
    entries: list[ProtocolEntry] = []

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{source}:{line_number}: expected 5 fields, got {len(fields)}"
                )
            speaker_id, utterance_id, _unused, attack_id, label = fields
            if label not in LABEL_TO_TARGET:
                raise ValueError(f"{source}:{line_number}: unknown label {label!r}")
            entries.append(
                ProtocolEntry(
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    attack_id=attack_id,
                    label=label,
                    target=LABEL_TO_TARGET[label],
                    audio_path=audio_dir / f"{utterance_id}.flac",
                )
            )
    return entries


def _check_split(split: str) -> None:
    if split not in PROTOCOL_FILENAMES:
        valid = ", ".join(PROTOCOL_FILENAMES)
        raise ValueError(f"Unknown split {split!r}; expected one of: {valid}")
