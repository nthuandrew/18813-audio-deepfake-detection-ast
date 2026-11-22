"""PyTorch dataset for ASVspoof 2019 LA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .features import LogMelFrontend
from .protocol import load_protocol


class ASVspoofDataset:
    """Return model features plus metadata needed by evaluation."""

    def __init__(self, la_root: str | Path, split: str, data_config: dict[str, Any]):
        try:
            import torch
            import torchaudio
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Dataset loading requires torch and torchaudio; install project deps."
            ) from exc

        self._torch = torch
        self._torchaudio = torchaudio
        self.entries = load_protocol(la_root, split)
        self.frontend = LogMelFrontend(data_config)
        self.target_sample_rate = int(data_config["sample_rate"])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        waveform, sample_rate = self._torchaudio.load(str(entry.audio_path))
        waveform = waveform.mean(dim=0)
        if sample_rate != self.target_sample_rate:
            waveform = self._torchaudio.functional.resample(
                waveform, sample_rate, self.target_sample_rate
            )

        return {
            "features": self.frontend(waveform),
            "target": self._torch.tensor(entry.target, dtype=self._torch.float32),
            "utterance_id": entry.utterance_id,
            "attack_id": entry.attack_id,
            "label": entry.label,
        }
