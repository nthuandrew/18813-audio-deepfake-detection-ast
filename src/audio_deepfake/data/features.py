"""Shared waveform-to-log-Mel preprocessing for CNN and AST models."""

from __future__ import annotations

from typing import Any


class LogMelFrontend:
    """Create fixed-size log-Mel features with no dataset-level leakage."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            import torchaudio
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Feature extraction requires torch and torchaudio; run "
                "`python3 -m pip install -e '.[dev]'`."
            ) from exc

        self._torch = torch
        self.sample_rate = int(config["sample_rate"])
        self.num_samples = round(
            self.sample_rate * float(config["clip_duration_seconds"])
        )
        self.normalize = config.get("normalize", "per_sample")
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(config["n_fft"]),
            win_length=int(config["win_length"]),
            hop_length=int(config["hop_length"]),
            n_mels=int(config["n_mels"]),
            f_min=float(config.get("f_min", 0)),
            f_max=float(config.get("f_max", self.sample_rate / 2)),
            power=2.0,
        )

    def __call__(self, waveform: Any) -> Any:
        torch = self._torch
        waveform = waveform.flatten()
        if waveform.numel() >= self.num_samples:
            waveform = waveform[: self.num_samples]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.num_samples - waveform.numel())
            )

        features = torch.log(self.mel(waveform) + 1e-6)
        if self.normalize == "per_sample":
            features = (features - features.mean()) / features.std().clamp_min(1e-6)
        return features.unsqueeze(0)  # [channel=1, mel bins, frames]
