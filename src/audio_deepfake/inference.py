"""Shared single-audio inference for trained CNN and AST checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def predict_audio(
    checkpoint_path: str | Path,
    audio_path: str | Path,
    threshold: float = 0.5,
    device_name: str = "auto",
    expected_model_name: str | None = None,
) -> dict[str, Any]:
    """Return a bona-fide score and binary prediction for one audio file."""
    try:
        import soundfile
        import torch
        import torchaudio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Inference requires the project dependencies; run "
            "`python3 -m pip install -e '.[dev]'`."
        ) from exc

    from .data.features import LogMelFrontend
    from .models import build_model
    from .training.reproducibility import select_device

    checkpoint_path = Path(checkpoint_path)
    audio_path = Path(audio_path)
    device = select_device(device_name)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise TypeError("Checkpoint does not contain the resolved training config")

    model_config = config.get("model", {})
    model_name = model_config.get("name")
    if expected_model_name is not None and model_name != expected_model_name:
        raise ValueError(
            f"Expected a {expected_model_name!r} checkpoint, got {model_name!r}"
        )

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_config = config["data"]
    audio, original_sample_rate = soundfile.read(
        audio_path, dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(audio).mean(dim=1)
    target_sample_rate = int(data_config["sample_rate"])
    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, original_sample_rate, target_sample_rate
        )

    features = LogMelFrontend(data_config)(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(features)).item()
    prediction = "bonafide" if score >= threshold else "spoof"
    return {
        "audio": str(audio_path),
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "score": score,
        "threshold": threshold,
        "prediction": prediction,
        "score_meaning": "probability-like bona-fide score",
    }
