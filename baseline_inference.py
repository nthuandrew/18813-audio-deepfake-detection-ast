"""Run a trained CNN baseline checkpoint on one audio file.

Example:
    python3 baseline_inference.py \
        --checkpoint artifacts/cnn_baseline/seed_42/best.pt \
        --audio path/to/audio.flac
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def predict_audio(
    checkpoint_path: str | Path,
    audio_path: str | Path,
    threshold: float = 0.5,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Return the bona-fide score and binary prediction for one audio file."""
    try:
        import soundfile
        import torch
        import torchaudio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Inference requires the project dependencies; run "
            "`python3 -m pip install -e '.[dev]'`."
        ) from exc

    from audio_deepfake.data.features import LogMelFrontend
    from audio_deepfake.models import build_model
    from audio_deepfake.training.reproducibility import select_device

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
    if model_config.get("name") != "cnn":
        raise ValueError("baseline_inference.py expects a CNN baseline checkpoint")

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
        "score": score,
        "threshold": threshold,
        "prediction": prediction,
        "score_meaning": "probability-like bona-fide score",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = predict_audio(
        args.checkpoint,
        args.audio,
        threshold=args.threshold,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
