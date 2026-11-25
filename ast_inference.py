"""Run a trained Audio Spectrogram Transformer on one audio file.

Example:
    python3 ast_inference.py \
        --checkpoint artifacts/ast/seed_42/best.pt \
        --audio path/to/audio.flac
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_deepfake.inference import predict_audio


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
        expected_model_name="ast",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
