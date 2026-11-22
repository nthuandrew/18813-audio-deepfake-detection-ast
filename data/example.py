"""Load ASVspoof metadata and one model-ready log-Mel sample.

Run this file from the repository root after installing the project. The local
audio files remain ignored by Git; this example is intentionally tracked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_deepfake.config import load_config
from audio_deepfake.data.dataset import ASVspoofDataset
from audio_deepfake.data.protocol import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_baseline.yaml"))
    parser.add_argument("--la-root", type=Path, help="Override data.la_root.")
    parser.add_argument("--split", choices=("train", "dev", "eval"), default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    la_root = args.la_root or Path(data_config["la_root"])

    entries = load_protocol(la_root, args.split)
    if not 0 <= args.index < len(entries):
        raise SystemExit(f"--index must be between 0 and {len(entries) - 1}")
    entry = entries[args.index]
    print(f"split:        {args.split}")
    print(f"samples:      {len(entries)}")
    print(f"utterance_id: {entry.utterance_id}")
    print(f"label/target: {entry.label}/{entry.target}")
    print(f"attack_id:    {entry.attack_id}")
    print(f"audio_path:   {entry.audio_path}")

    dataset = ASVspoofDataset(la_root, args.split, data_config)
    sample = dataset[args.index]
    features = sample["features"]
    print(f"feature shape: {tuple(features.shape)} [channel, mel, frames]")
    print(f"feature mean:  {features.mean().item():.6f}")
    print(f"feature std:   {features.std().item():.6f}")


if __name__ == "__main__":
    main()
