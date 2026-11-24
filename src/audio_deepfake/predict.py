"""Generate prediction CSVs from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def run(
    checkpoint_path: str | Path,
    split: str,
    output_path: str | Path,
    device_name: str = "auto",
) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Prediction requires the project's PyTorch dependencies.") from exc

    from .data.dataset import ASVspoofDataset
    from .models import build_model
    from .training.engine import save_predictions
    from .training.reproducibility import select_device

    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise TypeError("Checkpoint does not contain the resolved training config")

    data_config = saved_config["data"]
    train_config = saved_config.get("training", {})
    model = build_model(saved_config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ASVspoofDataset(data_config["la_root"], split, data_config)
    loader = DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"].to(device))
            scores = torch.sigmoid(logits).cpu().tolist()
            targets = batch["target"].to(torch.int64).tolist()
            rows.extend(
                {
                    "utterance_id": utterance_id,
                    "label": label,
                    "target": target,
                    "score": score,
                    "attack_id": attack_id,
                }
                for utterance_id, label, target, score, attack_id in zip(
                    batch["utterance_id"],
                    batch["label"],
                    targets,
                    scores,
                    batch["attack_id"],
                    strict=True,
                )
            )

    save_predictions(rows, output_path)
    print(f"Wrote {len(rows)} predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "dev", "eval"), default="eval")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.checkpoint, args.split, args.output, args.device)


if __name__ == "__main__":
    main()
