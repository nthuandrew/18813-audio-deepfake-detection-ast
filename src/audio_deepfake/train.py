"""Train a configured audio deepfake detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import load_config


def run(
    config_path: str | Path,
    seed_override: int | None = None,
    output_override: str | Path | None = None,
) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing; run `python3 -m pip install -e '.[dev]'`."
        ) from exc

    from .data.dataset import ASVspoofDataset
    from .models import build_model
    from .training.engine import (
        predict,
        save_history,
        save_predictions,
        simple_binary_metrics,
        train_one_epoch,
    )
    from .training.reproducibility import select_device, set_seed

    config = load_config(config_path)
    experiment = config["experiment"]
    data_config = config["data"]
    train_config = config["training"]
    seed = int(seed_override if seed_override is not None else experiment["seed"])
    experiment["seed"] = seed
    if output_override is not None:
        experiment["output_dir"] = str(output_override)
    set_seed(seed)

    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train_dataset = ASVspoofDataset(data_config["la_root"], "train", data_config)
    dev_dataset = ASVspoofDataset(data_config["la_root"], "dev", data_config)
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(train_config["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_kwargs)

    device = select_device(train_config.get("device", "auto"))
    model = build_model(config["model"]).to(device)
    positive_weight = None
    if train_config.get("use_class_weight", True):
        positives = sum(entry.target for entry in train_dataset.entries)
        negatives = len(train_dataset) - positives
        positive_weight = torch.tensor([negatives / positives], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_config.get("scheduler_factor", 0.5)),
        patience=int(train_config.get("scheduler_patience", 2)),
    )

    history = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    max_patience = int(train_config.get("early_stopping_patience", 6))
    for epoch in range(1, int(train_config["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, predictions = predict(model, dev_loader, criterion, device)
        dev_metrics = simple_binary_metrics(predictions)
        scheduler.step(dev_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
            "dev_bonafide_recall": dev_metrics["bonafide_recall"],
            "dev_spoof_recall": dev_metrics["spoof_recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        save_history(history, output_dir / "history.json")
        print(record)

        if dev_loss < best_loss:
            best_loss = dev_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "dev_loss": dev_loss,
                    "config": config,
                },
                output_dir / "best.pt",
            )
            save_predictions(predictions, output_dir / "dev_predictions.csv")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max_patience:
                print(f"Early stopping after {epoch} epochs")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--seed", type=int, help="Override experiment.seed for a reproducible run."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override experiment.output_dir."
    )
    args = parser.parse_args()
    run(args.config, seed_override=args.seed, output_override=args.output_dir)


if __name__ == "__main__":
    main()
