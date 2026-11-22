"""Training loop independent of the selected model architecture."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def train_one_epoch(model: Any, loader: Any, optimizer: Any, criterion: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def predict(model: Any, loader: Any, criterion: Any, device: Any) -> tuple[float, list[dict[str, Any]]]:
    """Return loss and the stable row schema consumed by evaluation."""
    import torch

    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["target"].to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            scores = torch.sigmoid(logits).cpu().tolist()
            target_values = targets.cpu().to(torch.int64).tolist()
            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
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
                    target_values,
                    scores,
                    batch["attack_id"],
                    strict=True,
                )
            )
    return total_loss / max(total_examples, 1), rows


def save_predictions(rows: list[dict[str, Any]], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["utterance_id", "label", "target", "score", "attack_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_history(history: list[dict[str, Any]], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")


def save_metrics(metrics: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
