"""Training loop independent of the selected model architecture."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def simple_binary_metrics(
    rows: list[dict[str, Any]], threshold: float = 0.5
) -> dict[str, float | int]:
    """Compute leakage-free, dependency-free metrics for quick monitoring.

    A score at or above ``threshold`` predicts bona fide (target 1). Balanced
    accuracy gives bona fide and spoof equal weight despite class imbalance.
    This is intentionally small; the full EER/CI analysis belongs to the
    evaluation pipeline.
    """
    if not rows:
        raise ValueError("Cannot compute metrics from zero predictions")

    true_positive = true_negative = false_positive = false_negative = 0
    for row in rows:
        target = int(row["target"])
        prediction = int(float(row["score"]) >= threshold)
        if target == 1 and prediction == 1:
            true_positive += 1
        elif target == 0 and prediction == 0:
            true_negative += 1
        elif target == 0 and prediction == 1:
            false_positive += 1
        elif target == 1 and prediction == 0:
            false_negative += 1
        else:
            raise ValueError(f"Binary target must be 0 or 1, got {target}")

    total = len(rows)
    bonafide_total = true_positive + false_negative
    spoof_total = true_negative + false_positive
    bonafide_recall = true_positive / bonafide_total if bonafide_total else 0.0
    spoof_recall = true_negative / spoof_total if spoof_total else 0.0
    return {
        "threshold": threshold,
        "accuracy": (true_positive + true_negative) / total,
        "balanced_accuracy": (bonafide_recall + spoof_recall) / 2,
        "bonafide_recall": bonafide_recall,
        "spoof_recall": spoof_recall,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


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
