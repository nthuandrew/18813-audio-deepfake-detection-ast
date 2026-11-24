"""Dependency-free binary metrics for bona-fide/spoof scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def equal_error_rate(targets: Sequence[int], scores: Sequence[float]) -> float:
    """Return interpolated EER for scores where larger means bona fide.

    False acceptance is the fraction of spoof samples classified as bona fide;
    false rejection is the fraction of bona-fide samples classified as spoof.
    """
    if len(targets) != len(scores) or not targets:
        raise ValueError("targets and scores must be non-empty and equally sized")
    pairs = []
    for target, score in zip(targets, scores, strict=True):
        target = int(target)
        if target not in {0, 1}:
            raise ValueError(f"Binary target must be 0 or 1, got {target}")
        pairs.append((float(score), target))

    positives = sum(target for _, target in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("EER requires both bona-fide and spoof samples")

    pairs.sort(key=lambda pair: pair[0], reverse=True)
    true_positive = false_positive = 0
    previous_far = 0.0
    previous_frr = 1.0
    previous_difference = previous_far - previous_frr
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                true_positive += 1
            else:
                false_positive += 1
            index += 1

        far = false_positive / negatives
        frr = (positives - true_positive) / positives
        difference = far - frr
        if difference == 0:
            return far
        if previous_difference < 0 < difference:
            fraction = -previous_difference / (difference - previous_difference)
            interpolated_far = previous_far + fraction * (far - previous_far)
            interpolated_frr = previous_frr + fraction * (frr - previous_frr)
            return (interpolated_far + interpolated_frr) / 2
        previous_far = far
        previous_frr = frr
        previous_difference = difference

    return (previous_far + previous_frr) / 2


def binary_classification_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float = 0.5
) -> dict[str, float | int]:
    """Compute EER and threshold-dependent classification metrics.

    Precision, recall, and F1 treat bona fide (target 1) as the positive class.
    A score at or above ``threshold`` predicts bona fide.
    """
    if not rows:
        raise ValueError("Cannot compute metrics from zero predictions")

    targets: list[int] = []
    scores: list[float] = []
    true_positive = true_negative = false_positive = false_negative = 0
    for row in rows:
        target = int(row["target"])
        score = float(row["score"])
        if target not in {0, 1}:
            raise ValueError(f"Binary target must be 0 or 1, got {target}")
        targets.append(target)
        scores.append(score)
        prediction = int(score >= threshold)
        if target == 1 and prediction == 1:
            true_positive += 1
        elif target == 0 and prediction == 0:
            true_negative += 1
        elif target == 0 and prediction == 1:
            false_positive += 1
        else:
            false_negative += 1

    bonafide_total = true_positive + false_negative
    spoof_total = true_negative + false_positive
    predicted_positive = true_positive + false_positive
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / bonafide_total if bonafide_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    spoof_recall = true_negative / spoof_total if spoof_total else 0.0
    return {
        "threshold": threshold,
        "eer": equal_error_rate(targets, scores),
        "accuracy": (true_positive + true_negative) / len(rows),
        "balanced_accuracy": (recall + spoof_recall) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bonafide_recall": recall,
        "spoof_recall": spoof_recall,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def attack_wise_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float = 0.5
) -> dict[str, dict[str, float | int]]:
    """Compute metrics for each spoof attack plus all bona-fide examples.

    ASVspoof bona-fide rows use ``attack_id = '-'``. Each attack-specific
    subset includes every bona-fide row and only the spoof rows for that attack,
    matching the evaluation report's per-attack construction.
    """
    attack_ids = sorted(
        {str(row["attack_id"]) for row in rows if row["attack_id"] != "-"}
    )
    if not attack_ids:
        raise ValueError("Prediction rows contain no spoof attack IDs")

    results: dict[str, dict[str, float | int]] = {}
    for attack_id in attack_ids:
        subset = [
            row
            for row in rows
            if row["attack_id"] == "-" or row["attack_id"] == attack_id
        ]
        results[attack_id] = binary_classification_metrics(
            subset, threshold=threshold
        )
    return results
