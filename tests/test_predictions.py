import csv
from pathlib import Path

from audio_deepfake.training.engine import save_predictions, simple_binary_metrics


def test_prediction_csv_schema(tmp_path: Path) -> None:
    destination = tmp_path / "predictions.csv"
    save_predictions(
        [
            {
                "utterance_id": "LA_D_0001",
                "label": "spoof",
                "target": 0,
                "score": 0.125,
                "attack_id": "A03",
            }
        ],
        destination,
    )
    with destination.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "utterance_id",
            "label",
            "target",
            "score",
            "attack_id",
        ]
        assert next(reader)["score"] == "0.125"


def test_simple_metrics_expose_class_imbalance() -> None:
    rows = [
        {"target": 1, "score": 0.9},
        {"target": 1, "score": 0.4},
        {"target": 0, "score": 0.2},
        {"target": 0, "score": 0.8},
        {"target": 0, "score": 0.1},
    ]
    metrics = simple_binary_metrics(rows)
    assert metrics["accuracy"] == 0.6
    assert metrics["bonafide_recall"] == 0.5
    assert metrics["spoof_recall"] == 2 / 3
    assert metrics["balanced_accuracy"] == (0.5 + 2 / 3) / 2
