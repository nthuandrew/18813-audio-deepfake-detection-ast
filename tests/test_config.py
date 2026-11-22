from pathlib import Path

import pytest

from audio_deepfake.config import ConfigError, load_config


def test_baseline_config_is_valid() -> None:
    config = load_config(Path("configs/cnn_baseline.yaml"))
    assert config["model"]["name"] == "cnn"
    assert config["data"]["sample_rate"] == 16000
    assert config["data"]["normalize"] == "per_sample"


def test_missing_sections_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing config sections"):
        load_config(path)
