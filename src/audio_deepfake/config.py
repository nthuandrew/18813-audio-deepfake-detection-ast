"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or invalid."""


REQUIRED_SECTIONS = ("experiment", "data", "model", "training")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and validate the shared team contract."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a mapping: {config_path}")

    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    data = config["data"]
    training = config["training"]
    if data.get("sample_rate", 0) <= 0:
        raise ConfigError("data.sample_rate must be positive")
    if data.get("clip_duration_seconds", 0) <= 0:
        raise ConfigError("data.clip_duration_seconds must be positive")
    if data.get("normalize") not in {"none", "per_sample"}:
        raise ConfigError("data.normalize must be 'none' or 'per_sample'")
    if training.get("batch_size", 0) <= 0:
        raise ConfigError("training.batch_size must be positive")
    if training.get("epochs", 0) <= 0:
        raise ConfigError("training.epochs must be positive")
    return config
