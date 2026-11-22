"""Central model construction point for all contributors."""

from __future__ import annotations

from typing import Any

from .cnn import CNNBaseline


def build_model(config: dict[str, Any]) -> Any:
    """Build a model from the ``model`` config section.

    Every model must accept ``[batch, 1, mel, time]`` float tensors and return
    one raw bona-fide logit per sample. Cassie's AST implementation should be
    added as another branch here without changing the training loop.
    """
    name = config.get("name")
    if name == "cnn":
        return CNNBaseline(
            channels=[int(value) for value in config.get("channels", [32, 64, 128])],
            dropout=float(config.get("dropout", 0.3)),
        )
    if name == "ast":
        raise NotImplementedError(
            "AST is reserved for Cassie's implementation. Add ASTModel and register "
            "it in audio_deepfake.models.factory.build_model."
        )
    raise ValueError(f"Unknown model name {name!r}; supported: cnn (ast extension reserved)")
