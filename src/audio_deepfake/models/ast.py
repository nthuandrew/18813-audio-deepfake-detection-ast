"""Hugging Face Audio Spectrogram Transformer model."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # Permit non-training utilities without PyTorch installed.
    torch = None
    nn = None


if nn is not None:

    class ASTModel(nn.Module):
        """Pretrained AST producing one raw bona-fide logit per sample."""

        def __init__(
            self,
            config: dict[str, Any] | None = None,
            *,
            backbone: Any | None = None,
        ) -> None:
            super().__init__()
            settings = config or {}
            if backbone is None:
                try:
                    from transformers import ASTForAudioClassification
                except ImportError as exc:  # pragma: no cover - environment dependent
                    raise RuntimeError(
                        "ASTModel requires transformers; install it with "
                        "`python3 -m pip install transformers`."
                    ) from exc
                checkpoint = settings.get(
                    "checkpoint", "MIT/ast-finetuned-audioset-10-10-0.4593"
                )
                backbone = ASTForAudioClassification.from_pretrained(
                    checkpoint,
                    num_labels=1,
                    ignore_mismatched_sizes=True,
                )

            self.backbone = backbone
            model_config = getattr(backbone, "config", None)
            self.max_length = int(
                settings.get(
                    "max_length", getattr(model_config, "max_length", 1024)
                )
            )
            self.num_mel_bins = int(
                settings.get(
                    "n_mels", getattr(model_config, "num_mel_bins", 128)
                )
            )

        def _prepare_input(self, features: Any) -> Any:
            if features.ndim != 4 or features.shape[1] != 1:
                raise ValueError(
                    "ASTModel expects features shaped "
                    "[batch, 1, mel_bins, frames]"
                )
            if features.shape[2] != self.num_mel_bins:
                raise ValueError(
                    f"AST expects {self.num_mel_bins} Mel bins, "
                    f"received {features.shape[2]}"
                )

            ast_input = features.squeeze(1).transpose(1, 2)
            frame_count = ast_input.shape[1]
            if frame_count > self.max_length:
                return ast_input[:, : self.max_length, :]
            if frame_count < self.max_length:
                padding = torch.zeros(
                    ast_input.shape[0],
                    self.max_length - frame_count,
                    self.num_mel_bins,
                    dtype=ast_input.dtype,
                    device=ast_input.device,
                )
                ast_input = torch.cat((ast_input, padding), dim=1)
            return ast_input

        def forward(self, features: Any) -> Any:
            """Return raw bona-fide logits for repo-format log-Mel inputs."""
            ast_input = self._prepare_input(features)
            logits = self.backbone(input_values=ast_input).logits
            return logits.squeeze(-1)

else:

    class ASTModel:  # pragma: no cover - only used for a clearer error
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("ASTModel requires PyTorch; install project dependencies.")
