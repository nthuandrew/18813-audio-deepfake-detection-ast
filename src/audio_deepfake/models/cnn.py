"""Lightweight convolutional baseline for log-Mel inputs."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # Permit protocol/config utilities without ML dependencies.
    torch = None
    nn = None


if nn is not None:

    class ConvBlock(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )


    class CNNBaseline(nn.Module):
        """Compact CNN producing one bona-fide logit per spectrogram."""

        def __init__(self, channels: list[int], dropout: float = 0.3) -> None:
            super().__init__()
            if not channels or any(channel <= 0 for channel in channels):
                raise ValueError("channels must contain positive integers")

            blocks = []
            in_channels = 1
            for out_channels in channels:
                blocks.append(ConvBlock(in_channels, out_channels))
                in_channels = out_channels

            self.encoder = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, 1),
            )

        def forward(self, features: Any) -> Any:
            """Return raw logits shaped [batch]; sigmoid is applied outside."""
            encoded = self.encoder(features)
            return self.classifier(self.pool(encoded)).squeeze(-1)

else:

    class CNNBaseline:  # pragma: no cover - only used for a clearer error
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("CNNBaseline requires PyTorch; install project dependencies.")
