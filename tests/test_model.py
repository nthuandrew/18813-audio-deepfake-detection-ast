import pytest

torch = pytest.importorskip("torch")

from audio_deepfake.models import build_model
from audio_deepfake.models.ast import ASTModel


def test_cnn_output_contract() -> None:
    model = build_model({"name": "cnn", "channels": [8, 16], "dropout": 0.0})
    inputs = torch.randn(3, 1, 32, 50)
    assert model(inputs).shape == (3,)


class FakeASTBackbone(torch.nn.Module):
    class Config:
        max_length = 8
        num_mel_bins = 4

    def __init__(self) -> None:
        super().__init__()
        self.config = self.Config()
        self.projection = torch.nn.Linear(1, 1)

    def forward(self, input_values: torch.Tensor):
        assert input_values.shape == (3, 8, 4)
        logits = self.projection(input_values.mean(dim=(1, 2)).unsqueeze(-1))
        return type("Output", (), {"logits": logits})()


def test_ast_output_contract() -> None:
    model = ASTModel({"n_mels": 4}, backbone=FakeASTBackbone())
    inputs = torch.randn(3, 1, 4, 6)
    assert model(inputs).shape == (3,)
