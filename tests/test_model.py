import pytest

torch = pytest.importorskip("torch")

from audio_deepfake.models import build_model


def test_cnn_output_contract() -> None:
    model = build_model({"name": "cnn", "channels": [8, 16], "dropout": 0.0})
    inputs = torch.randn(3, 1, 32, 50)
    assert model(inputs).shape == (3,)


def test_ast_extension_point_is_explicit() -> None:
    with pytest.raises(NotImplementedError, match="Cassie"):
        build_model({"name": "ast"})
