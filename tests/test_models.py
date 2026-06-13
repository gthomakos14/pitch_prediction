import torch
import pytest

from src.models.feedforward import FeedforwardModel
from src.models.sequence import LSTMModel
from src.models import build_model
from src.fetch.preprocess import PITCH_CLASSES

NUM_CLASSES = len(PITCH_CLASSES)
INPUT_DIM = 16
BATCH = 8


class TestFeedforwardModel:
    def test_output_shape(self):
        model = FeedforwardModel(INPUT_DIM, [64, 32], NUM_CLASSES)
        x = torch.randn(BATCH, INPUT_DIM)
        logits = model({"features": x})
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_output_dtype(self):
        model = FeedforwardModel(INPUT_DIM, [64], NUM_CLASSES)
        logits = model({"features": torch.randn(BATCH, INPUT_DIM)})
        assert logits.dtype == torch.float32

    def test_single_sample(self):
        model = FeedforwardModel(INPUT_DIM, [64], NUM_CLASSES)
        logits = model({"features": torch.randn(1, INPUT_DIM)})
        assert logits.shape == (1, NUM_CLASSES)

    def test_no_dropout_in_eval(self):
        model = FeedforwardModel(INPUT_DIM, [64], NUM_CLASSES, dropout=0.9)
        x = torch.randn(BATCH, INPUT_DIM)
        model.eval()
        with torch.no_grad():
            out1 = model({"features": x})
            out2 = model({"features": x})
        assert torch.allclose(out1, out2)


class TestLSTMModel:
    def test_output_shape(self):
        model = LSTMModel(INPUT_DIM, hidden_dim=32, num_classes=NUM_CLASSES, num_layers=1)
        seqs = torch.randn(BATCH, 5, INPUT_DIM)
        lengths = torch.tensor([5, 4, 3, 2, 1, 5, 4, 3], dtype=torch.long)
        logits = model({"sequences": seqs, "lengths": lengths})
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_padded_positions_ignored(self):
        model = LSTMModel(INPUT_DIM, hidden_dim=32, num_classes=NUM_CLASSES, num_layers=1)
        model.eval()
        seq1 = torch.randn(1, 5, INPUT_DIM)
        seq2 = seq1.clone()
        seq2[0, 3:] = 999.0  # corrupt padding positions
        with torch.no_grad():
            out1 = model({"sequences": seq1, "lengths": torch.tensor([3])})
            out2 = model({"sequences": seq2, "lengths": torch.tensor([3])})
        assert torch.allclose(out1, out2)


class TestBuildModelFactory:
    def test_feedforward(self):
        config = {"input_dim": INPUT_DIM, "num_classes": NUM_CLASSES, "hidden_dims": [64]}
        model = build_model("feedforward", config)
        assert isinstance(model, FeedforwardModel)

    def test_sequence(self):
        config = {"input_dim": INPUT_DIM, "num_classes": NUM_CLASSES, "hidden_dim": 32, "num_layers": 1}
        model = build_model("sequence", config)
        assert isinstance(model, LSTMModel)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model("transformer", {"input_dim": INPUT_DIM, "num_classes": NUM_CLASSES})
