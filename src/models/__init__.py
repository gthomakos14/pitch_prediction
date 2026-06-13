import torch.nn as nn

from src.models.feedforward import FeedforwardModel
from src.models.sequence import LSTMModel


def build_model(model_type: str, config: dict) -> nn.Module:
    input_dim: int = config["input_dim"]
    num_classes: int = config["num_classes"]
    if model_type == "feedforward":
        return FeedforwardModel(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [256, 128]),
            num_classes=num_classes,
            dropout=config.get("dropout", 0.3),
        )
    if model_type == "sequence":
        return LSTMModel(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            num_classes=num_classes,
            dropout=config.get("dropout", 0.3),
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")
