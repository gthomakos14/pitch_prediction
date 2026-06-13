import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        sequences = batch["sequences"]   # (B, T, D)
        lengths = batch["lengths"]       # (B,)
        packed = pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]            # (B, hidden_dim)
        return self.classifier(last_hidden)
