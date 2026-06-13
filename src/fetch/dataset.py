from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.fetch.preprocess import (
    PITCH_CLASSES,
    SPLITS_DIR,
    TARGET_COL,
    Preprocessor,
)

_PITCH_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(PITCH_CLASSES)}


def _encode_labels(series: pl.Series) -> np.ndarray:
    return np.array([_PITCH_TO_IDX[p] for p in series.to_list()], dtype=np.int64)


class PitchDataset(Dataset):
    """Per-pitch tabular dataset for the feedforward model.

    __getitem__ returns (feature_tensor, label_int).
    """

    def __init__(self, df: pl.DataFrame, preprocessor: Preprocessor) -> None:
        self._features = preprocessor.transform(df)
        self._labels = _encode_labels(df[TARGET_COL])

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self._features[idx])
        y = int(self._labels[idx])
        return x, y

    @classmethod
    def from_split(cls, split: str, preprocessor: Preprocessor) -> "PitchDataset":
        """Load from a named split file ('train', 'val', 'test')."""
        return cls(pl.read_parquet(SPLITS_DIR / f"{split}.parquet"), preprocessor)


class SequenceDataset(Dataset):
    """Per-at-bat sequence dataset for the LSTM / Transformer model.

    Each sample represents one at-bat. __getitem__ returns
    (sequence_tensor, length, label_int) where:
      - sequence_tensor: float32 tensor of shape (max_seq_len, feature_dim),
        zero-padded; contains the feature vectors for all pitches in the at-bat
      - length: number of pitches in the at-bat (unpadded)
      - label_int: pitch type of the LAST pitch in the at-bat
    """

    def __init__(self, df: pl.DataFrame, preprocessor: Preprocessor, max_seq_len: int = 20) -> None:
        feature_dim = preprocessor.feature_dim

        features_all = preprocessor.transform(df)
        labels_all = _encode_labels(df[TARGET_COL])

        groups_df = (
            df.with_row_index("_feat_idx")
            .sort(["game_pk", "at_bat_number", "pitch_number"])
            .group_by(["game_pk", "at_bat_number"], maintain_order=False)
            .agg(pl.col("_feat_idx"))
        )

        sequences: list[np.ndarray] = []
        lengths: list[int] = []
        targets: list[int] = []

        for idx_list in groups_df["_feat_idx"].to_list():
            n = min(len(idx_list), max_seq_len)
            seq = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
            seq[:n] = features_all[idx_list[:n]]
            sequences.append(seq)
            lengths.append(n)
            targets.append(int(labels_all[idx_list[-1]]))

        self._sequences = np.stack(sequences)
        self._lengths = np.array(lengths, dtype=np.int64)
        self._targets = np.array(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        seq = torch.from_numpy(self._sequences[idx])
        length = int(self._lengths[idx])
        label = int(self._targets[idx])
        return seq, length, label

    @classmethod
    def from_split(
        cls, split: str, preprocessor: Preprocessor, max_seq_len: int = 20
    ) -> "SequenceDataset":
        """Load from a named split file ('train', 'val', 'test')."""
        return cls(pl.read_parquet(SPLITS_DIR / f"{split}.parquet"), preprocessor, max_seq_len=max_seq_len)


def ff_collate(batch: list) -> tuple[dict, torch.Tensor]:
    xs, ys = zip(*batch)
    return {"features": torch.stack(xs)}, torch.tensor(ys, dtype=torch.long)


def seq_collate(batch: list) -> tuple[dict, torch.Tensor]:
    seqs, lengths, labels = zip(*batch)
    return (
        {
            "sequences": torch.stack(seqs),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        },
        torch.tensor(labels, dtype=torch.long),
    )
