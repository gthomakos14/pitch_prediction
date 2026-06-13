import numpy as np
import polars as pl
import pytest
import torch

from src.fetch.preprocess import PITCH_CLASSES, Preprocessor
from src.fetch.dataset import PitchDataset, SequenceDataset


@pytest.fixture()
def sample_df():
    rng = np.random.default_rng(1)
    n = 30
    games = np.repeat(np.arange(3), 10)
    at_bats = np.tile(np.repeat(np.arange(2), 5), 3)
    pitch_nums = np.tile(np.arange(1, 6), 6)
    return pl.DataFrame(
        {
            "game_pk": games.tolist(),
            "at_bat_number": at_bats.tolist(),
            "pitch_number": pitch_nums.tolist(),
            "p_throws": rng.choice(["R", "L"], size=n).tolist(),
            "stand": rng.choice(["R", "L"], size=n).tolist(),
            "balls": rng.integers(0, 4, size=n).tolist(),
            "strikes": rng.integers(0, 3, size=n).tolist(),
            "outs_when_up": rng.integers(0, 3, size=n).tolist(),
            "inning": rng.integers(1, 10, size=n).tolist(),
            "inning_topbot": rng.choice(["Top", "Bot"], size=n).tolist(),
            "home_score": rng.integers(0, 8, size=n).tolist(),
            "away_score": rng.integers(0, 8, size=n).tolist(),
            "on_1b": rng.choice([True, False], size=n).tolist(),
            "on_2b": rng.choice([True, False], size=n).tolist(),
            "on_3b": rng.choice([True, False], size=n).tolist(),
            "prev_pitches": [[] for _ in range(n)],
            "pitch_type": rng.choice(PITCH_CLASSES, size=n).tolist(),
        }
    )


@pytest.fixture()
def fitted_preprocessor(sample_df):
    pre = Preprocessor()
    pre.fit(sample_df)
    return pre


class TestPitchDataset:
    def test_len(self, sample_df, fitted_preprocessor):
        ds = PitchDataset(sample_df, fitted_preprocessor)
        assert len(ds) == len(sample_df)

    def test_getitem_shapes(self, sample_df, fitted_preprocessor):
        ds = PitchDataset(sample_df, fitted_preprocessor)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (fitted_preprocessor.feature_dim,)
        assert x.dtype == torch.float32
        assert isinstance(y, int)

    def test_label_in_range(self, sample_df, fitted_preprocessor):
        ds = PitchDataset(sample_df, fitted_preprocessor)
        for i in range(len(ds)):
            _, y = ds[i]
            assert 0 <= y < len(PITCH_CLASSES)


class TestSequenceDataset:
    def test_len_equals_num_at_bats(self, sample_df, fitted_preprocessor):
        n_at_bats = sample_df.select(["game_pk", "at_bat_number"]).unique().height
        ds = SequenceDataset(sample_df, fitted_preprocessor)
        assert len(ds) == n_at_bats

    def test_getitem_shapes(self, sample_df, fitted_preprocessor):
        max_seq_len = 10
        ds = SequenceDataset(sample_df, fitted_preprocessor, max_seq_len=max_seq_len)
        seq, length, label = ds[0]
        assert isinstance(seq, torch.Tensor)
        assert seq.shape == (max_seq_len, fitted_preprocessor.feature_dim)
        assert seq.dtype == torch.float32
        assert isinstance(length, int)
        assert 1 <= length <= max_seq_len
        assert isinstance(label, int)
        assert 0 <= label < len(PITCH_CLASSES)

    def test_padding_beyond_length_is_zero(self, sample_df, fitted_preprocessor):
        ds = SequenceDataset(sample_df, fitted_preprocessor, max_seq_len=20)
        for i in range(len(ds)):
            seq, length, _ = ds[i]
            if length < 20:
                assert seq[length:].sum().item() == 0.0

    def test_max_seq_len_clamps(self, sample_df, fitted_preprocessor):
        ds = SequenceDataset(sample_df, fitted_preprocessor, max_seq_len=2)
        for i in range(len(ds)):
            _, length, _ = ds[i]
            assert length <= 2
