import pickle

import numpy as np
import polars as pl
import pytest

from src.fetch.preprocess import (
    PITCH_CLASSES,
    Preprocessor,
    _split_by_game,
    run_preprocessing,
)


@pytest.fixture()
def processed_df():
    n = 60
    rng = np.random.default_rng(0)
    games = np.repeat(np.arange(10), 6)
    return pl.DataFrame(
        {
            "game_pk": games.tolist(),
            "at_bat_number": rng.integers(1, 5, size=n).tolist(),
            "pitch_number": rng.integers(1, 6, size=n).tolist(),
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


def test_preprocessor_fit_transform_shape(processed_df):
    pre = Preprocessor()
    X = pre.fit_transform(processed_df)
    assert X.shape == (len(processed_df), pre.feature_dim)
    assert X.dtype == np.float32


def test_preprocessor_transform_without_fit_raises(processed_df):
    pre = Preprocessor()
    with pytest.raises(RuntimeError):
        pre.transform(processed_df)


def test_preprocessor_boolean_encoding(processed_df):
    pre = Preprocessor()
    X = pre.fit_transform(processed_df)
    # boolean cols are at indices 3, 4, 5 (after 3 categorical cols)
    bool_cols = X[:, 3:6]
    assert set(bool_cols.flatten().tolist()).issubset({0.0, 1.0})


def test_split_by_game_no_leakage(processed_df):
    train, val, test = _split_by_game(processed_df)
    train_games = set(train["game_pk"])
    val_games = set(val["game_pk"])
    test_games = set(test["game_pk"])
    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)


def test_split_covers_all_rows(processed_df):
    train, val, test = _split_by_game(processed_df)
    assert len(train) + len(val) + len(test) == len(processed_df)


def test_run_preprocessing_writes_artifacts(processed_df, tmp_path):
    proc_path = tmp_path / "pitches.parquet"
    processed_df.write_parquet(proc_path)

    import src.fetch.preprocess as pp

    orig_splits = pp.SPLITS_DIR
    orig_artifacts = pp.ARTIFACTS_DIR
    try:
        pp.SPLITS_DIR = tmp_path / "splits"
        pp.ARTIFACTS_DIR = tmp_path / "artifacts"
        run_preprocessing(processed_path=proc_path)
        assert (tmp_path / "splits" / "train.parquet").exists()
        assert (tmp_path / "splits" / "val.parquet").exists()
        assert (tmp_path / "splits" / "test.parquet").exists()
        assert (tmp_path / "artifacts" / "preprocessor.pkl").exists()
        assert (tmp_path / "artifacts" / "label_encoder.pkl").exists()
    finally:
        pp.SPLITS_DIR = orig_splits
        pp.ARTIFACTS_DIR = orig_artifacts


def test_label_encoder_classes(processed_df, tmp_path):
    proc_path = tmp_path / "pitches.parquet"
    processed_df.write_parquet(proc_path)

    import src.fetch.preprocess as pp

    orig_splits = pp.SPLITS_DIR
    orig_artifacts = pp.ARTIFACTS_DIR
    try:
        pp.SPLITS_DIR = tmp_path / "splits"
        pp.ARTIFACTS_DIR = tmp_path / "artifacts"
        run_preprocessing(processed_path=proc_path)
        with open(tmp_path / "artifacts" / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        assert list(le.classes_) == PITCH_CLASSES
    finally:
        pp.SPLITS_DIR = orig_splits
        pp.ARTIFACTS_DIR = orig_artifacts
