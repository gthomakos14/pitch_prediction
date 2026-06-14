import polars as pl
import pytest

from src.features.engineer import _add_prev_pitches, build_features


@pytest.fixture()
def raw_df():
    return pl.DataFrame(
        {
            "game_pk": [1, 1, 1, 1, 2, 2],
            "at_bat_number": [1, 1, 1, 2, 1, 1],
            "pitch_number": [1, 2, 3, 1, 1, 2],
            "pitcher": [600000, 600000, 600000, 600000, 700000, 700000],
            "p_throws": ["R", "R", "R", "R", "L", "L"],
            "stand": ["R", "R", "R", "L", "R", "R"],
            "balls": [0, 1, 1, 0, 0, 1],
            "strikes": [0, 0, 1, 0, 0, 1],
            "outs_when_up": [0, 0, 0, 1, 0, 0],
            "inning": [1, 1, 1, 1, 1, 1],
            "inning_topbot": ["Top", "Top", "Top", "Top", "Bot", "Bot"],
            "home_score": [0, 0, 0, 0, 1, 1],
            "away_score": [0, 0, 0, 0, 0, 0],
            "on_1b": [None, 608369, None, None, None, None],
            "on_2b": [None, None, 543760, None, None, None],
            "on_3b": [None, None, None, None, None, None],
            "pitch_type": ["FF", "SL", "CH", "FF", "CU", None],
        },
        schema={
            "game_pk": pl.Int64,
            "at_bat_number": pl.Int64,
            "pitch_number": pl.Int64,
            "p_throws": pl.String,
            "stand": pl.String,
            "balls": pl.Int64,
            "strikes": pl.Int64,
            "outs_when_up": pl.Int64,
            "inning": pl.Int64,
            "inning_topbot": pl.String,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
            "pitcher": pl.Int64,
            "on_1b": pl.Int64,
            "on_2b": pl.Int64,
            "on_3b": pl.Int64,
            "pitch_type": pl.String,
        },
    )


def test_prev_pitches_first_pitch_empty(raw_df):
    df = raw_df.filter(pl.col("pitch_type").is_not_null())
    result = _add_prev_pitches(df)
    first_pitches = result.filter(pl.col("pitch_number") == 1)["prev_pitches"].to_list()
    assert all(p == [] for p in first_pitches)


def test_prev_pitches_accumulate(raw_df):
    df = raw_df.filter(pl.col("pitch_type").is_not_null())
    result = _add_prev_pitches(df).sort(["game_pk", "at_bat_number", "pitch_number"])
    ab1 = result.filter((pl.col("game_pk") == 1) & (pl.col("at_bat_number") == 1))
    pp = ab1["prev_pitches"].to_list()
    assert pp[0] == []
    assert pp[1] == ["FF"]
    assert pp[2] == ["FF", "SL"]


def test_null_pitch_type_dropped(raw_df, tmp_path):
    raw_parquet = tmp_path / "statcast_test.parquet"
    raw_df.write_parquet(raw_parquet)

    import src.features.engineer as eng

    orig_raw_dir = eng.RAW_DIR
    orig_proc_dir = eng.PROCESSED_DIR
    try:
        eng.RAW_DIR = tmp_path
        eng.PROCESSED_DIR = tmp_path
        result = eng.build_features(output_path=tmp_path / "pitches.parquet")
        assert result["pitch_type"].is_null().sum() == 0
        assert len(result) == raw_df.filter(pl.col("pitch_type").is_not_null()).height
    finally:
        eng.RAW_DIR = orig_raw_dir
        eng.PROCESSED_DIR = orig_proc_dir


def test_on_base_boolean(raw_df, tmp_path):
    raw_parquet = tmp_path / "statcast_test.parquet"
    raw_df.write_parquet(raw_parquet)

    import src.features.engineer as eng

    orig_raw_dir = eng.RAW_DIR
    orig_proc_dir = eng.PROCESSED_DIR
    try:
        eng.RAW_DIR = tmp_path
        eng.PROCESSED_DIR = tmp_path
        result = eng.build_features(output_path=tmp_path / "pitches.parquet")
        assert result["on_1b"].dtype == pl.Boolean
        assert result["on_2b"].dtype == pl.Boolean
        assert result["on_3b"].dtype == pl.Boolean
    finally:
        eng.RAW_DIR = orig_raw_dir
        eng.PROCESSED_DIR = orig_proc_dir


def test_output_columns(raw_df, tmp_path):
    raw_parquet = tmp_path / "statcast_test.parquet"
    raw_df.write_parquet(raw_parquet)

    import src.features.engineer as eng

    orig_raw_dir = eng.RAW_DIR
    orig_proc_dir = eng.PROCESSED_DIR
    try:
        eng.RAW_DIR = tmp_path
        eng.PROCESSED_DIR = tmp_path
        result = eng.build_features(output_path=tmp_path / "pitches.parquet")
        expected = {
            "game_pk", "at_bat_number", "pitch_number", "pitcher", "p_throws", "stand",
            "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
            "home_score", "away_score", "on_1b", "on_2b", "on_3b",
            "prev_pitches", "pitch_type",
        }
        assert set(result.columns) == expected
    finally:
        eng.RAW_DIR = orig_raw_dir
        eng.PROCESSED_DIR = orig_proc_dir
