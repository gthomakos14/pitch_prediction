from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "p_throws",
    "stand",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "home_score",
    "away_score",
    "on_1b",
    "on_2b",
    "on_3b",
    "pitch_type",
]


def _load_raw() -> pl.DataFrame:
    parquets = sorted(RAW_DIR.glob("statcast_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No raw parquets found in {RAW_DIR}")
    return pl.concat([pl.read_parquet(p, columns=REQUIRED_COLS) for p in parquets])


def _add_prev_pitches(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["game_pk", "at_bat_number", "pitch_number"])
        .with_columns(
            pl.col("pitch_type")
            .over(["game_pk", "at_bat_number"], mapping_strategy="join")
            .alias("_ab_seq")
        )
        .with_row_index("_row_idx")
        .with_columns(
            (pl.col("_row_idx") - pl.col("_row_idx").min().over(["game_pk", "at_bat_number"]))
            .alias("_idx_in_ab")
        )
        .with_columns(
            pl.col("_ab_seq").list.head(pl.col("_idx_in_ab")).alias("prev_pitches")
        )
        .drop(["_ab_seq", "_idx_in_ab", "_row_idx"])
    )


def build_features(output_path: Path | None = None) -> pl.DataFrame:
    if output_path is None:
        output_path = PROCESSED_DIR / "pitches.parquet"

    df = _load_raw()

    df = df.with_columns([
        pl.col("on_1b").is_not_null().alias("on_1b"),
        pl.col("on_2b").is_not_null().alias("on_2b"),
        pl.col("on_3b").is_not_null().alias("on_3b"),
    ])

    df = df.filter(pl.col("pitch_type").is_not_null())

    df = _add_prev_pitches(df)

    output_cols = [
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "p_throws",
        "stand",
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "inning_topbot",
        "home_score",
        "away_score",
        "on_1b",
        "on_2b",
        "on_3b",
        "prev_pitches",
        "pitch_type",
    ]
    df = df.select(output_cols)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Wrote {len(df):,} rows to {output_path}")
    return df


if __name__ == "__main__":
    build_features()
