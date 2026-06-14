import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

PITCH_CLASSES = ["CH", "CU", "FC", "FF", "FS", "KC", "KN", "SI", "SL", "ST"]

CATEGORICAL_COLS = ["p_throws", "stand", "inning_topbot"]
BOOLEAN_COLS = ["on_1b", "on_2b", "on_3b"]
NUMERIC_COLS = ["balls", "strikes", "outs_when_up", "inning", "home_score", "away_score", "pitch_number"]
TARGET_COL = "pitch_type"
PITCHER_COL = "pitcher"
PITCHER_MIN_PITCHES = 50
UNKNOWN_PITCHER = "UNKNOWN"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
RANDOM_SEED = 42


class Preprocessor:
    """Encodes and scales tabular pitch features. Fit only on train split."""

    def __init__(self):
        self.cat_encoders: dict[str, LabelEncoder] = {}
        self.pitcher_encoder = LabelEncoder()
        self._frequent_pitchers: set[str] = set()
        self.scaler = StandardScaler()
        self._fitted = False

    @property
    def feature_dim(self) -> int:
        return len(CATEGORICAL_COLS) + 1 + len(BOOLEAN_COLS) + len(NUMERIC_COLS)

    def fit(self, df: pl.DataFrame) -> "Preprocessor":
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            le.fit(df[col].to_numpy())
            self.cat_encoders[col] = le

        counts = df[PITCHER_COL].value_counts().filter(pl.col("count") >= PITCHER_MIN_PITCHES)
        self._frequent_pitchers = set(counts[PITCHER_COL].to_list())
        self.pitcher_encoder.fit([UNKNOWN_PITCHER] + sorted(self._frequent_pitchers))

        self.scaler.fit(df.select(NUMERIC_COLS).to_numpy().astype(float))
        self._fitted = True
        return self

    def _encode_pitcher(self, series: pl.Series) -> np.ndarray:
        labels = [
            p if p in self._frequent_pitchers else UNKNOWN_PITCHER
            for p in series.to_list()
        ]
        return self.pitcher_encoder.transform(labels).astype(np.float32)

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        cat = np.column_stack(
            [self.cat_encoders[c].transform(df[c].to_numpy()) for c in CATEGORICAL_COLS]
        ).astype(np.float32)
        pitcher = self._encode_pitcher(df[PITCHER_COL]).reshape(-1, 1)
        bools = df.select(BOOLEAN_COLS).to_numpy().astype(np.float32)
        nums = self.scaler.transform(df.select(NUMERIC_COLS).to_numpy().astype(float)).astype(np.float32)
        return np.concatenate([cat, pitcher, bools, nums], axis=1)

    def fit_transform(self, df: pl.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def _split_by_game(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    games = df["game_pk"].unique().to_numpy()
    rng.shuffle(games)
    n = len(games)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_games = games[:n_train].tolist()
    val_games = games[n_train: n_train + n_val].tolist()
    test_games = games[n_train + n_val:].tolist()
    return (
        df.filter(pl.col("game_pk").is_in(train_games)),
        df.filter(pl.col("game_pk").is_in(val_games)),
        df.filter(pl.col("game_pk").is_in(test_games)),
    )


def run_preprocessing(processed_path: Path | None = None) -> None:
    if processed_path is None:
        processed_path = PROCESSED_DIR / "pitches.parquet"

    df = pl.read_parquet(processed_path)
    df = df.filter(pl.col(TARGET_COL).is_in(PITCH_CLASSES))
    print(f"Loaded {len(df):,} rows covering {df['game_pk'].n_unique():,} games")

    train_df, val_df, test_df = _split_by_game(df)
    print(f"Split sizes — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(SPLITS_DIR / "train.parquet")
    val_df.write_parquet(SPLITS_DIR / "val.parquet")
    test_df.write_parquet(SPLITS_DIR / "test.parquet")
    print(f"Splits written to {SPLITS_DIR}")

    preprocessor = Preprocessor()
    preprocessor.fit(train_df)
    n_frequent = len(preprocessor._frequent_pitchers)
    n_total = train_df[PITCHER_COL].n_unique()
    print(
        f"Pitcher encoding: {n_frequent}/{n_total} pitchers kept "
        f"(threshold={PITCHER_MIN_PITCHES} pitches); rest → '{UNKNOWN_PITCHER}'"
    )

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(PITCH_CLASSES)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Artifacts written to {ARTIFACTS_DIR}")


def load_preprocessor() -> Preprocessor:
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "rb") as f:
        return pickle.load(f)


def load_label_encoder() -> LabelEncoder:
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    run_preprocessing()
