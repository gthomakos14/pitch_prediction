# Pitch Prediction

A pitch type predictor trained on MLB Statcast data. Given the current game situation — count, handedness, runners on base, score, inning — the model outputs a probability distribution over 10 pitch types.

## Architecture

```
Statcast API → data/raw/   →  src/features/engineer.py  →  data/processed/
                                                          ↓
                               src/fetch/preprocess.py  →  data/splits/  +  models/artifacts/
                                                          ↓
                               src/train/trainer.py     →  models/artifacts/model.pt
                                                          ↓
                               api/  (FastAPI)           →  frontend/  (Streamlit)
```

**Models**
- `feedforward` — per-pitch tabular features → MLP → pitch type logits
- `sequence` — LSTM over at-bat pitch sequences using `pack_padded_sequence`

**Pitch types:** CH · CU · FC · FF · FS · KC · KN · SI · SL · ST

## Quickstart

```bash
pip install -e ".[dev]"
# or: pip install -r requirements.txt

# 1. Fetch data
python cli.py fetch --season 2023

# 2. Build features + splits
python cli.py preprocess

# 3. Train
python cli.py train --model feedforward --config configs/ff_baseline.yaml

# 4. Evaluate on test split
python cli.py evaluate --run <mlflow-run-id>

# 5. Serve
uvicorn api.main:app --reload --port 8000
streamlit run frontend/app.py
```

## CLI Reference

| Command | Description |
|---|---|
| `python cli.py fetch --season YEAR` | Download Statcast data for a season |
| `python cli.py preprocess` | Build features and write train/val/test splits |
| `python cli.py train --model MODEL --config CONFIG` | Train a model, log to MLflow |
| `python cli.py evaluate --run RUN_ID` | Score the test split, log metrics to MLflow |
| `python cli.py export --output DIR` | Copy model artifacts to a directory |

## Project Layout

```
cli.py                  # top-level entry point
configs/
  ff_baseline.yaml      # feedforward hyperparameters
  lstm.yaml             # LSTM hyperparameters
src/
  fetch/
    statcast_fetch.py   # pybaseball download
    preprocess.py       # Preprocessor, game-level splits, artifact serialization
    dataset.py          # PitchDataset, SequenceDataset, collate functions
  features/
    engineer.py         # raw → processed (boolean runners, prev_pitches list)
  models/
    feedforward.py      # FeedforwardModel (nn.Module)
    sequence.py         # LSTMModel (nn.Module)
    __init__.py         # build_model() factory
  train/
    trainer.py          # training loop, MLflow logging, best-checkpoint saving
  evaluate/
    metrics.py          # test-split evaluation, per-class F1
api/
  main.py               # FastAPI app (lifespan model loading)
  model_loader.py       # loads artifacts into ModelBundle
  schemas/predict.py    # PitchContext + PredictResponse (Pydantic v2)
  routes/
    health.py           # GET /health
    predict.py          # POST /predict
frontend/
  app.py                # Streamlit home page
  pages/
    predict.py          # single-pitch prediction form
    explore.py          # at-bat sequence explorer
  components/           # sidebar, pitch form, probability chart
data/
  raw/                  # immutable Statcast parquets (gitignored)
  processed/            # pitches.parquet (gitignored); schema in data/processed/README.md
  splits/               # train/val/test parquets (gitignored)
models/
  artifacts/            # model.pt, preprocessor.pkl, label_encoder.pkl, config.yaml (gitignored)
```

## API

**`POST /predict`**

```json
{
  "p_throws": "R",
  "stand": "R",
  "inning_topbot": "Top",
  "balls": 1,
  "strikes": 1,
  "outs_when_up": 1,
  "inning": 5,
  "home_score": 2,
  "away_score": 1,
  "pitch_number": 3,
  "on_1b": true,
  "on_2b": false,
  "on_3b": false,
  "prev_pitches": ["FF", "SL"]
}
```

Response:

```json
{
  "pitch_type": "FF",
  "probabilities": {
    "FF": 0.41, "SL": 0.22, "CH": 0.12, "SI": 0.09,
    "CU": 0.06, "FC": 0.05, "ST": 0.03, "FS": 0.01,
    "KC": 0.01, "KN": 0.00
  }
}
```

**`GET /health`** → `{"status": "ok"}`

## Docker

```bash
cp .env.example .env
docker compose up --build
# API:      http://localhost:8000
# Frontend: http://localhost:8501
```

## Experiments

Training runs are tracked with MLflow (local `mlruns/`). To open the UI:

```bash
mlflow ui
```

Each run logs: hyperparameters, per-epoch train loss, val loss, mean val F1, and per-class val F1 for all 10 pitch types. `evaluate` logs final test-split metrics back to the same run.

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov=api
```

Tests cover: feature engineering, preprocessing + splits, dataset shapes and padding, feedforward and LSTM forward passes, and the full API request/response cycle (health + predict).
