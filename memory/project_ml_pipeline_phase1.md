---
name: project-ml-pipeline-phase1
description: Status of the pitch prediction ML pipeline build — what's done and what remains
metadata:
  type: project
---

All core pipeline components are complete and all 36 tests pass.

**Done:**
- `src/fetch/` — statcast fetch, preprocess, dataset (PitchDataset + SequenceDataset + ff_collate/seq_collate)
- `src/features/engineer.py` — feature engineering
- `src/models/feedforward.py` — MLP (batch dict interface)
- `src/models/sequence.py` — LSTM (batch dict interface, pack_padded_sequence)
- `src/models/__init__.py` — `build_model()` factory
- `src/train/trainer.py` — training loop with MLflow logging, saves best checkpoint
- `src/evaluate/metrics.py` — test-split evaluation with per-class F1, logs back to MLflow run
- `api/` — FastAPI service: `main.py`, `model_loader.py`, `/health` and `/predict` routes, Pydantic v2 schemas
- `cli.py` — fetch / preprocess / train / evaluate / export commands
- `configs/ff_baseline.yaml`, `configs/lstm.yaml` — hyperparameter configs
- `docker/api.Dockerfile`, `docker/frontend.Dockerfile`, `docker-compose.yml`
- `.env.example`
- `data/processed/README.md` — already existed and is current
- `tests/test_models.py`, `tests/test_api.py` — new test files
- `requirements.txt` — updated with fastapi, uvicorn, pydantic, mlflow, pyyaml, httpx, altair, pytest-asyncio

**Next logical steps:**
- Fetch real Statcast data: `python cli.py fetch --season 2023`
- Run preprocessing: `python cli.py preprocess`
- Train feedforward baseline: `python cli.py train --model feedforward --config configs/ff_baseline.yaml`
- Evaluate: `python cli.py evaluate --run <run_id>`
- Then train sequence model and compare

**Why:** Project goal is a trained pitch-type predictor served via FastAPI + Streamlit. No blockers remain before first real training run.
