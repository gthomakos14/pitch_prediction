import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from api.main import app
from api.model_loader import ModelBundle
from src.fetch.preprocess import PITCH_CLASSES, Preprocessor


@pytest.fixture()
def mock_bundle():
    preprocessor = MagicMock(spec=Preprocessor)
    preprocessor.transform.return_value = np.zeros((1, 16), dtype=np.float32)

    model = MagicMock()
    fake_logits = torch.zeros(1, len(PITCH_CLASSES))
    fake_logits[0, 0] = 10.0  # FF gets highest logit
    model.return_value = fake_logits

    return ModelBundle(
        model=model,
        preprocessor=preprocessor,
        label_encoder=MagicMock(),
        config={"model_type": "feedforward", "input_dim": 16, "num_classes": len(PITCH_CLASSES)},
    )


@pytest.fixture()
def client(mock_bundle):
    with patch("api.main.load_artifacts", return_value=mock_bundle):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


VALID_CONTEXT = {
    "p_throws": "R",
    "stand": "R",
    "inning_topbot": "Top",
    "on_1b": False,
    "on_2b": False,
    "on_3b": False,
    "balls": 1,
    "strikes": 1,
    "outs_when_up": 1,
    "inning": 1,
    "home_score": 0,
    "away_score": 0,
    "pitch_number": 1,
}


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


class TestPredictEndpoint:
    def test_returns_200(self, client):
        resp = client.post("/predict", json=VALID_CONTEXT)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post("/predict", json=VALID_CONTEXT)
        body = resp.json()
        assert "pitch_type" in body
        assert "probabilities" in body
        assert body["pitch_type"] in PITCH_CLASSES
        assert set(body["probabilities"].keys()) == set(PITCH_CLASSES)

    def test_probabilities_sum_to_one(self, client):
        resp = client.post("/predict", json=VALID_CONTEXT)
        probs = resp.json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_optional_prev_pitches(self, client):
        ctx = {**VALID_CONTEXT, "prev_pitches": ["FF", "SL"]}
        resp = client.post("/predict", json=ctx)
        assert resp.status_code == 200

    def test_invalid_balls_rejected(self, client):
        ctx = {**VALID_CONTEXT, "balls": 5}
        resp = client.post("/predict", json=ctx)
        assert resp.status_code == 422

    def test_invalid_strikes_rejected(self, client):
        ctx = {**VALID_CONTEXT, "strikes": 3}
        resp = client.post("/predict", json=ctx)
        assert resp.status_code == 422
