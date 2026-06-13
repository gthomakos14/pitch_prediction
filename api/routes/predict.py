import numpy as np
import polars as pl
import torch
from fastapi import APIRouter, Request

from api.schemas.predict import PitchContext, PredictResponse
from src.fetch.preprocess import PITCH_CLASSES

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(context: PitchContext, request: Request) -> PredictResponse:
    bundle = request.app.state.bundle
    config: dict = bundle.config
    model_type: str = config["model_type"]

    df = pl.DataFrame(
        {
            "p_throws": [context.p_throws],
            "stand": [context.stand],
            "inning_topbot": [context.inning_topbot],
            "on_1b": [context.on_1b],
            "on_2b": [context.on_2b],
            "on_3b": [context.on_3b],
            "balls": [context.balls],
            "strikes": [context.strikes],
            "outs_when_up": [context.outs_when_up],
            "inning": [context.inning],
            "home_score": [context.home_score],
            "away_score": [context.away_score],
            "pitch_number": [context.pitch_number],
        }
    )

    features = bundle.preprocessor.transform(df)
    x = torch.from_numpy(features)

    if model_type == "feedforward":
        batch = {"features": x}
    else:
        # Single-pitch sequence: shape (1, 1, D)
        batch = {
            "sequences": x.unsqueeze(0),
            "lengths": torch.tensor([1], dtype=torch.long),
        }

    with torch.no_grad():
        probs = torch.softmax(bundle.model(batch), dim=1).squeeze(0).numpy()

    pred_idx = int(np.argmax(probs))
    return PredictResponse(
        pitch_type=PITCH_CLASSES[pred_idx],
        probabilities={pt: float(probs[i]) for i, pt in enumerate(PITCH_CLASSES)},
    )
