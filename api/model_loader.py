import pickle
from dataclasses import dataclass

import torch
import yaml
from sklearn.preprocessing import LabelEncoder

from src.fetch.preprocess import ARTIFACTS_DIR, Preprocessor
from src.models import build_model


@dataclass
class ModelBundle:
    model: torch.nn.Module
    preprocessor: Preprocessor
    label_encoder: LabelEncoder
    config: dict


def load_artifacts() -> ModelBundle:
    config_path = ARTIFACTS_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(ARTIFACTS_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor: Preprocessor = pickle.load(f)
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder: LabelEncoder = pickle.load(f)

    model = build_model(config["model_type"], config)
    model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    return ModelBundle(
        model=model,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        config=config,
    )
