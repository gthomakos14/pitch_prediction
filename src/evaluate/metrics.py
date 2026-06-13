from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.fetch.dataset import PitchDataset, SequenceDataset, ff_collate, seq_collate
from src.fetch.preprocess import ARTIFACTS_DIR, PITCH_CLASSES, load_preprocessor
from src.models import build_model


def evaluate(run_id: str) -> None:
    config_path = ARTIFACTS_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    preprocessor = load_preprocessor()
    model_type: str = config["model_type"]
    num_classes = len(PITCH_CLASSES)

    if model_type == "feedforward":
        test_ds = PitchDataset.from_split("test", preprocessor)
        collate_fn = ff_collate
    else:
        test_ds = SequenceDataset.from_split(
            "test", preprocessor, max_seq_len=config.get("max_seq_len", 20)
        )
        collate_fn = seq_collate

    loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
    model = build_model(model_type, config)
    model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch, labels in loader:
            all_preds.extend(model(batch).argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())

    per_class_f1 = f1_score(
        all_labels, all_preds, labels=list(range(num_classes)), average=None, zero_division=0
    )
    mean_f1 = float(np.mean(per_class_f1))
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    metrics = {
        "test_accuracy": accuracy,
        "test_f1_mean": mean_f1,
        **{f"test_f1_{PITCH_CLASSES[i]}": float(per_class_f1[i]) for i in range(num_classes)},
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    print(f"\nTest accuracy: {accuracy:.4f}   mean F1: {mean_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=PITCH_CLASSES, zero_division=0))
