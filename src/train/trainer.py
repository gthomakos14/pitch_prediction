import shutil
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.fetch.dataset import PitchDataset, SequenceDataset, ff_collate, seq_collate
from src.fetch.preprocess import ARTIFACTS_DIR, PITCH_CLASSES, load_label_encoder, load_preprocessor
from src.models import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train(model_type: str, config_path: Path) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    preprocessor = load_preprocessor()
    load_label_encoder()  # validate artifacts exist
    num_classes = len(PITCH_CLASSES)
    input_dim = preprocessor.feature_dim

    if model_type == "feedforward":
        train_ds = PitchDataset.from_split("train", preprocessor)
        val_ds = PitchDataset.from_split("val", preprocessor)
        collate_fn = ff_collate
    elif model_type == "sequence":
        max_seq_len: int = config.get("max_seq_len", 20)
        train_ds = SequenceDataset.from_split("train", preprocessor, max_seq_len=max_seq_len)
        val_ds = SequenceDataset.from_split("val", preprocessor, max_seq_len=max_seq_len)
        collate_fn = seq_collate
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    batch_size: int = config.get("batch_size", 512)
    num_workers: int = config.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_config = {**config, "model_type": model_type, "input_dim": input_dim, "num_classes": num_classes}
    model = build_model(model_type, full_config).to(device)

    lr: float = config.get("lr", 1e-3)
    epochs: int = config.get("epochs", 20)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("pitch_prediction")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({"model_type": model_type, "config_path": str(config_path), **config})

        best_val_loss = float("inf")
        tmp_ckpt = ARTIFACTS_DIR / f"_tmp_{run_id}.pt"
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch, labels in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            all_preds: list[int] = []
            all_labels: list[int] = []
            with torch.no_grad():
                for batch, labels in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = labels.to(device)
                    logits = model(batch)
                    val_loss += criterion(logits, labels).item() * labels.size(0)
                    all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            val_loss /= len(val_ds)

            per_class_f1 = f1_score(
                all_labels, all_preds,
                labels=list(range(num_classes)), average=None, zero_division=0,
            )
            mean_f1 = float(np.mean(per_class_f1))

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1_mean": mean_f1,
                    **{f"val_f1_{PITCH_CLASSES[i]}": float(per_class_f1[i]) for i in range(num_classes)},
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={mean_f1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), tmp_ckpt)

        model.load_state_dict(torch.load(tmp_ckpt, map_location="cpu", weights_only=True))
        tmp_ckpt.unlink(missing_ok=True)

        torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")
        with open(ARTIFACTS_DIR / "config.yaml", "w") as f:
            yaml.safe_dump(full_config, f)

        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")
        print(f"\nRun complete — run_id={run_id}")
        return run_id
