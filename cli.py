import argparse
import shutil
from pathlib import Path


def _fetch(args: argparse.Namespace) -> None:
    from src.fetch.statcast_fetch import refresh_statcast_data
    refresh_statcast_data(min_year=args.season, override_refresh=True)


def _preprocess(args: argparse.Namespace) -> None:
    from src.features.engineer import build_features
    from src.fetch.preprocess import run_preprocessing
    build_features()
    run_preprocessing()


def _train(args: argparse.Namespace) -> None:
    from src.train.trainer import train
    run_id = train(args.model, Path(args.config))
    print(f"Saved run_id: {run_id}")


def _evaluate(args: argparse.Namespace) -> None:
    from src.evaluate.metrics import evaluate
    evaluate(args.run)


def _export(args: argparse.Namespace) -> None:
    from src.fetch.preprocess import ARTIFACTS_DIR
    dest = Path(args.output)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ARTIFACTS_DIR, dest)
    print(f"Artifacts exported to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="cli", description="Pitch Prediction CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="Download Statcast data")
    fetch_p.add_argument("--season", type=int, required=True, help="Season year (e.g. 2023)")

    sub.add_parser("preprocess", help="Build features and write train/val/test splits")

    train_p = sub.add_parser("train", help="Train a model and log to MLflow")
    train_p.add_argument("--model", required=True, choices=["feedforward", "sequence"])
    train_p.add_argument("--config", required=True, help="Path to YAML config file")

    eval_p = sub.add_parser("evaluate", help="Evaluate a model run on the test split")
    eval_p.add_argument("--run", required=True, help="MLflow run ID")

    export_p = sub.add_parser("export", help="Copy model artifacts to an output directory")
    export_p.add_argument("--output", required=True, help="Destination directory path")

    args = parser.parse_args()
    {"fetch": _fetch, "preprocess": _preprocess, "train": _train, "evaluate": _evaluate, "export": _export}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
