"""Evaluate a saved classifier using only the manifest test split."""

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import OUTPUT_DIR
from data import make_dataset, read_manifest
from models import load_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task", choices=("eye", "yawn"), required=True)
    parser.add_argument("--arch", choices=("simple", "improved", "mobilenet"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate the model, export error analysis, subgroup scores, and summary."""
    args = parse_args(argv)
    if args.batch < 1:
        raise ValueError("--batch must be a positive integer.")
    prefix = f"{args.task}_{args.arch}"
    model_path = args.model or args.output_dir / f"{prefix}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}; run train.py first.")
    frame = read_manifest(args.manifest, args.task)
    test_rows = frame[frame["split"] == "test"].copy()
    if test_rows.empty:
        raise ValueError("Manifest contains no test rows.")
    model = load_model(str(model_path))
    dataset = make_dataset(test_rows, args.batch, training=False)
    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    labels = test_rows["label"].astype(int).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
    }

    image_ds = make_dataset(test_rows.iloc[:1], 1, training=False)
    sample, _ = next(iter(image_ds))
    model(sample, training=False)
    start = time.perf_counter()
    for _ in range(20):
        model(sample, training=False)
    latency_ms = (time.perf_counter() - start) * 1000.0 / 20
    metrics["latency_ms_batch1"] = float(latency_ms)
    metrics["fps_batch1"] = float(1000.0 / latency_ms) if latency_ms else float("inf")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
             xlabel="Predicted", ylabel="Actual", title="Test confusion matrix")
    for (row, col), value in np.ndenumerate(matrix):
        axis.text(col, row, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close(figure)

    result = test_rows.copy()
    result["probability"] = probabilities
    result["prediction"] = predictions
    result["confidence"] = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    result["correct"] = predictions == labels
    result.loc[~result["correct"]].nlargest(50, "confidence").to_csv(
        args.output_dir / f"{prefix}_wrong_predictions.csv", index=False
    )
    subgroup_accuracy: dict[str, dict[str, float]] = {}
    for column in ("glasses", "lighting", "reflections", "gender"):
        if column not in result.columns:
            continue
        for group, rows in result.groupby(column, dropna=False):
            subgroup_accuracy.setdefault(column, {})[str(group)] = float(
                accuracy_score(rows["label"].astype(int), rows["prediction"])
            )

    summary_path = args.output_dir / "summary.csv"
    summary_row = {
        "task": args.task,
        "arch": args.arch,
        **metrics,
        "robustness_accuracy": json.dumps(subgroup_accuracy, sort_keys=True),
    }
    summary_columns = [
        "task", "arch", "accuracy", "precision", "recall", "f1", "auc",
        "latency_ms_batch1", "fps_batch1", "robustness_accuracy",
    ]
    pd.DataFrame([summary_row], columns=summary_columns).to_csv(
        summary_path, mode="a", header=not summary_path.exists(), index=False
    )
    print(pd.Series(metrics).to_string())
    if subgroup_accuracy:
        print("Robustness accuracy:")
        print(json.dumps(subgroup_accuracy, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
