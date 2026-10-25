"""Train an image classifier from a CSV manifest."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from config import OUTPUT_DIR, SEED
from data import class_weights, make_dataset, read_manifest
from models import build_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task", choices=("eye", "yawn"), required=True)
    parser.add_argument("--arch", choices=("simple", "improved", "mobilenet"), default="improved")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run training and save model, history, and learning curves."""
    args = parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or args.lr <= 0:
        raise ValueError("--epochs and --batch must be positive, and --lr must be greater than zero.")
    tf.keras.utils.set_random_seed(SEED)
    frame = read_manifest(args.manifest, args.task)
    train_rows = frame[frame["split"] == "train"]
    validation_rows = frame[frame["split"] == "validation"]
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Manifest must contain non-empty train and validation splits.")
    train_ds = make_dataset(train_rows, args.batch, training=True, seed=SEED)
    validation_ds = make_dataset(validation_rows, args.batch, training=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.task}_{args.arch}"
    model_path = args.output_dir / f"{prefix}.keras"
    model = build_model(args.arch)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        class_weight=class_weights(train_rows),
        callbacks=callbacks,
    )
    if not model_path.exists():
        model.save(model_path)
    with (args.output_dir / f"{prefix}_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history.history, stream, indent=2)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{prefix}_curves.png", dpi=150)
    plt.close(figure)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()

