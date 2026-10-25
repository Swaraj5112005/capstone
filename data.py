"""Manifest validation and tf.data input pipeline."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from config import IMAGE_SIZE
from utils import resolve_manifest_path


def read_manifest(path: str | Path, task: str) -> pd.DataFrame:
    """Load and validate the image manifest for a classification task."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    required = {"path", "label", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
    frame = frame[frame["split"].isin(["train", "validation", "test"])].copy()
    frame["resolved_path"] = frame["path"].map(
        lambda value: str(resolve_manifest_path(value, manifest_path))
    )
    absent = [value for value in frame["resolved_path"] if not Path(value).is_file()]
    if absent:
        raise FileNotFoundError(f"Manifest image not found (first missing path): {absent[0]}")
    if frame.empty:
        raise ValueError(f"The {task} manifest contains no usable rows.")
    labels = set(frame["label"].dropna().astype(int).unique())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError("Manifest labels must be binary integers (0 or 1).")
    return frame


def make_dataset(
    frame: pd.DataFrame, batch_size: int, training: bool, seed: int = 42
) -> tf.data.Dataset:
    """Create a deterministic grayscale image dataset from manifest rows."""
    paths = frame["resolved_path"].astype(str).to_numpy()
    labels = frame["label"].astype(np.float32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=1, expand_animations=False)
        image.set_shape([None, None, 1])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, tf.reshape(label, (1,))

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(frame: pd.DataFrame) -> dict[int, float]:
    """Calculate balanced weights for the classes present in training data."""
    counts = frame["label"].astype(int).value_counts().to_dict()
    if len(counts) < 2:
        raise ValueError("Training split must contain examples from both classes.")
    total = sum(counts.values())
    return {int(label): total / (len(counts) * count) for label, count in counts.items()}

