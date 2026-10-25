"""Keras model builders for eye-state and yawning classification."""

from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomBrightness(0.15, value_range=(0, 255)),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model(arch: str, input_shape: tuple[int, int, int] = (64, 64, 1)) -> keras.Model:
    """Build a binary classifier with the requested architecture."""
    if arch not in {"simple", "improved", "mobilenet"}:
        raise ValueError(f"Unknown architecture '{arch}'. Choose simple, improved, or mobilenet.")
    inputs = keras.Input(shape=input_shape, name="image")
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    if arch == "simple":
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    elif arch == "improved":
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation="relu")(x)
    else:
        x = layers.Resizing(96, 96, name="mobilenet_resize")(x)
        x = layers.Concatenate(name="grayscale_to_rgb")([x, x, x])
        x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(x)
        base = keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, weights="imagenet"
        )
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="probability")(x)
    return keras.Model(inputs, outputs, name=f"{arch}_binary_classifier")


def load_model(path: str) -> Any:
    """Load a saved Keras model without custom objects."""
    return keras.models.load_model(path)

