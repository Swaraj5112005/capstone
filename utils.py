"""Small shared helpers for manifests, paths, and reproducibility."""

from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd


def resolve_manifest_path(value: str | Path, manifest: Path) -> Path:
    """Resolve a manifest path relative to its CSV when it is not absolute."""
    path = Path(str(value))
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def require_file(path: str | Path, description: str = "File") -> Path:
    """Return an existing file path or raise a useful error."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def subject_splits(subjects: list[str], seed: int = 42) -> dict[str, str]:
    """Assign unique subjects to deterministic 70/15/15 splits."""
    from sklearn.model_selection import train_test_split

    unique = np.asarray(sorted(set(subjects)), dtype=object)
    if len(unique) < 4:
        raise ValueError("At least 4 distinct subjects are required for 70/15/15 splits.")
    train, rest = train_test_split(unique, test_size=0.30, random_state=seed, shuffle=True)
    validation, test = train_test_split(rest, test_size=0.50, random_state=seed, shuffle=True)
    return {
        **{str(item): "train" for item in train},
        **{str(item): "validation" for item in validation},
        **{str(item): "test" for item in test},
    }


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append a dictionary as one CSV row, creating the header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
