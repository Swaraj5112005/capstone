"""Create a subject-independent manifest for the MRL Eye Dataset."""

import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from utils import subject_splits


def parse_filename(path: Path) -> dict[str, object]:
    """Parse MRL metadata from its underscore-delimited filename."""
    parts = path.stem.split("_")
    if len(parts) != 8:
        raise ValueError(f"Unexpected MRL filename format: {path.name}")
    try:
        eye_state = int(parts[4])
    except ValueError as exc:
        raise ValueError(f"Invalid eyeState in filename: {path.name}") from exc
    if eye_state not in (0, 1):
        raise ValueError(f"eyeState must be 0 or 1 in filename: {path.name}")
    return {
        "subject": parts[0],
        "gender": parts[2],
        "glasses": parts[3],
        "label": 1 if eye_state == 0 else 0,
        "reflections": parts[5],
        "lighting": parts[6],
        "sensor": parts[7],
    }


def prepare(input_dir: Path, output: Path) -> pd.DataFrame:
    """Scan images, assign subject-level splits, and save a manifest."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"MRL dataset directory not found: {input_dir}")
    images = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() == ".png")
    if not images:
        raise ValueError(f"No PNG images found under {input_dir}")
    rows = []
    for image in images:
        metadata = parse_filename(image)
        rows.append(
            {
                "path": Path(os.path.relpath(image.resolve(), output.parent.resolve())).as_posix(),
                **metadata,
            }
        )
    frame = pd.DataFrame(rows)
    split_map = subject_splits(frame["subject"].astype(str).tolist(), seed=42)
    frame["split"] = frame["subject"].astype(str).map(split_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} image rows from {frame['subject'].nunique()} subjects to {output}")
    print(frame["split"].value_counts().to_string())
    return frame


def main(argv: Sequence[str] | None = None) -> None:
    """Run the MRL manifest preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/mrl_manifest.csv"))
    args = parser.parse_args(argv)
    prepare(args.input_dir, args.output)


if __name__ == "__main__":
    main()
