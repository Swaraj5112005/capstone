"""Extract mouth crops and weak labels from YawDD videos."""

import argparse
import os
import random
import re
from pathlib import Path
from typing import Sequence

import cv2
import pandas as pd

from landmarks import MOUTH, create_face_mesh, crop_landmark_region, mouth_aspect_ratio
from utils import subject_splits

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}


def video_metadata(path: Path) -> tuple[str, str]:
    """Return subject identifier and recording type (normal or mixed)."""
    parts = re.split(r"[-_ ]+", path.stem)
    collection = next(
        (
            parent.name.casefold()
            for parent in path.parents
            if parent.name.casefold() in {"dash", "mirror"}
        ),
        "unknown",
    )
    gender_match = re.search(r"female|male", path.stem, flags=re.IGNORECASE)
    gender = gender_match.group(0).casefold() if gender_match else "unknown"
    subject = f"{collection}_{gender}_{parts[0].strip()}"
    conditions = {part.lower() for part in parts[1:]}
    if conditions.intersection({"normal", "talking"}):
        return subject, "normal"
    return subject, "mixed"


def label_mixed_frames(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply three-sampled-frame MAR thresholds; leave ambiguous frames out."""
    output = []
    ratios = [float(row["mar"]) for row in records]
    for index, row in enumerate(records):
        ratio = ratios[index]
        if ratio < 0.35:
            label: int | None = 0
        elif (
            ratio > 0.50
            and index >= 2
            and all(value > 0.50 for value in ratios[index - 2 : index + 1])
        ):
            label = 1
        else:
            label = None
        if label is not None:
            output.append({**row, "label": label})
    return output


def prepare(
    input_dir: Path, output: Path, sample_every: int = 5, negatives_per_video: int = 500
) -> pd.DataFrame:
    """Extract crops, weakly label frames, and create a subject split manifest."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"YawDD dataset directory not found: {input_dir}")
    videos = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise ValueError(f"No supported video files found under {input_dir}")
    if sample_every < 1 or negatives_per_video < 1:
        raise ValueError("sample_every and negatives_per_video must be positive.")
    crop_dir = output.parent / "yawdd_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for video_number, video_path in enumerate(videos):
        subject, recording_type = video_metadata(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        records: list[dict[str, object]] = []
        mesh = create_face_mesh()
        frame_index = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % sample_every == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = mesh.process(rgb)
                    if result.multi_face_landmarks:
                        landmarks = result.multi_face_landmarks[0].landmark
                        crop = crop_landmark_region(frame, landmarks, MOUTH, padding=0.65)
                        if crop is not None:
                            crop_path = crop_dir / f"video{video_number:04d}_frame{frame_index:07d}.png"
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                            if not cv2.imwrite(str(crop_path), gray):
                                raise OSError(f"Could not write mouth crop: {crop_path}")
                            records.append(
                                {
                                    "path": Path(
                                        os.path.relpath(crop_path.resolve(), output.parent.resolve())
                                    ).as_posix(),
                                    "subject": subject,
                                    "video": video_path.name,
                                    "mar": mouth_aspect_ratio(landmarks, frame.shape[1], frame.shape[0]),
                                }
                            )
                frame_index += 1
        finally:
            capture.release()
            mesh.close()

        if recording_type == "normal":
            labelled = [{**row, "label": 0} for row in records]
        else:
            labelled = label_mixed_frames(records)
        negatives = [row for row in labelled if row["label"] == 0]
        positives = [row for row in labelled if row["label"] == 1]
        if len(negatives) > negatives_per_video:
            negatives = random.Random(42 + video_number).sample(negatives, negatives_per_video)
        rows.extend(positives + negatives)
        print(f"{video_path.name}: retained {len(positives)} positive, {len(negatives)} negative crops")

    if not rows:
        raise ValueError("No labelled mouth crops were extracted; check the videos and FaceMesh input.")
    frame = pd.DataFrame(rows)
    split_map = subject_splits(frame["subject"].astype(str).tolist(), seed=42)
    frame["split"] = frame["subject"].astype(str).map(split_map)
    frame.drop(columns=["mar"]).to_csv(output, index=False)
    print(f"Saved {len(frame)} labelled crops from {frame['subject'].nunique()} subjects to {output}")
    print(frame["split"].value_counts().to_string())
    return frame


def main(argv: Sequence[str] | None = None) -> None:
    """Run YawDD video extraction and manifest generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/yawdd_manifest.csv"))
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--negatives-per-video", type=int, default=500)
    args = parser.parse_args(argv)
    prepare(args.input_dir, args.output, args.sample_every, args.negatives_per_video)


if __name__ == "__main__":
    main()
