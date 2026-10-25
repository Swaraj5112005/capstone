"""Live webcam/video drowsiness demo."""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from config import OUTPUT_DIR
from landmarks import (
    LEFT_EYE,
    MOUTH,
    RIGHT_EYE,
    create_face_mesh,
    crop_landmark_region,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    points_from_landmarks,
    preprocess_crop,
)
from temporal import DrowsinessMonitor, MonitorResult
from utils import append_csv_row

STATE_COLORS = {"OK": (40, 190, 40), "CAUTION": (0, 190, 255), "ALERT": (0, 0, 255)}


def _source(value: str) -> int | str:
    """Interpret a numeric source as a webcam index, otherwise a path."""
    try:
        return int(value)
    except ValueError:
        return value


def _beep() -> None:
    """Play a short alert tone with a dependency-free console fallback."""
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except (ImportError, RuntimeError):
        sys.stdout.write("\a")
        sys.stdout.flush()


def _predict(model: Any, crop: np.ndarray) -> float:
    """Run one normalized crop through a Keras sigmoid classifier."""
    image = preprocess_crop(crop)[np.newaxis, ...]
    return float(model(image, training=False).numpy().reshape(-1)[0])


def _draw_status(
    frame: np.ndarray,
    result: MonitorResult,
    no_face: bool,
    alerts: tuple[str, ...],
) -> None:
    """Render current monitor values on the camera frame."""
    state = "NO FACE DETECTED" if no_face else result.state
    color = (255, 255, 255) if no_face else STATE_COLORS[result.state]
    cv2.putText(frame, state, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"PERCLOS: {result.perclos:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Closure: {result.closure_s:.1f} s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Yawns/min: {result.yawns}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for index, reason in enumerate(alerts[:3]):
        cv2.putText(frame, reason, (20, 165 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if result.state == "ALERT" and not no_face:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)


def run_demo(
    source: int | str, eye_model_path: Path | None, yawn_model_path: Path | None, heuristic: bool
) -> None:
    """Run frame inference and temporal monitoring until q is pressed."""
    eye_model = yawn_model = None
    if not heuristic:
        if eye_model_path is None or yawn_model_path is None:
            raise ValueError("Provide both --eye-model and --yawn-model unless --heuristic is enabled.")
        for model_path in (eye_model_path, yawn_model_path):
            if not model_path.is_file():
                raise FileNotFoundError(f"Model not found: {model_path}")
        from models import load_model

        eye_model = load_model(str(eye_model_path))
        yawn_model = load_model(str(yawn_model_path))

    is_video = isinstance(source, str)
    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    mesh = create_face_mesh()
    monitor = DrowsinessMonitor()
    result = MonitorResult("OK", 0.0, 0.0, 0, ())
    last_face_time: float | None = None
    start_timestamp: float | None = None
    last_beep_time = -float("inf")
    previous_state = "OK"
    frame_index = 0
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    alert_log = output_dir / "alerts_log.csv"
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / fps if is_video else time.monotonic()
            if start_timestamp is None:
                start_timestamp = timestamp
            now_wall = time.monotonic()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = mesh.process(rgb)
            reasons: tuple[str, ...] = result.reasons
            if detected.multi_face_landmarks:
                last_face_time = timestamp
                face = detected.multi_face_landmarks[0].landmark
                height, width = frame.shape[:2]
                left_points = points_from_landmarks(face, LEFT_EYE, width, height)
                right_points = points_from_landmarks(face, RIGHT_EYE, width, height)
                ear = (eye_aspect_ratio(left_points) + eye_aspect_ratio(right_points)) / 2.0
                left_crop = crop_landmark_region(frame, face, LEFT_EYE, padding=0.55)
                right_crop = crop_landmark_region(frame, face, RIGHT_EYE, padding=0.55)
                mouth_crop = crop_landmark_region(frame, face, MOUTH, padding=0.65)
                if heuristic:
                    eye_closed = ear < 0.20
                    mar = mouth_aspect_ratio(face, width, height)
                    yawning = mar > 0.55
                elif left_crop is not None and right_crop is not None and mouth_crop is not None:
                    eye_probability = (_predict(eye_model, left_crop) + _predict(eye_model, right_crop)) / 2.0
                    eye_closed = eye_probability >= 0.5
                    yawning = _predict(yawn_model, mouth_crop) >= 0.5
                else:
                    eye_closed, yawning = False, False
                result = monitor.update(timestamp, eye_closed, yawning)
                reasons = result.reasons
                if result.state == "ALERT" and previous_state != "ALERT":
                    append_csv_row(
                        alert_log,
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "source": str(source),
                            "perclos": f"{result.perclos:.4f}",
                            "closure_s": f"{result.closure_s:.3f}",
                            "yawns": result.yawns,
                            "reasons": "; ".join(result.reasons),
                        },
                    )
                if result.state == "ALERT" and now_wall - last_beep_time >= 2.0:
                    _beep()
                    last_beep_time = now_wall
                previous_state = result.state

            reference_time = last_face_time if last_face_time is not None else start_timestamp
            no_face = reference_time is not None and timestamp - reference_time >= 2.0
            _draw_status(frame, result, no_face, reasons)
            cv2.imshow("Driver Drowsiness Detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    finally:
        capture.release()
        mesh.close()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse options and launch the live demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Webcam index (for example 0) or video file")
    parser.add_argument("--eye-model", type=Path)
    parser.add_argument("--yawn-model", type=Path)
    parser.add_argument("--heuristic", action="store_true", help="Use EAR/MAR rules instead of CNN models")
    args = parser.parse_args(argv)
    run_demo(_source(args.source), args.eye_model, args.yawn_model, args.heuristic)


if __name__ == "__main__":
    main()
