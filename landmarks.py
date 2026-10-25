"""MediaPipe FaceMesh helpers for eye and mouth geometry."""

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
MOUTH = (61, 291, 13, 14)


def create_face_mesh(max_faces: int = 1) -> Any:
    """Create a FaceMesh instance using the legacy solutions API."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required. Install requirements.txt first.") from exc
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def points_from_landmarks(
    landmarks: Sequence[Any], indices: Sequence[int], width: int, height: int
) -> np.ndarray:
    """Convert normalized landmark indices into pixel coordinates."""
    return np.asarray(
        [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in indices],
        dtype=np.int32,
    )


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Calculate EAR from six points in the conventional eye order."""
    if len(points) != 6:
        raise ValueError("Eye aspect ratio requires exactly six points.")
    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    horizontal = 2.0 * np.linalg.norm(points[0] - points[3])
    return float(vertical / horizontal) if horizontal else 0.0


def mouth_aspect_ratio(landmarks: Sequence[Any], width: int, height: int) -> float:
    """Calculate mouth opening height divided by mouth width."""
    points = points_from_landmarks(landmarks, MOUTH, width, height)
    mouth_width = np.linalg.norm(points[0] - points[1])
    return float(np.linalg.norm(points[2] - points[3]) / mouth_width) if mouth_width else 0.0


def crop_landmark_region(
    frame: np.ndarray, landmarks: Sequence[Any], indices: Sequence[int], padding: float = 0.35
) -> np.ndarray | None:
    """Crop a padded landmark bounding box from a frame."""
    height, width = frame.shape[:2]
    points = points_from_landmarks(landmarks, indices, width, height)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    pad_x = max(2, int((x_max - x_min) * padding))
    pad_y = max(2, int((y_max - y_min) * padding))
    x_min, y_min = max(0, x_min - pad_x), max(0, y_min - pad_y)
    x_max, y_max = min(width, x_max + pad_x + 1), min(height, y_max + pad_y + 1)
    crop = frame[y_min:y_max, x_min:x_max]
    return crop if crop.size else None


def preprocess_crop(crop: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Resize a crop to grayscale float pixels in the model's 0-255 input range."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)[..., np.newaxis]
