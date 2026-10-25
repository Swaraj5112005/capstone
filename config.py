"""Shared project configuration."""

from dataclasses import dataclass
from pathlib import Path

IMAGE_SIZE = (64, 64)
SEED = 42
OUTPUT_DIR = Path("outputs")
EYE_CLOSED_LABEL = 1


@dataclass(frozen=True)
class TemporalConfig:
    """Thresholds for smoothing frame predictions into driver states."""

    closure_alert_seconds: float = 1.5
    perclos_window_seconds: float = 10.0
    perclos_alert_threshold: float = 0.40
    perclos_minimum_coverage: float = 0.50
    yawn_sustain_seconds: float = 1.0
    yawn_window_seconds: float = 60.0
    yawn_alert_count: int = 2

