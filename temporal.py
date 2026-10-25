"""Pure-Python temporal logic for driver drowsiness state decisions."""

from collections import deque
from typing import NamedTuple

from config import TemporalConfig


class MonitorResult(NamedTuple):
    """Drowsiness state and its interpretable temporal measurements."""

    state: str
    perclos: float
    closure_s: float
    yawns: int
    reasons: tuple[str, ...]


class DrowsinessMonitor:
    """Combine frame-level events over rolling time windows."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()
        self._eye_samples: deque[tuple[float, bool]] = deque()
        self._closure_start: float | None = None
        self._yawn_start: float | None = None
        self._yawn_active = False
        self._yawn_times: deque[float] = deque()
        self._last_time: float | None = None

    def update(self, t: float, eye_closed: bool, yawning: bool) -> MonitorResult:
        """Update from one timestamped observation and return current status."""
        if self._last_time is not None and t < self._last_time:
            raise ValueError("Timestamps must be non-decreasing.")
        self._last_time = t
        eye_closed = bool(eye_closed)
        yawning = bool(yawning)

        if eye_closed:
            if self._closure_start is None:
                self._closure_start = t
        else:
            self._closure_start = None
        closure_s = max(0.0, t - self._closure_start) if self._closure_start is not None else 0.0

        if yawning:
            if self._yawn_start is None:
                self._yawn_start = t
            if not self._yawn_active and t - self._yawn_start >= self.config.yawn_sustain_seconds:
                self._yawn_times.append(self._yawn_start)
                self._yawn_active = True
        else:
            self._yawn_start = None
            self._yawn_active = False
        while self._yawn_times and t - self._yawn_times[0] > self.config.yawn_window_seconds:
            self._yawn_times.popleft()

        self._eye_samples.append((t, eye_closed))
        window_start = t - self.config.perclos_window_seconds
        while len(self._eye_samples) > 1 and self._eye_samples[1][0] <= window_start:
            self._eye_samples.popleft()
        coverage = min(self.config.perclos_window_seconds, max(0.0, t - self._eye_samples[0][0]))
        closed_duration = 0.0
        samples = list(self._eye_samples)
        for index in range(len(samples) - 1):
            sample_t, closed = samples[index]
            next_t = samples[index + 1][0]
            if closed:
                closed_duration += max(0.0, min(next_t, t) - max(sample_t, window_start))
        perclos = closed_duration / coverage if coverage > 0 else 0.0

        reasons: list[str] = []
        if closure_s >= self.config.closure_alert_seconds:
            reasons.append("continuous eye closure")
        perclos_ready = coverage >= (
            self.config.perclos_window_seconds * self.config.perclos_minimum_coverage
        )
        if perclos_ready and perclos >= self.config.perclos_alert_threshold:
            reasons.append("high PERCLOS")
        if len(self._yawn_times) >= self.config.yawn_alert_count:
            reasons.append("repeated sustained yawning")

        if reasons:
            state = "ALERT"
        elif (perclos_ready and perclos > self.config.perclos_alert_threshold / 2) or len(
            self._yawn_times
        ) == 1:
            state = "CAUTION"
            if len(self._yawn_times) == 1:
                reasons.append("one sustained yawn")
            if perclos_ready and perclos > self.config.perclos_alert_threshold / 2:
                reasons.append("elevated PERCLOS")
        else:
            state = "OK"
        return MonitorResult(state, perclos, closure_s, len(self._yawn_times), tuple(reasons))
