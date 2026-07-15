"""
perclos_tracker.py
PERCLOS: Percentage of Eye Closure over a rolling time window.
PERCLOS > 0.80 indicates drowsiness.
"""
from collections import deque
import time

EAR_CLOSED_THRESHOLD = 0.20   # eye considered closed below this
PERCLOS_THRESHOLD    = 0.80   # drowsy if > 80% closure in window


class PerclosTracker:
    def __init__(self, window_seconds: int = 60, fps: int = 30):
        self.max_frames = window_seconds * fps
        self.window: deque = deque(maxlen=self.max_frames)

    def update(self, ear: float) -> float:
        """
        Add one frame EAR reading. Returns current PERCLOS (0.0 - 1.0).
        """
        self.window.append(1 if ear < EAR_CLOSED_THRESHOLD else 0)
        if len(self.window) == 0:
            return 0.0
        return sum(self.window) / len(self.window)

    @property
    def is_drowsy(self) -> bool:
        return self.update.__wrapped__ if hasattr(self.update, "__wrapped__")                else (sum(self.window) / max(len(self.window), 1)) > PERCLOS_THRESHOLD
