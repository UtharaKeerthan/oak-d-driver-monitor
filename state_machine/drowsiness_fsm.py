"""
drowsiness_fsm.py
Driver drowsiness finite state machine.
States: ALERT -> WARNING -> DROWSY
"""
from enum import Enum

EAR_WARN_THRESHOLD   = 0.20
EAR_WARN_FRAMES      = 60     # 2s at 30fps
EAR_DROWSY_FRAMES    = 120    # 4s at 30fps
PERCLOS_WARN         = 0.40
PERCLOS_DROWSY       = 0.80
PITCH_DROWSY_DEG     = 25.0


class DrowsinessState(Enum):
    ALERT   = "ALERT"
    WARNING = "WARNING"
    DROWSY  = "DROWSY"


class DrowsinessFSM:
    def __init__(self):
        self.state       = DrowsinessState.ALERT
        self._ear_frames = 0

    def update(self, ear: float, perclos: float, pitch: float) -> DrowsinessState:
        eye_closed = ear < EAR_WARN_THRESHOLD
        self._ear_frames = (self._ear_frames + 1) if eye_closed else 0

        if (self._ear_frames >= EAR_DROWSY_FRAMES or
                perclos >= PERCLOS_DROWSY or pitch >= PITCH_DROWSY_DEG):
            self.state = DrowsinessState.DROWSY
        elif (self._ear_frames >= EAR_WARN_FRAMES or perclos >= PERCLOS_WARN):
            self.state = DrowsinessState.WARNING
        else:
            self.state = DrowsinessState.ALERT

        return self.state
