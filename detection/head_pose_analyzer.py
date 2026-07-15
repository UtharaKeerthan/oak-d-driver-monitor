"""
head_pose_analyzer.py
Parses head-pose-estimation-adas-0001 output and classifies head nod events.
"""

PITCH_NOD_THRESHOLD_DEG  = 25.0   # chin drops toward chest
NOD_DURATION_FRAMES      = 45     # sustained nod over ~1.5s at 30fps


class HeadPoseAnalyzer:
    def __init__(self):
        self._nod_count = 0

    def parse_model_output(self, raw_output) -> tuple:
        """
        head-pose-estimation-adas-0001 outputs [yaw, pitch, roll] in degrees.
        Returns (yaw, pitch, roll) as floats.
        """
        import numpy as np
        out = np.array(raw_output).flatten()
        if len(out) >= 3:
            return float(out[0]), float(out[1]), float(out[2])
        return 0.0, 0.0, 0.0

    def is_nodding(self, pitch_deg: float) -> bool:
        """
        Returns True if head has been pitched down beyond threshold
        for NOD_DURATION_FRAMES consecutive frames.
        """
        if pitch_deg > PITCH_NOD_THRESHOLD_DEG:
            self._nod_count += 1
        else:
            self._nod_count = 0
        return self._nod_count >= NOD_DURATION_FRAMES
