"""
yawn_detector.py
Mouth Aspect Ratio (MAR) for yawn detection from FaceMesh landmarks.
"""
import math
from typing import List, Tuple

# MediaPipe mouth landmark indices (subset)
MOUTH_IDX = [61, 291, 17, 0, 267, 37]
MAR_THRESHOLD = 0.5

Point = Tuple[int, int]

def _dist(a: Point, b: Point) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class YawnDetector:
    def is_yawning(self, pts: List[Point]) -> bool:
        if len(pts) < 300:
            return False
        p = [pts[i] for i in MOUTH_IDX]
        vertical   = (_dist(p[2], p[5]) + _dist(p[3], p[4])) / 2.0
        horizontal = _dist(p[0], p[1])
        mar = vertical / (horizontal + 1e-6)
        return mar > MAR_THRESHOLD
