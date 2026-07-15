"""
ear_calculator.py
Eye Aspect Ratio (EAR) from MediaPipe FaceMesh 468-landmark indices.
"""
import math
from typing import List, Tuple

# MediaPipe FaceMesh eye landmark indices
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158,  133, 153, 144]

Point = Tuple[int, int]

def _dist(a: Point, b: Point) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _ear(pts: List[Point], indices: List[int]) -> float:
    """
    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    """
    p = [pts[i] for i in indices]
    vertical   = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horizontal = 2.0 * _dist(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


class EarCalculator:
    def compute_left(self,  pts: List[Point]) -> float:
        return _ear(pts, LEFT_EYE_IDX)

    def compute_right(self, pts: List[Point]) -> float:
        return _ear(pts, RIGHT_EYE_IDX)
