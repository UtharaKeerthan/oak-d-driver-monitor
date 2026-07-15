"""
dashboard.py
2D OpenCV dashboard showing live camera feed + drowsiness metrics overlay.
"""
import cv2
import numpy as np
from collections import deque
from state_machine.drowsiness_fsm import DrowsinessState

STATE_COLORS = {
    DrowsinessState.ALERT:   (0, 200, 0),    # green
    DrowsinessState.WARNING: (0, 165, 255),  # orange
    DrowsinessState.DROWSY:  (0, 0, 255),    # red
}
EAR_HISTORY = 300  # 10s at 30fps


class Dashboard:
    def __init__(self, width: int = 1280, height: int = 480):
        self.w = width
        self.h = height
        self.ear_history: deque = deque(maxlen=EAR_HISTORY)

    def render(self, frame: np.ndarray, metrics: dict, fps: float) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Left panel: camera feed
        cam_resized = cv2.resize(frame, (640, self.h))
        canvas[:, :640] = cam_resized

        # Right panel: metrics
        state  = metrics.get("state", DrowsinessState.ALERT)
        color  = STATE_COLORS[state]
        self.ear_history.append((metrics.get("ear_left", 0) +
                                  metrics.get("ear_right", 0)) / 2.0)

        lines = [
            (f"FPS: {fps:.1f}",                  (255, 255, 255)),
            (f"EAR Left:   {metrics.get('ear_left',  0):.2f}", (200,200,200)),
            (f"EAR Right:  {metrics.get('ear_right', 0):.2f}", (200,200,200)),
            (f"PERCLOS:    {metrics.get('perclos',   0)*100:.0f}%", (200,200,200)),
            (f"Head Pitch: {metrics.get('head_pitch',0):.1f} deg", (200,200,200)),
            ("", (0,0,0)),
            (f"STATUS: {state.value}", color),
        ]
        for i, (text, col) in enumerate(lines):
            cv2.putText(canvas, text, (660, 40 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        # EAR time-series graph
        self._draw_ear_graph(canvas, 660, 340, 580, 120)
        return canvas

    def _draw_ear_graph(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (40,40,40), -1)
        cv2.putText(canvas, "EAR (10s)", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        if len(self.ear_history) > 1:
            pts = []
            for i, v in enumerate(self.ear_history):
                px = x + int(i * w / EAR_HISTORY)
                py = y + h - int(min(v, 0.5) / 0.5 * h)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], (0, 200, 0), 1)
