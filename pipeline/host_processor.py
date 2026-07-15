"""
host_processor.py
Host-side processing: runs MediaPipe FaceMesh on frames received from OAK-D.
Coordinates with detection modules to compute drowsiness metrics.
"""
import cv2
import numpy as np
from detection.ear_calculator   import EarCalculator
from detection.perclos_tracker  import PerclosTracker
from detection.head_pose_analyzer import HeadPoseAnalyzer
from detection.yawn_detector    import YawnDetector
from state_machine.drowsiness_fsm import DrowsinessState, DrowsinessFSM

try:
    import mediapipe as mp
    MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except ImportError:
    MP_FACE_MESH = None
    print("[WARN] mediapipe not installed - landmark detection disabled")


class HostProcessor:
    def __init__(self):
        self.ear_calc    = EarCalculator()
        self.perclos     = PerclosTracker(window_seconds=60)
        self.head_pose   = HeadPoseAnalyzer()
        self.yawn_det    = YawnDetector()
        self.fsm         = DrowsinessFSM()

    def process_frame(self, frame: np.ndarray,
                      head_yaw: float, head_pitch: float, head_roll: float) -> dict:
        """
        Process one RGB frame. Returns a dict of all metrics and the current state.
        """
        result = {
            "ear_left": 0.0, "ear_right": 0.0, "perclos": 0.0,
            "head_pitch": head_pitch, "yaw": head_yaw, "roll": head_roll,
            "mouth_open": False, "state": DrowsinessState.ALERT
        }

        if MP_FACE_MESH is None:
            return result

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = MP_FACE_MESH.process(rgb)

        if mesh_result.multi_face_landmarks:
            landmarks = mesh_result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            result["ear_left"]  = self.ear_calc.compute_left(pts)
            result["ear_right"] = self.ear_calc.compute_right(pts)
            ear_avg = (result["ear_left"] + result["ear_right"]) / 2.0

            result["perclos"]    = self.perclos.update(ear_avg)
            result["mouth_open"] = self.yawn_det.is_yawning(pts)

        result["state"] = self.fsm.update(
            ear=(result["ear_left"] + result["ear_right"]) / 2.0,
            perclos=result["perclos"],
            pitch=head_pitch
        )
        return result
