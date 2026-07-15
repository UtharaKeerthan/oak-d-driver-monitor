"""
session_recorder.py
Records OAK-D camera sessions to disk for offline SIL replay.

Usage:
    python recorder/session_recorder.py --output tests/mock_frames/session.bin --duration 30
"""
import argparse
import pickle
import time
import numpy as np
from pathlib import Path


class SessionRecorder:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.frames: list = []

    def add_frame(self, rgb: np.ndarray, head_yaw: float,
                  head_pitch: float, head_roll: float):
        self.frames.append({
            "rgb": rgb.copy(),
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "head_roll": head_roll,
            "timestamp": time.time()
        })

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(self.frames, f)
        print(f"[OK] Saved {len(self.frames)} frames to {self.output_path}")

    @staticmethod
    def load(path: str) -> list:
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   required=True)
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
    args = parser.parse_args()

    print(f"Recording {args.duration}s session...")
    print("NOTE: Requires OAK-D hardware. For SIL, use pre-recorded sessions in tests/mock_frames/")
    # TODO: integrate with oak_pipeline.py for live recording


if __name__ == "__main__":
    main()
