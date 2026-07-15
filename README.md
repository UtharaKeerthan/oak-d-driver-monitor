# oak-d-driver-monitor

A real-time driver drowsiness detection system built with the Luxonis OAK-D stereo camera, combining Intel OpenVINO neural network models running on the OAK-D's on-device VPU with Google MediaPipe landmark analysis on the host. Outputs a live 2D OpenCV dashboard showing drowsiness metrics and safety alerts.

Designed as a proof-of-concept for in-cabin driver monitoring systems (DMS) — the type of application that runs on NXP S32G or i.MX8 class ECUs in production vehicles.

---

## What it detects

The system computes three complementary drowsiness signals in real time:

| Signal | Method | Alert threshold |
|---|---|---|
| Eye closure | Eye Aspect Ratio (EAR) from MediaPipe FaceMesh | EAR < 0.20 for more than 2 seconds |
| Sustained fatigue | PERCLOS over a rolling 60-second window | Eyes closed more than 80% of the window |
| Head nodding | Pitch angle from OpenVINO head-pose model | Pitch > 25° downward for more than 1.5 seconds |

When any signal crosses its threshold, the drowsiness state machine advances from ALERT → WARNING → DROWSY and triggers a visual alarm on the dashboard.

---

## System architecture

```
OAK-D Camera Hardware
        │
        ▼
 DepthAI Pipeline (on device — Myriad X VPU)
 ┌────────────────────────────────────────────┐
 │  ColorCamera node   RGB frame at 30 fps    │
 │         │                                  │
 │         ▼                                  │
 │  NeuralNetwork node                        │
 │  face-detection-adas-0001.blob             │  OpenVINO model
 │         │                                  │  runs ON the VPU
 │         ▼                                  │  no host CPU used
 │  NeuralNetwork node                        │
 │  head-pose-estimation-adas-0001.blob       │
 └───────────────┬────────────────────────────┘
                 │  face bbox + head pose angles
                 ▼  (sent to host via XLink)
 Host Processing (Python, CPU)
 ┌────────────────────────────────────────────┐
 │  MediaPipe FaceMesh                        │
 │  468 facial landmarks (3D)                 │
 │         │                                  │
 │  ┌──────▼──────┐  ┌─────────────────────┐  │
 │  │ EAR         │  │ Head Pose Analyzer   │  │
 │  │ Calculator  │  │ pitch / yaw / roll   │  │
 │  └──────┬──────┘  └──────────┬──────────┘  │
 │         │                    │              │
 │  PERCLOS Tracker             │              │
 │  60-second rolling window    │              │
 │         │                    │              │
 │  Yawn Detector               │              │
 │  mouth aspect ratio          │              │
 │         │                    │              │
 │  ┌──────▼────────────────────▼──────────┐  │
 │  │     Drowsiness State Machine          │  │
 │  │     ALERT → WARNING → DROWSY          │  │
 │  └──────────────────┬───────────────────┘  │
 └─────────────────────┼──────────────────────┘
                       │
                       ▼
            2D OpenCV Dashboard
```

---

## Models used

### On the OAK-D VPU (OpenVINO .blob format)

| Model | Source | Purpose |
|---|---|---|
| `face-detection-adas-0001` | Intel Open Model Zoo | Detects driver face bounding box |
| `head-pose-estimation-adas-0001` | Intel Open Model Zoo | Outputs yaw, pitch, roll in degrees |

These models are compiled to `.blob` format for the Myriad X VPU using the OpenVINO Model Optimizer. They run entirely on the OAK-D — no GPU or heavy CPU is needed on the host machine.

### On the host CPU (Python)

| Model / Library | Source | Purpose |
|---|---|---|
| MediaPipe FaceMesh | Google MediaPipe | 468 3D facial landmarks for EAR and yawn |
| MediaPipe Pose | Google MediaPipe | 33 body keypoints for shoulder/neck angle |

---

## Drowsiness metrics explained

### EAR — Eye Aspect Ratio

Derived from 6 landmark coordinates around each eye:

```
EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
```

- Open eye: EAR ≈ 0.30
- Closed eye: EAR ≈ 0.00
- Drowsiness threshold: EAR < 0.20

EAR is invariant to image scale and in-plane face rotation, making it robust in a vehicle environment.

### PERCLOS

Percentage of Eye Closure over time. Fraction of frames within a rolling 60-second window in which the eyes are more than 80% closed. The gold-standard fatigue metric used in NHTSA driver fatigue research.

- PERCLOS > 0.80 = drowsy classification

### Head Pitch (nodding off)

The `head-pose-estimation-adas-0001` model outputs pitch, yaw, and roll angles in degrees. Pitch > 25° downward (chin dropping toward chest) indicates the head is falling forward — a classic sign of micro-sleep onset.

---

## 2D Dashboard layout

```
┌─────────────────────────────────────────────────────────┐
│  oak-d-driver-monitor                     FPS: 28.4     │
├────────────────────────┬────────────────────────────────┤
│                        │  METRICS                        │
│   Live RGB feed        │  EAR Left :  0.31  ●●●○○       │
│                        │  EAR Right:  0.29  ●●●○○       │
│   ┌──────────────┐     │  PERCLOS :   12%   ✓            │
│   │  [face box]  │     │  Head Pitch: -8°   ✓            │
│   │  👁 ●  👁 ●  │     │  Yaw     :  +3°   ✓            │
│   │              │     │                                  │
│   └──────────────┘     │  STATUS :  ● ALERT              │
│                        │                                  │
│   Head pose arrow →    │  EAR over time (10s):           │
│                        │  ▁▂▃▄▅▄▃▂▁▁▁▁▂▃▄▅▄             │
└────────────────────────┴────────────────────────────────┘
```

---

## Folder structure

```
oak-d-driver-monitor/
│
├── pipeline/
│   ├── oak_pipeline.py             DepthAI node graph — runs on OAK-D VPU
│   ├── host_processor.py           MediaPipe FaceMesh and Pose on host CPU
│   └── models/
│       ├── face-detection-adas-0001.blob         OpenVINO compiled for Myriad X
│       └── head-pose-estimation-adas-0001.blob
│
├── detection/
│   ├── ear_calculator.py           EAR from 6 landmark coordinates per eye
│   ├── perclos_tracker.py          Rolling 60-second window closure metric
│   ├── head_pose_analyzer.py       Pitch/yaw/roll → drowsiness classification
│   └── yawn_detector.py            Mouth aspect ratio for yawn detection
│
├── state_machine/
│   └── drowsiness_fsm.py           Finite state machine: ALERT → WARNING → DROWSY
│
├── visualization/
│   └── dashboard.py                2D OpenCV window with metrics, EAR graph, alerts
│
├── recorder/
│   └── session_recorder.py         Records OAK-D sessions for offline replay and CI
│
├── tests/
│   └── mock_frames/                Pre-recorded sessions — CI runs without hardware
│       ├── alert_session.bin       Session where driver is alert
│       ├── drowsy_session.bin      Session with sustained eye closure
│       └── nodding_session.bin     Session with head pitch events
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourname/oak-d-driver-monitor.git
cd oak-d-driver-monitor

# Install Python dependencies
pip install depthai opencv-python mediapipe numpy

# Download OpenVINO model blobs (compiled for Myriad X)
python pipeline/download_models.py
```

---

## Running with OAK-D hardware

Connect the OAK-D camera via USB 3.0, then:

```bash
python pipeline/oak_pipeline.py
```

The 2D dashboard opens automatically. Press `q` to quit, `r` to start recording a session.

---

## Running without hardware (offline replay / CI mode)

The system supports a `SIL_MODE` that replays pre-recorded sessions. GitHub Actions CI uses this mode so the pipeline can be tested without any camera attached.

```bash
# Replay a pre-recorded drowsy session
python pipeline/oak_pipeline.py --sil-mode --session tests/mock_frames/drowsy_session.bin

# Record a new session for replay (requires hardware)
python recorder/session_recorder.py --output tests/mock_frames/my_session.bin --duration 60
```

---

## Alerts and state machine

```
State: ALERT   (normal — driver is awake)
    │
    │  EAR < 0.20 for 2s  OR  PERCLOS > 0.40
    ▼
State: WARNING  (amber overlay on dashboard, audio beep)
    │
    │  EAR < 0.20 for 4s  OR  PERCLOS > 0.80  OR  pitch > 25°
    ▼
State: DROWSY   (red overlay, continuous alarm)
    │
    │  Driver movement detected / EAR returns above threshold
    ▼
State: ALERT    (reset)
```
---

## Hardware requirements

| Component | Specification |
|---|---|
| Camera | Luxonis OAK-D (any revision) |
| Connection | USB 3.0 (USB 2.0 works at reduced fps) |
| Host OS | Ubuntu 20.04+ / macOS 12+ / Windows 10+ |
| Host Python | 3.9 or higher |
| Host RAM | 4 GB minimum |
| GPU | Not required — VPU handles inference on-device |
