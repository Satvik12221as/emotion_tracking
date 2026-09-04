# Real-Time Context-Aware Assistive Vision System (Raspberry Pi 5 / Edge AI)

An edge-native, real-time, camera-based multimodal user perception and decision assistance system running locally on **Raspberry Pi 5** and cross-platform desktop operating systems.

The system combines multiple visual signals to estimate user state and provide intelligent context-aware assistance:
1. **Facial Expression / Emotion Estimation** (7-class ONNX model)
2. **Face Detection & ROI Tracking** (Single-user primary face selection policy)
3. **Head Orientation & Screen Attention Estimation** (3D Yaw, Pitch, Roll via solvePnP)
4. **Seating Alignment** (Calibrated Left / Center / Right horizontal zones)
5. **Camera Distance Estimation** (Face geometry ratio & pinhole depth calculation)
6. **Pose & Posture Analysis** (Upright, Leaning Left/Right, Slouching, Head Tilted)
7. **Multi-User Ambiguity Detection** (Suppresses false alerts when multiple users are present)
8. **Temporal Stabilization & Filtering** (Sliding-window majority voting & persistence duration)
9. **Priority-Ranked Decision Engine** (Rule evaluator with cooldown protection)
10. **Assistance Engine** (HUD Overlay & Local TTS Voice Notifications)

```text
CAMERA → PERCEPTION → USER STATE → CONTEXT FUSION → DECISION ENGINE → ASSISTANCE ENGINE
```

---

## 1. System Architecture

```text
                                 +------------------+
                                 |  Camera Manager  |
                                 | (Picamera2/OpenCV|
                                 +--------+---------+
                                          |
                                          v
                                 Frame Buffer (640x480)
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
               FACE PIPELINE                             BODY PIPELINE
                     |                                         |
            +--------+--------+                       +--------+--------+
            |                 |                       |                 |
            v                 v                       v                 v
      Face Detector     Landmarks/Mesh          Person Detector    Pose Estimator
       (Face ROI)     (Head Orient/Attention)   (Seating/Dist)    (Posture Feats)
            |                 |                       |                 |
            v                 v                       v                 v
      Emotion Model    Head Pose Classifier    Seating Zone     Posture Analyzer
       (ONNX FER)      (Yaw/Pitch/Attention)  (Left/Center/R)  (Upright/Leaning)
            |                 |                       |                 |
            +--------+--------+-----------------------+-----------------+
                     |
                     v
             Perception State (Raw frame features)
                     |
                     v
             Temporal Filter (State stability & hysteresis window)
                     |
                     v
             User State Manager (Normalized UserState object)
                     |
                     v
             Context Fusion & Decision Engine (Priority-ranked rules)
                     |
                     v
             Assistance Engine (Cooldown-managed HUD / Audio Output)
```

---

## 2. Directory Structure

```text
emotion tracking aaruush/
├── config/
│   └── config.yaml             # System configuration & calibrated thresholds
├── docs/
│   ├── ARCHITECTURE.md          # Architectural specification & dataflows
│   ├── EXISTING_PROJECT_AUDIT.md # Audit of legacy EMO-Bot reference repo
│   └── PERFORMANCE.md           # Empirical performance benchmark results
├── models/
│   └── emotion/                 # ONNX facial expression model storage
├── src/
│   ├── main.py                  # Main orchestrator entry point
│   ├── camera/
│   │   └── camera_manager.py    # Threaded capture supporting Picamera2 & OpenCV
│   ├── face/
│   │   ├── detector.py          # Face detection & ROI cropper
│   │   ├── landmarks.py         # Head pose (Yaw/Pitch/Roll) & attention tracker
│   │   └── emotion.py           # Decoupled ONNX emotion estimator
│   ├── body/
│   │   ├── person_detector.py   # Person detection & primary user selection
│   │   ├── pose.py              # MediaPipe Pose Lite estimator
│   │   └── posture.py           # Human-readable posture analyzer
│   ├── seating/
│   │   ├── position.py          # Horizontal seating zone classifier
│   │   ├── distance.py          # Camera-based distance estimator
│   │   └── calibration.py       # Calibration utility
│   ├── state/
│   │   ├── user_state.py        # Central UserState dataclass contract
│   │   └── temporal_filter.py   # Majority voting & persistence filter
│   ├── decision/
│   │   ├── rules.py             # AssistanceAction definitions & explainability
│   │   └── engine.py            # Priority decision engine & cooldown manager
│   ├── assistance/
│   │   ├── display.py           # Live HUD overlay renderer
│   │   ├── voice.py             # Local text-to-speech module
│   │   └── actions.py           # Assistance Engine orchestrator
│   └── utils/
│       ├── config_loader.py     # YAML configuration loader
│       └── logger.py            # Structured logging setup
├── scripts/
│   ├── camera_test.py           # Camera acquisition benchmark script
│   ├── face_test.py             # Face detector benchmark script
│   └── benchmark.py            # End-to-end performance benchmarking tool
├── tests/                       # Complete unit & integration test suite (20 tests)
└── requirements.txt             # Clean Python dependencies
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.11+
* Raspberry Pi 5 running 64-bit Raspberry Pi OS (Bookworm) or Linux/Windows desktop
* Camera: Raspberry Pi Camera Module 3 (CSI) or standard USB webcam

### Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run system test suite:
   ```bash
   python -m pytest tests/
   ```

3. Launch live Assistive Vision System:
   ```bash
   python src/main.py
   ```

4. Run in headless mode (for background deployment or server mode):
   ```bash
   python src/main.py --headless
   ```

---

## 4. Calibration & Configuration

All thresholds are fully configurable in `config/config.yaml`.

```yaml
seating:
  left_boundary: 0.35   # Normalized frame X boundary
  right_boundary: 0.65

distance:
  too_close_face_ratio: 0.35  # Bounding box width / frame width
  too_far_face_ratio: 0.10

posture:
  shoulder_tilt_threshold: 0.06
  lean_x_threshold: 0.10
  slouch_head_drop_threshold: -0.05
  persistence_seconds: 2.0

assistance:
  cooldown_seconds: 5.0
```

To recalibrate seating or distance boundaries, run python interactive calibrator:
```python
from src.seating.calibration import SystemCalibrator

calibrator = SystemCalibrator()
calibrator.update_seating_zones(0.30, 0.70)
```

---

## 5. Performance Engineering Results

Benchmark measurements on 640x480 video feed:
* **End-to-End Throughput**: 10.0+ FPS
* **Average Processing Latency**: ~99.8 ms
* **Test Suite**: 20/20 unit and integration tests passing cleanly.

Detailed breakdown available in [docs/PERFORMANCE.md](file:///c:/Users/Satvik%20singh/OneDrive/Desktop/emotion%20tracking%20aaruush/docs/PERFORMANCE.md).

---

## 6. Privacy & Edge Principles

* **100% Local Inference**: Zero cloud API dependencies (no external HTTP calls, no cloud STT/TTS).
* **Privacy First**: Video frames are processed in-memory and are never written to disk or transmitted over the network.
* **Explainable AI**: The decision engine outputs human-readable reasoning for every assistance action triggered.
