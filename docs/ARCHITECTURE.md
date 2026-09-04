# System Architecture: Real-Time Context-Aware Assistive Vision System

## 1. High-Level Concept
The system transforms raw camera input into actionable assistive feedback through a modular 6-stage edge computer vision pipeline running locally on a Raspberry Pi 5.

```text
CAMERA → PERCEPTION → USER STATE → CONTEXT FUSION → DECISION → ASSISTANCE
```

---

## 2. End-to-End Dataflow Diagram

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

## 3. Core Modules & Responsibilities

### 1. Camera Module (`src/camera/`)
* **`camera_manager.py`**: Provides unified threaded capture interface. Supports `Picamera2` (native Pi 5 libcamera) with fallback to `cv2.VideoCapture` (V4L2/USB/Video file).
* Resolution: 640x480 @ 30 FPS target.

### 2. Face Pipeline (`src/face/`)
* **`detector.py`**: Lightweight MediaPipe / OpenCV face detector. Extracts single primary face ROI using size and confidence weighting.
* **`landmarks.py`**: Facial landmarks for head orientation estimation (Yaw, Pitch, Roll) and screen attention calculation (`FACING_SCREEN` vs `LOOKING_AWAY`).
* **`emotion.py`**: Decoupled ONNX model wrapper for facial expression estimation (`neutral`, `happy`, `sad`, `surprise`, `confused`/`frustrated`, `fear`, `angry`).

### 3. Body Pipeline (`src/body/`)
* **`person_detector.py`**: Identifies person presence, bounding boxes, and detects multiple user conditions (`MULTIPLE_USERS`).
* **`pose.py`**: Lightweight MediaPipe Pose Lite (`model_complexity=0`).
* **`posture.py`**: Evaluates posture state (`UPRIGHT`, `LEANING_LEFT`, `LEANING_RIGHT`, `SLOUCHING`, `HEAD_TILTED`).

### 4. Seating & Distance Module (`src/seating/`)
* **`position.py`**: Evaluates horizontal seating alignment relative to calibrated horizontal zones (`LEFT`, `CENTER`, `RIGHT`).
* **`distance.py`**: Calibrated face/person bounding box geometry for distance classification (`TOO_CLOSE`, `NORMAL`, `TOO_FAR`).
* **`calibration.py`**: Interactive calibration tool to record user-specific geometric thresholds into `config/config.yaml`.

### 5. User State & Temporal Filtering (`src/state/`)
* **`user_state.py`**: Strongly typed data structure representing complete user perception snapshot.
* **`temporal_filter.py`**: Smoothing filter using sliding-window majority voting, confidence thresholding, and state persistence (e.g., condition must persist for > 2.0 seconds) to prevent false triggers.

### 6. Decision & Assistance Engine (`src/decision/`, `src/assistance/`)
* **`rules.py` & `engine.py`**: Priority-ranked rule evaluator. Resolves conflicting signals (e.g., Seating > Posture > Confusion Assistance) into a single deterministic action.
* **`display.py` & `voice.py`**: Modular HUD renderer with on-screen status overlay and local TTS audio alert dispatcher.
