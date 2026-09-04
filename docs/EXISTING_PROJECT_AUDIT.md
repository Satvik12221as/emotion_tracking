# Phase 0 Audit: Existing EMO-Bot Repository Analysis

## 1. Executive Summary
This document provides a thorough audit of the reference repository (`EMO-Bot` by Ekansh Patil). The purpose is to evaluate existing vision, posture, control, and intelligence modules, determine what can be reused for the **Real-Time Context-Aware Assistive Vision System**, identify missing components (notably the emotion recognition model), and outline a clear migration strategy targeting the **Raspberry Pi 5**.

---

## 2. Component Audit Table

| Existing Component | File / Path | Technology / Stack | Purpose in EMO-Bot | Reusable? | Required Changes | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Main Orchestrator** | `main.py` | Python `asyncio` + `threading` | Launches serial, audio, API, vision, and behavior tree tasks | **REPLACE** | Replace with structured perception-state-decision pipeline | Robot-centric main loop tied to MQTT and Arduino hardware. |
| **Camera Acquisition** | `vision_posture_module.py` | OpenCV `cv2.VideoCapture` + V4L2 | Captures `/dev/video0` frames at 640x480 @ 30 FPS | **MODIFY** | Encapsulate in `CameraManager`; add `Picamera2` fallback for Pi 5 | V4L2 raw call fails on native Pi 5 libcamera standard without wrapper. |
| **Face Detection** | `vision_posture_module.py` | MediaPipe FaceDetection (`model_selection=0`) | Detects faces and selects primary face by area score | **KEEP** | Refactor into `FaceDetector` module in `src/face/` | Fast (~5ms), effective single-user face selection. |
| **Emotion Model** | N/A | **NONE** | N/A | **MISSING** | Implement lightweight ONNX / TFLite FER model in `src/face/emotion.py` | EMO-Bot README claimed vision features, but source code contains NO emotion model. |
| **Pose Estimation** | `vision_posture_module.py` | MediaPipe Pose (`model_complexity=0`) | Runs pose landmark extraction every 2 frames | **KEEP** | Refactor into `PoseDetector` module in `src/body/` | BlazePose Lite is lightweight and efficient on Pi 5 ARM64. |
| **Posture Analysis** | `vision_posture_module.py` | Manual geometric heuristic (`is_poor_posture`) | Checks shoulder tilt, head drop, neck offset | **MODIFY** | Expand into structured posture classification with configurable thresholds | Existing function only returns binary boolean based on hardcoded numbers. |
| **Seating / Distance** | N/A | N/A | N/A | **MISSING** | Add `src/seating/position.py` & `distance.py` | Not present in EMO-Bot. |
| **Behavior Tree** | `behavior_tree_module.py` | `py-trees` + MQTT | 10 Hz tree for E-Stop, conversation, tracking, idle | **REMOVE** | Replace with rule-based `DecisionEngine` | Behavior trees with robot motor queues add unnecessary complexity for assistive HUD/voice. |
| **Serial / Actuators** | `serial_module.py` | PySerial + Arduino Nano | Sends `J,joint,angle` commands to 16 servos | **REMOVE** | Exclude from core MVP | Servos and physical motion are out of scope for initial vision assistance system. |
| **Audio / Voice** | `audio_trigger_task.py`, `api_routing_task.py` | Porcupine + Whisper + GPT-4o + ElevenLabs | Wake word listening, cloud STT, LLM, cloud TTS | **REMOVE** | Exclude cloud dependencies; use local pyttsx3/espeak or screen HUD | Cloud APIs violate edge-first, offline, low-latency requirement. |

---

## 3. Detailed Technical Analysis

### A. Repository Map
* `main.py`: Asynchronous script running 5 concurrent worker tasks via `asyncio.to_thread` and task queues.
* `vision_posture_module.py`: Standalone script and library for camera frames, MediaPipe FaceDetection, MediaPipe Pose, primary face tracking error calculation, and posture alert publishing over MQTT.
* `behavior_tree_module.py`: `py-trees` implementation running a 10 Hz priority loop receiving MQTT events and driving a motor queue (`CommandBus`).
* `serial_module.py`: Non-blocking serial port writer with ACK verification for Arduino Nano commands over `/dev/ttyUSB0`.
* `audio_trigger_task.py`: Porcupine wake-word engine and 5-second PCM audio recorder.
* `api_routing_task.py`: Cascade of HTTP calls to OpenAI Whisper, OpenAI GPT-4o, and ElevenLabs TTS with a 3.0s timeout.

### B. Emotion Pipeline Status: **MISSING**
* **Inspection Result**: Deep inspection of `EMO-Bot` reveals **no emotion detection code, no model files, no CNN weights, and no emotion inference logic**.
* **Action Required**: We must introduce a lightweight, real-time Facial Expression Recognition (FER) model (e.g., ONNX-quantized MiniXception / MobileNetV2 FER model) in `src/face/emotion.py` accepting face ROIs normalized to 64x64 grayscale or RGB.

### C. Face Pipeline Analysis
* MediaPipe Face Detection (`model_selection=0`) performs well for desk distances (0.5m – 2m).
* `select_primary_face()` uses `detection.score * bounding_box_area` to prioritize the largest, highest-confidence face in view.
* **Gap**: Lacks facial mesh / landmark detail needed to calculate head pose (yaw, pitch, roll) for attention estimation. We will supplement this with MediaPipe Face Mesh or landmark geometry in `src/face/landmarks.py`.

### D. Posture Pipeline Analysis
* Uses MediaPipe Pose Lite (`model_complexity=0`, `smooth_landmarks=True`).
* Evaluates 3 key metrics:
  1. `shoulder_tilt = |left_shoulder.y - right_shoulder.y| > 0.07`
  2. `neck_forward_offset = |nose.x - shoulder_mid_x| > 0.12`
  3. `head_drop = nose.y - shoulder_mid_y > -0.06`
* Streak filter: Triggers alert only after 3 consecutive positive frames (`POSTURE_CONFIRM_FRAMES = 3`).
* **Gap**: Thresholds are hardcoded and non-calibrated; fails to distinguish leaning left vs leaning right or slouching vs standing.

### E. Raspberry Pi 5 ARM64 Compatibility Concerns
1. **Camera Backend**: `cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)` fails on Raspberry Pi OS Bookworm with CSI Camera Module 3 unless `Picamera2` or `libcamerify` is used.
2. **Dependencies**: `py-trees`, `pvporcupine`, `pyaudio`, `paho-mqtt` add heavy native binary and system daemon dependencies (`mosquitto`).
3. **Threading Model**: OpenCV GUI calls + MediaPipe processing in sub-threads must pass data cleanly to the asyncio loop without thread contention.

---

## 4. Migration & Reuse Plan

### What We KEEP
1. Primary face selection heuristic logic.
2. MediaPipe Pose Lite configuration.
3. Posture feature measurement math (shoulder tilt, head drop offset).

### What We MODIFY
1. Wrap camera access into `CameraManager` supporting both `Picamera2` (CSI on Pi 5) and OpenCV (`V4L2` / USB / Video file fallback for dev).
2. Move hardcoded vision thresholds to `config/config.yaml`.
3. Separate pose detection from posture classification.

### What We REMOVE
1. Mosquitto MQTT broker and `paho-mqtt` topics.
2. `py-trees` behavior tree framework.
3. Serial communications and Arduino hardware task.
4. Porcupine, Whisper, GPT-4o, and ElevenLabs cloud audio pipeline.

### What We ADD (New Modules)
1. `src/face/emotion.py`: ONNX-based lightweight facial expression estimator.
2. `src/face/landmarks.py`: Head pose (Yaw, Pitch, Roll) and attention tracker (`FACING_SCREEN` vs `LOOKING_AWAY`).
3. `src/seating/position.py` & `distance.py`: Calibrated left/center/right seating zone and face-box distance classifier.
4. `src/state/user_state.py` & `temporal_filter.py`: Central `UserState` dataclass with temporal majority voting filter.
5. `src/decision/engine.py` & `rules.py`: Priority-ranked assistance decision engine.
6. `src/assistance/display.py` & `voice.py`: Modular assistance output interface (HUD / Local Audio).
