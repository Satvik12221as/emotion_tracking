"""Main Orchestrator Entry Point for Real-Time Assistive Vision System."""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.assistance.actions import AssistanceEngine
from src.body.person_detector import PersonDetector
from src.body.pose import PoseEstimator
from src.body.posture import PostureAnalyzer
from src.camera.camera_manager import CameraManager
from src.decision.engine import DecisionEngine
from src.face.detector import FaceDetector
from src.face.emotion import EmotionEstimator
from src.face.landmarks import FaceLandmarkAnalyzer
from src.seating.distance import DistanceEstimator
from src.seating.position import SeatingPositionAnalyzer
from src.state.temporal_filter import TemporalFilter
from src.state.user_state import UserState
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("main")


class AssistiveVisionPipeline:
    """Integrated Multimodal Perception and Decision Pipeline."""

    def __init__(self, config_path: str = "config/config.yaml", headless: bool = False):
        self.config = load_config(config_path)
        self.headless = headless

        cam_cfg = self.config.get("camera", {})
        face_cfg = self.config.get("face", {})
        emotion_cfg = self.config.get("emotion", {})
        pose_cfg = self.config.get("pose", {})
        seating_cfg = self.config.get("seating", {})
        distance_cfg = self.config.get("distance", {})
        posture_cfg = self.config.get("posture", {})
        attention_cfg = self.config.get("attention", {})
        temporal_cfg = self.config.get("temporal_filter", {})
        decision_cfg = self.config.get("decision", {})
        assistance_cfg = self.config.get("assistance", {})

        # 1. Camera Manager
        self.camera = CameraManager(
            source=cam_cfg.get("source", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            flip_h=cam_cfg.get("flip_h", False),
            flip_v=cam_cfg.get("flip_v", False),
            use_picamera2=cam_cfg.get("use_picamera2", False),
        )

        # 2. Perception Modules
        self.face_detector = FaceDetector(
            min_detection_confidence=face_cfg.get("min_detection_confidence", 0.5),
            model_selection=face_cfg.get("model_selection", 0),
            roi_padding=face_cfg.get("roi_padding", 0.15),
        )
        self.landmark_analyzer = FaceLandmarkAnalyzer(
            yaw_threshold_deg=attention_cfg.get("yaw_threshold_deg", 25.0),
            pitch_threshold_deg=attention_cfg.get("pitch_threshold_deg", 20.0),
        )
        self.emotion_estimator = EmotionEstimator(
            model_path=emotion_cfg.get("model_path", "models/emotion/mini_xception_fer.onnx"),
            confidence_threshold=emotion_cfg.get("confidence_threshold", 0.50),
        )
        self.person_detector = PersonDetector()
        self.pose_estimator = PoseEstimator(
            model_complexity=pose_cfg.get("model_complexity", 0),
        )
        self.posture_analyzer = PostureAnalyzer(
            shoulder_tilt_threshold=posture_cfg.get("shoulder_tilt_threshold", 0.06),
            lean_x_threshold=posture_cfg.get("lean_x_threshold", 0.10),
            slouch_head_drop_threshold=posture_cfg.get("slouch_head_drop_threshold", -0.05),
        )
        self.seating_analyzer = SeatingPositionAnalyzer(
            left_boundary=seating_cfg.get("left_boundary", 0.35),
            right_boundary=seating_cfg.get("right_boundary", 0.65),
        )
        self.distance_estimator = DistanceEstimator(
            too_close_ratio=distance_cfg.get("too_close_face_ratio", 0.35),
            too_far_ratio=distance_cfg.get("too_far_face_ratio", 0.10),
        )

        # 3. State & Temporal Smoothing
        self.temporal_filter = TemporalFilter(
            window_size_sec=temporal_cfg.get("smoothing_window_sec", 1.5),
        )

        # 4. Decision Engine
        self.decision_engine = DecisionEngine(
            cooldown_seconds=decision_cfg.get("cooldown_seconds", 5.0),
            posture_persistence_sec=posture_cfg.get("persistence_seconds", 2.0),
        )

        # 5. Assistance Engine
        self.assistance_engine = AssistanceEngine(
            enable_display=not self.headless and assistance_cfg.get("enable_display", True),
            enable_audio=assistance_cfg.get("enable_audio", False),
        )

        self._running = False
        self._pose_every_n = pose_cfg.get("process_every_n_frames", 2)
        self._frame_idx = 0
        self._last_pose_res = None

    def start(self) -> None:
        """Starts processing pipeline loop."""
        logger.info("Starting Assistive Vision Pipeline...")
        if not self.camera.start():
            logger.error("Camera failed to start. Aborting.")
            return

        self._running = True

        def _signal_handler(sig, frame):
            logger.info("Shutdown signal received.")
            self.stop()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        logger.info("Pipeline running. Press 'q' or Ctrl+C to exit.")

        try:
            while self._running:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                self.process_frame(frame)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")
        finally:
            self.stop()

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Executes full multimodal perception and decision pipeline on a single frame."""
        self._frame_idx += 1
        h, w = frame.shape[:2]

        # 1. Face Perception
        face_res = self.face_detector.detect(frame)
        face_center = (face_res.center_x, face_res.center_y) if face_res.detected else None

        # 2. Landmark & Attention Analysis
        head_res = self.landmark_analyzer.analyze(frame) if face_res.detected else self.landmark_analyzer.analyze(None)

        # 3. Emotion Estimation
        emotion_res = self.emotion_estimator.predict(face_res.roi) if face_res.detected else self.emotion_estimator.predict(None)

        # 4. Person & Multi-User Detection
        person_res = self.person_detector.detect(frame, face_bbox_center=face_center)

        # 5. Pose & Posture (Interleaved execution for efficiency)
        if self._frame_idx % self._pose_every_n == 0:
            self._last_pose_res = self.pose_estimator.estimate(frame)
        pose_res = self._last_pose_res if self._last_pose_res is not None else self.pose_estimator.estimate(None)

        posture_res = self.posture_analyzer.analyze(pose_res)

        # 6. Seating Position & Distance
        seating_res = self.seating_analyzer.analyze(face_res.center_x if face_res.detected else person_res.center_x)
        distance_res = self.distance_estimator.estimate(face_res.normalized_bbox if face_res.detected else None, frame_width=w)

        # 7. Build Raw Perception UserState
        raw_state = UserState(
            face_detected=face_res.detected,
            face_confidence=face_res.confidence,
            face_center_x=face_res.center_x,
            face_center_y=face_res.center_y,
            emotion=emotion_res.label,
            emotion_confidence=emotion_res.confidence,
            emotion_probabilities=emotion_res.probabilities,
            seating_position=seating_res.zone,
            distance=distance_res.classification,
            distance_meters=distance_res.estimated_distance_m,
            posture=posture_res.state,
            yaw_deg=head_res.yaw_deg,
            pitch_deg=head_res.pitch_deg,
            roll_deg=head_res.roll_deg,
            attention=head_res.attention_state,
            is_facing_screen=head_res.is_facing_screen,
            multiple_users=person_res.multiple_users,
        )

        # 8. Temporal Filtering & Persistence
        stabilized_state, durations = self.temporal_filter.push(raw_state)

        # 9. Decision Engine Evaluation
        action = self.decision_engine.evaluate(stabilized_state, durations)

        # 10. Assistance Engine Output & HUD Rendering
        output_frame = self.assistance_engine.process(
            frame=frame,
            user_state=stabilized_state,
            action=action,
            fps=self.camera.get_fps(),
            durations=durations,
        )

        if not self.headless and output_frame is not None:
            cv2.imshow("Assistive Vision System (Press 'q' to exit)", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False

        return output_frame

    def stop(self) -> None:
        """Clean shutdown of all modules."""
        self._running = False
        self.camera.stop()
        self.face_detector.close()
        self.landmark_analyzer.close()
        self.pose_estimator.close()
        if not self.headless:
            cv2.destroyAllWindows()
        logger.info("Pipeline shut down cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Run Assistive Vision System")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()

    pipeline = AssistiveVisionPipeline(config_path=args.config, headless=args.headless)
    pipeline.start()


if __name__ == "__main__":
    main()
