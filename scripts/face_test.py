"""Face Detection pipeline test script."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.camera.camera_manager import CameraManager
from src.face.detector import FaceDetector
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("face_test")


def run_face_test(headless: bool = False, duration_sec: float = 5.0) -> bool:
    config = load_config()
    cam_cfg = config.get("camera", {})
    face_cfg = config.get("face", {})

    cam = CameraManager(
        source=cam_cfg.get("source", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
        flip_h=cam_cfg.get("flip_h", False),
        flip_v=cam_cfg.get("flip_v", False),
        use_picamera2=cam_cfg.get("use_picamera2", False),
    )

    detector = FaceDetector(
        min_detection_confidence=face_cfg.get("min_detection_confidence", 0.5),
        model_selection=face_cfg.get("model_selection", 0),
        roi_padding=face_cfg.get("roi_padding", 0.15),
    )

    if not cam.start():
        logger.error("Failed to start CameraManager.")
        return False

    start_time = time.monotonic()
    total_frames = 0
    detected_count = 0
    total_latency_ms = 0.0

    try:
        while (time.monotonic() - start_time) < duration_sec:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            total_frames += 1
            res = detector.detect(frame)

            if res.detected:
                detected_count += 1
                total_latency_ms += res.inference_time_ms

            if not headless:
                vis_frame = detector.draw_detection(frame, res)
                fps_text = f"FPS: {cam.get_fps():.1f} | Face Latency: {res.inference_time_ms:.1f}ms"
                cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Face Detection Test", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Face test interrupted.")
    finally:
        detector.close()
        cam.stop()
        if not headless:
            cv2.destroyAllWindows()

    avg_latency = (total_latency_ms / detected_count) if detected_count > 0 else 0.0
    logger.info(
        f"Face Test Results: Processed {total_frames} frames, Detected Face in {detected_count} frames. Avg Detection Latency: {avg_latency:.2f}ms"
    )
    return total_frames > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Face Detection Pipeline")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    args = parser.parse_args()

    success = run_face_test(headless=args.headless, duration_sec=args.duration)
    sys.exit(0 if success else 1)
