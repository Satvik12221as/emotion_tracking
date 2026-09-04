"""Camera testing script to benchmark capture FPS and display/verify feed."""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.camera.camera_manager import CameraManager
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("camera_test")


def run_camera_test(headless: bool = False, duration_sec: float = 5.0) -> bool:
    config = load_config()
    cam_cfg = config.get("camera", {})

    source = cam_cfg.get("source", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    fps = cam_cfg.get("fps", 30)
    flip_h = cam_cfg.get("flip_h", False)
    flip_v = cam_cfg.get("flip_v", False)
    use_picam2 = cam_cfg.get("use_picamera2", False)

    logger.info(f"Starting camera test (Source: {source}, Configured Res: {width}x{height} @ {fps} FPS)...")

    cam = CameraManager(
        source=source,
        width=width,
        height=height,
        fps=fps,
        flip_h=flip_h,
        flip_v=flip_v,
        use_picamera2=use_picam2,
    )

    if not cam.start():
        logger.error("Failed to start CameraManager.")
        return False

    start_time = time.monotonic()
    frames_received = 0

    try:
        while (time.monotonic() - start_time) < duration_sec:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frames_received += 1
            measured_fps = cam.get_fps()
            h, w = frame.shape[:2]

            if not headless:
                cv2.putText(
                    frame,
                    f"Res: {w}x{h} | Measured FPS: {measured_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Camera Test Feed (Press 'q' to exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.02)

    except KeyboardInterrupt:
        logger.info("Camera test interrupted by user.")
    finally:
        cam.stop()
        if not headless:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    avg_fps = frames_received / elapsed if elapsed > 0 else 0
    logger.info(f"Camera Test Complete: Captured {frames_received} frames in {elapsed:.2f}s (Avg FPS: {avg_fps:.1f})")
    return frames_received > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Camera Acquisition Pipeline")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds")
    args = parser.parse_args()

    success = run_camera_test(headless=args.headless, duration_sec=args.duration)
    sys.exit(0 if success else 1)
