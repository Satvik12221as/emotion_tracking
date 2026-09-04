"""Camera Manager module supporting Picamera2 (Raspberry Pi 5) and OpenCV capture."""

import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Check Picamera2 availability for Pi 5
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class CameraManager:
    """Threaded Camera Manager providing continuous frame acquisition."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_h: bool = False,
        flip_v: bool = False,
        use_picamera2: bool = False,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.use_picamera2 = use_picamera2 and PICAMERA2_AVAILABLE

        self._cap = None
        self._picam2 = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._current_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # FPS metrics
        self._fps_counter = 0
        self._fps_start_time = time.monotonic()
        self._measured_fps = 0.0

    def start(self) -> bool:
        """Initializes hardware camera and starts background capture thread."""
        if self._running:
            return True

        if self.use_picamera2:
            try:
                logger.info("Initializing Picamera2 (Raspberry Pi Camera Module)...")
                self._picam2 = Picamera2()
                video_config = self._picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self._picam2.configure(video_config)
                self._picam2.start()
                logger.info("Picamera2 initialized successfully.")
            except Exception as e:
                logger.warning(f"Picamera2 failed to initialize: {e}. Falling back to OpenCV.")
                self.use_picamera2 = False

        if not self.use_picamera2:
            logger.info(f"Initializing OpenCV VideoCapture with source: {self.source}...")
            # On Linux try V4L2 backend if int source
            if isinstance(self.source, int) and hasattr(cv2, "CAP_V4L2"):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                if not self._cap.isOpened():
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.source)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if not self._cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._fps_start_time = time.monotonic()
        self._fps_counter = 0

        self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        logger.info("Camera capture thread started.")
        return True

    def _capture_loop(self) -> None:
        """Background thread grabbing frames continuously."""
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.033

        while self._running:
            start_t = time.monotonic()
            frame = None

            if self.use_picamera2 and self._picam2 is not None:
                try:
                    # Picamera2 returns RGB numpy array
                    rgb_frame = self._picam2.capture_array()
                    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"Error capturing from Picamera2: {e}")
                    time.sleep(0.01)
                    continue
            elif self._cap is not None:
                ok, raw_frame = self._cap.read()
                if not ok:
                    # Loop video files if applicable
                    if isinstance(self.source, str) and not self.source.isdigit():
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, raw_frame = self._cap.read()
                    if not ok:
                        time.sleep(0.01)
                        continue
                frame = raw_frame

            if frame is not None:
                if self.flip_h and self.flip_v:
                    frame = cv2.flip(frame, -1)
                elif self.flip_h:
                    frame = cv2.flip(frame, 1)
                elif self.flip_v:
                    frame = cv2.flip(frame, 0)

                with self._lock:
                    self._current_frame = frame

                # Measure FPS
                self._fps_counter += 1
                now = time.monotonic()
                elapsed = now - self._fps_start_time
                if elapsed >= 1.0:
                    self._measured_fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_start_time = now

            # Pacing
            elapsed_frame = time.monotonic() - start_t
            sleep_time = frame_interval - elapsed_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the latest captured frame and status."""
        with self._lock:
            if self._current_frame is None:
                return False, None
            return True, self._current_frame.copy()

    def get_fps(self) -> float:
        """Returns measured frames per second."""
        return self._measured_fps

    def stop(self) -> None:
        """Stops thread and releases hardware resources."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self.use_picamera2 and self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception as e:
                logger.error(f"Error stopping Picamera2: {e}")
            self._picam2 = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("Camera Manager stopped cleanly.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
