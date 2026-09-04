"""Lightweight Pose Estimation Module wrapping MediaPipe Pose Lite."""

from dataclasses import dataclass, field
import time
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseResult:
    """Raw landmark result from Pose Estimator."""

    detected: bool = False
    pose_landmarks: Optional[Any] = field(default=None, repr=False)
    inference_time_ms: float = 0.0


class PoseEstimator:
    """Wrapper around MediaPipe Pose Lite (model_complexity=0) for Pi 5 optimization."""

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_complexity = model_complexity

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame: np.ndarray) -> PoseResult:
        """Processes RGB frame and extracts 3D/2D pose landmarks."""
        if frame is None or frame.size == 0:
            return PoseResult(detected=False)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.monotonic()
        results = self._pose.process(rgb)
        dt_ms = (time.monotonic() - t0) * 1000.0

        if not results or not results.pose_landmarks:
            return PoseResult(detected=False, inference_time_ms=dt_ms)

        return PoseResult(
            detected=True,
            pose_landmarks=results.pose_landmarks,
            inference_time_ms=dt_ms,
        )

    def close(self) -> None:
        """Releases MediaPipe pose resources."""
        if hasattr(self, "_pose") and self._pose:
            self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
