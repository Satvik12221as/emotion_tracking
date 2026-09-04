"""Person / Body Detection & Primary User Selection Policy Module."""

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PersonDetectionResult:
    """Structured result for person detection & multi-user policy."""

    person_detected: bool = False
    multiple_users: bool = False
    person_count: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (xmin, ymin, w, h) in pixels
    normalized_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = 0.0
    center_x: float = 0.5
    center_y: float = 0.5
    inference_time_ms: float = 0.0


class PersonDetector:
    """Detects person presence and applies primary user selection policy."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_confidence = min_detection_confidence
        # Lightweight HOG People Detector fallback + MediaPipe Pose region detection
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray, face_bbox_center: Optional[Tuple[float, float]] = None) -> PersonDetectionResult:
        """Detects people in frame and selects primary user based on box area and proximity to face/center."""
        if frame is None or frame.size == 0:
            return PersonDetectionResult()

        h, w = frame.shape[:2]
        t0 = time.monotonic()

        # Execute HOG multi-person detection
        boxes, weights = self._hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )
        dt_ms = (time.monotonic() - t0) * 1000.0

        valid_detections = []
        for i, (bx, by, bw, bh) in enumerate(boxes):
            weight = float(weights[i]) if len(weights) > i else 0.5
            if weight >= 0.2:  # Threshold
                valid_detections.append((bx, by, bw, bh, weight))

        count = len(valid_detections)
        if count == 0:
            return PersonDetectionResult(inference_time_ms=dt_ms)

        multiple_users = count > 1

        # Primary user selection policy:
        # 1. Favor person containing face center (if available)
        # 2. Favor largest bounding box area * confidence score
        best_score = -1.0
        best_box = None

        for bx, by, bw, bh, weight in valid_detections:
            area_norm = (bw * bh) / (w * h)
            score = weight * area_norm

            if face_bbox_center is not None:
                fc_x, fc_y = face_bbox_center
                # Check if face center falls inside person box
                px_min, px_max = bx / w, (bx + bw) / w
                py_min, py_max = by / h, (by + bh) / h
                if (px_min <= fc_x <= px_max) and (py_min <= fc_y <= py_max):
                    score *= 2.0  # Priority boost for matching face person

            if score > best_score:
                best_score = score
                best_box = (bx, by, bw, bh, weight)

        if best_box is None:
            return PersonDetectionResult(multiple_users=multiple_users, person_count=count, inference_time_ms=dt_ms)

        bx, by, bw, bh, weight = best_box
        cx = (bx + (bw * 0.5)) / w
        cy = (by + (bh * 0.5)) / h

        return PersonDetectionResult(
            person_detected=True,
            multiple_users=multiple_users,
            person_count=count,
            bbox=(int(bx), int(by), int(bw), int(bh)),
            normalized_bbox=(bx / w, by / h, bw / w, bh / h),
            confidence=weight,
            center_x=cx,
            center_y=cy,
            inference_time_ms=dt_ms,
        )
