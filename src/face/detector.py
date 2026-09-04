"""Face Detection module providing ROI extraction and primary face selection."""

from dataclasses import dataclass, field
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceDetectionResult:
    """Structured result for single-user face detection."""

    detected: bool = False
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (xmin, ymin, width, height) in pixels
    normalized_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (0.0 to 1.0)
    confidence: float = 0.0
    center_x: float = 0.5  # Normalized X center
    center_y: float = 0.5  # Normalized Y center
    roi: Optional[np.ndarray] = field(default=None, repr=False)
    inference_time_ms: float = 0.0


class FaceDetector:
    """Lightweight single-user Face Detector wrapping MediaPipe Face Detection."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        roi_padding: float = 0.15,
    ):
        self.min_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.roi_padding = roi_padding

        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self.min_confidence,
            model_selection=self.model_selection,
        )

    def detect(self, frame: np.ndarray) -> FaceDetectionResult:
        """Processes RGB frame and selects primary face ROI."""
        if frame is None or frame.size == 0:
            return FaceDetectionResult(detected=False)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.monotonic()
        results = self._detector.process(rgb)
        dt_ms = (time.monotonic() - t0) * 1000.0

        if not results or not results.detections:
            return FaceDetectionResult(detected=False, inference_time_ms=dt_ms)

        # Primary face selection policy: Maximum (Confidence * BoundingBoxArea)
        best_score = -1.0
        best_det = None

        for det in results.detections:
            bbox_rel = det.location_data.relative_bounding_box
            score = float(det.score[0]) * (bbox_rel.width * bbox_rel.height)
            if score > best_score:
                best_score = score
                best_det = det

        if best_det is None:
            return FaceDetectionResult(detected=False, inference_time_ms=dt_ms)

        bbox_rel = best_det.location_data.relative_bounding_box
        xmin_rel = max(0.0, min(1.0, bbox_rel.xmin))
        ymin_rel = max(0.0, min(1.0, bbox_rel.ymin))
        w_rel = max(0.0, min(1.0 - xmin_rel, bbox_rel.width))
        h_rel = max(0.0, min(1.0 - ymin_rel, bbox_rel.height))

        cx = xmin_rel + (w_rel * 0.5)
        cy = ymin_rel + (h_rel * 0.5)

        # Convert to pixel coords
        xmin_px = int(xmin_rel * w)
        ymin_px = int(ymin_rel * h)
        w_px = int(w_rel * w)
        h_px = int(h_rel * h)

        # Extract padded ROI
        pad_w = int(w_px * self.roi_padding)
        pad_h = int(h_px * self.roi_padding)

        crop_xmin = max(0, xmin_px - pad_w)
        crop_ymin = max(0, ymin_px - pad_h)
        crop_xmax = min(w, xmin_px + w_px + pad_w)
        crop_ymax = min(h, ymin_px + h_px + pad_h)

        face_roi = None
        if (crop_xmax > crop_xmin) and (crop_ymax > crop_ymin):
            face_roi = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()

        return FaceDetectionResult(
            detected=True,
            bbox=(xmin_px, ymin_px, w_px, h_px),
            normalized_bbox=(xmin_rel, ymin_rel, w_rel, h_rel),
            confidence=float(best_det.score[0]),
            center_x=cx,
            center_y=cy,
            roi=face_roi,
            inference_time_ms=dt_ms,
        )

    def draw_detection(self, frame: np.ndarray, result: FaceDetectionResult) -> np.ndarray:
        """Visualizes detection bounding box and stats on frame copy."""
        output = frame.copy()
        if not result.detected:
            cv2.putText(
                output,
                "Face: NOT DETECTED",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            return output

        x, y, w, h = result.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Face: {result.confidence:.2f} ({result.inference_time_ms:.1f}ms)"
        cv2.putText(
            output,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        return output

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if hasattr(self, "_detector") and self._detector:
            self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
