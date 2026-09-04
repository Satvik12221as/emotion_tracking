"""Camera-based User Distance Estimator Module."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DistanceResult:
    """Estimated user distance classification and ratio metrics."""

    classification: str = "UNKNOWN"  # "TOO_CLOSE", "NORMAL", "TOO_FAR", "UNKNOWN"
    face_ratio: float = 0.0          # Normalized face width relative to frame width
    estimated_distance_m: Optional[float] = None


class DistanceEstimator:
    """Estimates distance using face bounding-box width ratio and calibrated focal length."""

    def __init__(
        self,
        too_close_ratio: float = 0.35,
        too_far_ratio: float = 0.10,
        known_face_width_m: float = 0.15,
        focal_length_px: float = 600.0,
    ):
        self.too_close_ratio = too_close_ratio
        self.too_far_ratio = too_far_ratio
        self.known_face_width_m = known_face_width_m
        self.focal_length_px = focal_length_px

    def estimate(self, face_bbox_norm: Optional[Tuple[float, float, float, float]], frame_width: int = 640) -> DistanceResult:
        """Estimates distance classification and numerical distance in meters."""
        if face_bbox_norm is None:
            return DistanceResult(classification="UNKNOWN")

        _, _, norm_w, _ = face_bbox_norm
        if norm_w <= 0.0:
            return DistanceResult(classification="UNKNOWN")

        # Classification based on calibrated ratio thresholds
        if norm_w > self.too_close_ratio:
            classification = "TOO_CLOSE"
        elif norm_w < self.too_far_ratio:
            classification = "TOO_FAR"
        else:
            classification = "NORMAL"

        # Pinhole camera geometry estimation: distance = (known_width * focal_length) / width_px
        face_width_px = norm_w * frame_width
        dist_m = None
        if face_width_px > 0:
            dist_m = (self.known_face_width_m * self.focal_length_px) / face_width_px

        return DistanceResult(
            classification=classification,
            face_ratio=norm_w,
            estimated_distance_m=dist_m,
        )
