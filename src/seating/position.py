"""Seating Position Analysis Module."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SeatingPositionResult:
    """Seating alignment relative to calibrated horizontal zones."""

    zone: str = "UNKNOWN"  # "LEFT", "CENTER", "RIGHT", "UNKNOWN"
    normalized_x: float = 0.5
    is_centered: bool = True


class SeatingPositionAnalyzer:
    """Evaluates horizontal seating position based on calibrated boundary thresholds."""

    def __init__(self, left_boundary: float = 0.35, right_boundary: float = 0.65):
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def analyze(self, center_x: Optional[float]) -> SeatingPositionResult:
        """Determines if user is in LEFT, CENTER, or RIGHT seating zone."""
        if center_x is None:
            return SeatingPositionResult(zone="UNKNOWN", is_centered=False)

        cx = max(0.0, min(1.0, center_x))

        if cx < self.left_boundary:
            zone = "LEFT"
            centered = False
        elif cx > self.right_boundary:
            zone = "RIGHT"
            centered = False
        else:
            zone = "CENTER"
            centered = True

        return SeatingPositionResult(
            zone=zone,
            normalized_x=cx,
            is_centered=centered,
        )
