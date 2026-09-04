"""Central User State Contract Module."""

from dataclasses import dataclass, field
import time
from typing import Dict, Optional


@dataclass
class UserState:
    """Central perception snapshot contract representing unified user state."""

    timestamp: float = field(default_factory=time.monotonic)

    # Face Perception
    face_detected: bool = False
    face_confidence: float = 0.0
    face_center_x: float = 0.5
    face_center_y: float = 0.5

    # Emotion Perception
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    emotion_probabilities: Dict[str, float] = field(default_factory=dict)

    # Seating Alignment
    seating_position: str = "UNKNOWN"  # LEFT, CENTER, RIGHT, UNKNOWN

    # Distance Perception
    distance: str = "UNKNOWN"          # TOO_CLOSE, NORMAL, TOO_FAR, UNKNOWN
    distance_meters: Optional[float] = None

    # Posture Perception
    posture: str = "UNKNOWN"           # UPRIGHT, LEANING_LEFT, LEANING_RIGHT, SLOUCHING, HEAD_TILTED, UNKNOWN

    # Head Pose & Attention
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    attention: str = "UNKNOWN"         # FACING_SCREEN, LOOKING_LEFT, LOOKING_RIGHT, LOOKING_UP, LOOKING_DOWN, LOOKING_AWAY, UNKNOWN
    is_facing_screen: bool = False

    # Multi-User Signal
    multiple_users: bool = False

    def to_dict() -> Dict[str, str]:
        """Human-readable dictionary summary for display and logging."""
        return {
            "Face": f"{'DETECTED' if self.face_detected else 'NO_FACE'} ({self.face_confidence:.2f})",
            "Emotion": f"{self.emotion} ({self.emotion_confidence:.2f})",
            "Seating": self.seating_position,
            "Distance": self.distance,
            "Posture": self.posture,
            "Attention": self.attention,
            "MultiUser": str(self.multiple_users),
        }
