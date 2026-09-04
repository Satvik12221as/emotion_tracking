"""Posture Analysis Module converting raw pose landmarks into human-readable posture states."""

from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
from src.body.pose import PoseResult


@dataclass
class PostureResult:
    """Human-readable posture classification and feature metrics."""

    state: str = "UNKNOWN"  # UPRIGHT, LEANING_LEFT, LEANING_RIGHT, SLOUCHING, HEAD_TILTED, UNKNOWN
    shoulder_tilt: float = 0.0
    lean_offset_x: float = 0.0
    head_drop_y: float = 0.0
    is_poor: bool = False


class PostureAnalyzer:
    """Analyzes geometric pose features to classify human body posture."""

    def __init__(
        self,
        shoulder_tilt_threshold: float = 0.06,
        lean_x_threshold: float = 0.10,
        slouch_head_drop_threshold: float = -0.05,
    ):
        self.shoulder_tilt_threshold = shoulder_tilt_threshold
        self.lean_x_threshold = lean_x_threshold
        self.slouch_head_drop_threshold = slouch_head_drop_threshold

    def analyze(self, pose_res: PoseResult) -> PostureResult:
        """Evaluates posture landmarks and returns discrete posture state."""
        if not pose_res.detected or pose_res.pose_landmarks is None:
            return PostureResult(state="UNKNOWN", is_poor=False)

        lm = mp.solutions.pose.PoseLandmark
        landmarks = pose_res.pose_landmarks.landmark

        left_shoulder = landmarks[lm.LEFT_SHOULDER.value]
        right_shoulder = landmarks[lm.RIGHT_SHOULDER.value]
        nose = landmarks[lm.NOSE.value]

        # Require minimum visibility
        if (
            left_shoulder.visibility < 0.5
            or right_shoulder.visibility < 0.5
            or nose.visibility < 0.5
        ):
            return PostureResult(state="UNKNOWN", is_poor=False)

        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) * 0.5
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) * 0.5

        shoulder_tilt = left_shoulder.y - right_shoulder.y  # + when left shoulder is lower
        abs_shoulder_tilt = abs(shoulder_tilt)
        lean_offset_x = nose.x - shoulder_mid_x            # - when leaning left, + when leaning right
        head_drop_y = nose.y - shoulder_mid_y              # Higher values (less negative) mean head is lower

        # Posture state classification logic
        if lean_offset_x < -self.lean_x_threshold:
            state = "LEANING_LEFT"
            is_poor = True
        elif lean_offset_x > self.lean_x_threshold:
            state = "LEANING_RIGHT"
            is_poor = True
        elif head_drop_y > self.slouch_head_drop_threshold:
            state = "SLOUCHING"
            is_poor = True
        elif abs_shoulder_tilt > self.shoulder_tilt_threshold:
            state = "HEAD_TILTED"
            is_poor = True
        else:
            state = "UPRIGHT"
            is_poor = False

        return PostureResult(
            state=state,
            shoulder_tilt=float(shoulder_tilt),
            lean_offset_x=float(lean_offset_x),
            head_drop_y=float(head_drop_y),
            is_poor=is_poor,
        )
