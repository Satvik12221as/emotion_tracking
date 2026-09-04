"""Facial Landmarks & Head Orientation / Attention Estimation Module."""

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HeadPoseResult:
    """Head orientation angles and screen attention status."""

    yaw_deg: float = 0.0    # Horizontal rotation (-left, +right)
    pitch_deg: float = 0.0  # Vertical rotation (-down, +up)
    roll_deg: float = 0.0   # Head tilt (-left, +right)
    attention_state: str = "NO_FACE"  # FACING_SCREEN, LOOKING_LEFT, LOOKING_RIGHT, LOOKING_UP, LOOKING_DOWN, LOOKING_AWAY
    is_facing_screen: bool = False
    inference_time_ms: float = 0.0


class FaceLandmarkAnalyzer:
    """Estimates head pose (Yaw/Pitch/Roll) and screen attention using MediaPipe FaceMesh & solvePnP."""

    # Generic 3D facial model points (in mm)
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0),    # Right mouth corner
        ],
        dtype=np.float64,
    )

    # Key landmark indices in MediaPipe Face Mesh
    LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

    def __init__(
        self,
        yaw_threshold_deg: float = 25.0,
        pitch_threshold_deg: float = 20.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.yaw_threshold = yaw_threshold_deg
        self.pitch_threshold = pitch_threshold_deg

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def analyze(self, frame: np.ndarray) -> HeadPoseResult:
        """Processes RGB frame and returns head pose angles & attention classification."""
        if frame is None or frame.size == 0:
            return HeadPoseResult(attention_state="NO_FACE")

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.monotonic()
        mesh_results = self._mesh.process(rgb)
        dt_ms = (time.monotonic() - t0) * 1000.0

        if not mesh_results or not mesh_results.multi_face_landmarks:
            return HeadPoseResult(attention_state="NO_FACE", inference_time_ms=dt_ms)

        face_landmarks = mesh_results.multi_face_landmarks[0]
        image_points = []

        for idx in self.LANDMARK_IDS:
            lm = face_landmarks.landmark[idx]
            px = int(lm.x * w)
            py = int(lm.y * h)
            image_points.append([px, py])

        image_points_2d = np.array(image_points, dtype=np.float64)

        # Camera intrinsic matrix estimation
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rot_vec, trans_vec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return HeadPoseResult(attention_state="NO_FACE", inference_time_ms=dt_ms)

        # Convert rotation vector to matrix and euler angles
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        pose_mat = cv2.hconcat((rot_mat, trans_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        # Normalize yaw/pitch/roll to [-180, 180]
        yaw = (yaw + 180) % 360 - 180
        pitch = (pitch + 180) % 360 - 180
        roll = (roll + 180) % 360 - 180

        # Classify attention
        is_facing = abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold

        if is_facing:
            attention_state = "FACING_SCREEN"
        elif yaw > self.yaw_threshold:
            attention_state = "LOOKING_RIGHT"
        elif yaw < -self.yaw_threshold:
            attention_state = "LOOKING_LEFT"
        elif pitch > self.pitch_threshold:
            attention_state = "LOOKING_UP"
        elif pitch < -self.pitch_threshold:
            attention_state = "LOOKING_DOWN"
        else:
            attention_state = "LOOKING_AWAY"

        return HeadPoseResult(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            attention_state=attention_state,
            is_facing_screen=is_facing,
            inference_time_ms=dt_ms,
        )

    def close(self) -> None:
        """Releases FaceMesh resources."""
        if hasattr(self, "_mesh") and self._mesh:
            self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
