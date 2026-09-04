"""HUD Overlay & Graphical Display Assistance Output Module."""

from typing import Optional

import cv2
import numpy as np
from src.decision.rules import AssistanceAction
from src.state.user_state import UserState


class DisplayRenderer:
    """Renders real-time assistive HUD overlays onto video frames."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    def render_hud(
        self,
        frame: np.ndarray,
        user_state: UserState,
        action: AssistanceAction,
        fps: float = 0.0,
        durations: Optional[dict] = None,
    ) -> np.ndarray:
        """Renders comprehensive assistive HUD dashboard onto frame copy."""
        output = frame.copy()
        h, w = output.shape[:2]

        # Semi-transparent side panel overlay
        panel_w = 260
        overlay = output.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        # Header Title
        cv2.putText(output, "ASSISTIVE VISION", (w - panel_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.line(output, (w - panel_w + 10, 32), (w - 10, 32), (0, 255, 255), 1)

        # Draw Seating Boundary Lines
        cv2.line(output, (int(w * 0.35), 0), (int(w * 0.35), h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.line(output, (int(w * 0.65), 0), (int(w * 0.65), h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(output, "L", (int(w * 0.15), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(output, "CENTER", (int(w * 0.45), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(output, "R", (int(w * 0.80), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Metrics status list
        metrics = [
            f"FPS: {fps:.1f}",
            f"Face: {'YES' if user_state.face_detected else 'NO'}",
            f"Emotion: {user_state.emotion}",
            f"Seating: {user_state.seating_position}",
            f"Distance: {user_state.distance}",
            f"Posture: {user_state.posture}",
            f"Attention: {user_state.attention}",
            f"Yaw/Pitch: {user_state.yaw_deg:.0f}/{user_state.pitch_deg:.0f}",
        ]

        y_offset = 55
        for m in metrics:
            cv2.putText(output, m, (w - panel_w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            y_offset += 22

        # Assistance Trigger Banner
        if action.triggered:
            banner_h = 70
            cv2.rectangle(output, (10, h - banner_h - 10), (w - panel_w - 10, h - 10), (0, 0, 180), -1)
            cv2.rectangle(output, (10, h - banner_h - 10), (w - panel_w - 10, h - 10), (0, 255, 255), 2)
            cv2.putText(output, f"ALERT: {action.title}", (20, h - banner_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, action.message, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 255, 220), 1)

        return output
