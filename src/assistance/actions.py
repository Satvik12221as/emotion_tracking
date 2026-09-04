"""Assistance Engine orchestrating screen HUD, audio, and logging outputs."""

import logging
from typing import Optional

import numpy as np
from src.assistance.display import DisplayRenderer
from src.assistance.voice import VoiceAssistance
from src.decision.rules import AssistanceAction
from src.state.user_state import UserState

logger = logging.getLogger(__name__)


class AssistanceEngine:
    """Central Assistance Engine orchestrating all output modalities."""

    def __init__(self, enable_display: bool = True, enable_audio: bool = False):
        self.display_renderer = DisplayRenderer()
        self.voice_assistance = VoiceAssistance(enabled=enable_audio)
        self.enable_display = enable_display

    def process(
        self,
        frame: np.ndarray,
        user_state: UserState,
        action: AssistanceAction,
        fps: float = 0.0,
        durations: Optional[dict] = None,
    ) -> np.ndarray:
        """Processes assistance output, logs explanations, triggers speech, and renders HUD."""
        if action.triggered:
            # Print structured explanation (Principle 7)
            logger.info(action.explain())
            # Voice notification
            self.voice_assistance.speak(action.message)

        # Render HUD on frame
        if self.enable_display and frame is not None:
            return self.display_renderer.render_hud(frame, user_state, action, fps=fps, durations=durations)

        return frame
