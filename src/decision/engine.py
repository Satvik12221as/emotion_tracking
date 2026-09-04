"""Priority-Ranked Rule-Based Decision Engine with Cooldown Management."""

import logging
import time
from typing import Dict, List, Optional

from src.decision.rules import AssistanceAction
from src.state.user_state import UserState

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Evaluates UserState against priority rules and resolves triggered assistance actions."""

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        seating_persistence_sec: float = 2.0,
        distance_persistence_sec: float = 2.0,
        posture_persistence_sec: float = 2.5,
        confusion_persistence_sec: float = 3.0,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.seating_persistence_sec = seating_persistence_sec
        self.distance_persistence_sec = distance_persistence_sec
        self.posture_persistence_sec = posture_persistence_sec
        self.confusion_persistence_sec = confusion_persistence_sec

        self._last_trigger_time: float = 0.0
        self._last_action_type: str = "NONE"

    def evaluate(self, user_state: UserState, durations: Dict[str, float]) -> AssistanceAction:
        """Evaluates user state and persistence durations, returning highest-priority assistance action."""
        now = time.monotonic()

        if not user_state.face_detected:
            return AssistanceAction(triggered=False, action_type="NONE")

        candidates: List[AssistanceAction] = []

        # Priority 1: Multi-User Ambiguity Alert
        if user_state.multiple_users:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="MULTI_USER_ALERT",
                    priority=1,
                    title="Multiple Users Detected",
                    message="Multiple people detected in view. Suppressing individual assistance.",
                    reasons=[
                        "Multiple people visible in camera feed",
                        "Primary user selection active",
                    ],
                )
            )

        # Priority 2: Seating Alignment Correction
        seating_dur = durations.get("seating_position", 0.0)
        if user_state.seating_position == "LEFT" and seating_dur >= self.seating_persistence_sec:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="SEATING_LEFT",
                    priority=2,
                    title="Adjust Seating Position",
                    message="Please shift slightly toward your right (center).",
                    reasons=[
                        f"Seating position is LEFT for {seating_dur:.1f}s (threshold: {self.seating_persistence_sec}s)",
                        f"User horizontal center X = {user_state.face_center_x:.2f}",
                    ],
                )
            )
        elif user_state.seating_position == "RIGHT" and seating_dur >= self.seating_persistence_sec:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="SEATING_RIGHT",
                    priority=2,
                    title="Adjust Seating Position",
                    message="Please shift slightly toward your left (center).",
                    reasons=[
                        f"Seating position is RIGHT for {seating_dur:.1f}s (threshold: {self.seating_persistence_sec}s)",
                        f"User horizontal center X = {user_state.face_center_x:.2f}",
                    ],
                )
            )

        # Priority 3: Distance Correction
        dist_dur = durations.get("distance", 0.0)
        if user_state.distance == "TOO_CLOSE" and dist_dur >= self.distance_persistence_sec:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="DISTANCE_CLOSE",
                    priority=3,
                    title="Distance Alert",
                    message="You are sitting too close to the camera/screen. Please step back slightly.",
                    reasons=[
                        f"Distance state is TOO_CLOSE for {dist_dur:.1f}s",
                        f"Face ratio = {user_state.face_confidence:.2f}",
                    ],
                )
            )
        elif user_state.distance == "TOO_FAR" and dist_dur >= self.distance_persistence_sec:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="DISTANCE_FAR",
                    priority=3,
                    title="Distance Alert",
                    message="You are sitting too far from the camera/screen. Please move closer.",
                    reasons=[
                        f"Distance state is TOO_FAR for {dist_dur:.1f}s",
                    ],
                )
            )

        # Priority 4: Posture Correction
        posture_dur = durations.get("posture", 0.0)
        if user_state.posture == "LEANING_LEFT" and posture_dur >= self.posture_persistence_sec:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="POSTURE_LEAN_LEFT",
                    priority=4,
                    title="Posture Warning",
                    message="You are leaning to the left. Please sit upright.",
                    reasons=[
                        f"Posture is LEANING_LEFT for {posture_dur:.1f}s (threshold: {self.posture_persistence_sec}s)",
                    ],
                )
            )
        elif user_state.posture == "LEANING_RIGHT" and posture_dur >= self.posture_persistence_sec:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="POSTURE_LEAN_RIGHT",
                    priority=4,
                    title="Posture Warning",
                    message="You are leaning to the right. Please sit upright.",
                    reasons=[
                        f"Posture is LEANING_RIGHT for {posture_dur:.1f}s (threshold: {self.posture_persistence_sec}s)",
                    ],
                )
            )
        elif user_state.posture == "SLOUCHING" and posture_dur >= self.posture_persistence_sec:
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="POSTURE_SLOUCH",
                    priority=4,
                    title="Posture Warning",
                    message="Slouching detected. Please adjust your back posture.",
                    reasons=[
                        f"Posture is SLOUCHING for {posture_dur:.1f}s (threshold: {self.posture_persistence_sec}s)",
                    ],
                )
            )

        # Priority 5: Possible Confusion / Frustration Assistance
        emotion_dur = durations.get("emotion", 0.0)
        if (
            user_state.emotion in ("sad", "angry", "fear", "surprise")
            and user_state.is_facing_screen
            and emotion_dur >= self.confusion_persistence_sec
        ):
            candidates.append(
                AssistanceAction(
                    triggered=True,
                    action_type="CONFUSION_ASSIST",
                    priority=5,
                    title="Assistance Offer",
                    message="It looks like you might need assistance. Would you like help?",
                    reasons=[
                        f"Facial expression indicates possible {user_state.emotion} state",
                        f"User is FACING_SCREEN and engaged",
                        f"Expression persistent for {emotion_dur:.1f}s (threshold: {self.confusion_persistence_sec}s)",
                    ],
                )
            )

        if not candidates:
            return AssistanceAction(triggered=False, action_type="NONE")

        # Sort candidates by priority (lowest priority number = highest importance)
        candidates.sort(key=lambda a: a.priority)
        selected = candidates[0]

        # Enforce Cooldown timer
        if (now - self._last_trigger_time) < self.cooldown_seconds:
            # If same action triggered during cooldown, suppress
            if selected.action_type == self._last_action_type:
                return AssistanceAction(triggered=False, action_type="COOLDOWN_ACTIVE")

        # Update last trigger
        self._last_trigger_time = now
        self._last_action_type = selected.action_type
        logger.info(f"Decision Engine Triggered: {selected.title} ({selected.action_type})")
        return selected
