"""Temporal Filter & Hysteresis Smoothing Module."""

from collections import Counter, deque
import time
from typing import Dict, List, Tuple

from src.state.user_state import UserState


class TemporalFilter:
    """Sliding-window majority voting filter with temporal state duration persistence."""

    def __init__(self, window_size_sec: float = 1.5, max_history_len: int = 45):
        self.window_size_sec = window_size_sec
        self.max_history_len = max_history_len

        self._history: deque[UserState] = deque(maxlen=self.max_history_len)

        # State persistence trackers (field_name -> (current_stable_value, start_time))
        self._state_start_times: Dict[str, Tuple[str, float]] = {}

    def push(self, raw_state: UserState) -> Tuple[UserState, Dict[str, float]]:
        """Appends new raw perception frame and returns stabilized state with persistence durations."""
        now = time.monotonic()
        raw_state.timestamp = now
        self._history.append(raw_state)

        # Remove snapshots older than window_size_sec
        cutoff = now - self.window_size_sec
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

        if not self._history:
            return raw_state, {}

        # Majority voting over history
        stabilized = UserState(
            timestamp=now,
            face_detected=self._history_majority_bool("face_detected"),
            face_confidence=self._history_average_float("face_confidence"),
            face_center_x=self._history_average_float("face_center_x"),
            face_center_y=self._history_average_float("face_center_y"),
            emotion=self._history_majority_str("emotion"),
            emotion_confidence=self._history_average_float("emotion_confidence"),
            emotion_probabilities=self._history[-1].emotion_probabilities,
            seating_position=self._history_majority_str("seating_position"),
            distance=self._history_majority_str("distance"),
            distance_meters=self._history[-1].distance_meters,
            posture=self._history_majority_str("posture"),
            yaw_deg=self._history_average_float("yaw_deg"),
            pitch_deg=self._history_average_float("pitch_deg"),
            roll_deg=self._history_average_float("roll_deg"),
            attention=self._history_majority_str("attention"),
            is_facing_screen=self._history_majority_bool("is_facing_screen"),
            multiple_users=self._history_majority_bool("multiple_users"),
        )

        # Update persistence durations
        durations: Dict[str, float] = {}
        fields_to_track = ["emotion", "seating_position", "distance", "posture", "attention"]

        for field in fields_to_track:
            val = getattr(stabilized, field)
            if field not in self._state_start_times or self._state_start_times[field][0] != val:
                self._state_start_times[field] = (val, now)
                durations[field] = 0.0
            else:
                durations[field] = now - self._state_start_times[field][1]

        return stabilized, durations

    def _history_majority_str(self, field: str) -> str:
        values = [getattr(s, field) for s in self._history]
        if not values:
            return "UNKNOWN"
        return Counter(values).most_common(1)[0][0]

    def _history_majority_bool(self, field: str) -> bool:
        values = [getattr(s, field) for s in self._history]
        if not values:
            return False
        return Counter(values).most_common(1)[0][0]

    def _history_average_float(self, field: str) -> float:
        values = [getattr(s, field) for s in self._history if getattr(s, field) is not None]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def reset(self) -> None:
        """Clears state history."""
        self._history.clear()
        self._state_start_times.clear()
