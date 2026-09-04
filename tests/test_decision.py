"""Unit tests for Decision Engine and Assistance Action prioritization."""

import pytest
from src.decision.engine import DecisionEngine
from src.decision.rules import AssistanceAction
from src.state.user_state import UserState


def test_decision_engine_no_face():
    engine = DecisionEngine()
    state = UserState(face_detected=False)

    action = engine.evaluate(state, {})
    assert isinstance(action, AssistanceAction)
    assert action.triggered is False


def test_decision_engine_seating_trigger():
    engine = DecisionEngine(seating_persistence_sec=2.0)

    # Trigger seating left after 2.5s duration
    state = UserState(face_detected=True, seating_position="LEFT", face_center_x=0.20)
    durations = {"seating_position": 2.5}

    action = engine.evaluate(state, durations)
    assert action.triggered is True
    assert action.action_type == "SEATING_LEFT"
    assert action.priority == 2


def test_decision_engine_posture_trigger():
    engine = DecisionEngine(posture_persistence_sec=2.0, cooldown_seconds=0.0)

    state = UserState(face_detected=True, posture="LEANING_LEFT")
    durations = {"posture": 2.5}

    action = engine.evaluate(state, durations)
    assert action.triggered is True
    assert action.action_type == "POSTURE_LEAN_LEFT"
    assert action.priority == 4


def test_decision_engine_priority_resolution():
    """Verify higher priority rule (Seating P2) overrides lower priority rule (Posture P4)."""
    engine = DecisionEngine(seating_persistence_sec=2.0, posture_persistence_sec=2.0, cooldown_seconds=0.0)

    state = UserState(
        face_detected=True,
        seating_position="LEFT",
        posture="LEANING_LEFT",
    )
    durations = {"seating_position": 2.5, "posture": 2.5}

    action = engine.evaluate(state, durations)
    assert action.triggered is True
    assert action.action_type == "SEATING_LEFT"
    assert action.priority == 2  # Priority 2 beats Priority 4
