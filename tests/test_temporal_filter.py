"""Unit tests for TemporalFilter module."""

import time
import pytest
from src.state.temporal_filter import TemporalFilter
from src.state.user_state import UserState


def test_temporal_filter_majority_voting():
    tf = TemporalFilter(window_size_sec=2.0)

    # Push 3 LEANING_LEFT and 1 UPRIGHT
    tf.push(UserState(posture="LEANING_LEFT"))
    tf.push(UserState(posture="LEANING_LEFT"))
    tf.push(UserState(posture="LEANING_LEFT"))
    tf.push(UserState(posture="UPRIGHT"))

    stabilized, durations = tf.push(UserState(posture="LEANING_LEFT"))
    assert stabilized.posture == "LEANING_LEFT"
    assert "posture" in durations
