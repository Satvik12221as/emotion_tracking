"""Unit tests for FaceLandmarkAnalyzer module."""

import numpy as np
import pytest
from src.face.landmarks import FaceLandmarkAnalyzer, HeadPoseResult


def test_landmark_analyzer_empty_frame():
    analyzer = FaceLandmarkAnalyzer()
    res = analyzer.analyze(np.array([]))
    assert res.attention_state == "NO_FACE"
    assert res.is_facing_screen is False

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_blank = analyzer.analyze(blank_frame)
    assert res_blank.attention_state == "NO_FACE"
    analyzer.close()
