"""Unit tests for PersonDetector module."""

import numpy as np
import pytest
from src.body.person_detector import PersonDetector, PersonDetectionResult


def test_person_detector_blank_frame():
    detector = PersonDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = detector.detect(frame)
    assert isinstance(res, PersonDetectionResult)
    assert res.person_detected is False
    assert res.multiple_users is False
    assert res.person_count == 0
