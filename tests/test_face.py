"""Unit tests for FaceDetector module."""

import numpy as np
import pytest
from src.face.detector import FaceDetector, FaceDetectionResult


def test_face_detector_no_face():
    """Test face detector behavior when no face is present."""
    detector = FaceDetector(min_detection_confidence=0.5)
    # Synthetic black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = detector.detect(frame)
    assert isinstance(res, FaceDetectionResult)
    assert res.detected is False
    assert res.roi is None
    assert res.confidence == 0.0
    detector.close()


def test_face_detector_empty_frame():
    """Test face detector handling of invalid inputs."""
    detector = FaceDetector()
    res = detector.detect(np.array([]))
    assert res.detected is False

    res_none = detector.detect(None)
    assert res_none.detected is False
    detector.close()
