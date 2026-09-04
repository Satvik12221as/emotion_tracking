"""Unit tests for PoseEstimator and PostureAnalyzer modules."""

import numpy as np
import pytest
from src.body.pose import PoseEstimator, PoseResult
from src.body.posture import PostureAnalyzer, PostureResult


def test_pose_estimator_empty_frame():
    estimator = PoseEstimator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = estimator.estimate(frame)
    assert isinstance(res, PoseResult)
    assert res.detected is False
    assert res.pose_landmarks is None
    estimator.close()


def test_posture_analyzer_no_landmarks():
    analyzer = PostureAnalyzer()
    res = analyzer.analyze(PoseResult(detected=False))
    assert isinstance(res, PostureResult)
    assert res.state == "UNKNOWN"
    assert res.is_poor is False
