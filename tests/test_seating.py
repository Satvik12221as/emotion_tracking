"""Unit tests for seating position, distance estimation, and calibration."""

import pytest
from src.seating.position import SeatingPositionAnalyzer, SeatingPositionResult
from src.seating.distance import DistanceEstimator, DistanceResult
from src.seating.calibration import SystemCalibrator


def test_seating_position():
    analyzer = SeatingPositionAnalyzer(left_boundary=0.35, right_boundary=0.65)

    res_left = analyzer.analyze(0.20)
    assert res_left.zone == "LEFT"
    assert res_left.is_centered is False

    res_center = analyzer.analyze(0.50)
    assert res_center.zone == "CENTER"
    assert res_center.is_centered is True

    res_right = analyzer.analyze(0.80)
    assert res_right.zone == "RIGHT"
    assert res_right.is_centered is False

    res_none = analyzer.analyze(None)
    assert res_none.zone == "UNKNOWN"


def test_distance_estimator():
    estimator = DistanceEstimator(too_close_ratio=0.35, too_far_ratio=0.10)

    # Face ratio 0.45 -> TOO_CLOSE
    res_close = estimator.estimate((0.1, 0.1, 0.45, 0.45), frame_width=640)
    assert res_close.classification == "TOO_CLOSE"

    # Face ratio 0.20 -> NORMAL
    res_normal = estimator.estimate((0.1, 0.1, 0.20, 0.20), frame_width=640)
    assert res_normal.classification == "NORMAL"

    # Face ratio 0.05 -> TOO_FAR
    res_far = estimator.estimate((0.1, 0.1, 0.05, 0.05), frame_width=640)
    assert res_far.classification == "TOO_FAR"


def test_calibration_tool(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("seating:\n  left_boundary: 0.3\n  right_boundary: 0.7\n")

    calibrator = SystemCalibrator(str(config_file))
    assert calibrator.update_seating_zones(0.35, 0.65) is True

    updated_config = calibrator.config
    assert updated_config["seating"]["left_boundary"] == 0.35
    assert updated_config["seating"]["right_boundary"] == 0.65
