"""Unit tests for EmotionEstimator module."""

import numpy as np
import pytest
from src.face.emotion import EmotionEstimator, EmotionResult


def test_emotion_estimator_none_roi():
    estimator = EmotionEstimator()
    res = estimator.predict(None)
    assert isinstance(res, EmotionResult)
    assert res.label == "neutral"
    assert res.confidence == 0.0
    assert "neutral" in res.probabilities


def test_emotion_estimator_synthetic_roi():
    estimator = EmotionEstimator(confidence_threshold=0.4)
    # Synthetic face ROI image (64x64)
    synthetic_roi = np.zeros((64, 64, 3), dtype=np.uint8)
    synthetic_roi[20:40, 20:40] = (200, 200, 200)

    res = estimator.predict(synthetic_roi)
    assert isinstance(res, EmotionResult)
    assert res.label in estimator.labels or res.label == "uncertain"
    assert len(res.probabilities) == len(estimator.labels)
