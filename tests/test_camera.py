"""Unit tests for CameraManager module."""

import time
import numpy as np
import pytest
from src.camera.camera_manager import CameraManager


class DummyCapture:
    """Mock OpenCV Capture for headless automated unit testing."""

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.opened = True

    def isOpened(self):
        return self.opened

    def set(self, prop, val):
        pass

    def read(self):
        # Generate synthetic test frame
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = (100, 150, 200)
        return True, img

    def release(self):
        self.opened = False


def test_camera_manager_mocked(monkeypatch):
    """Test CameraManager lifecycle using mocked OpenCV VideoCapture."""
    monkeypatch.setattr("cv2.VideoCapture", lambda source, *args: DummyCapture(640, 480))

    cam = CameraManager(source=0, width=640, height=480, fps=30, flip_h=True)
    assert cam.start() is True

    time.sleep(0.3)
    ok, frame = cam.read()
    assert ok is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)

    cam.stop()
    assert cam._running is False
