"""Integration test for full Assistive Vision Pipeline."""

import numpy as np
import pytest
from src.main import AssistiveVisionPipeline


def test_pipeline_single_frame():
    """Test full pipeline execution on a single synthetic frame."""
    pipeline = AssistiveVisionPipeline(headless=True)

    # Synthetic test frame (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (120, 120, 120)

    output = pipeline.process_frame(frame)
    assert output is not None
    assert output.shape == (480, 640, 3)

    pipeline.stop()
