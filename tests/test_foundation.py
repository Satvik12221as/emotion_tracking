"""Foundation test for Phase 1 verification."""

import pytest
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def test_load_config():
    config = load_config("config/config.yaml")
    assert isinstance(config, dict)
    assert "system" in config
    assert "camera" in config
    assert "face" in config
    assert "posture" in config
    assert config["camera"]["width"] == 640
    assert config["camera"]["height"] == 480


def test_logger():
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    logger.info("Foundation test logger message.")
