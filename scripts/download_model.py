"""Pretrained ONNX Emotion Model Downloader Script."""

import logging
from pathlib import Path
import sys
import urllib.request

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("download_model")

# Public ONNX model URL (FERPlus 64x64 ONNX model)
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
TARGET_PATH = Path("models/emotion/mini_xception_fer.onnx")


def download_emotion_model() -> bool:
    """Downloads pretrained ONNX facial expression model if not present."""
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_PATH.exists():
        logger.info(f"Emotion model already exists at {TARGET_PATH.resolve()}")
        return True

    logger.info(f"Downloading pretrained ONNX emotion model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, TARGET_PATH)
        logger.info(f"Model downloaded successfully to {TARGET_PATH.resolve()} ({TARGET_PATH.stat().st_size / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download emotion model: {e}")
        return False


if __name__ == "__main__":
    download_emotion_model()
