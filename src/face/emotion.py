"""Facial Expression / Emotion Estimation Module using ONNX Runtime."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ONNX Runtime availability check
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


DEFAULT_EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


@dataclass
class EmotionResult:
    """Structured result for facial expression / emotion estimation."""

    label: str = "neutral"
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0


class EmotionEstimator:
    """Decoupled Facial Expression / Emotion Estimator."""

    def __init__(
        self,
        model_path: str = "models/emotion/mini_xception_fer.onnx",
        labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.50,
        input_size: Tuple[int, int] = (64, 64),
        is_grayscale: bool = True,
    ):
        self.model_path = Path(model_path)
        self.labels = labels or DEFAULT_EMOTION_LABELS
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.is_grayscale = is_grayscale

        self._session = None
        self._input_name = None
        self._output_name = None

        self._init_model()

    def _init_model(self) -> None:
        """Initializes ONNX inference session if model file exists."""
        if not ONNX_AVAILABLE:
            logger.warning("onnxruntime is not installed. Running in heuristic/fallback mode.")
            return

        if not self.model_path.exists():
            logger.info(
                f"Emotion model file not found at '{self.model_path}'. Running in lightweight heuristic mode."
            )
            return

        try:
            # Use CPU execution provider (ARM64 / Raspberry Pi optimized)
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info(f"Loaded ONNX emotion model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model at {self.model_path}: {e}")
            self._session = None

    def predict(self, face_roi: Optional[np.ndarray]) -> EmotionResult:
        """Predicts emotional state from face ROI."""
        if face_roi is None or face_roi.size == 0:
            probs = {label: (1.0 if label == "neutral" else 0.0) for label in self.labels}
            return EmotionResult(label="neutral", confidence=0.0, probabilities=probs)

        t0 = time.monotonic()

        if self._session is not None:
            # ONNX Inference Pipeline
            try:
                processed_roi = self._preprocess(face_roi)
                outputs = self._session.run([self._output_name], {self._input_name: processed_roi})
                logits = outputs[0][0]
                probs_vec = self._softmax(logits)
            except Exception as e:
                logger.error(f"ONNX emotion inference error: {e}")
                probs_vec = self._fallback_probs()
        else:
            # Fast heuristic fallback for testing/demo when model file is not present
            probs_vec = self._fallback_probs()

        dt_ms = (time.monotonic() - t0) * 1000.0

        probs_dict = {
            self.labels[i]: float(probs_vec[i])
            for i in range(min(len(self.labels), len(probs_vec)))
        }

        best_idx = int(np.argmax(probs_vec))
        best_label = self.labels[best_idx] if best_idx < len(self.labels) else "neutral"
        best_confidence = float(probs_vec[best_idx])

        # Qualifier wording to avoid overclaiming direct emotion measurement
        if best_confidence < self.confidence_threshold:
            best_label = "uncertain"

        return EmotionResult(
            label=best_label,
            confidence=best_confidence,
            probabilities=probs_dict,
            inference_time_ms=dt_ms,
        )

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Preprocesses image crop to model input dimensions."""
        resized = cv2.resize(roi, self.input_size)
        if self.is_grayscale:
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            normalized = (gray.astype(np.float32) - 128.0) / 128.0
            # Reshape to (1, 1, H, W) or (1, H, W, 1) depending on model schema
            return np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = (rgb.astype(np.float32) - 128.0) / 128.0
            # Reshape to (1, C, H, W)
            transposed = np.transpose(normalized, (2, 0, 1))
            return np.expand_dims(transposed, axis=0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _fallback_probs(self) -> np.ndarray:
        """Default neutral probabilities distribution."""
        probs = np.zeros(len(self.labels), dtype=np.float32)
        neutral_idx = self.labels.index("neutral") if "neutral" in self.labels else 0
        probs[neutral_idx] = 0.85
        remaining = 0.15 / (len(self.labels) - 1) if len(self.labels) > 1 else 0.0
        for i in range(len(self.labels)):
            if i != neutral_idx:
                probs[i] = remaining
        return probs
