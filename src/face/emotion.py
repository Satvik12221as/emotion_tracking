"""Facial Expression / Emotion Estimation Module supporting both ONNX and Keras (.h5) models."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Framework availability checks
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

KERAS_AVAILABLE = False
try:
    import tensorflow as tf
    KERAS_AVAILABLE = True
except Exception:
    try:
        import keras
        KERAS_AVAILABLE = True
    except Exception:
        KERAS_AVAILABLE = False


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
    """Decoupled Facial Expression / Emotion Estimator supporting ONNX and Keras .h5 models."""

    def __init__(
        self,
        model_path: str = "models/emotion/face_model.h5",
        labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.50,
        input_size: Tuple[int, int] = (48, 48),
        is_grayscale: bool = True,
    ):
        self.model_path = Path(model_path)
        self.labels = labels or DEFAULT_EMOTION_LABELS
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.is_grayscale = is_grayscale

        self._model_type = "none"  # "onnx", "keras", or "none"
        self._session = None
        self._keras_model = None
        self._input_name = None
        self._output_name = None

        self._init_model()

    def _init_model(self) -> None:
        """Initializes ONNX or Keras inference session based on model file extension."""
        if not self.model_path.exists():
            # Fallback to alternate ONNX if .h5 is missing or vice versa
            alt_onnx = self.model_path.parent / "mini_xception_fer.onnx"
            if alt_onnx.exists():
                self.model_path = alt_onnx

        if not self.model_path.exists():
            logger.info(
                f"Emotion model file not found at '{self.model_path}'. Running in lightweight fallback mode."
            )
            return

        suffix = self.model_path.suffix.lower()

        if suffix in [".h5", ".keras"]:
            self._init_keras_model()
        elif suffix == ".onnx":
            self._init_onnx_model()
        else:
            logger.warning(f"Unsupported model extension '{suffix}'. Expected .h5, .keras, or .onnx.")

    def _init_keras_model(self) -> None:
        """Loads Keras .h5 / .keras model."""
        try:
            import h5py
            # Use h5py/keras or tensorflow.keras
            try:
                import keras
                self._keras_model = keras.models.load_model(str(self.model_path), compile=False)
            except Exception:
                import tensorflow as tf
                self._keras_model = tf.keras.models.load_model(str(self.model_path), compile=False)

            self._model_type = "keras"
            # Auto-detect input size from model shape if available (e.g. (None, 48, 48, 1))
            if hasattr(self._keras_model, "input_shape") and self._keras_model.input_shape:
                shape = self._keras_model.input_shape
                if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                    self.input_size = (int(shape[1]), int(shape[2]))
                    self.is_grayscale = (shape[3] == 1)

            logger.info(f"Loaded Keras .h5 emotion model from {self.model_path} (Input shape: {self.input_size})")
        except Exception as e:
            logger.error(f"Failed to load Keras model at {self.model_path}: {e}")
            # Try ONNX fallback if available
            alt_onnx = self.model_path.parent / "mini_xception_fer.onnx"
            if alt_onnx.exists():
                self.model_path = alt_onnx
                self._init_onnx_model()

    def _init_onnx_model(self) -> None:
        """Loads ONNX inference model."""
        if not ONNX_AVAILABLE:
            logger.warning("onnxruntime is not installed.")
            return

        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._model_type = "onnx"
            logger.info(f"Loaded ONNX emotion model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model at {self.model_path}: {e}")

    def predict(self, face_roi: Optional[np.ndarray]) -> EmotionResult:
        """Predicts emotional state from face ROI."""
        if face_roi is None or face_roi.size == 0:
            probs = {label: (1.0 if label == "neutral" else 0.0) for label in self.labels}
            return EmotionResult(label="neutral", confidence=0.0, probabilities=probs)

        t0 = time.monotonic()

        if self._model_type == "keras" and self._keras_model is not None:
            try:
                processed_roi = self._preprocess_keras(face_roi)
                preds = self._keras_model.predict(processed_roi, verbose=0)
                probs_vec = preds[0]
            except Exception as e:
                logger.error(f"Keras emotion prediction error: {e}")
                probs_vec = self._fallback_probs()

        elif self._model_type == "onnx" and self._session is not None:
            try:
                processed_roi = self._preprocess_onnx(face_roi)
                outputs = self._session.run([self._output_name], {self._input_name: processed_roi})
                logits = outputs[0][0]
                probs_vec = self._softmax(logits)
            except Exception as e:
                logger.error(f"ONNX emotion prediction error: {e}")
                probs_vec = self._fallback_probs()
        else:
            probs_vec = self._fallback_probs()

        dt_ms = (time.monotonic() - t0) * 1000.0

        probs_dict = {
            self.labels[i]: float(probs_vec[i])
            for i in range(min(len(self.labels), len(probs_vec)))
        }

        best_idx = int(np.argmax(probs_vec))
        best_label = self.labels[best_idx] if best_idx < len(self.labels) else "neutral"
        best_confidence = float(probs_vec[best_idx])

        if best_confidence < self.confidence_threshold:
            best_label = "uncertain"

        return EmotionResult(
            label=best_label,
            confidence=best_confidence,
            probabilities=probs_dict,
            inference_time_ms=dt_ms,
        )

    def _preprocess_keras(self, roi: np.ndarray) -> np.ndarray:
        """Preprocesses face crop for Keras CNN (1, H, W, 1) or (1, H, W, 3)."""
        resized = cv2.resize(roi, self.input_size)
        if self.is_grayscale:
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            normalized = gray.astype(np.float32) / 255.0
            return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            return np.expand_dims(normalized, axis=0)

    def _preprocess_onnx(self, roi: np.ndarray) -> np.ndarray:
        """Preprocesses face crop for ONNX model (1, 1, H, W) or (1, C, H, W)."""
        resized = cv2.resize(roi, (64, 64))
        if self.is_grayscale:
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            normalized = (gray.astype(np.float32) - 128.0) / 128.0
            return np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = (rgb.astype(np.float32) - 128.0) / 128.0
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
