from __future__ import annotations

import numpy as np
import tensorflow as tf

from backend.config import MODEL_PATH, PREDICTION_THRESHOLD
from backend.preprocessing import preprocess_image_bytes
from src.models import efficientnetb0_preprocess  # noqa: F401 - registers custom Lambda for loading


class InferenceService:
    def __init__(self, model_path=MODEL_PATH, threshold: float = PREDICTION_THRESHOLD) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.model: tf.keras.Model | None = None

    def load_model(self) -> None:
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)

    def predict_array(self, image_array: np.ndarray) -> dict[str, float | str]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        probability = float(self.model.predict(image_array[np.newaxis, ...], verbose=0).reshape(-1)[0])
        prediction = "PNEUMONIA" if probability >= self.threshold else "NORMAL"
        confidence = probability if prediction == "PNEUMONIA" else 1.0 - probability
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probability": probability,
            "threshold": self.threshold,
        }

    def predict_bytes(self, data: bytes) -> dict[str, float | str]:
        return self.predict_array(preprocess_image_bytes(data))
