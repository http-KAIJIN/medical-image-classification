from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import tensorflow as tf
from PIL import Image

from backend.config import GRADCAM_STATIC_DIR
from src.gradcam import create_overlay, find_final_conv_layer, generate_gradcam_heatmap, heatmap_to_rgb, resize_heatmap


class GradCAMService:
    def __init__(self, model: tf.keras.Model, output_dir: Path = GRADCAM_STATIC_DIR) -> None:
        self.model = model
        self.output_dir = output_dir
        self.conv_layer_name = find_final_conv_layer(model)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, image: Image.Image, image_array) -> dict[str, float | str]:
        result = generate_gradcam_heatmap(self.model, image_array, conv_layer_name=self.conv_layer_name)

        file_id = uuid4().hex
        heatmap = resize_heatmap(result.heatmap, image.size)
        heatmap_path = self.output_dir / f"{file_id}_heatmap.png"
        overlay_path = self.output_dir / f"{file_id}_overlay.png"

        heatmap_to_rgb(heatmap).save(heatmap_path)
        create_overlay(image.convert("RGB"), result.heatmap).save(overlay_path)

        return {
            "prediction": result.predicted_class,
            "confidence": result.confidence,
            "probability": result.predicted_probability,
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "gradcam_image": f"/static/gradcam/{overlay_path.name}",
        }
