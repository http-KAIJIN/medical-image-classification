from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import IMAGE_SIZE


@dataclass(frozen=True)
class GradCAMResult:
    heatmap: np.ndarray
    predicted_probability: float
    predicted_class: str
    confidence: float
    attention_summary: dict[str, object]


def load_image_array(image_path: str | Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_original_rgb(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def find_final_conv_layer(model: tf.keras.Model, base_model_name: str = "efficientnetb0") -> str:
    base_model = model.get_layer(base_model_name)
    for layer in reversed(base_model.layers):
        if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            continue
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = layer.output.shape
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
    raise ValueError(f"No convolution-like 4D layer found in {base_model_name}")


def build_efficientnetb0_gradcam_parts(
    model: tf.keras.Model,
    conv_layer_name: str = "top_conv",
    base_model_name: str = "efficientnetb0",
) -> tuple[tf.keras.layers.Layer, tf.keras.Model, list[tf.keras.layers.Layer]]:
    preprocess_layer = model.get_layer("efficientnetb0_preprocess_input")
    base_model = model.get_layer(base_model_name)
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(conv_layer_name).output, base_model.output],
        name=f"{base_model_name}_{conv_layer_name}_feature_model",
    )
    head_layers = [
        model.get_layer("global_average_pooling"),
        model.get_layer("classification_dense"),
        model.get_layer("classification_dropout"),
        model.get_layer("pneumonia_probability"),
    ]
    return preprocess_layer, feature_model, head_layers


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_array: np.ndarray,
    conv_layer_name: str = "top_conv",
    explain_class: str | None = None,
) -> GradCAMResult:
    preprocess_layer, feature_model, head_layers = build_efficientnetb0_gradcam_parts(model, conv_layer_name=conv_layer_name)
    input_tensor = tf.convert_to_tensor(image_array[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        preprocessed = preprocess_layer(input_tensor, training=False)
        conv_outputs, base_outputs = feature_model(preprocessed, training=False)
        x = base_outputs
        for layer in head_layers:
            x = layer(x, training=False) if isinstance(layer, tf.keras.layers.Dropout) else layer(x)
        probability = tf.reshape(x, [-1])[0]
        predicted_class = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
        class_to_explain = explain_class or predicted_class
        class_score = probability if class_to_explain == "PNEUMONIA" else 1.0 - probability

    gradients = tape.gradient(class_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
    heatmap_array = heatmap.numpy()
    probability_value = float(probability.numpy())
    confidence = probability_value if predicted_class == "PNEUMONIA" else 1.0 - probability_value

    return GradCAMResult(
        heatmap=heatmap_array,
        predicted_probability=probability_value,
        predicted_class=predicted_class,
        confidence=float(confidence),
        attention_summary=summarize_attention(heatmap_array),
    )


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), mode="L")
    heatmap_image = heatmap_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(heatmap_image, dtype=np.float32) / 255.0


def heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    colormap = plt.get_cmap("jet")
    colored = colormap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return Image.fromarray(np.uint8(colored * 255)).convert("RGB")


def create_overlay(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.38) -> Image.Image:
    resized_heatmap = resize_heatmap(heatmap, original_image.size)
    heatmap_rgb = heatmap_to_rgb(resized_heatmap)
    return Image.blend(original_image.convert("RGB"), heatmap_rgb, alpha=alpha)


def save_gradcam_outputs(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_dir: str | Path,
    stem: str = "example",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_original_rgb(image_path)
    resized_heatmap = resize_heatmap(heatmap, original.size)
    heatmap_rgb = heatmap_to_rgb(resized_heatmap)
    overlay = create_overlay(original, heatmap)

    original_path = output_dir / f"{stem}_original.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    original.save(original_path)
    heatmap_rgb.save(heatmap_path)
    overlay.save(overlay_path)

    return {
        "original_image": str(original_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
    }


def summarize_attention(heatmap: np.ndarray) -> dict[str, object]:
    if heatmap.size == 0 or float(np.max(heatmap)) <= 0:
        return {
            "highest_attention_region": "No positive Grad-CAM activation",
            "centroid_x": None,
            "centroid_y": None,
            "inside_likely_lung_field": False,
            "border_attention_fraction": None,
            "artifact_attention_flag": True,
            "medical_plausibility": "Indeterminate because no positive activation was produced.",
        }

    threshold = np.quantile(heatmap, 0.85)
    mask = heatmap >= threshold
    y_indices, x_indices = np.where(mask)
    weights = heatmap[mask]
    centroid_x = float(np.average(x_indices, weights=weights) / max(heatmap.shape[1] - 1, 1))
    centroid_y = float(np.average(y_indices, weights=weights) / max(heatmap.shape[0] - 1, 1))

    horizontal = "left" if centroid_x < 0.33 else "right" if centroid_x > 0.67 else "central"
    vertical = "upper" if centroid_y < 0.33 else "lower" if centroid_y > 0.67 else "middle"

    border_mask = np.zeros_like(mask, dtype=bool)
    margin_y = max(1, int(mask.shape[0] * 0.15))
    margin_x = max(1, int(mask.shape[1] * 0.15))
    border_mask[:margin_y, :] = True
    border_mask[-margin_y:, :] = True
    border_mask[:, :margin_x] = True
    border_mask[:, -margin_x:] = True
    border_attention_fraction = float(np.mean(border_mask[mask])) if np.any(mask) else 1.0
    inside_likely_lung_field = 0.15 <= centroid_x <= 0.85 and 0.10 <= centroid_y <= 0.90
    artifact_attention_flag = border_attention_fraction > 0.35
    plausibility = "Plausible screening explanation" if inside_likely_lung_field and not artifact_attention_flag else "Needs cautious interpretation"

    return {
        "highest_attention_region": f"{vertical} {horizontal} chest region",
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "inside_likely_lung_field": bool(inside_likely_lung_field),
        "border_attention_fraction": border_attention_fraction,
        "artifact_attention_flag": bool(artifact_attention_flag),
        "medical_plausibility": plausibility,
    }
