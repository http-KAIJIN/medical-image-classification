from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from backend.config import ALLOWED_EXTENSIONS, IMAGE_SIZE


class ImageValidationError(ValueError):
    pass


async def read_validated_image(upload: UploadFile) -> tuple[bytes, Image.Image]:
    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Invalid image format")

    data = await upload.read()
    if not data:
        raise ImageValidationError("Empty image upload")

    try:
        image = Image.open(BytesIO(data))
        image.verify()
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Corrupted image file") from exc

    return data, image


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def preprocess_image_bytes(data: bytes) -> np.ndarray:
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()
