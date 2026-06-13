from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.config import BATCH_SIZE, CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED, SPLITS_DIR


AUTOTUNE = tf.data.AUTOTUNE
LABEL_TO_INDEX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {idx: class_name for class_name, idx in LABEL_TO_INDEX.items()}


def load_split_dataframe(split_name: str, splits_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    csv_path = splits_dir / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["readable"] == True].copy()  # noqa: E712
    df["label_index"] = df["label"].map(LABEL_TO_INDEX).astype("int32")
    return df


def compute_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    counts = train_df["label_index"].value_counts().to_dict()
    total = len(train_df)
    num_classes = len(CLASS_NAMES)
    return {class_idx: total / (num_classes * counts[class_idx]) for class_idx in sorted(counts)}


def decode_and_preprocess_image(filepath: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def make_augmentation_model() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
            tf.keras.layers.RandomRotation(0.04, fill_mode="nearest", seed=RANDOM_SEED),
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest", seed=RANDOM_SEED),
            tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="nearest", seed=RANDOM_SEED),
            tf.keras.layers.RandomContrast(0.08, seed=RANDOM_SEED),
        ],
        name="medical_safe_augmentation",
    )


def build_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    filepaths = df["filepath"].astype(str).values
    labels = df["label_index"].astype("float32").values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    def load_item(filepath: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = decode_and_preprocess_image(filepath)
        return image, tf.expand_dims(label, axis=-1)

    dataset = dataset.map(load_item, num_parallel_calls=AUTOTUNE)

    if augment:
        augmentation = make_augmentation_model()

        def augment_item(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            image = augmentation(image, training=True)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

        dataset = dataset.map(augment_item, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def load_prepared_datasets(batch_size: int = BATCH_SIZE) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[int, float], dict[str, pd.DataFrame]]:
    train_df = load_split_dataframe("train")
    val_df = load_split_dataframe("val")
    test_df = load_split_dataframe("test")
    class_weights = compute_class_weights(train_df)

    train_ds = build_dataset(train_df, batch_size=batch_size, training=True, augment=True)
    val_ds = build_dataset(val_df, batch_size=batch_size, training=False, augment=False)
    test_ds = build_dataset(test_df, batch_size=batch_size, training=False, augment=False)

    return train_ds, val_ds, test_ds, class_weights, {"train": train_df, "val": val_df, "test": test_df}


def preprocess_for_inference(image_path: str | Path) -> tf.Tensor:
    image = decode_and_preprocess_image(tf.constant(str(image_path)))
    return tf.expand_dims(image, axis=0)
