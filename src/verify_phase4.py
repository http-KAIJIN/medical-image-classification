import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR
from src.preprocessing import (
    build_dataset,
    compute_class_weights,
    load_split_dataframe,
    make_augmentation_model,
    preprocess_for_inference,
)


def save_augmented_samples(train_df: pd.DataFrame) -> str:
    sample_df = train_df.groupby("label", group_keys=False).head(3).copy()
    base_ds = build_dataset(sample_df, batch_size=len(sample_df), training=False, augment=False)
    images, labels = next(iter(base_ds))
    augmentation = make_augmentation_model()
    augmented = augmentation(images, training=True)
    augmented = augmented.numpy().clip(0.0, 1.0)

    n = len(sample_df)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for idx in range(n):
        label = sample_df.iloc[idx]["label"]
        axes[0, idx].imshow(images[idx].numpy())
        axes[0, idx].set_title(f"Original\n{label}")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(augmented[idx])
        axes[1, idx].set_title("Augmented")
        axes[1, idx].axis("off")
    fig.tight_layout()
    output_path = FIGURES_DIR / "phase4_augmented_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split_dataframe("train")
    val_df = load_split_dataframe("val")
    test_df = load_split_dataframe("test")
    class_weights = compute_class_weights(train_df)

    train_ds = build_dataset(train_df, training=True, augment=True)
    val_ds = build_dataset(val_df, training=False, augment=False)
    test_ds = build_dataset(test_df, training=False, augment=False)

    train_images, train_labels = next(iter(train_ds))
    val_images, val_labels = next(iter(val_ds))
    test_images, test_labels = next(iter(test_ds))

    sample_path = save_augmented_samples(train_df)
    inference_sample = preprocess_for_inference(train_df.iloc[0]["filepath"])

    report = {
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "class_weight_labels": {"0": "NORMAL", "1": "PNEUMONIA"},
        "train_shape": list(train_images.shape),
        "val_shape": list(val_images.shape),
        "test_shape": list(test_images.shape),
        "train_label_shape": list(train_labels.shape),
        "val_label_shape": list(val_labels.shape),
        "test_label_shape": list(test_labels.shape),
        "train_value_range": [float(train_images.numpy().min()), float(train_images.numpy().max())],
        "val_value_range": [float(val_images.numpy().min()), float(val_images.numpy().max())],
        "test_value_range": [float(test_images.numpy().min()), float(test_images.numpy().max())],
        "inference_shape": list(inference_sample.shape),
        "inference_value_range": [float(inference_sample.numpy().min()), float(inference_sample.numpy().max())],
        "augmentation_applied_to_train_only": True,
        "validation_augmented": False,
        "test_augmented": False,
        "preprocessing_shared_with_inference": "decode_and_preprocess_image is used by both dataset pipeline and preprocess_for_inference",
        "augmented_samples_png": sample_path,
    }

    report_path = REPORTS_DIR / "phase4_preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = REPORTS_DIR / "PHASE_4_PREPROCESSING_REPORT.md"
    md_path.write_text(
        f"""# Phase 4 - Preprocessing and Augmentation Report

## Class Weights

- `0` NORMAL: `{class_weights[0]:.6f}`
- `1` PNEUMONIA: `{class_weights[1]:.6f}`

## Dataset Tensor Verification

- Train batch shape: `{list(train_images.shape)}`
- Validation batch shape: `{list(val_images.shape)}`
- Test batch shape: `{list(test_images.shape)}`
- Inference sample shape: `{list(inference_sample.shape)}`

## Pixel Normalization

- Train value range: `{report['train_value_range']}`
- Validation value range: `{report['val_value_range']}`
- Test value range: `{report['test_value_range']}`
- Inference value range: `{report['inference_value_range']}`

## Augmentation Policy

- Training augmented: `True`
- Validation augmented: `False`
- Test augmented: `False`

Augmentations used: horizontal flip, small rotation, zoom, translation, contrast adjustment.

## Shared Inference Preprocessing

Training/evaluation and future inference both use `decode_and_preprocess_image`, ensuring RGB decoding, resize to `224x224`, and pixel normalization to `[0, 1]`.

## Generated Visualization

- `{sample_path}`
""",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
