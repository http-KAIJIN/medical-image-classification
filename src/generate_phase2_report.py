import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR


def main() -> None:
    dataset_report = json.loads((REPORTS_DIR / "dataset_report.json").read_text(encoding="utf-8"))
    split_report = json.loads((REPORTS_DIR / "prepared_split_report.json").read_text(encoding="utf-8"))
    original_distribution = pd.read_csv(REPORTS_DIR / "class_distribution.csv")
    prepared_distribution = pd.read_csv(REPORTS_DIR / "prepared_split_distribution.csv")

    report_path = REPORTS_DIR / "PHASE_2_DATASET_REPORT.md"
    content = f"""# Phase 2 - Dataset Preparation and Validation Report

Official implementation dataset: Chest X-Ray Pneumonia Dataset.

## Dataset Acquisition

Dataset was acquired automatically using `kagglehub` and linked into the project at:

`data/raw/chest_xray`

The project uses a symlink to avoid duplicating the dataset and wasting disk space.

## Raw Dataset Validation

- Dataset available: `{dataset_report['dataset_available']}`
- Total images: `{dataset_report['total_images']}`
- Number of classes: `{dataset_report['number_of_classes']}`
- Classes: `{', '.join(dataset_report['expected_classes'])}`
- Image modes found: `{', '.join(dataset_report['image_stats']['modes'])}`
- Width range: `{dataset_report['image_stats']['min_width']}` to `{dataset_report['image_stats']['max_width']}` pixels
- Height range: `{dataset_report['image_stats']['min_height']}` to `{dataset_report['image_stats']['max_height']}` pixels
- Data quality issues found: `{dataset_report['issues_count']}`

## Original Dataset Split Distribution

```text
{original_distribution.to_string(index=False)}
```

Important issue found: the original validation split contains only `16` images. This is too small for reliable validation during training.

## Prepared Split Strategy

{split_report['strategy']}.

Validation size from original train+val pool: `{split_report['validation_size_from_train_val']}`.

Random seed: `{split_report['random_seed']}`.

The official test split is preserved and not used for validation or model selection.

## Prepared Split Distribution

```text
{prepared_distribution.to_string(index=False)}
```

## Final Counts For Training Pipeline

- Total images: `{split_report['total_images']}`
- Number of classes: `2`
- NORMAL images: `{split_report['class_counts'].get('NORMAL', 0)}`
- PNEUMONIA images: `{split_report['class_counts'].get('PNEUMONIA', 0)}`
- Train count: `{split_report['split_counts'].get('train', 0)}`
- Validation count: `{split_report['split_counts'].get('val', 0)}`
- Test count: `{split_report['split_counts'].get('test', 0)}`

## Generated Evidence Files

- Dataset records: `{dataset_report['records_csv']}`
- Original class distribution: `{dataset_report['class_distribution_csv']}`
- Dataset issues: `{dataset_report['issues_path']}`
- Sample image visualization: `{dataset_report['sample_images_png']}`
- Class distribution plot: `{dataset_report['class_distribution_png']}`
- Prepared train CSV: `{split_report['train_csv']}`
- Prepared validation CSV: `{split_report['val_csv']}`
- Prepared test CSV: `{split_report['test_csv']}`
- Prepared split distribution: `{split_report['prepared_split_distribution_csv']}`

## Data Quality Issues

- No unreadable/corrupted images were found by PIL verification.
- Images have mixed modes: grayscale `L` and `RGB`; preprocessing must convert all images to RGB for ResNet50 and EfficientNetB0.
- Image sizes vary significantly; preprocessing must resize all images to `224x224`.
- Dataset is imbalanced: pneumonia images are much more frequent than normal images; training should use class weights and metrics beyond accuracy.
- Original validation split is too small; prepared split CSV files must be used for training.

## Phase 2 Decision

Dataset preparation is validated for modeling, provided that training uses the prepared CSV splits in `data/splits/` rather than the original folder validation split.
"""
    report_path.write_text(content, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
