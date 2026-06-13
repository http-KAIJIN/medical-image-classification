# Speaker Notes

## Opening

This project builds a pneumonia screening-support classifier for pediatric chest X-rays. The final submission emphasizes scientific validity by correcting data leakage before final model selection.

## Dataset and Splitting

The dataset contains NORMAL and PNEUMONIA X-rays. The final split is patient/group-aware, preserves the official test set, and removes train/validation images that collide with test groups or hashes.

## Models

Custom CNN provides a lightweight baseline. ResNet50 and EfficientNetB0 provide transfer-learning baselines using ImageNet initialization, frozen-head training, and fine tuning.

## Results

EfficientNetB0 is the final winner with accuracy `0.887821`, recall `0.956410`, F1 `0.914216`, and AUC `0.957868`.

## Grad-CAM

Grad-CAM examples are used to show influential regions, especially for correct pneumonia, correct normal, false positive, and false negative cases. They are not clinical proof.

## Closing

The final model is academically defensible as a screening-support prototype, but external validation and radiologist review are required before any clinical use.
