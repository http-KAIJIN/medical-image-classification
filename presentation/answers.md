# Answers

1. Leakage can inflate validation performance by exposing related patient images across splits, making the model appear more generalizable than it really is.
2. I grouped images by patient-like filename identifiers and hashes, then split groups deterministically with zero overlap.
3. Preserving the official test set keeps the benchmark comparable while cleaning train/validation contamination.
4. Transfer learning is appropriate because medical datasets are limited and ImageNet backbones provide useful low-level visual features.
5. The three models test increasing capacity and prior knowledge: a baseline CNN, a large residual model, and an efficient modern backbone.
6. EfficientNetB0 won on clean-split default-threshold accuracy, F1, AUC, and deployability.
7. I would not claim strong statistical significance without confidence intervals; the margin is small.
8. Threshold 0.50 is simple, reproducible, and preserves high screening recall; optimized thresholds can overfit the test set if used carelessly.
9. Recall is prioritized because missed pneumonia is the more serious screening failure.
10. False negatives are pneumonia cases predicted normal, which could delay clinical review.
11. False positives burden review workflows but are safer than missed pneumonia in a screening context.
12. AUC-ROC measures ranking discrimination across thresholds, independent of one fixed threshold.
13. Grad-CAM shows influential image regions but does not prove medical causality or radiologist-like reasoning.
14. No. It is a decision-support prototype, not an autonomous diagnostic device.
15. Yes. Pediatric-only data limits claims for adults and different hospitals.
16. No. The labels are binary NORMAL/PNEUMONIA only.
17. Class weights were used during training to compensate for class imbalance.
18. Images were decoded, resized to 224x224, normalized, and augmented safely during training.
19. Early stopping and validation monitoring reduced overfitting risk.
20. Add external validation, confidence intervals, calibration, segmentation/quality checks, and stronger clinical review.
21. Test on images from another hospital or dataset without retraining, then recalibrate only if clinically justified.
22. Risks include domain shift, hidden confounders, poor calibration, missed pneumonia, and misuse as a diagnostic replacement.
