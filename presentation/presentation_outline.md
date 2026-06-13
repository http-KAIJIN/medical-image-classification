# Presentation Outline

1. Project objective: binary pneumonia screening from chest X-ray images.
2. Dataset: pediatric Kaggle chest X-ray dataset with NORMAL and PNEUMONIA classes.
3. Leakage correction: patient/group-aware split and official test preservation.
4. Preprocessing: resize to 224x224, RGB conversion, normalization, medical-safe augmentation.
5. Custom CNN baseline: architecture, role, and clean-split performance.
6. ResNet50 transfer learning: frozen head, fine tuning, performance.
7. EfficientNetB0 transfer learning: frozen head, fine tuning, final selected model.
8. Clean-split comparison: show metrics table and ranking.
9. Threshold analysis: explain default threshold and best-F1 thresholds.
10. Grad-CAM: correct and error-case interpretability examples.
11. Limitations: pediatric dataset, binary labels, single public dataset, Grad-CAM limits.
12. Conclusion: EfficientNetB0 selected for final application and academic submission.

Final model: `/home/Oussama/medical-image-classification/models/efficientnetb0_final.keras`.
