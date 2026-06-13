# Defense Preparation

## Strict Evaluation: Weaknesses and Risks

- The dataset is pediatric and from a public benchmark, so external generalization is unproven.
- The final winner margin over ResNet50 is small.
- The official test set has only 624 images.
- Binary labels hide disease subtype and comorbidity complexity.
- Grad-CAM is low-resolution and can be misleading.
- Image-level labels do not provide pathology localization.
- Dataset acquisition artifacts may influence predictions.
- Threshold `0.50` may not be clinically optimal outside this dataset.
- No calibration curve or confidence interval is included in the core training scripts.
- The frontend/backend demonstrate a prototype, not regulated medical software.

## Difficult Questions and Strong Responses

| Criticism | Strong response |
|---|---|
| Your old results were contaminated by leakage. | Correct, and that is why final claims are based only on clean patient/group-aware retraining. The leakage fix is documented and verified with zero group/hash overlap. |
| EfficientNetB0 barely beats ResNet50. | I do not overclaim. EfficientNetB0 is selected because it has the best clean default-threshold profile and smaller size, but ResNet50 remains a close benchmark. |
| The model is not clinically validated. | Correct. I present it as a screening-support academic prototype requiring external validation and radiologist review before clinical use. |
| Grad-CAM does not prove the model sees pneumonia. | Correct. I use Grad-CAM only as interpretability support and explicitly state its limitations. |
| Pediatric data cannot generalize to adults. | Correct. The limitation is included in the notebook, final selection, and defense material. |
| Why not use the best-F1 threshold? | The best-F1 threshold is reported for analysis, but threshold `0.50` is kept for reproducibility and high screening recall. |

## Final Defense Position

The strongest defensible claim is: after leakage removal and clean-split retraining, `EfficientNetB0` is the best model in this project for a prototype pediatric pneumonia screening application, but it is not clinically deployable without external validation.
