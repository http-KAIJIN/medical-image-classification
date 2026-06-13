import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR


MODEL_ORDER = [
    ("custom_cnn", "Custom CNN"),
    ("resnet50", "ResNet50"),
    ("efficientnetb0", "EfficientNetB0"),
]

OLD_RESULTS = {
    "custom_cnn": {
        "accuracy": 0.6394230769230769,
        "precision": 0.634584013050571,
        "recall": 0.9974358974358974,
        "f1_score": 0.7756729810568295,
        "auc_roc": 0.9071005917159763,
        "confusion_matrix": [[10, 224], [1, 389]],
    },
    "resnet50": {
        "accuracy": 0.8862179487179487,
        "precision": 0.8788598574821853,
        "recall": 0.9487179487179487,
        "f1_score": 0.9124537607891492,
        "auc_roc": 0.9583607275914968,
        "confusion_matrix": [[183, 51], [20, 370]],
    },
    "efficientnetb0": {
        "accuracy": 0.8910256410256411,
        "precision": 0.8779342723004695,
        "recall": 0.958974358974359,
        "f1_score": 0.9166666666666666,
        "auc_roc": 0.9574293771914575,
        "confusion_matrix": [[182, 52], [16, 374]],
    },
}

METRICS = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "auc_roc": "AUC-ROC",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def format_metric(value: float) -> str:
    return f"`{value:.6f}`"


def confusion_summary(cm: list[list[int]]) -> str:
    return f"TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}"


def ensure_custom_threshold_analysis() -> None:
    predictions_path = REPORTS_DIR / "custom_cnn_test_predictions.csv"
    df = pd.read_csv(predictions_path)
    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["pneumonia_probability"].to_numpy(dtype=float)
    rows = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(REPORTS_DIR / "custom_cnn_threshold_analysis.csv", index=False)
    best = threshold_df.loc[threshold_df["f1_score"].idxmax()].to_dict()
    write_text(REPORTS_DIR / "custom_cnn_threshold_analysis.json", json.dumps({"best_f1_threshold": best}, indent=2))


def load_clean_results() -> dict:
    results = {}
    for key, label in MODEL_ORDER:
        evaluation = load_json(REPORTS_DIR / f"{key}_evaluation.json")
        threshold = load_json(REPORTS_DIR / f"{key}_threshold_analysis.json")
        results[key] = {
            "label": label,
            "evaluation": evaluation,
            "metrics": evaluation["test_metrics"],
            "threshold": threshold["best_f1_threshold"],
            "confusion_matrix": evaluation["confusion_matrix"],
        }
    return results


def score_models(results: dict) -> list[dict]:
    weights = {"recall": 0.35, "f1_score": 0.25, "auc_roc": 0.20, "accuracy": 0.10, "precision": 0.10}
    rows = []
    for key, data in results.items():
        row = {"model_key": key, "model": data["label"], **{metric: float(data["metrics"][metric]) for metric in METRICS}}
        rows.append(row)
    for row in rows:
        score = 0.0
        for metric, weight in weights.items():
            values = np.array([item[metric] for item in rows], dtype=float)
            if values.max() == values.min():
                normalized = 1.0
            else:
                normalized = (row[metric] - values.min()) / (values.max() - values.min())
            score += normalized * weight
        row["screening_weighted_score"] = float(score)
    rows.sort(key=lambda item: item["screening_weighted_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def comparison_dataframe(results: dict, ranking: list[dict]) -> pd.DataFrame:
    rank_by_key = {row["model_key"]: row for row in ranking}
    rows = []
    for key, data in results.items():
        eval_report = data["evaluation"]
        threshold = data["threshold"]
        row = {
            "model_key": key,
            "model": data["label"],
            **{metric: float(data["metrics"][metric]) for metric in METRICS},
            "confusion_matrix": confusion_summary(data["confusion_matrix"]),
            "best_threshold": float(threshold["threshold"]),
            "best_threshold_f1_score": float(threshold["f1_score"]),
            "parameters": int(eval_report["total_parameters"]),
            "training_minutes": float(eval_report["training_duration_minutes"]),
            "rank": rank_by_key[key]["rank"],
            "screening_weighted_score": rank_by_key[key]["screening_weighted_score"],
            "final_model": eval_report["final_model"],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rank")


def markdown_metrics_table(df: pd.DataFrame) -> str:
    lines = [
        "| Rank | Model | Accuracy | Precision | Recall | F1-score | AUC-ROC | Best F1 threshold | Confusion matrix |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.model} | `{row.accuracy:.6f}` | `{row.precision:.6f}` | `{row.recall:.6f}` | "
            f"`{row.f1_score:.6f}` | `{row.auc_roc:.6f}` | `{row.best_threshold:.2f}` | `{row.confusion_matrix}` |"
        )
    return "\n".join(lines)


def old_new_table(results: dict) -> str:
    lines = [
        "| Model | Metric | Old result | Clean-split result | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for key, data in results.items():
        for metric in METRICS:
            old = OLD_RESULTS[key][metric]
            new = float(data["metrics"][metric])
            lines.append(f"| {data['label']} | {METRIC_LABELS[metric]} | `{old:.6f}` | `{new:.6f}` | `{new - old:+.6f}` |")
    return "\n".join(lines)


def write_clean_split_report(results: dict, df: pd.DataFrame) -> None:
    winner = df.iloc[0]
    content = f"""# Clean-Split Comparison Report

## Objective

This report replaces previous model-selection evidence with results generated after patient/group-level leakage removal. The clean split uses `data/splits/train.csv`, `data/splits/val.csv`, and `data/splits/test.csv`, with zero group/hash overlap documented in `data/reports/LEAKAGE_FIX_REPORT.md`.

## Clean-Split Results

{markdown_metrics_table(df)}

## Old vs Clean-Split Results

{old_new_table(results)}

## Impact of Leakage Removal

Leakage removal changed the validation population and excluded train/validation images that collided with official test images by patient group or hash. The clean split is academically stronger because model selection no longer benefits from duplicated or near-duplicated patient evidence across splits.

The most important scientific effect is confidence calibration, not only headline score movement. Custom CNN improved strongly versus the archived result because the retraining run selected a less degenerate checkpoint with far fewer false positives. Both transfer-learning models moved slightly downward on F1 and AUC, which is consistent with a harder, cleaner validation process.

## Impact on Model Ranking

EfficientNetB0 remains ranked first on clean-split results, but the margin is small. ResNet50 has the best optimized-threshold F1 (`{df.loc[df['model'] == 'ResNet50', 'best_threshold_f1_score'].iloc[0]:.6f}`), while EfficientNetB0 has the best default-threshold accuracy, F1-score, and AUC-ROC among the final clean runs.

## Impact on Confidence

Confidence in the final academic claim increased because leakage was explicitly removed and verified. Confidence in model superiority should remain moderate, not absolute, because EfficientNetB0 and ResNet50 differ by only `{winner.f1_score - df.loc[df['model'] == 'ResNet50', 'f1_score'].iloc[0]:+.6f}` F1 at threshold 0.50 and share the same false-negative count in this run.
"""
    write_text(PROJECT_ROOT / "CLEAN_SPLIT_COMPARISON_REPORT.md", content)


def write_final_selection_report(df: pd.DataFrame) -> None:
    winner = df.iloc[0]
    resnet = df.loc[df["model"] == "ResNet50"].iloc[0]
    custom = df.loc[df["model"] == "Custom CNN"].iloc[0]
    content = f"""# Final Model Selection

## Winner

The final selected model is **{winner.model}**, using `{winner.final_model}` with production threshold `0.50`.

## Clean-Split Justification

{markdown_metrics_table(df)}

EfficientNetB0 is selected because it has the strongest clean-split default-threshold profile: accuracy `{winner.accuracy:.6f}`, F1-score `{winner.f1_score:.6f}`, AUC-ROC `{winner.auc_roc:.6f}`, and confusion matrix `{winner.confusion_matrix}`. ResNet50 remains a credible academic competitor, but its default-threshold accuracy and F1 are lower in the clean run. Custom CNN is useful as a baseline but is not competitive with transfer learning.

## Limitations

- The performance difference between EfficientNetB0 and ResNet50 is small and should not be overstated.
- The dataset is pediatric chest X-ray data, so generalization to adult populations is not established.
- The task is binary and does not distinguish viral pneumonia, bacterial pneumonia, other lung diseases, or poor image quality.
- The official test set is modest in size, so confidence intervals would be useful in future work.
- Grad-CAM is explanatory support, not clinical proof of causal reasoning.

## Deployment Recommendation

Use EfficientNetB0 as a screening-support model only, not as an autonomous diagnostic tool. Keep threshold `0.50` for the submitted application because it preserves a high recall of `{winner.recall:.6f}` while maintaining better precision and specificity than the Custom CNN baseline. Any real deployment would require external validation, radiologist review, monitoring for domain shift, and local threshold calibration.

## Rejected Alternatives

- ResNet50: F1 `{resnet.f1_score:.6f}` and AUC `{resnet.auc_roc:.6f}` are strong, but it is much larger and lower than EfficientNetB0 at threshold `0.50` in this clean run.
- Custom CNN: F1 `{custom.f1_score:.6f}` and AUC `{custom.auc_roc:.6f}` improved after retraining, but it remains below both transfer-learning models.
"""
    write_text(PROJECT_ROOT / "FINAL_MODEL_SELECTION.md", content)


def write_phase8_report(df: pd.DataFrame) -> None:
    winner = df.iloc[0]
    content = f"""# Phase 8 - Final Clean-Split Model Comparison and Model Selection

## Objective

This report compares all retrained models using the leakage-fixed clean split and preserved official test set.

## Final Comparison Table

{markdown_metrics_table(df)}

## Final Decision

The final selected model is **{winner.model}** with model file `{winner.final_model}` and production threshold `0.50`.

## Decision Basis

- EfficientNetB0 has the best clean-split default-threshold accuracy, precision, F1-score, and AUC-ROC in the final run.
- EfficientNetB0 and ResNet50 have the same default-threshold recall and false-negative count, so the smaller, stronger default-threshold model is preferred.
- ResNet50 has a slightly lower default-threshold F1 and much larger parameter count.
- Custom CNN remains an important baseline but ranks third on all core clean-split metrics.

## Threshold Analysis

| Model | Threshold 0.50 F1 | Best F1 threshold | Best threshold F1 |
|---|---:|---:|---:|
"""
    for row in df.itertuples(index=False):
        content += f"| {row.model} | `{row.f1_score:.6f}` | `{row.best_threshold:.2f}` | `{row.best_threshold_f1_score:.6f}` |\n"
    content += """

The production threshold remains `0.50` for reproducibility and screening-oriented recall. Best-F1 thresholds are reported as analysis, not as post-hoc deployment tuning.

## Limitations

- The difference between EfficientNetB0 and ResNet50 is small and should be reported cautiously.
- The dataset is pediatric and single-source.
- Binary labels do not support subtype classification.
- Grad-CAM supports interpretation but does not prove clinical reasoning.
"""
    write_text(REPORTS_DIR / "PHASE_8_FINAL_COMPARISON_REPORT.md", content)


def presentation_package(df: pd.DataFrame) -> None:
    presentation_dir = PROJECT_ROOT / "presentation"
    winner = df.iloc[0]
    outline = f"""# Presentation Outline

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

Final model: `{winner.final_model}`.
"""
    write_text(presentation_dir / "presentation_outline.md", outline)

    notes = f"""# Speaker Notes

## Opening

This project builds a pneumonia screening-support classifier for pediatric chest X-rays. The final submission emphasizes scientific validity by correcting data leakage before final model selection.

## Dataset and Splitting

The dataset contains NORMAL and PNEUMONIA X-rays. The final split is patient/group-aware, preserves the official test set, and removes train/validation images that collide with test groups or hashes.

## Models

Custom CNN provides a lightweight baseline. ResNet50 and EfficientNetB0 provide transfer-learning baselines using ImageNet initialization, frozen-head training, and fine tuning.

## Results

EfficientNetB0 is the final winner with accuracy `{winner.accuracy:.6f}`, recall `{winner.recall:.6f}`, F1 `{winner.f1_score:.6f}`, and AUC `{winner.auc_roc:.6f}`.

## Grad-CAM

Grad-CAM examples are used to show influential regions, especially for correct pneumonia, correct normal, false positive, and false negative cases. They are not clinical proof.

## Closing

The final model is academically defensible as a screening-support prototype, but external validation and radiologist review are required before any clinical use.
"""
    write_text(presentation_dir / "speaker_notes.md", notes)

    questions = [
        "Why was leakage a serious issue?",
        "How did you prevent patient-level leakage?",
        "Why preserve the official test set?",
        "Why use transfer learning?",
        "Why compare Custom CNN, ResNet50, and EfficientNetB0?",
        "Why did EfficientNetB0 win?",
        "Is the difference between EfficientNetB0 and ResNet50 statistically significant?",
        "Why use threshold 0.50 instead of the best-F1 threshold?",
        "Why prioritize recall in pneumonia screening?",
        "What are false negatives and why are they dangerous?",
        "What are false positives and why do they matter?",
        "What does AUC-ROC tell you?",
        "What does Grad-CAM prove?",
        "Can this model diagnose pneumonia clinically?",
        "Does the pediatric dataset limit generalization?",
        "Can the model distinguish viral and bacterial pneumonia?",
        "How did you handle class imbalance?",
        "What preprocessing was applied?",
        "Why not train more epochs?",
        "What would you improve with more time?",
        "How would you validate this externally?",
        "What are the deployment risks?",
    ]
    write_text(presentation_dir / "likely_questions.md", "# Likely Jury Questions\n\n" + "\n".join(f"{idx}. {q}" for idx, q in enumerate(questions, 1)))

    answers = """# Answers

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
"""
    write_text(presentation_dir / "answers.md", answers)


def defense_preparation(df: pd.DataFrame) -> None:
    winner = df.iloc[0]
    content = f"""# Defense Preparation

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

The strongest defensible claim is: after leakage removal and clean-split retraining, `{winner.model}` is the best model in this project for a prototype pediatric pneumonia screening application, but it is not clinically deployable without external validation.
"""
    write_text(PROJECT_ROOT / "presentation" / "defense_preparation.md", content)


def update_notebook(df: pd.DataFrame) -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "training_notebook.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    marker = "<!-- FINAL_CLEAN_SPLIT_UPDATE -->"
    notebook.cells = [cell for cell in notebook.cells if marker not in cell.get("source", "")]
    winner = df.iloc[0]
    source = f"""{marker}
# Final Clean-Split Update

The final training run uses the leakage-fixed split from `data/splits/`. Patient/group and hash overlap were removed before model selection, while the official test set was preserved.

## Clean-Split Metrics

{markdown_metrics_table(df)}

## Final Winner

The final selected model is **{winner.model}** with accuracy `{winner.accuracy:.6f}`, recall `{winner.recall:.6f}`, F1-score `{winner.f1_score:.6f}`, and AUC-ROC `{winner.auc_roc:.6f}`.

## Leakage Discussion

Previous model-selection results are treated as outdated because patient-level or duplicate leakage can inflate validation evidence. Final conclusions use only the clean-split retraining artifacts.

## Pediatric Dataset Limitation

The dataset is pediatric chest X-ray data. Results should not be generalized to adult populations, other hospitals, different machines, or other acquisition protocols without external validation.

## Figure References

- Training histories: `results/figures/custom_cnn_training_curves.png`, `results/figures/resnet50_training_curves.png`, `results/figures/efficientnetb0_training_curves.png`
- Confusion matrices: `results/confusion_matrices/`
- ROC curves: `results/roc_curves/`
- Grad-CAM summary: `results/figures/gradcam_summary_grid.png`
"""
    notebook.cells.append(nbformat.v4.new_markdown_cell(source))
    nbformat.write(notebook, notebook_path)


def final_readiness_report(df: pd.DataFrame) -> None:
    def json_success(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            return bool(json.loads(path.read_text(encoding="utf-8")).get("success"))
        except Exception:
            return False

    checks = {
        "notebook_exists": PROJECT_ROOT.joinpath("notebooks", "training_notebook.ipynb").exists(),
        "clean_split_report_exists": PROJECT_ROOT.joinpath("CLEAN_SPLIT_COMPARISON_REPORT.md").exists(),
        "final_selection_report_exists": PROJECT_ROOT.joinpath("FINAL_MODEL_SELECTION.md").exists(),
        "model_files_exist": all(MODELS_DIR.joinpath(f"{key}_final.keras").exists() for key, _ in MODEL_ORDER),
        "confusion_matrices_exist": all(RESULTS_DIR.joinpath("confusion_matrices", f"{key}_confusion_matrix.png").exists() for key, _ in MODEL_ORDER),
        "roc_curves_exist": all(RESULTS_DIR.joinpath("roc_curves", f"{key}_roc_curve.png").exists() for key, _ in MODEL_ORDER),
        "training_histories_exist": all(REPORTS_DIR.joinpath(f"{key}_training_history.csv").exists() for key, _ in MODEL_ORDER),
        "frontend_exists": PROJECT_ROOT.joinpath("frontend", "index.html").exists() and PROJECT_ROOT.joinpath("frontend", "app.js").exists(),
        "backend_exists": PROJECT_ROOT.joinpath("backend", "main.py").exists(),
        "screenshots_exist": PROJECT_ROOT.joinpath("results", "frontend", "frontend_01_initial.png").exists(),
        "presentation_package_exists": all(PROJECT_ROOT.joinpath("presentation", name).exists() for name in ["presentation_outline.md", "speaker_notes.md", "likely_questions.md", "answers.md", "defense_preparation.md"]),
        "gradcam_grid_exists": PROJECT_ROOT.joinpath("results", "figures", "gradcam_summary_grid.png").exists(),
        "backend_validation_passed": json_success(REPORTS_DIR / "api_test_results.json"),
        "frontend_validation_passed": json_success(REPORTS_DIR / "frontend_test_results.json"),
        "integration_validation_passed": json_success(REPORTS_DIR / "end_to_end_test_results.json"),
    }
    readiness = sum(checks.values()) / len(checks) * 100
    winner = df.iloc[0]
    content = "# Final Readiness Report\n\n"
    content += f"Readiness percentage: `{readiness:.1f}%`\n\n"
    content += f"Recommended final model: **{winner.model}** (`{winner.final_model}`)\n\n"
    content += "## Audit Checks\n\n| Check | Status |\n|---|---:|\n"
    for key, value in checks.items():
        content += f"| {key} | `{value}` |\n"
    content += "\n## Remaining Issues\n\n"
    missing = [key for key, value in checks.items() if not value]
    content += "- None detected in automated file-level audit.\n" if not missing else "\n".join(f"- {item}" for item in missing) + "\n"
    content += "\n## Recommended Files to Submit\n\n"
    for path in [
        "notebooks/training_notebook.ipynb",
        "CLEAN_SPLIT_COMPARISON_REPORT.md",
        "FINAL_MODEL_SELECTION.md",
        "FINAL_READINESS_REPORT.md",
        "models/efficientnetb0_final.keras",
        "data/reports/final_model_comparison.csv",
        "results/figures/gradcam_summary_grid.png",
        "presentation/",
        "backend/",
        "frontend/",
        "README.md",
    ]:
        content += f"- `{path}`\n"
    write_text(PROJECT_ROOT / "FINAL_READINESS_REPORT.md", content)


def main() -> None:
    ensure_custom_threshold_analysis()
    results = load_clean_results()
    ranking = score_models(results)
    df = comparison_dataframe(results, ranking)
    df.to_csv(REPORTS_DIR / "final_model_comparison.csv", index=False)
    write_text(REPORTS_DIR / "final_model_comparison.json", json.dumps({"ranking": ranking, "comparison": df.to_dict(orient="records")}, indent=2))
    write_clean_split_report(results, df)
    write_final_selection_report(df)
    write_phase8_report(df)
    presentation_package(df)
    defense_preparation(df)
    update_notebook(df)
    final_readiness_report(df)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
