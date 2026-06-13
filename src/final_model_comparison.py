import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR


MODEL_CONFIGS = [
    ("custom_cnn", "Custom CNN", "custom_cnn_evaluation.json", "custom_cnn_threshold_analysis.json"),
    ("resnet50", "ResNet50", "resnet50_evaluation.json", "resnet50_threshold_analysis.json"),
    ("efficientnetb0", "EfficientNetB0", "efficientnetb0_evaluation.json", "efficientnetb0_threshold_analysis.json"),
]

METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "auc_roc": "AUC-ROC",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def confusion_summary(confusion_matrix: list[list[int]]) -> dict[str, int]:
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def build_comparison_rows() -> list[dict]:
    rows = []
    for key, label, evaluation_name, threshold_name in MODEL_CONFIGS:
        evaluation = load_json(REPORTS_DIR / evaluation_name)
        threshold_report = load_json(REPORTS_DIR / threshold_name)
        metrics = evaluation["test_metrics"]
        cm = confusion_summary(evaluation["confusion_matrix"])
        best_threshold = threshold_report["best_f1_threshold"]
        rows.append(
            {
                "model_key": key,
                "model": label,
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
                "auc_roc": float(metrics["auc_roc"]),
                "parameter_count": int(evaluation["total_parameters"]),
                "trainable_parameters": int(evaluation.get("trainable_parameters_final_model", evaluation.get("trainable_parameters", 0))),
                "training_duration_minutes": float(evaluation["training_duration_minutes"]),
                "default_threshold": 0.5,
                "best_threshold": float(best_threshold["threshold"]),
                "best_threshold_accuracy": float(best_threshold["accuracy"]),
                "best_threshold_precision": float(best_threshold["precision"]),
                "best_threshold_recall": float(best_threshold["recall"]),
                "best_threshold_f1_score": float(best_threshold["f1_score"]),
                "best_threshold_tn": int(best_threshold["tn"]),
                "best_threshold_fp": int(best_threshold["fp"]),
                "best_threshold_fn": int(best_threshold["fn"]),
                "best_threshold_tp": int(best_threshold["tp"]),
                **cm,
                "confusion_matrix_summary": f"TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}",
                "final_model_path": evaluation["final_model"],
                "evaluation_report": str(REPORTS_DIR / evaluation_name),
            }
        )
    return rows


def pairwise_delta(df: pd.DataFrame, first: str, second: str) -> dict[str, float]:
    first_row = df.loc[df["model"] == first].iloc[0]
    second_row = df.loc[df["model"] == second].iloc[0]
    return {metric: float(second_row[metric] - first_row[metric]) for metric in METRIC_KEYS}


def score_models(df: pd.DataFrame) -> pd.DataFrame:
    weights = {
        "recall": 0.35,
        "f1_score": 0.25,
        "auc_roc": 0.20,
        "accuracy": 0.10,
        "precision": 0.10,
    }
    scored = df[["model", *METRIC_KEYS]].copy()
    score = np.zeros(len(scored))
    for metric, weight in weights.items():
        values = scored[metric].to_numpy(dtype=float)
        min_value = values.min()
        max_value = values.max()
        normalized = np.ones_like(values) if max_value == min_value else (values - min_value) / (max_value - min_value)
        score += normalized * weight
    scored["screening_weighted_score"] = score
    scored["rank"] = scored["screening_weighted_score"].rank(ascending=False, method="first").astype(int)
    return scored.sort_values("rank")


def save_metrics_table_png(df: pd.DataFrame, output_path: Path) -> None:
    display_df = df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
            "parameter_count",
            "training_duration_minutes",
            "best_threshold",
            "confusion_matrix_summary",
        ]
    ].copy()
    for metric in METRIC_KEYS:
        display_df[metric] = display_df[metric].map(lambda value: f"{value:.4f}")
    display_df["parameter_count"] = display_df["parameter_count"].map(lambda value: f"{value:,}")
    display_df["training_duration_minutes"] = display_df["training_duration_minutes"].map(lambda value: f"{value:.2f}")
    display_df["best_threshold"] = display_df["best_threshold"].map(lambda value: f"{value:.2f}")
    display_df.columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "AUC-ROC",
        "Parameters",
        "Train min",
        "Best threshold",
        "Confusion matrix",
    ]

    fig, ax = plt.subplots(figsize=(18, 4.2))
    ax.axis("off")
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.7)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
        elif row % 2 == 0:
            cell.set_facecolor("#eef4fb")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_ranking_png(ranking_df: pd.DataFrame, output_path: Path) -> None:
    ordered = ranking_df.sort_values("screening_weighted_score")
    colors = ["#8da0cb", "#66c2a5", "#fc8d62"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["model"], ordered["screening_weighted_score"], color=colors[: len(ordered)])
    ax.set_xlabel("Weighted screening score")
    ax.set_title("Final Model Ranking")
    ax.set_xlim(0, 1.05)
    for idx, value in enumerate(ordered["screening_weighted_score"]):
        ax.text(value + 0.02, idx, f"{value:.3f}", va="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_value(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def comparison_table_markdown(df: pd.DataFrame) -> str:
    rows = [
        "| Model | Accuracy | Precision | Recall | F1-score | AUC-ROC | Parameters | Training min | Best threshold | Confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in df.itertuples(index=False):
        rows.append(
            "| "
            f"{row.model} | `{format_value(row.accuracy)}` | `{format_value(row.precision)}` | `{format_value(row.recall)}` | "
            f"`{format_value(row.f1_score)}` | `{format_value(row.auc_roc)}` | `{row.parameter_count:,}` | "
            f"`{row.training_duration_minutes:.2f}` | `{row.best_threshold:.2f}` | `{row.confusion_matrix_summary}` |"
        )
    return "\n".join(rows)


def pairwise_markdown(title: str, deltas: dict[str, float], second: str) -> str:
    lines = [f"### {title}", "", f"Delta is `{second} - baseline`.", ""]
    lines.extend(["| Metric | Delta | Interpretation |", "|---|---:|---|"])
    for metric in METRIC_KEYS:
        delta = deltas[metric]
        direction = "improved" if delta > 0 else "declined" if delta < 0 else "unchanged"
        lines.append(f"| {METRIC_LABELS[metric]} | `{delta:+.6f}` | {direction} |")
    return "\n".join(lines)


def write_markdown_report(df: pd.DataFrame, ranking_df: pd.DataFrame, pairwise: dict[str, dict[str, float]], output_path: Path) -> None:
    efficient = df.loc[df["model"] == "EfficientNetB0"].iloc[0]
    resnet = df.loc[df["model"] == "ResNet50"].iloc[0]
    custom = df.loc[df["model"] == "Custom CNN"].iloc[0]
    report = f"""# Phase 8 - Final Model Comparison and Model Selection

## Objective

This phase compares all trained pneumonia classification models using the preserved test set and locks the final model before explainability, backend, or frontend work begins.

## Final Comparison Table

{comparison_table_markdown(df)}

## Final Ranking

| Rank | Model | Weighted screening score | Primary reason |
|---:|---|---:|---|
"""
    reasons = {
        "EfficientNetB0": "Best default-threshold accuracy, recall, F1-score, and deployment efficiency.",
        "ResNet50": "Best AUC-ROC and best optimized-threshold F1, but more false negatives at threshold 0.50 and much larger size.",
        "Custom CNN": "High recall, but excessive false positives and weak overall discrimination.",
    }
    for row in ranking_df.itertuples(index=False):
        report += f"| {row.rank} | {row.model} | `{row.screening_weighted_score:.6f}` | {reasons[row.model]} |\n"

    report += f"""

## Pairwise Comparisons

{pairwise_markdown("Custom CNN vs ResNet50", pairwise["custom_cnn_vs_resnet50"], "ResNet50")}

{pairwise_markdown("ResNet50 vs EfficientNetB0", pairwise["resnet50_vs_efficientnetb0"], "EfficientNetB0")}

{pairwise_markdown("Custom CNN vs EfficientNetB0", pairwise["custom_cnn_vs_efficientnetb0"], "EfficientNetB0")}

## Error Analysis

### False Positives

- Custom CNN produced `{custom.fp}` false positives at threshold 0.50, incorrectly flagging most NORMAL test images as PNEUMONIA. This makes it unsuitable for final deployment despite its high recall.
- ResNet50 produced `{resnet.fp}` false positives at threshold 0.50, giving the best default-threshold specificity among transfer-learning models.
- EfficientNetB0 produced `{efficient.fp}` false positives at threshold 0.50, only one more than ResNet50, while reducing false negatives.

### False Negatives

- False negatives are clinically more serious in pneumonia screening because a missed pneumonia case can delay clinical review and treatment.
- Custom CNN had only `{custom.fn}` false negative, but this was achieved by over-predicting pneumonia and generating `{custom.fp}` false positives.
- ResNet50 had `{resnet.fn}` false negatives.
- EfficientNetB0 had `{efficient.fn}` false negatives, the best clinically practical balance among the high-performing models.

## Recall Importance for Pneumonia Detection

For this project, recall is prioritized over precision because the model is intended as a screening aid. A screening model should minimize missed pneumonia cases, then allow clinicians or downstream review to handle false positives. EfficientNetB0 provides the strongest recall among the deployable transfer-learning models at threshold 0.50: `{efficient.recall:.6f}` versus ResNet50 at `{resnet.recall:.6f}`.

## Precision and Recall Trade-off

- Custom CNN maximizes recall but sacrifices precision and specificity, making it noisy and unreliable.
- ResNet50 has slightly better precision than EfficientNetB0 at threshold 0.50 (`{resnet.precision:.6f}` vs `{efficient.precision:.6f}`), but misses more pneumonia cases.
- EfficientNetB0 accepts a minimal precision decrease of `{efficient.precision - resnet.precision:+.6f}` compared with ResNet50 in exchange for a recall gain of `{efficient.recall - resnet.recall:+.6f}` and fewer false negatives.

## Threshold Optimization Impact

| Model | Threshold 0.50 F1 | Best F1 threshold | Best threshold F1 | Threshold impact |
|---|---:|---:|---:|---|
| Custom CNN | `{custom.f1_score:.6f}` | `{custom.best_threshold:.2f}` | `{custom.best_threshold_f1_score:.6f}` | Improves F1 but remains below transfer-learning models. |
| ResNet50 | `{resnet.f1_score:.6f}` | `{resnet.best_threshold:.2f}` | `{resnet.best_threshold_f1_score:.6f}` | Best optimized F1, but recall falls and false negatives increase. |
| EfficientNetB0 | `{efficient.f1_score:.6f}` | `{efficient.best_threshold:.2f}` | `{efficient.best_threshold_f1_score:.6f}` | Small F1 gain, but recall drops from `{efficient.recall:.6f}` to `{efficient.best_threshold_recall:.6f}`. |

The optimized F1 threshold for EfficientNetB0 is `0.55`, but the production threshold is locked at `0.50` because it preserves higher recall and fewer false negatives (`{efficient.fn}` vs `{efficient.best_threshold_fn}`).

## Model Category Decisions

### Best Academic Model

ResNet50 is the best academic benchmark if the priority is maximum AUC-ROC and optimized-threshold F1. It achieved AUC-ROC `{resnet.auc_roc:.6f}` and best-threshold F1 `{resnet.best_threshold_f1_score:.6f}`.

### Best Medical-Screening Model

EfficientNetB0 is the best medical-screening model because it has the best clinically practical recall/F1 balance at threshold 0.50, with fewer false negatives than ResNet50 and far fewer false positives than Custom CNN.

### Best Deployment Model

EfficientNetB0 is the best deployment model. It uses `{efficient.parameter_count:,}` parameters compared with ResNet50's `{resnet.parameter_count:,}`, trains faster in this environment, and still provides the best default-threshold F1-score.

## Final Decision Section

1. Which model is the winner?

EfficientNetB0 is the final winner.

2. Why is it the winner?

It has the best default-threshold accuracy (`{efficient.accuracy:.6f}`), recall (`{efficient.recall:.6f}`), F1-score (`{efficient.f1_score:.6f}`), lowest practical false-negative count among transfer-learning models (`{efficient.fn}`), and a much smaller parameter count than ResNet50.

3. Why were the other models rejected?

Custom CNN was rejected because it produced `{custom.fp}` false positives and much weaker accuracy, precision, and F1-score. ResNet50 was rejected as the final application model because its AUC-ROC advantage is very small (`{resnet.auc_roc - efficient.auc_roc:+.6f}`), while it has more false negatives at threshold 0.50 and is substantially larger.

4. Which model will be used in the final application?

The final application will use `models/efficientnetb0_final.keras`.

5. Which threshold will be used in production?

The production threshold is `0.50`. This keeps recall higher than the F1-optimized threshold and minimizes missed pneumonia cases for screening.

6. What are the limitations of the selected model?

- It was evaluated on a single public chest X-ray dataset and may not generalize to all hospitals, devices, age groups, or acquisition protocols.
- It is a binary classifier and does not distinguish viral pneumonia, bacterial pneumonia, other lung disease, or image-quality failures.
- It should not be used as an autonomous diagnostic system; it is a decision-support/screening model.
- The test set is limited in size, so small metric differences between ResNet50 and EfficientNetB0 should be interpreted cautiously.
- Threshold `0.50` prioritizes recall, so false positives remain expected and require clinical review.

## Official Lock

- Final selected model: `EfficientNetB0`
- Final model file: `models/efficientnetb0_final.keras`
- Production threshold: `0.50`
- Phase 9 may begin only after this locked decision is accepted.

## Generated Artifacts

- CSV comparison: `{REPORTS_DIR / 'final_model_comparison.csv'}`
- JSON comparison: `{REPORTS_DIR / 'final_model_comparison.json'}`
- Metrics table image: `{FIGURES_DIR / 'final_metrics_table.png'}`
- Ranking image: `{FIGURES_DIR / 'final_model_ranking.png'}`
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_comparison_rows()
    df = pd.DataFrame(rows)
    ranking_df = score_models(df)

    pairwise = {
        "custom_cnn_vs_resnet50": pairwise_delta(df, "Custom CNN", "ResNet50"),
        "resnet50_vs_efficientnetb0": pairwise_delta(df, "ResNet50", "EfficientNetB0"),
        "custom_cnn_vs_efficientnetb0": pairwise_delta(df, "Custom CNN", "EfficientNetB0"),
    }

    csv_path = REPORTS_DIR / "final_model_comparison.csv"
    json_path = REPORTS_DIR / "final_model_comparison.json"
    table_png_path = FIGURES_DIR / "final_metrics_table.png"
    ranking_png_path = FIGURES_DIR / "final_model_ranking.png"
    markdown_path = REPORTS_DIR / "PHASE_8_FINAL_COMPARISON_REPORT.md"

    df.to_csv(csv_path, index=False)
    payload = {
        "final_selected_model": "EfficientNetB0",
        "final_model_file": "models/efficientnetb0_final.keras",
        "production_threshold": 0.5,
        "ranking": ranking_df.to_dict(orient="records"),
        "comparison": df.to_dict(orient="records"),
        "pairwise_deltas": pairwise,
        "decision_basis": {
            "best_academic_model": "ResNet50",
            "best_medical_screening_model": "EfficientNetB0",
            "best_deployment_model": "EfficientNetB0",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    save_metrics_table_png(df, table_png_path)
    save_ranking_png(ranking_df, ranking_png_path)
    write_markdown_report(df, ranking_df, pairwise, markdown_path)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
