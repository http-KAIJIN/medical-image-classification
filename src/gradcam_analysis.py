import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.config import FIGURES_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR
from src.gradcam import find_final_conv_layer, generate_gradcam_heatmap, load_image_array, save_gradcam_outputs
from src.models import efficientnetb0_preprocess  # noqa: F401 - registers custom Lambda for model loading


MODEL_PATH = MODELS_DIR / "efficientnetb0_final.keras"
PREDICTIONS_PATH = REPORTS_DIR / "efficientnetb0_test_predictions.csv"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
SUMMARY_GRID_PATH = FIGURES_DIR / "gradcam_summary_grid.png"
EXAMPLES_JSON_PATH = REPORTS_DIR / "gradcam_examples.json"
REPORT_PATH = REPORTS_DIR / "PHASE_9_GRADCAM_REPORT.md"

CASE_CONFIGS = [
    {
        "case_key": "correct_pneumonia",
        "case_title": "Correct Pneumonia Prediction",
        "true_label": "PNEUMONIA",
        "predicted_label": "PNEUMONIA",
        "sort_column": "pneumonia_probability",
        "ascending": False,
    },
    {
        "case_key": "correct_normal",
        "case_title": "Correct Normal Prediction",
        "true_label": "NORMAL",
        "predicted_label": "NORMAL",
        "sort_column": "pneumonia_probability",
        "ascending": True,
    },
    {
        "case_key": "false_positive",
        "case_title": "False Positive",
        "true_label": "NORMAL",
        "predicted_label": "PNEUMONIA",
        "sort_column": "pneumonia_probability",
        "ascending": False,
    },
    {
        "case_key": "false_negative",
        "case_title": "False Negative",
        "true_label": "PNEUMONIA",
        "predicted_label": "NORMAL",
        "sort_column": "pneumonia_probability",
        "ascending": True,
    },
]


def label_from_index(index: int) -> str:
    return "PNEUMONIA" if int(index) == 1 else "NORMAL"


def select_examples(predictions_df: pd.DataFrame) -> list[dict]:
    selected = []
    for config in CASE_CONFIGS:
        matches = predictions_df[
            (predictions_df["label"] == config["true_label"])
            & (predictions_df["y_pred"].map(label_from_index) == config["predicted_label"])
        ].copy()
        if matches.empty:
            raise ValueError(f"No available example for {config['case_key']}")
        row = matches.sort_values(config["sort_column"], ascending=config["ascending"]).iloc[0]
        selected.append({**config, "row": row})
    return selected


def image_path_from_row(row: pd.Series) -> Path:
    path = Path(str(row["filepath"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_example(example: dict) -> dict[str, bool]:
    paths = example["paths"]
    return {
        "heatmap_exists": Path(paths["heatmap"]).exists(),
        "overlay_exists": Path(paths["overlay"]).exists(),
        "prediction_exists": example.get("predicted_class") is not None,
        "confidence_exists": example.get("confidence") is not None,
    }


def create_summary_grid(examples: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 15))
    columns = ["Original", "Grad-CAM Heatmap", "Overlay"]
    for col_idx, title in enumerate(columns):
        axes[0, col_idx].set_title(title, fontsize=12, weight="bold")

    for row_idx, example in enumerate(examples):
        image_paths = [
            example["paths"]["original_image"],
            example["paths"]["heatmap"],
            example["paths"]["overlay"],
        ]
        for col_idx, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(
            f"{example['case_title']}\nTrue: {example['true_class']}\nPred: {example['predicted_class']}\nConf: {example['confidence']:.3f}",
            fontsize=9,
            rotation=0,
            labelpad=60,
            va="center",
        )

    fig.suptitle("EfficientNetB0 Grad-CAM Examples", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def interpretation_text(example: dict) -> str:
    summary = example["attention_summary"]
    inside_lungs = "Yes, approximately within the central chest/lung field." if summary["inside_likely_lung_field"] else "Not clearly; attention is near an edge or uncertain area."
    artifact_focus = "Yes, possible border/artifact influence is present." if summary["artifact_attention_flag"] else "No major border-dominated artifact focus was detected by the heuristic."
    plausible = "Yes, cautiously." if summary["medical_plausibility"] == "Plausible screening explanation" else "Only partially; interpret cautiously."
    return f"""- Highest attention region: `{summary['highest_attention_region']}`
- Is the attention inside the lungs? {inside_lungs}
- Is attention focused on irrelevant artifacts? {artifact_focus}
- Does the explanation appear medically plausible? {plausible}
- Limitations: This is a coarse Grad-CAM localization from a 7x7 feature map. It does not identify pathology boundaries and should not be interpreted as a radiologist-level explanation."""


def write_report(examples: list[dict], conv_layer_name: str) -> None:
    report = f"""# Phase 9 - Grad-CAM Explainability Report

## Objective

This phase implements Grad-CAM explainability for the locked final model, `EfficientNetB0`, using `models/efficientnetb0_final.keras`.

Grad-CAM visualizes regions that influenced the model prediction, but it does not prove that the model is reasoning like a medical expert.

## Model and Layer Selection

- Final model: `models/efficientnetb0_final.keras`
- Target convolutional layer: `{conv_layer_name}`
- Target layer output resolution: `7x7x1280`
- Production threshold used for class labels: `0.50`

The selected layer is the final convolutional layer inside the EfficientNetB0 backbone. It is appropriate for Grad-CAM because it preserves spatial activation information immediately before global pooling and classification.

## Generated Examples

| Case | True Class | Predicted Class | Confidence | Pneumonia Probability | Original | Heatmap | Overlay |
|---|---|---|---:|---:|---|---|---|
"""
    for example in examples:
        report += (
            f"| {example['case_title']} | {example['true_class']} | {example['predicted_class']} | "
            f"`{example['confidence']:.6f}` | `{example['pneumonia_probability']:.6f}` | "
            f"`{example['paths']['original_image']}` | `{example['paths']['heatmap']}` | `{example['paths']['overlay']}` |\n"
        )

    report += """

## Academic Interpretation

"""
    for example in examples:
        report += f"### {example['case_title']}\n\n"
        report += f"- Source image: `{example['source_image']}`\n"
        report += f"- True class: `{example['true_class']}`\n"
        report += f"- Predicted class: `{example['predicted_class']}`\n"
        report += f"- Confidence score: `{example['confidence']:.6f}`\n"
        report += f"- Pneumonia probability: `{example['pneumonia_probability']:.6f}`\n"
        report += interpretation_text(example)
        report += "\n\n"

    report += """## Verification

| Case | Heatmap exists | Overlay exists | Prediction exists | Confidence exists |
|---|---:|---:|---:|---:|
"""
    for example in examples:
        verification = example["verification"]
        report += (
            f"| {example['case_title']} | `{verification['heatmap_exists']}` | `{verification['overlay_exists']}` | "
            f"`{verification['prediction_exists']}` | `{verification['confidence_exists']}` |\n"
        )

    report += f"""

## Key Observations

- Grad-CAM successfully generated explanations for correct pneumonia, correct normal, false positive, and false negative predictions.
- The transfer-learning model generally focuses on broad chest regions rather than isolated non-image metadata.
- Error cases are especially useful for presentation because they show that plausible-looking confidence does not guarantee clinical correctness.
- The false negative example is important because it demonstrates the remaining risk of missed pneumonia even with the selected high-recall model.

## Limitations

- Grad-CAM visualizations are low-resolution and approximate because the final convolutional layer is `7x7`.
- Grad-CAM highlights influential regions, not confirmed disease locations.
- Heatmaps can be affected by image borders, contrast, positioning, and dataset-specific artifacts.
- A medically plausible heatmap does not prove the model used medically valid reasoning.
- A medically implausible heatmap does not automatically prove the classification is wrong.
- Expert radiologist review would be required for clinical validation.

## Final Presentation Suitability

The generated Grad-CAM explanations are suitable for the final academic presentation if they are presented as interpretability aids rather than proof of medical reasoning. The report and figures should explicitly state that Grad-CAM visualizes model influence, not clinical causality.

## Generated Artifacts

- Example metadata: `{EXAMPLES_JSON_PATH}`
- Summary grid: `{SUMMARY_GRID_PATH}`
- Correct pneumonia directory: `{GRADCAM_DIR / 'correct_pneumonia'}`
- Correct normal directory: `{GRADCAM_DIR / 'correct_normal'}`
- False positive directory: `{GRADCAM_DIR / 'false_positive'}`
- False negative directory: `{GRADCAM_DIR / 'false_negative'}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    conv_layer_name = find_final_conv_layer(model)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    selected = select_examples(predictions_df)

    examples = []
    for item in selected:
        row = item["row"]
        source_image = image_path_from_row(row)
        image_array = load_image_array(source_image)
        gradcam_result = generate_gradcam_heatmap(model, image_array, conv_layer_name=conv_layer_name)
        output_dir = GRADCAM_DIR / item["case_key"]
        paths = save_gradcam_outputs(source_image, gradcam_result.heatmap, output_dir, stem=item["case_key"])
        example = {
            "case_key": item["case_key"],
            "case_title": item["case_title"],
            "source_image": str(source_image),
            "filename": row["filename"],
            "true_class": row["label"],
            "predicted_class": gradcam_result.predicted_class,
            "pneumonia_probability": gradcam_result.predicted_probability,
            "confidence": gradcam_result.confidence,
            "target_conv_layer": conv_layer_name,
            "paths": paths,
            "attention_summary": gradcam_result.attention_summary,
        }
        example["verification"] = verify_example(example)
        examples.append(example)

    create_summary_grid(examples, SUMMARY_GRID_PATH)

    payload = {
        "model": "EfficientNetB0",
        "model_file": str(MODEL_PATH),
        "target_conv_layer": conv_layer_name,
        "production_threshold": 0.5,
        "gradcam_caution": "Grad-CAM visualizes regions that influenced the model prediction, but it does not prove that the model is reasoning like a medical expert.",
        "summary_grid": str(SUMMARY_GRID_PATH),
        "examples": examples,
    }
    EXAMPLES_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(examples, conv_layer_name)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
