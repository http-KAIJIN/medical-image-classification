import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RESULTS_DIR
from src.models import build_resnet50_transfer, set_resnet50_fine_tuning
from src.preprocessing import load_prepared_datasets


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["loss"], label="train_loss")
    axes[0].plot(history_df["val_loss"], label="val_loss")
    axes[0].set_title("ResNet50 Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["binary_accuracy"], label="train_accuracy")
    axes[1].plot(history_df["val_binary_accuracy"], label="val_accuracy")
    axes[1].set_title("ResNet50 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true_batches = []
    y_prob_batches = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_prob_batches.append(probs)
        y_true_batches.append(labels.numpy().reshape(-1))
    return np.concatenate(y_true_batches).astype(int), np.concatenate(y_prob_batches)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, threshold: float = 0.5) -> list[list[int]]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"], ax=ax)
    ax.set_title("ResNet50 Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return cm.tolist()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ResNet50 ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    df = pd.DataFrame(rows)
    best_f1 = df.loc[df["f1_score"].idxmax()].to_dict()
    return df, {"best_f1_threshold": best_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ResNet50 transfer learning model.")
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "roc_curves").mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_weights, dfs = load_prepared_datasets(batch_size=args.batch_size)
    model = build_resnet50_transfer()

    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (REPORTS_DIR / "resnet50_model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    checkpoint_path = MODELS_DIR / "resnet50_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]

    histories = []
    start = time.perf_counter()

    compile_model(model, args.head_learning_rate)
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    head_df = pd.DataFrame(head_history.history)
    head_df["stage"] = "frozen_head"
    histories.append(head_df)

    set_resnet50_fine_tuning(model)
    compile_model(model, args.fine_tune_learning_rate)
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    fine_df = pd.DataFrame(fine_history.history)
    fine_df["stage"] = "fine_tuning"
    histories.append(fine_df)

    training_duration_seconds = time.perf_counter() - start
    history_df = pd.concat(histories, ignore_index=True)
    history_path = REPORTS_DIR / "resnet50_training_history.csv"
    history_df.to_csv(history_path, index=False)

    curves_path = FIGURES_DIR / "resnet50_training_curves.png"
    plot_training_curves(history_df, curves_path)

    best_model = tf.keras.models.load_model(checkpoint_path)
    y_true, y_prob = collect_predictions(best_model, test_ds)
    metrics = compute_metrics(y_true, y_prob)

    cm_path = RESULTS_DIR / "confusion_matrices" / "resnet50_confusion_matrix.png"
    cm = plot_confusion_matrix(y_true, y_prob, cm_path)
    roc_path = RESULTS_DIR / "roc_curves" / "resnet50_roc_curve.png"
    plot_roc_curve(y_true, y_prob, roc_path)

    threshold_df, threshold_report = threshold_analysis(y_true, y_prob)
    threshold_csv = REPORTS_DIR / "resnet50_threshold_analysis.csv"
    threshold_json = REPORTS_DIR / "resnet50_threshold_analysis.json"
    threshold_df.to_csv(threshold_csv, index=False)
    threshold_json.write_text(json.dumps(threshold_report, indent=2), encoding="utf-8")

    predictions_path = REPORTS_DIR / "resnet50_test_predictions.csv"
    test_df = dfs["test"].copy().reset_index(drop=True)
    test_df["y_true"] = y_true
    test_df["pneumonia_probability"] = y_prob
    test_df["y_pred"] = (y_prob >= 0.5).astype(int)
    test_df.to_csv(predictions_path, index=False)

    final_model_path = MODELS_DIR / "resnet50_final.keras"
    best_model.save(final_model_path)

    baseline = json.loads((REPORTS_DIR / "custom_cnn_evaluation.json").read_text(encoding="utf-8"))
    baseline_metrics = baseline["test_metrics"]
    comparison = {
        metric: {
            "custom_cnn": float(baseline_metrics[metric]),
            "resnet50": float(metrics[metric]),
            "delta": float(metrics[metric] - baseline_metrics[metric]),
        }
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    }

    report = {
        "model": "resnet50_transfer",
        "head_epochs_requested": args.head_epochs,
        "fine_tune_epochs_requested": args.fine_tune_epochs,
        "epochs_ran": int(len(history_df)),
        "batch_size": args.batch_size,
        "head_learning_rate": args.head_learning_rate,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "training_duration_seconds": float(training_duration_seconds),
        "training_duration_minutes": float(training_duration_seconds / 60),
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "train_count": int(len(dfs["train"])),
        "val_count": int(len(dfs["val"])),
        "test_count": int(len(dfs["test"])),
        "trainable_parameters_final_model": int(sum(tf.keras.backend.count_params(w) for w in best_model.trainable_weights)),
        "total_parameters": int(best_model.count_params()),
        "test_metrics": metrics,
        "confusion_matrix": cm,
        "threshold_report": threshold_report,
        "comparison_vs_custom_cnn": comparison,
        "model_checkpoint": str(checkpoint_path),
        "final_model": str(final_model_path),
        "model_summary": str(REPORTS_DIR / "resnet50_model_summary.txt"),
        "training_history": str(history_path),
        "training_curves": str(curves_path),
        "confusion_matrix_png": str(cm_path),
        "roc_curve_png": str(roc_path),
        "threshold_analysis_csv": str(threshold_csv),
        "threshold_analysis_json": str(threshold_json),
        "test_predictions_csv": str(predictions_path),
    }

    report_path = REPORTS_DIR / "resnet50_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    best_f1 = threshold_report["best_f1_threshold"]
    md_path = REPORTS_DIR / "PHASE_6_RESNET50_REPORT.md"
    md_path.write_text(
        f"""# Phase 6 - ResNet50 Transfer Learning Report

## Training Configuration

- Frozen-head epochs requested: `{args.head_epochs}`
- Fine-tuning epochs requested: `{args.fine_tune_epochs}`
- Epochs ran: `{report['epochs_ran']}`
- Batch size: `{args.batch_size}`
- Head learning rate: `{args.head_learning_rate}`
- Fine-tuning learning rate: `{args.fine_tune_learning_rate}`
- Training duration: `{report['training_duration_minutes']:.2f}` minutes
- Class weights: `{report['class_weights']}`

## Model Size

- Total parameters: `{report['total_parameters']}`
- Trainable parameters in saved model: `{report['trainable_parameters_final_model']}`

## Test Metrics At Threshold 0.50

- Accuracy: `{metrics['accuracy']:.6f}`
- Precision: `{metrics['precision']:.6f}`
- Recall: `{metrics['recall']:.6f}`
- F1-score: `{metrics['f1_score']:.6f}`
- AUC-ROC: `{metrics['auc_roc']:.6f}`

## Confusion Matrix At Threshold 0.50

Rows = true labels, columns = predicted labels. Label order: NORMAL, PNEUMONIA.

```text
{np.array(cm)}
```

## Best F1 Threshold Analysis

- Threshold: `{best_f1['threshold']}`
- Accuracy: `{best_f1['accuracy']:.6f}`
- Precision: `{best_f1['precision']:.6f}`
- Recall: `{best_f1['recall']:.6f}`
- F1-score: `{best_f1['f1_score']:.6f}`
- TN / FP / FN / TP: `{int(best_f1['tn'])}` / `{int(best_f1['fp'])}` / `{int(best_f1['fn'])}` / `{int(best_f1['tp'])}`

## Comparison Versus Custom CNN

| Metric | Custom CNN | ResNet50 | Delta |
|---|---:|---:|---:|
| Accuracy | `{comparison['accuracy']['custom_cnn']:.6f}` | `{comparison['accuracy']['resnet50']:.6f}` | `{comparison['accuracy']['delta']:.6f}` |
| Precision | `{comparison['precision']['custom_cnn']:.6f}` | `{comparison['precision']['resnet50']:.6f}` | `{comparison['precision']['delta']:.6f}` |
| Recall | `{comparison['recall']['custom_cnn']:.6f}` | `{comparison['recall']['resnet50']:.6f}` | `{comparison['recall']['delta']:.6f}` |
| F1-score | `{comparison['f1_score']['custom_cnn']:.6f}` | `{comparison['f1_score']['resnet50']:.6f}` | `{comparison['f1_score']['delta']:.6f}` |
| AUC-ROC | `{comparison['auc_roc']['custom_cnn']:.6f}` | `{comparison['auc_roc']['resnet50']:.6f}` | `{comparison['auc_roc']['delta']:.6f}` |

## Generated Artifacts

- Model summary: `{report['model_summary']}`
- Best model checkpoint: `{report['model_checkpoint']}`
- Final saved model: `{report['final_model']}`
- Training history: `{report['training_history']}`
- Training curves: `{report['training_curves']}`
- Confusion matrix plot: `{report['confusion_matrix_png']}`
- ROC curve plot: `{report['roc_curve_png']}`
- Threshold analysis CSV: `{report['threshold_analysis_csv']}`
- Test predictions: `{report['test_predictions_csv']}`
""",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
