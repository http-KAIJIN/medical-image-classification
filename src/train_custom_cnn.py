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
from src.models import build_custom_cnn
from src.preprocessing import load_prepared_datasets


def plot_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    hist = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist["loss"], label="train_loss")
    axes[0].plot(hist["val_loss"], label="val_loss")
    axes[0].set_title("Custom CNN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["binary_accuracy"], label="train_accuracy")
    axes[1].plot(hist["val_binary_accuracy"], label="val_accuracy")
    axes[1].set_title("Custom CNN Accuracy")
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> list[list[int]]:
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"], ax=ax)
    ax.set_title("Custom CNN Confusion Matrix")
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
    ax.set_title("Custom CNN ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Custom CNN baseline.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "roc_curves").mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_weights, dfs = load_prepared_datasets(batch_size=args.batch_size)
    model = build_custom_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (REPORTS_DIR / "custom_cnn_model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    checkpoint_path = MODELS_DIR / "custom_cnn_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration_seconds = time.perf_counter() - start

    history_df = pd.DataFrame(history.history)
    history_path = REPORTS_DIR / "custom_cnn_training_history.csv"
    history_df.to_csv(history_path, index=False)

    curves_path = FIGURES_DIR / "custom_cnn_training_curves.png"
    plot_training_curves(history, curves_path)

    best_model = tf.keras.models.load_model(checkpoint_path)
    y_true, y_prob = collect_predictions(best_model, test_ds)
    metrics = compute_metrics(y_true, y_prob)

    cm_path = RESULTS_DIR / "confusion_matrices" / "custom_cnn_confusion_matrix.png"
    cm = plot_confusion_matrix(y_true, y_prob, cm_path)

    roc_path = RESULTS_DIR / "roc_curves" / "custom_cnn_roc_curve.png"
    plot_roc_curve(y_true, y_prob, roc_path)

    predictions_path = REPORTS_DIR / "custom_cnn_test_predictions.csv"
    test_df = dfs["test"].copy().reset_index(drop=True)
    test_df["y_true"] = y_true
    test_df["pneumonia_probability"] = y_prob
    test_df["y_pred"] = (y_prob >= 0.5).astype(int)
    test_df.to_csv(predictions_path, index=False)

    final_model_path = MODELS_DIR / "custom_cnn_final.keras"
    best_model.save(final_model_path)

    report = {
        "model": "custom_cnn_baseline",
        "epochs_requested": args.epochs,
        "epochs_ran": int(len(history_df)),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_duration_seconds": float(training_duration_seconds),
        "training_duration_minutes": float(training_duration_seconds / 60),
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "train_count": int(len(dfs["train"])),
        "val_count": int(len(dfs["val"])),
        "test_count": int(len(dfs["test"])),
        "trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in best_model.trainable_weights)),
        "total_parameters": int(best_model.count_params()),
        "test_metrics": metrics,
        "confusion_matrix": cm,
        "model_checkpoint": str(checkpoint_path),
        "final_model": str(final_model_path),
        "model_summary": str(REPORTS_DIR / "custom_cnn_model_summary.txt"),
        "training_history": str(history_path),
        "training_curves": str(curves_path),
        "confusion_matrix_png": str(cm_path),
        "roc_curve_png": str(roc_path),
        "test_predictions_csv": str(predictions_path),
    }

    report_path = REPORTS_DIR / "custom_cnn_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = REPORTS_DIR / "PHASE_5_CUSTOM_CNN_REPORT.md"
    md_path.write_text(
        f"""# Phase 5 - Custom CNN Baseline Report

## Training Configuration

- Epochs requested: `{report['epochs_requested']}`
- Epochs ran: `{report['epochs_ran']}`
- Batch size: `{report['batch_size']}`
- Learning rate: `{report['learning_rate']}`
- Training duration: `{report['training_duration_minutes']:.2f}` minutes
- Class weights: `{report['class_weights']}`
- Train / validation / test counts: `{report['train_count']}` / `{report['val_count']}` / `{report['test_count']}`

## Model Size

- Total parameters: `{report['total_parameters']}`
- Trainable parameters: `{report['trainable_parameters']}`

## Test Metrics

- Accuracy: `{metrics['accuracy']:.6f}`
- Precision: `{metrics['precision']:.6f}`
- Recall: `{metrics['recall']:.6f}`
- F1-score: `{metrics['f1_score']:.6f}`
- AUC-ROC: `{metrics['auc_roc']:.6f}`

## Confusion Matrix

Rows = true labels, columns = predicted labels. Label order: NORMAL, PNEUMONIA.

```text
{np.array(cm)}
```

## Generated Artifacts

- Model summary: `{report['model_summary']}`
- Best model checkpoint: `{report['model_checkpoint']}`
- Final saved model: `{report['final_model']}`
- Training history: `{report['training_history']}`
- Training curves: `{report['training_curves']}`
- Confusion matrix plot: `{report['confusion_matrix_png']}`
- ROC curve plot: `{report['roc_curve_png']}`
- Test predictions: `{report['test_predictions_csv']}`
""",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
