import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR


def save_image_size_plots(df: pd.DataFrame) -> None:
    readable = df[df["readable"] == True].copy()  # noqa: E712
    readable["aspect_ratio"] = readable["width"] / readable["height"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(readable["width"], bins=40, color="#2563eb", alpha=0.8)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Count")

    axes[1].hist(readable["height"], bins=40, color="#16a34a", alpha=0.8)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height")

    axes[2].hist(readable["aspect_ratio"], bins=40, color="#dc2626", alpha=0.8)
    axes[2].set_title("Aspect Ratio Distribution")
    axes[2].set_xlabel("Width / Height")

    fig.tight_layout()
    output = FIGURES_DIR / "image_size_distributions.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_prepared_distribution_plot(prepared_distribution: pd.DataFrame) -> None:
    pivot = prepared_distribution.pivot(index="prepared_split", columns="label", values="count")
    pivot = pivot.reindex(["train", "val", "test"]).fillna(0)
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#2563eb", "#f97316"])
    ax.set_title("Prepared Class Distribution by Split")
    ax.set_xlabel("Prepared Split")
    ax.set_ylabel("Image Count")
    ax.legend(title="Class")
    plt.tight_layout()
    output = FIGURES_DIR / "prepared_class_distribution.png"
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPORTS_DIR / "dataset_records.csv")
    prepared_distribution = pd.read_csv(REPORTS_DIR / "prepared_split_distribution.csv")

    readable = df[df["readable"] == True].copy()  # noqa: E712
    readable["aspect_ratio"] = readable["width"] / readable["height"]

    save_image_size_plots(df)
    save_prepared_distribution_plot(prepared_distribution)

    mode_distribution = readable.groupby(["split", "mode"], as_index=False).size().rename(columns={"size": "count"})
    extension_distribution = readable.groupby("extension", as_index=False).size().rename(columns={"size": "count"})

    class_counts = readable.groupby("label").size().to_dict()
    imbalance_ratio = class_counts.get("PNEUMONIA", 0) / max(class_counts.get("NORMAL", 1), 1)

    eda_report = {
        "total_readable_images": int(readable.shape[0]),
        "mode_distribution": mode_distribution.to_dict("records"),
        "extension_distribution": extension_distribution.to_dict("records"),
        "width_mean": float(readable["width"].mean()),
        "height_mean": float(readable["height"].mean()),
        "aspect_ratio_mean": float(readable["aspect_ratio"].mean()),
        "normal_count": int(class_counts.get("NORMAL", 0)),
        "pneumonia_count": int(class_counts.get("PNEUMONIA", 0)),
        "pneumonia_to_normal_ratio": float(imbalance_ratio),
        "recommended_training_controls": [
            "resize all images to 224x224",
            "convert all images to RGB",
            "use class weights or weighted loss",
            "evaluate with recall, precision, F1, AUC, and confusion matrix",
            "use prepared CSV splits instead of original val folder",
        ],
        "figures": {
            "image_size_distributions": str(FIGURES_DIR / "image_size_distributions.png"),
            "prepared_class_distribution": str(FIGURES_DIR / "prepared_class_distribution.png"),
        },
    }

    (REPORTS_DIR / "eda_report.json").write_text(json.dumps(eda_report, indent=2), encoding="utf-8")

    report_md = f"""# Phase 3 - Exploratory Data Analysis

## Key Findings

- Total readable images: `{eda_report['total_readable_images']}`
- NORMAL images: `{eda_report['normal_count']}`
- PNEUMONIA images: `{eda_report['pneumonia_count']}`
- Pneumonia/Normal imbalance ratio: `{eda_report['pneumonia_to_normal_ratio']:.2f}`
- Mean width: `{eda_report['width_mean']:.1f}` pixels
- Mean height: `{eda_report['height_mean']:.1f}` pixels
- Mean aspect ratio: `{eda_report['aspect_ratio_mean']:.2f}`

## Mode Distribution

```text
{mode_distribution.to_string(index=False)}
```

## File Extension Distribution

```text
{extension_distribution.to_string(index=False)}
```

## Important EDA Conclusions

- The dataset is significantly imbalanced toward `PNEUMONIA`.
- All training must use metrics beyond accuracy.
- Class weights should be used during training.
- Images have variable sizes and mixed channel modes, so preprocessing must standardize every image.
- The prepared split CSV files are mandatory because the original validation folder is too small.

## Generated EDA Figures

- `results/figures/image_size_distributions.png`
- `results/figures/prepared_class_distribution.png`
"""
    (REPORTS_DIR / "PHASE_3_EDA_REPORT.md").write_text(report_md, encoding="utf-8")
    print(json.dumps(eda_report, indent=2))


if __name__ == "__main__":
    main()
