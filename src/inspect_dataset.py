import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile

from src.config import CLASS_NAMES, FIGURES_DIR, IMAGE_EXTENSIONS, REPORTS_DIR, SPLIT_NAMES


ImageFile.LOAD_TRUNCATED_IMAGES = True


def collect_records(dataset_dir: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    issues: list[dict] = []

    for split in SPLIT_NAMES:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            issues.append({"type": "missing_split", "path": str(split_dir)})
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                issues.append({"type": "missing_class_dir", "path": str(class_dir)})
                continue

            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    issues.append({"type": "unsupported_extension", "path": str(file_path)})
                    continue

                record = {
                    "filepath": str(file_path),
                    "filename": file_path.name,
                    "split": split,
                    "label": class_name,
                    "extension": file_path.suffix.lower(),
                    "width": None,
                    "height": None,
                    "mode": None,
                    "readable": False,
                }

                try:
                    with Image.open(file_path) as img:
                        record["width"], record["height"] = img.size
                        record["mode"] = img.mode
                        img.verify()
                    record["readable"] = True
                except Exception as exc:
                    issues.append({"type": "unreadable_image", "path": str(file_path), "error": str(exc)})

                records.append(record)

    return records, issues


def create_sample_grid(df: pd.DataFrame, output_path: Path, max_per_class: int = 4) -> None:
    readable_df = df[df["readable"]].copy()
    if readable_df.empty:
        return

    rows = []
    for class_name in CLASS_NAMES:
        class_rows = readable_df[readable_df["label"] == class_name].head(max_per_class)
        rows.extend(class_rows.to_dict("records"))

    if not rows:
        return

    n_cols = max_per_class
    n_rows = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = [axes]

    for row_idx, class_name in enumerate(CLASS_NAMES):
        class_examples = [r for r in rows if r["label"] == class_name]
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx >= len(class_examples):
                continue
            example = class_examples[col_idx]
            with Image.open(example["filepath"]) as img:
                ax.imshow(img.convert("L"), cmap="gray")
            ax.set_title(f"{class_name}\n{example['width']}x{example['height']}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_distribution_plot(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    pivot = summary.pivot(index="split", columns="label", values="count").reindex(SPLIT_NAMES)
    pivot = pivot.fillna(0)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Class Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image Count")
    ax.legend(title="Class")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def inspect_dataset(dataset_dir: Path) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records, issues = collect_records(dataset_dir)
    df = pd.DataFrame(records)

    if df.empty:
        summary = pd.DataFrame(columns=["split", "label", "count"])
        total_images = 0
        image_stats = {}
    else:
        summary = (
            df.groupby(["split", "label"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["split", "label"])
        )
        total_images = int(len(df))
        image_stats = {
            "min_width": int(df["width"].dropna().min()) if df["width"].notna().any() else None,
            "max_width": int(df["width"].dropna().max()) if df["width"].notna().any() else None,
            "min_height": int(df["height"].dropna().min()) if df["height"].notna().any() else None,
            "max_height": int(df["height"].dropna().max()) if df["height"].notna().any() else None,
            "modes": sorted(df["mode"].dropna().unique().tolist()),
        }

    records_path = REPORTS_DIR / "dataset_records.csv"
    summary_path = REPORTS_DIR / "class_distribution.csv"
    json_path = REPORTS_DIR / "dataset_report.json"
    issues_path = REPORTS_DIR / "dataset_issues.json"
    sample_grid_path = FIGURES_DIR / "sample_images.png"
    distribution_plot_path = FIGURES_DIR / "class_distribution.png"

    df.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    create_sample_grid(df, sample_grid_path)
    create_distribution_plot(summary, distribution_plot_path)

    split_counts = df.groupby("split").size().to_dict() if not df.empty else {}
    class_counts = df.groupby("label").size().to_dict() if not df.empty else {}

    report = {
        "dataset_dir": str(dataset_dir),
        "dataset_available": dataset_dir.exists(),
        "total_images": total_images,
        "number_of_classes": len([c for c in CLASS_NAMES if c in class_counts]),
        "expected_classes": CLASS_NAMES,
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "image_stats": image_stats,
        "issues_count": len(issues),
        "issues_path": str(issues_path),
        "records_csv": str(records_path),
        "class_distribution_csv": str(summary_path),
        "sample_images_png": str(sample_grid_path),
        "class_distribution_png": str(distribution_plot_path),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with issues_path.open("w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Chest X-Ray Pneumonia dataset structure and quality.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    report = inspect_dataset(args.dataset_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
