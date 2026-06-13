import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLASS_NAMES, REPORTS_DIR, SPLITS_DIR


def prepare_splits(records_csv: Path, val_size: float, seed: int) -> dict:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(records_csv)
    df = df[df["readable"] == True].copy()  # noqa: E712

    official_test = df[df["split"] == "test"].copy()
    train_val = df[df["split"].isin(["train", "val"])].copy()

    train_df, val_df = train_test_split(
        train_val,
        test_size=val_size,
        random_state=seed,
        stratify=train_val["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    official_test = official_test.copy()

    train_df["prepared_split"] = "train"
    val_df["prepared_split"] = "val"
    official_test["prepared_split"] = "test"

    prepared = pd.concat([train_df, val_df, official_test], ignore_index=True)
    prepared = prepared.sort_values(["prepared_split", "label", "filepath"]).reset_index(drop=True)

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"
    all_path = SPLITS_DIR / "prepared_splits.csv"
    summary_path = REPORTS_DIR / "prepared_split_distribution.csv"
    report_path = REPORTS_DIR / "prepared_split_report.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    official_test.to_csv(test_path, index=False)
    prepared.to_csv(all_path, index=False)

    summary = (
        prepared.groupby(["prepared_split", "label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["prepared_split", "label"])
    )
    summary.to_csv(summary_path, index=False)

    split_counts = prepared.groupby("prepared_split").size().to_dict()
    class_counts = prepared.groupby("label").size().to_dict()
    per_split_class = {
        split: {
            class_name: int(
                prepared[(prepared["prepared_split"] == split) & (prepared["label"] == class_name)].shape[0]
            )
            for class_name in CLASS_NAMES
        }
        for split in ["train", "val", "test"]
    }

    report = {
        "source_records_csv": str(records_csv),
        "strategy": "official test split preserved; original train+val recombined and stratified into new train/val",
        "validation_size_from_train_val": val_size,
        "random_seed": seed,
        "total_images": int(prepared.shape[0]),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "per_split_class_counts": per_split_class,
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "prepared_splits_csv": str(all_path),
        "prepared_split_distribution_csv": str(summary_path),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create robust train/validation/test CSV splits.")
    parser.add_argument("--records-csv", type=Path, default=REPORTS_DIR / "dataset_records.csv")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = prepare_splits(args.records_csv, args.val_size, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
