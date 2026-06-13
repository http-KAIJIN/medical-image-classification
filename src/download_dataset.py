import argparse
from pathlib import Path

import requests


KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset acquisition helper for Chest X-Ray Pneumonia.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Official implementation dataset: Chest X-Ray Pneumonia")
    print(f"Primary public page: {KAGGLE_DATASET_URL}")
    print("Automatic download requires Kaggle credentials. If unavailable, download manually and extract as:")
    print(args.output_dir / "chest_xray")
    print("Expected folders: train/NORMAL, train/PNEUMONIA, val/NORMAL, val/PNEUMONIA, test/NORMAL, test/PNEUMONIA")

    try:
        response = requests.get(KAGGLE_DATASET_URL, timeout=20)
        print(f"Dataset page HTTP status: {response.status_code}")
    except Exception as exc:
        print(f"Could not reach dataset page: {exc}")


if __name__ == "__main__":
    main()
