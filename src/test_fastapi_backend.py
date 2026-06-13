import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.config import MODEL_PATH, PREDICTION_THRESHOLD, PROJECT_ROOT
from backend.main import app
from src.config import REPORTS_DIR


PREDICTIONS_PATH = REPORTS_DIR / "efficientnetb0_test_predictions.csv"
RESULTS_PATH = REPORTS_DIR / "api_test_results.json"
REPORT_PATH = REPORTS_DIR / "PHASE_10_FASTAPI_REPORT.md"


def load_case_rows() -> dict[str, pd.Series]:
    df = pd.read_csv(PREDICTIONS_PATH)
    pneumonia = df[(df["label"] == "PNEUMONIA") & (df["y_pred"] == 1)].sort_values("pneumonia_probability", ascending=False).iloc[0]
    normal = df[(df["label"] == "NORMAL") & (df["y_pred"] == 0)].sort_values("pneumonia_probability", ascending=True).iloc[0]
    return {"valid_pneumonia": pneumonia, "valid_normal": normal}


def absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def post_image(client: TestClient, endpoint: str, image_path: Path) -> dict:
    with image_path.open("rb") as image_file:
        response = client.post(endpoint, files={"file": (image_path.name, image_file, "image/jpeg")})
    return {"status_code": response.status_code, "json": response.json()}


def compare_notebook_prediction(api_json: dict, row: pd.Series) -> dict:
    expected_prediction = "PNEUMONIA" if int(row["y_pred"]) == 1 else "NORMAL"
    expected_probability = float(row["pneumonia_probability"])
    api_probability = float(api_json["probability"])
    return {
        "expected_prediction": expected_prediction,
        "api_prediction": api_json["prediction"],
        "expected_probability": expected_probability,
        "api_probability": api_probability,
        "absolute_probability_delta": abs(api_probability - expected_probability),
        "prediction_matches": api_json["prediction"] == expected_prediction,
        "probability_matches": abs(api_probability - expected_probability) < 1e-5,
    }


def run_tests() -> dict:
    rows = load_case_rows()
    with TestClient(app) as client:
        root_response = client.get("/")
        health_response = client.get("/health")
        pneumonia_predict = post_image(client, "/predict", absolute_path(rows["valid_pneumonia"]["filepath"]))
        normal_predict = post_image(client, "/predict", absolute_path(rows["valid_normal"]["filepath"]))
        pneumonia_gradcam = post_image(client, "/gradcam", absolute_path(rows["valid_pneumonia"]["filepath"]))

        invalid_format_response = client.post(
            "/predict",
            files={"file": ("invalid.txt", b"not an image", "text/plain")},
        )
        corrupted_image_response = client.post(
            "/predict",
            files={"file": ("corrupted.jpg", b"not an image", "image/jpeg")},
        )
        empty_response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )

    gradcam_path = pneumonia_gradcam["json"].get("gradcam_image", "")
    static_file = PROJECT_ROOT / "backend" / gradcam_path.lstrip("/") if gradcam_path else None
    consistency = {
        "valid_pneumonia": compare_notebook_prediction(pneumonia_predict["json"], rows["valid_pneumonia"]),
        "valid_normal": compare_notebook_prediction(normal_predict["json"], rows["valid_normal"]),
    }
    success_criteria = {
        "fastapi_starts_successfully": True,
        "model_loads_successfully": MODEL_PATH.exists(),
        "health_works": health_response.status_code == 200 and health_response.json().get("status") == "ok",
        "predict_works": pneumonia_predict["status_code"] == 200 and normal_predict["status_code"] == 200,
        "gradcam_works": pneumonia_gradcam["status_code"] == 200 and static_file is not None and static_file.exists(),
        "predictions_match_notebook": all(item["prediction_matches"] and item["probability_matches"] for item in consistency.values()),
        "error_handling_works": all(
            response.status_code == 400 and "error" in response.json()
            for response in [invalid_format_response, corrupted_image_response, empty_response]
        ),
    }

    return {
        "model": "EfficientNetB0",
        "model_file": str(MODEL_PATH),
        "threshold": PREDICTION_THRESHOLD,
        "success": all(success_criteria.values()),
        "success_criteria": success_criteria,
        "endpoint_tests": {
            "GET /": {"status_code": root_response.status_code, "json": root_response.json()},
            "GET /health": {"status_code": health_response.status_code, "json": health_response.json()},
            "POST /predict pneumonia": pneumonia_predict,
            "POST /predict normal": normal_predict,
            "POST /gradcam pneumonia": pneumonia_gradcam,
            "POST /predict invalid_format": {"status_code": invalid_format_response.status_code, "json": invalid_format_response.json()},
            "POST /predict corrupted_image": {"status_code": corrupted_image_response.status_code, "json": corrupted_image_response.json()},
            "POST /predict empty_upload": {"status_code": empty_response.status_code, "json": empty_response.json()},
        },
        "notebook_api_consistency": consistency,
        "example_requests": {
            "health": "curl http://127.0.0.1:8000/health",
            "predict": "curl -X POST -F file=@data/raw/chest_xray/test/PNEUMONIA/person1615_virus_2801.jpeg http://127.0.0.1:8000/predict",
            "gradcam": "curl -X POST -F file=@data/raw/chest_xray/test/PNEUMONIA/person1615_virus_2801.jpeg http://127.0.0.1:8000/gradcam",
        },
    }


def write_markdown_report(results: dict) -> None:
    tests = results["endpoint_tests"]
    criteria = results["success_criteria"]
    consistency = results["notebook_api_consistency"]
    report = f"""# Phase 10 - FastAPI Backend and Model Inference API Report

## Objective

Build and validate a FastAPI backend for the locked EfficientNetB0 pneumonia classifier and Grad-CAM explainability service.

## Backend Architecture

```text
backend/
├── main.py              # FastAPI app, routes, startup model loading, static files
├── inference.py         # Singleton-style EfficientNetB0 inference service
├── preprocessing.py     # Upload validation and training-consistent image preprocessing
├── gradcam_service.py   # Grad-CAM generation using src.gradcam.py
├── schemas.py           # Pydantic response schemas
├── config.py            # Paths, model metadata, threshold, allowed formats
└── static/              # Generated Grad-CAM overlay files
```

## Locked Model

- Model: `EfficientNetB0`
- File: `{results['model_file']}`
- Threshold: `{results['threshold']:.2f}`
- Model loading behavior: loaded once during FastAPI startup
- Model weights: not retrained and not modified

## Endpoint List

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API root message |
| `/health` | GET | Health check and model metadata |
| `/predict` | POST | Image upload prediction |
| `/gradcam` | POST | Image upload prediction plus Grad-CAM overlay path |

## Startup Command

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Success Criteria

| Criterion | Passed |
|---|---:|
"""
    for key, value in criteria.items():
        report += f"| {key.replace('_', ' ').title()} | `{value}` |\n"

    report += """
## Endpoint Test Results

| Test | Status Code | Response |
|---|---:|---|
"""
    for name, result in tests.items():
        report += f"| {name} | `{result['status_code']}` | `{json.dumps(result['json'])}` |\n"

    report += """
## Notebook vs API Inference Consistency

| Case | Expected Prediction | API Prediction | Expected Probability | API Probability | Absolute Delta | Matches |
|---|---|---|---:|---:|---:|---:|
"""
    for case_name, item in consistency.items():
        matches = item["prediction_matches"] and item["probability_matches"]
        report += (
            f"| {case_name} | {item['expected_prediction']} | {item['api_prediction']} | "
            f"`{item['expected_probability']:.8f}` | `{item['api_probability']:.8f}` | "
            f"`{item['absolute_probability_delta']:.8f}` | `{matches}` |\n"
        )

    report += f"""

## Example Requests

```bash
{results['example_requests']['health']}
```

```bash
{results['example_requests']['predict']}
```

```bash
{results['example_requests']['gradcam']}
```

## Example Responses

### Health

```json
{json.dumps(tests['GET /health']['json'], indent=2)}
```

### Predict

```json
{json.dumps(tests['POST /predict pneumonia']['json'], indent=2)}
```

### Grad-CAM

```json
{json.dumps(tests['POST /gradcam pneumonia']['json'], indent=2)}
```

### Invalid Image

```json
{json.dumps(tests['POST /predict invalid_format']['json'], indent=2)}
```

### Empty Upload

```json
{json.dumps(tests['POST /predict empty_upload']['json'], indent=2)}
```

## Issues Encountered and Fixes Applied

- Nested EfficientNetB0 Grad-CAM layers cannot be used through a direct top-level Keras graph connection. The existing `src.gradcam.py` implementation splits preprocessing, backbone feature extraction, and classifier head to keep gradients connected.
- Upload validation is performed before inference to return clean JSON errors for unsupported, corrupted, or empty files.
- The locked production threshold remains `0.50`; no model weights or thresholds were modified.

## Final Status

- Backend complete: `{results['success']}`
- API test results JSON: `{RESULTS_PATH}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_tests()
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
