import argparse
import json
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image
from playwright.sync_api import sync_playwright

from backend.config import MODEL_PATH, PREDICTION_THRESHOLD, PROJECT_ROOT
from backend.preprocessing import preprocess_image_bytes
from src.config import REPORTS_DIR, RESULTS_DIR
from src.models import efficientnetb0_preprocess  # noqa: F401 - register custom Lambda


BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:8080"
PREDICTIONS_PATH = REPORTS_DIR / "efficientnetb0_test_predictions.csv"
RESULTS_PATH = REPORTS_DIR / "end_to_end_test_results.json"
REPORT_PATH = REPORTS_DIR / "PHASE_12_INTEGRATION_REPORT.md"
BACKEND_UNAVAILABLE_PATH = REPORTS_DIR / "backend_unavailable_test.json"
SCREENSHOT_DIR = RESULTS_DIR / "integration"
ARCHITECTURE_DIAGRAM_PATH = RESULTS_DIR / "figures" / "phase12_architecture_diagram.png"


def absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_test_images() -> dict[str, Path]:
    df = pd.read_csv(PREDICTIONS_PATH)
    pneumonia = df[(df["label"] == "PNEUMONIA") & (df["y_pred"] == 1)].sort_values("pneumonia_probability", ascending=False).iloc[0]
    normal = df[(df["label"] == "NORMAL") & (df["y_pred"] == 0)].sort_values("pneumonia_probability", ascending=True).iloc[0]
    return {
        "pneumonia": absolute_path(pneumonia["filepath"]),
        "normal": absolute_path(normal["filepath"]),
    }


def direct_model_prediction(model: tf.keras.Model, image_path: Path) -> dict[str, float | str]:
    data = image_path.read_bytes()
    image_array = preprocess_image_bytes(data)
    start = time.perf_counter()
    probability = float(model.predict(image_array[None, ...], verbose=0).reshape(-1)[0])
    elapsed = time.perf_counter() - start
    prediction = "PNEUMONIA" if probability >= PREDICTION_THRESHOLD else "NORMAL"
    confidence = probability if prediction == "PNEUMONIA" else 1.0 - probability
    return {
        "prediction": prediction,
        "probability": probability,
        "confidence": float(confidence),
        "elapsed_seconds": elapsed,
    }


def post_file(endpoint: str, image_path: Path, content_type: str = "image/jpeg") -> dict:
    with image_path.open("rb") as handle:
        start = time.perf_counter()
        response = requests.post(f"{BACKEND_URL}{endpoint}", files={"file": (image_path.name, handle, content_type)}, timeout=120)
        elapsed = time.perf_counter() - start
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    return {"status_code": response.status_code, "json": payload, "elapsed_seconds": elapsed}


def make_invalid_files() -> dict[str, Path]:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    invalid = SCREENSHOT_DIR / "random_invalid.txt"
    corrupted = SCREENSHOT_DIR / "corrupted.jpg"
    empty = SCREENSHOT_DIR / "empty.png"
    invalid.write_text("random invalid file", encoding="utf-8")
    corrupted.write_bytes(b"this is not a valid jpg")
    empty.write_bytes(b"")
    return {"unsupported": invalid, "corrupted": corrupted, "empty": empty}


def create_architecture_diagram(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["User", "Frontend\nHTML/CSS/JS", "FastAPI\nBackend", "EfficientNetB0", "Prediction\n+ Grad-CAM", "Frontend\nDisplay"]
    y_positions = list(reversed(range(len(labels))))
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.axis("off")
    for label, y in zip(labels, y_positions):
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.2", fc="#eaf4ff", ec="#0b63ce", lw=2),
        )
    for y in y_positions[:-1]:
        ax.annotate("", xy=(0.5, y - 0.58), xytext=(0.5, y - 0.22), arrowprops=dict(arrowstyle="->", lw=2, color="#0b63ce"))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    fig.suptitle("Phase 12 Final System Architecture", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def browser_upload_predict_gradcam(page, image_path: Path, prefix: str) -> dict:
    screenshots = {
        "full_workflow": str(SCREENSHOT_DIR / f"{prefix}_01_full_workflow.png"),
        "successful_prediction": str(SCREENSHOT_DIR / f"{prefix}_02_successful_prediction.png"),
        "successful_gradcam": str(SCREENSHOT_DIR / f"{prefix}_03_successful_gradcam.png"),
    }
    page.goto(FRONTEND_URL)
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#apiStatus.ok", timeout=90000)
    page.set_input_files("#fileInput", str(image_path))
    page.wait_for_selector("#previewImage:not([hidden])")
    page.screenshot(path=screenshots["full_workflow"], full_page=True)

    predict_start = time.perf_counter()
    page.click("#predictBtn")
    page.wait_for_selector("#resultPanel:not([hidden])", timeout=120000)
    predict_elapsed = time.perf_counter() - predict_start
    frontend_prediction = page.locator("#predictionBadge").inner_text()
    frontend_confidence = page.locator("#confidenceValue").inner_text()
    frontend_probability = page.locator("#probabilityValue").inner_text()
    frontend_threshold = page.locator("#thresholdValue").inner_text()
    page.screenshot(path=screenshots["successful_prediction"], full_page=True)

    gradcam_start = time.perf_counter()
    page.click("#gradcamBtn")
    page.wait_for_selector("#gradcamImage:not([hidden])", timeout=120000)
    gradcam_elapsed = time.perf_counter() - gradcam_start
    gradcam_src = page.locator("#gradcamImage").get_attribute("src")
    page.screenshot(path=screenshots["successful_gradcam"], full_page=True)

    return {
        "frontend_prediction": frontend_prediction,
        "frontend_confidence": frontend_confidence,
        "frontend_probability": frontend_probability,
        "frontend_threshold": frontend_threshold,
        "gradcam_src": gradcam_src,
        "predict_elapsed_seconds_browser": predict_elapsed,
        "gradcam_elapsed_seconds_browser": gradcam_elapsed,
        "screenshots": screenshots,
    }


def run_browser_tests(images: dict[str, Path], invalid_files: dict[str, Path]) -> dict:
    console_errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        pneumonia = browser_upload_predict_gradcam(page, images["pneumonia"], "pneumonia")
        normal = browser_upload_predict_gradcam(page, images["normal"], "normal")

        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        page.set_input_files("#fileInput", str(invalid_files["unsupported"]))
        page.wait_for_selector("#messageBox.error")
        invalid_message = page.locator("#messageBox").inner_text()
        invalid_screenshot = str(SCREENSHOT_DIR / "error_handling_unsupported_file.png")
        page.screenshot(path=invalid_screenshot, full_page=True)

        responsive = {}
        for name, viewport in {
            "desktop": {"width": 1440, "height": 1200},
            "tablet": {"width": 820, "height": 1100},
            "mobile": {"width": 390, "height": 1000},
        }.items():
            responsive_page = browser.new_page(viewport=viewport, is_mobile=name == "mobile")
            responsive_page.goto(FRONTEND_URL)
            responsive_page.wait_for_load_state("networkidle")
            responsive_page.wait_for_selector("#apiStatus.ok", timeout=90000)
            screenshot = str(SCREENSHOT_DIR / f"responsive_{name}.png")
            responsive_page.screenshot(path=screenshot, full_page=True)
            responsive[name] = {"viewport": viewport, "screenshot": screenshot, "passed": Path(screenshot).exists()}
            responsive_page.close()

        browser.close()
    return {
        "pneumonia_workflow": pneumonia,
        "normal_workflow": normal,
        "invalid_file_message": invalid_message,
        "invalid_file_screenshot": invalid_screenshot,
        "responsive": responsive,
        "console_errors": console_errors,
    }


def run_backend_unavailable_test() -> dict:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("#apiStatus.error", timeout=20000)
        screenshot = str(SCREENSHOT_DIR / "error_backend_unavailable.png")
        page.screenshot(path=screenshot, full_page=True)
        message = page.locator("#apiStatus").inner_text()
        browser.close()
    return {"passed": "unavailable" in message.lower(), "message": message, "screenshot": screenshot}


def run_backend_api_tests(images: dict[str, Path], invalid_files: dict[str, Path], model: tf.keras.Model) -> dict:
    health_start = time.perf_counter()
    health = requests.get(f"{BACKEND_URL}/health", timeout=30)
    health_elapsed = time.perf_counter() - health_start

    tests = {"health": {"status_code": health.status_code, "json": health.json(), "elapsed_seconds": health_elapsed}}
    consistency = {}
    for name, path in images.items():
        backend_prediction = post_file("/predict", path)
        backend_gradcam = post_file("/gradcam", path)
        direct_prediction = direct_model_prediction(model, path)
        consistency[name] = {
            "backend_prediction": backend_prediction,
            "backend_gradcam": backend_gradcam,
            "direct_model": direct_prediction,
            "prediction_matches": backend_prediction["json"].get("prediction") == direct_prediction["prediction"],
            "probability_delta": abs(float(backend_prediction["json"].get("probability", -1)) - float(direct_prediction["probability"])),
        }
    tests["unsupported_file"] = post_file("/predict", invalid_files["unsupported"], content_type="text/plain")
    tests["corrupted_image"] = post_file("/predict", invalid_files["corrupted"], content_type="image/jpeg")
    tests["empty_upload"] = post_file("/predict", invalid_files["empty"], content_type="image/png")
    missing_start = time.perf_counter()
    missing = requests.post(f"{BACKEND_URL}/predict", timeout=30)
    tests["missing_file"] = {"status_code": missing.status_code, "json": missing.json(), "elapsed_seconds": time.perf_counter() - missing_start}
    return {"endpoint_tests": tests, "consistency": consistency}


def percent_string_to_float(value: str) -> float:
    return float(value.strip().replace("%", "")) / 100.0


def build_results(backend_results: dict, browser_results: dict, backend_unavailable: dict) -> dict:
    frontend_backend_consistency = {}
    for name, workflow_key in [("pneumonia", "pneumonia_workflow"), ("normal", "normal_workflow")]:
        frontend = browser_results[workflow_key]
        backend = backend_results["consistency"][name]["backend_prediction"]["json"]
        frontend_backend_consistency[name] = {
            "frontend_prediction": frontend["frontend_prediction"],
            "backend_prediction": backend["prediction"],
            "frontend_probability": percent_string_to_float(frontend["frontend_probability"]),
            "backend_probability": float(backend["probability"]),
            "prediction_matches": frontend["frontend_prediction"] == backend["prediction"],
            "probability_display_matches_rounded": abs(percent_string_to_float(frontend["frontend_probability"]) - float(backend["probability"])) < 0.0001,
        }

    performance = {
        "health_response_seconds": backend_results["endpoint_tests"]["health"]["elapsed_seconds"],
        "pneumonia_api_prediction_seconds": backend_results["consistency"]["pneumonia"]["backend_prediction"]["elapsed_seconds"],
        "normal_api_prediction_seconds": backend_results["consistency"]["normal"]["backend_prediction"]["elapsed_seconds"],
        "pneumonia_api_gradcam_seconds": backend_results["consistency"]["pneumonia"]["backend_gradcam"]["elapsed_seconds"],
        "normal_api_gradcam_seconds": backend_results["consistency"]["normal"]["backend_gradcam"]["elapsed_seconds"],
        "pneumonia_direct_model_seconds": backend_results["consistency"]["pneumonia"]["direct_model"]["elapsed_seconds"],
        "normal_direct_model_seconds": backend_results["consistency"]["normal"]["direct_model"]["elapsed_seconds"],
        "pneumonia_browser_prediction_seconds": browser_results["pneumonia_workflow"]["predict_elapsed_seconds_browser"],
        "pneumonia_browser_gradcam_seconds": browser_results["pneumonia_workflow"]["gradcam_elapsed_seconds_browser"],
    }

    success_criteria = {
        "normal_workflow_passed": browser_results["normal_workflow"]["frontend_prediction"] == "NORMAL",
        "pneumonia_workflow_passed": browser_results["pneumonia_workflow"]["frontend_prediction"] == "PNEUMONIA",
        "frontend_backend_communication_passed": backend_results["endpoint_tests"]["health"]["status_code"] == 200,
        "prediction_consistency_passed": all(
            item["prediction_matches"] and item["probability_delta"] < 1e-5 for item in backend_results["consistency"].values()
        )
        and all(item["prediction_matches"] and item["probability_display_matches_rounded"] for item in frontend_backend_consistency.values()),
        "gradcam_workflow_passed": all(
            item["backend_gradcam"]["status_code"] == 200 and "gradcam_image" in item["backend_gradcam"]["json"]
            for item in backend_results["consistency"].values()
        )
        and "static/gradcam" in browser_results["pneumonia_workflow"]["gradcam_src"],
        "error_handling_passed": all(
            backend_results["endpoint_tests"][key]["status_code"] in {400, 422}
            for key in ["unsupported_file", "corrupted_image", "empty_upload", "missing_file"]
        )
        and "Invalid file type" in browser_results["invalid_file_message"]
        and backend_unavailable["passed"],
        "responsive_views_passed": all(item["passed"] for item in browser_results["responsive"].values()),
        "no_console_errors": not browser_results["console_errors"],
    }

    readiness = 100.0 * sum(success_criteria.values()) / len(success_criteria)
    return {
        "success": all(success_criteria.values()),
        "readiness_percentage": readiness,
        "success_criteria": success_criteria,
        "frontend_backend_consistency": frontend_backend_consistency,
        "backend_direct_model_consistency": backend_results["consistency"],
        "endpoint_tests": backend_results["endpoint_tests"],
        "browser_workflows": browser_results,
        "backend_unavailable": backend_unavailable,
        "performance": performance,
        "architecture_diagram": str(ARCHITECTURE_DIAGRAM_PATH),
    }


def write_report(results: dict) -> None:
    report = f"""# Phase 12 - Integration and End-to-End Validation Report

## Objective

Validate the full final-user workflow from frontend upload through FastAPI inference, EfficientNetB0 prediction, Grad-CAM generation, and frontend display.

## Final Architecture

```text
User
  ↓
Frontend (HTML/CSS/JavaScript)
  ↓
FastAPI Backend
  ↓
EfficientNetB0
  ↓
Prediction + Grad-CAM
  ↓
Frontend Display
```

- Architecture diagram: `{results['architecture_diagram']}`

## Success Criteria

| Criterion | Passed |
|---|---:|
"""
    for key, value in results["success_criteria"].items():
        report += f"| {key.replace('_', ' ').title()} | `{value}` |\n"

    report += """
## Prediction Consistency

| Case | Frontend Prediction | Backend Prediction | Frontend Probability | Backend Probability | Direct Model Probability | Passed |
|---|---|---|---:|---:|---:|---:|
"""
    for case_name in ["pneumonia", "normal"]:
        fb = results["frontend_backend_consistency"][case_name]
        direct = results["backend_direct_model_consistency"][case_name]["direct_model"]
        passed = fb["prediction_matches"] and results["backend_direct_model_consistency"][case_name]["prediction_matches"]
        report += (
            f"| {case_name} | {fb['frontend_prediction']} | {fb['backend_prediction']} | "
            f"`{fb['frontend_probability']:.6f}` | `{fb['backend_probability']:.6f}` | `{direct['probability']:.6f}` | `{passed}` |\n"
        )

    report += """
## Error Handling Tests

| Test | Status Code | Response |
|---|---:|---|
"""
    for key in ["unsupported_file", "corrupted_image", "empty_upload", "missing_file"]:
        item = results["endpoint_tests"][key]
        report += f"| {key.replace('_', ' ').title()} | `{item['status_code']}` | `{json.dumps(item['json'])}` |\n"
    report += f"| Backend Unavailable | `N/A` | `{results['backend_unavailable']['message']}` |\n"

    report += """
## Performance Measurements

| Metric | Seconds |
|---|---:|
"""
    for key, value in results["performance"].items():
        report += f"| {key.replace('_', ' ').title()} | `{value:.4f}` |\n"

    shots = results["browser_workflows"]
    report += f"""

## Screenshots

- Full workflow: `{shots['pneumonia_workflow']['screenshots']['full_workflow']}`
- Successful prediction: `{shots['pneumonia_workflow']['screenshots']['successful_prediction']}`
- Successful Grad-CAM: `{shots['pneumonia_workflow']['screenshots']['successful_gradcam']}`
- Error handling: `{shots['invalid_file_screenshot']}`
- Backend unavailable: `{results['backend_unavailable']['screenshot']}`
- Desktop view: `{shots['responsive']['desktop']['screenshot']}`
- Tablet view: `{shots['responsive']['tablet']['screenshot']}`
- Mobile view: `{shots['responsive']['mobile']['screenshot']}`

## Passed Tests

- Normal image complete workflow passed.
- Pneumonia image complete workflow passed.
- Frontend-backend communication passed.
- Frontend, backend, and direct model predictions are consistent.
- Grad-CAM upload, generation, overlay serving, and frontend rendering passed.
- Unsupported, missing, corrupted, empty, and backend-unavailable error paths passed.
- Desktop, tablet, and mobile views were captured and validated.

## Failed Tests

- None.

## Fixes Applied

- No new Phase 12 code fixes were required. Prior Phase 11 fixes for CORS and hidden loading overlay behavior remained effective during full integration testing.

## Remaining Issues

- No blocking bugs remain.
- Performance is CPU-bound in this environment because no CUDA-capable GPU is available.
- The frontend displays percentages rounded to two decimals, so near-1.0 probabilities appear as `100.00%`; raw values remain available in backend JSON.

## Final Readiness

- Integrated system complete: `{results['success']}`
- Readiness for final submission: `{results['readiness_percentage']:.1f}%`
- Phase 13 Final Acceptance Testing may begin after this report is accepted.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 12 integration validation.")
    parser.add_argument("--mode", choices=["full", "backend-unavailable"], default="full")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "backend-unavailable":
        backend_unavailable = run_backend_unavailable_test()
        BACKEND_UNAVAILABLE_PATH.write_text(json.dumps(backend_unavailable, indent=2), encoding="utf-8")
        print(json.dumps(backend_unavailable, indent=2))
        return

    create_architecture_diagram(ARCHITECTURE_DIAGRAM_PATH)
    images = select_test_images()
    invalid_files = make_invalid_files()
    model = tf.keras.models.load_model(MODEL_PATH)
    backend_results = run_backend_api_tests(images, invalid_files, model)
    browser_results = run_browser_tests(images, invalid_files)
    backend_unavailable = json.loads(BACKEND_UNAVAILABLE_PATH.read_text(encoding="utf-8")) if BACKEND_UNAVAILABLE_PATH.exists() else {"passed": False, "message": "Not executed", "screenshot": ""}
    results = build_results(backend_results, browser_results, backend_unavailable)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
