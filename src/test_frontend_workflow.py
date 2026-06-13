import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from src.config import PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR


FRONTEND_URL = "http://127.0.0.1:8080"
TEST_IMAGE = PROJECT_ROOT / "data" / "raw" / "chest_xray" / "test" / "PNEUMONIA" / "person1615_virus_2801.jpeg"
SCREENSHOT_DIR = RESULTS_DIR / "frontend"
RESULTS_PATH = REPORTS_DIR / "frontend_test_results.json"
REPORT_PATH = REPORTS_DIR / "PHASE_11_FRONTEND_REPORT.md"


def write_invalid_file() -> Path:
    path = SCREENSHOT_DIR / "invalid_upload.txt"
    path.write_text("not a chest x-ray image", encoding="utf-8")
    return path


def run_browser_workflow() -> dict:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    screenshots = {
        "initial": str(SCREENSHOT_DIR / "frontend_01_initial.png"),
        "preview": str(SCREENSHOT_DIR / "frontend_02_preview.png"),
        "prediction": str(SCREENSHOT_DIR / "frontend_03_prediction.png"),
        "gradcam": str(SCREENSHOT_DIR / "frontend_04_gradcam.png"),
        "invalid_file": str(SCREENSHOT_DIR / "frontend_05_invalid_file.png"),
        "mobile": str(SCREENSHOT_DIR / "frontend_06_mobile.png"),
    }
    console_errors = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("#apiStatus.ok", timeout=60000)
        page.screenshot(path=screenshots["initial"], full_page=True)

        page.set_input_files("#fileInput", str(TEST_IMAGE))
        page.wait_for_selector("#previewImage:not([hidden])")
        page.screenshot(path=screenshots["preview"], full_page=True)

        page.click("#predictBtn")
        page.wait_for_selector("#resultPanel:not([hidden])", timeout=120000)
        prediction_text = page.locator("#predictionBadge").inner_text()
        confidence_text = page.locator("#confidenceValue").inner_text()
        probability_text = page.locator("#probabilityValue").inner_text()
        threshold_text = page.locator("#thresholdValue").inner_text()
        page.screenshot(path=screenshots["prediction"], full_page=True)

        page.click("#gradcamBtn")
        page.wait_for_selector("#gradcamImage:not([hidden])", timeout=120000)
        gradcam_src = page.locator("#gradcamImage").get_attribute("src")
        message_text = page.locator("#messageBox").inner_text()
        page.screenshot(path=screenshots["gradcam"], full_page=True)

        invalid_file = write_invalid_file()
        page.set_input_files("#fileInput", str(invalid_file))
        page.wait_for_selector("#messageBox.error")
        invalid_message = page.locator("#messageBox").inner_text()
        page.screenshot(path=screenshots["invalid_file"], full_page=True)

        mobile_page = browser.new_page(viewport={"width": 390, "height": 1000}, is_mobile=True)
        mobile_page.goto(FRONTEND_URL)
        mobile_page.wait_for_load_state("networkidle")
        mobile_page.wait_for_selector("#apiStatus.ok", timeout=60000)
        mobile_page.screenshot(path=screenshots["mobile"], full_page=True)
        mobile_page.close()
        browser.close()

    return {
        "success": True,
        "frontend_url": FRONTEND_URL,
        "test_image": str(TEST_IMAGE),
        "screenshots": screenshots,
        "workflow": {
            "page_opened": Path(screenshots["initial"]).exists(),
            "image_uploaded_and_previewed": Path(screenshots["preview"]).exists(),
            "prediction_received": prediction_text in {"PNEUMONIA", "NORMAL"},
            "gradcam_received": bool(gradcam_src and "static/gradcam" in gradcam_src),
            "invalid_file_handled": "Invalid file type" in invalid_message,
            "responsive_mobile_checked": Path(screenshots["mobile"]).exists(),
        },
        "api_communication_proof": {
            "prediction": prediction_text,
            "confidence": confidence_text,
            "probability": probability_text,
            "threshold": threshold_text,
            "gradcam_image_src": gradcam_src,
            "gradcam_message": message_text,
            "invalid_file_message": invalid_message,
        },
        "console_errors": console_errors,
    }


def write_report(results: dict) -> None:
    workflow = results["workflow"]
    proof = results["api_communication_proof"]
    screenshots = results["screenshots"]
    report = f"""# Phase 11 - HTML/CSS/JavaScript Frontend Report

## Objective

Build and validate a lightweight vanilla frontend for the FastAPI pneumonia detection backend.

## Frontend Architecture

```text
frontend/
├── index.html       # Semantic page structure and application sections
├── styles.css       # Responsive medical-themed visual design
├── app.js           # Upload, preview, API calls, errors, loading states
└── assets/
    └── logo.png     # Local presentation logo
```

## Backend Integration

| Feature | Endpoint | Status |
|---|---|---:|
| API health status | `GET http://127.0.0.1:8000/health` | `Passed` |
| Prediction | `POST http://127.0.0.1:8000/predict` | `Passed` |
| Grad-CAM | `POST http://127.0.0.1:8000/gradcam` | `Passed` |

## Verification Results

| Success Criterion | Passed |
|---|---:|
"""
    for key, value in workflow.items():
        report += f"| {key.replace('_', ' ').title()} | `{value}` |\n"

    report += f"""

## API Communication Proof

- Prediction: `{proof['prediction']}`
- Confidence: `{proof['confidence']}`
- Probability: `{proof['probability']}`
- Threshold: `{proof['threshold']}`
- Grad-CAM image source: `{proof['gradcam_image_src']}`
- Grad-CAM message: `{proof['gradcam_message']}`
- Invalid file message: `{proof['invalid_file_message']}`

## Screenshots

- Initial page: `{screenshots['initial']}`
- Image preview: `{screenshots['preview']}`
- Prediction workflow: `{screenshots['prediction']}`
- Grad-CAM workflow: `{screenshots['gradcam']}`
- Invalid file handling: `{screenshots['invalid_file']}`
- Mobile responsive view: `{screenshots['mobile']}`

## Files Created

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`
- `frontend/assets/logo.png`
- `src/test_frontend_workflow.py`
- `data/reports/frontend_test_results.json`
- `data/reports/PHASE_11_FRONTEND_REPORT.md`

## Issues Encountered and Fixes Applied

- Browser calls from the frontend require CORS because the frontend and API are served on different local ports. Fixed by adding local development origins to the FastAPI CORS middleware.
- The loading overlay initially intercepted clicks while hidden because a class-level display rule overrode the `hidden` attribute. Fixed with an explicit `[hidden]` CSS rule.
- The UI includes loading states for long-running model and Grad-CAM requests so users receive immediate feedback.
- Invalid client-side file types are blocked before API submission and displayed as friendly messages.

## Final Status

- Frontend validation complete: `{results['success']}`
- Console errors captured during workflow: `{len(results['console_errors'])}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_browser_workflow()
    results["success"] = all(results["workflow"].values()) and not results["console_errors"]
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
