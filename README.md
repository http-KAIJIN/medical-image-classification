# Medical Image Classification for Pneumonia Detection

> Deep-learning pneumonia detection from chest X-rays — **EfficientNetB0 at 88.8% accuracy / 0.96 AUC** — with **Grad-CAM explainability** so predictions aren't a black box. Served via a FastAPI inference API and a web demo.

Binary chest X-ray classification (`NORMAL` vs `PNEUMONIA`) covering the full lifecycle: dataset preparation, training and comparison of three architectures, a final selected model, Grad-CAM explainability, a FastAPI backend, and a lightweight web frontend for live demonstration.

**Why it matters:** it demonstrates trustworthy applied ML — rigorous evaluation *and* visual explainability, the two things that make a model deployable in a high-stakes domain.

> 🔗 **Live demo:** _coming soon_ — hosted FastAPI demo with sample X-rays (see deployment notes below).

## Dataset

The project uses the Chest X-Ray Pneumonia dataset organized into `train`, `val`, and `test` splits with two classes:

```text
NORMAL
PNEUMONIA
```

Expected local dataset layout for training or re-evaluation:

```text
data/raw/chest_xray/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
```

The dataset itself is not included in this repository.

## Models Used

Three architectures were trained and compared:

| Model | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Custom CNN | `0.826923` | `0.813333` | `0.938462` | `0.871429` | `0.907287` |
| ResNet50 | `0.876603` | `0.861432` | `0.956410` | `0.906440` | `0.954175` |
| EfficientNetB0 | `0.887821` | `0.875587` | `0.956410` | `0.914216` | `0.957868` |

## Final Selected Model

The final selected model is **EfficientNetB0**.

Final model artifact:

```text
models/efficientnetb0_final.keras
```

Final metrics for EfficientNetB0:

| Metric | Value |
|---|---:|
| Accuracy | `0.887821` |
| Precision | `0.875587` |
| Recall | `0.956410` |
| F1-score | `0.914216` |
| AUC-ROC | `0.957868` |
| Confusion Matrix | `TN=181, FP=53, FN=17, TP=373` |
| Production Threshold | `0.50` |

EfficientNetB0 was selected because it achieved the strongest clean-split accuracy, F1-score, and AUC-ROC while remaining smaller for deployment than ResNet50.

## Project Architecture

```text
medical-image-classification/
  README.md
  requirements.txt
  src/                         # Training, evaluation, preprocessing, Grad-CAM, and validation scripts
  backend/                     # FastAPI inference and Grad-CAM API
  frontend/                    # Static web interface for predictions and Grad-CAM generation
  notebooks/
    training_notebook.ipynb    # Academic training notebook
  models/
    efficientnetb0_final.keras # Final selected model
  presentation/                # Presentation and defense materials
  MASTER_FINAL_SUMMARY.md
  FINAL_READINESS_REPORT.md
```

## Installation

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Backend

Start the FastAPI application from the repository root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```text
POST http://127.0.0.1:8000/predict
```

Grad-CAM endpoint:

```text
POST http://127.0.0.1:8000/gradcam
```

## Running Frontend

Serve the static frontend locally:

```bash
python -m http.server 8080 --directory frontend
```

Then open:

```text
http://127.0.0.1:8080
```

The frontend expects the backend to be running at `http://127.0.0.1:8000`.

## Example Usage

Use the API directly with a chest X-ray image:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/chest_xray.png"
```

Expected response format:

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.95,
  "probability": 0.95,
  "threshold": 0.5
}
```

## Limitations

- This project is for academic evaluation and demonstration only.
- It is not a medical diagnostic device and must not be used for clinical decisions.
- Model performance depends on the dataset distribution and image quality.
- EfficientNetB0 and ResNet50 had close results, so conclusions should be reported cautiously.
- External validation on independent clinical data was not performed.

## Authors

- Oussama
