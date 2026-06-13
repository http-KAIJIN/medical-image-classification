from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import API_VERSION, MODEL_NAME, PREDICTION_THRESHOLD, STATIC_DIR
from backend.gradcam_service import GradCAMService
from backend.inference import InferenceService
from backend.preprocessing import ImageValidationError, preprocess_image_bytes, read_validated_image
from backend.schemas import ErrorResponse, GradCAMResponse, HealthResponse, PredictionResponse, RootResponse


app = FastAPI(title="Pneumonia Detection API", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

inference_service = InferenceService()
gradcam_service: GradCAMService | None = None


@app.on_event("startup")
def startup() -> None:
    global gradcam_service
    inference_service.load_model()
    if inference_service.model is None:
        raise RuntimeError("Model failed to load")
    gradcam_service = GradCAMService(inference_service.model)


def error_response(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


@app.get("/", response_model=RootResponse)
def root() -> dict[str, str]:
    return {"message": "Pneumonia Detection API"}


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME, "version": API_VERSION}


@app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict(file: UploadFile = File(...)):
    try:
        data, _ = await read_validated_image(file)
        return inference_service.predict_bytes(data)
    except ImageValidationError as exc:
        return error_response(str(exc), status_code=400)
    except Exception:
        return error_response("Prediction failed", status_code=500)


@app.post("/gradcam", response_model=GradCAMResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def gradcam(file: UploadFile = File(...)):
    try:
        data, image = await read_validated_image(file)
        if gradcam_service is None:
            raise RuntimeError("Grad-CAM service is not initialized")
        result = gradcam_service.generate(image, preprocess_image_bytes(data))
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "gradcam_image": result["gradcam_image"],
        }
    except ImageValidationError as exc:
        return error_response(str(exc), status_code=400)
    except Exception:
        return error_response("Grad-CAM generation failed", status_code=500)
