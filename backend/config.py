from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetb0_final.keras"
STATIC_DIR = Path(__file__).resolve().parent / "static"
GRADCAM_STATIC_DIR = STATIC_DIR / "gradcam"

MODEL_NAME = "EfficientNetB0"
API_VERSION = "1.0"
PREDICTION_THRESHOLD = 0.50
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
