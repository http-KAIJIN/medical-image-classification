from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class RootResponse(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: float
    threshold: float


class GradCAMResponse(BaseModel):
    prediction: str
    confidence: float
    gradcam_image: str


class ErrorResponse(BaseModel):
    error: str
