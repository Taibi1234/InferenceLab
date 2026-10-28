import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.model_service import ModelService
from app.schemas import PredictionRequest, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

model_service = ModelService(MODEL_PATH)


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="InferenceLab API",
    description="Serve predictions from the selected wine-classification model.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    predicted_class, confidence = model_service.predict(request.features)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
    )