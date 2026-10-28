import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    PREDICTIONS_TOTAL,
)
from app.model_service import ModelService
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)


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
    description="Serve monitored individual and batch predictions.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def collect_request_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start_time
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=path,
            status_code=str(status_code),
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            path=path,
        ).observe(duration)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded,
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    predicted_class, confidence = model_service.predict(request.features)
    PREDICTIONS_TOTAL.labels(endpoint="single").inc()

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    results = model_service.predict_batch(request.rows)
    PREDICTIONS_TOTAL.labels(endpoint="batch").inc(len(results))

    predictions = [
        PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
        )
        for predicted_class, confidence in results
    ]

    return BatchPredictionResponse(
        count=len(predictions),
        predictions=predictions,
    )