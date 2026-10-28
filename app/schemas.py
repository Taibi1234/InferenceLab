from typing import Annotated, Literal

from pydantic import BaseModel, Field, FiniteFloat


FeatureVector = Annotated[
    list[FiniteFloat],
    Field(min_length=13, max_length=13),
]

DriftStatus = Literal[
    "no_significant_drift",
    "moderate_drift",
    "significant_drift",
]


class PredictionRequest(BaseModel):
    features: FeatureVector


class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float = Field(ge=0.0, le=1.0)


class BatchPredictionRequest(BaseModel):
    rows: list[FeatureVector] = Field(
        min_length=1,
        max_length=1000,
    )


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class DriftCheckRequest(BaseModel):
    rows: list[FeatureVector] = Field(
        min_length=30,
        max_length=1000,
    )


class FeatureDriftResult(BaseModel):
    feature_name: str
    psi: float = Field(ge=0.0)
    status: DriftStatus


class DriftCheckResponse(BaseModel):
    sample_count: int
    overall_status: DriftStatus
    moderate_threshold: float
    significant_threshold: float
    features: list[FeatureDriftResult]