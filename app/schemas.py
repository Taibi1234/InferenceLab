from typing import Annotated

from pydantic import BaseModel, Field, FiniteFloat


FeatureVector = Annotated[
    list[FiniteFloat],
    Field(min_length=13, max_length=13),
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