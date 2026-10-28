from pydantic import BaseModel, Field, FiniteFloat


class PredictionRequest(BaseModel):
    features: list[FiniteFloat] = Field(
        min_length=13,
        max_length=13,
        description="The 13 wine features in the dataset's original order.",
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float = Field(ge=0.0, le=1.0)