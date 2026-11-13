from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    """Input schema for single-row real-time inference."""

    feature_2: float = Field(..., description="Leading predictive feature 2")
    feature_13: float = Field(..., description="Leading predictive feature 13")
    feature_9: float = Field(..., description="Leading predictive feature 9")
    feature_11: float = Field(..., description="Leading predictive feature 11")
    feature_18: float = Field(..., description="Non-linear predictive feature 18")


class PredictionOutput(BaseModel):
    """Output schema for single-row inference."""

    target_pred: float


class HealthResponse(BaseModel):
    """Service health and active model metadata."""

    status: str
    model_name: str
    model_version: str
    required_features: list[str]