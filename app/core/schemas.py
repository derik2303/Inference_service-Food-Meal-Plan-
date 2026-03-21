from pydantic import BaseModel
from typing import List, Optional, Any


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    predictions: List[Prediction]
    dish_name: str | None = None
    calories_kcal: float | None = None
    latency_ms: int


class AsyncPredictResponse(BaseModel):
    job_id: str
    request_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
