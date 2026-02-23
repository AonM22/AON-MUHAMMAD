from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

try:
    from src.predict import load_model, predict_price
except ModuleNotFoundError:
    from predict import load_model, predict_price


MODEL_PATH = Path("models/best_model.pkl")
MODEL = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location: str = Field(min_length=1)
    baths: int = Field(ge=0, le=20)
    property_type: str = Field(min_length=1)
    city: str = Field(min_length=1)
    province_name: str = Field(min_length=1)
    bedrooms: int = Field(ge=0, le=20)
    Total_Area: float = Field(gt=0)
    date_added: str = Field(min_length=6)


class PredictionResponse(BaseModel):
    predicted_price_in_lac: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model once at application startup."""
    global MODEL
    try:
        MODEL = load_model(MODEL_PATH)
    except FileNotFoundError:
        MODEL = None
    yield


app = FastAPI(title="Housing Price ML System", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")

    prediction = predict_price(request.model_dump(), model_path=MODEL_PATH)
    return PredictionResponse(predicted_price_in_lac=prediction)
