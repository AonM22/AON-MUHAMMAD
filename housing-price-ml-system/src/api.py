from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

try:
    from src.predict import load_model, predict_price
    from src.schema import HousePriceInput
except ModuleNotFoundError:
    from predict import load_model, predict_price
    from schema import HousePriceInput


MODEL_PATH = Path("models/best_model.pkl")
MODEL = None


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "message": "Invalid request payload. Please provide all required fields with valid types.",
            "errors": exc.errors(),
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
def predict(request: HousePriceInput) -> dict[str, float]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")

    prediction = predict_price(request.model_dump(), model_path=MODEL_PATH)
    return {"predicted_price_in_lac": prediction}
