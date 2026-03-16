"""
GetAround — Pricing API
FastAPI with 4 endpoints:
  GET  /         → welcome message
  GET  /health   → API health check
  GET  /cars/stats → dataset statistics
  POST /predict  → predict daily rental price (named fields, named response)
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
import os

# ─── App definition ───────────────────────────────────────────────────────────
app = FastAPI(
    title="GetAround Pricing API",
    description=(
        "Machine Learning API for optimal daily rental price prediction. "
        "Trained on the GetAround pricing dataset (4,843 vehicles). "
        "Built with Scikit-Learn RandomForestRegressor + GridSearchCV."
    ),
    version="1.0.0",
    contact={"name": "Philippe TOSO", "email": "contact@example.com"},
)

# ─── Load model at startup ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
STATS_PATH = os.path.join(BASE_DIR, "stats.json")

model = None
stats = None

@app.on_event("startup")
def load_model():
    global model, stats
    model = joblib.load(MODEL_PATH)
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            stats = json.load(f)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class CarFeatures(BaseModel):
    model_key: str
    mileage: float
    engine_power: float
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

    class Config:
        json_schema_extra = {
            "example": {
                "model_key": "Renault",
                "mileage": 50000,
                "engine_power": 120,
                "fuel": "diesel",
                "paint_color": "grey",
                "car_type": "sedan",
                "private_parking_available": True,
                "has_gps": True,
                "has_air_conditioning": True,
                "automatic_car": False,
                "has_getaround_connect": False,
                "has_speed_regulator": True,
                "winter_tires": True,
            }
        }


class BatchCarFeatures(BaseModel):
    cars: list[CarFeatures]


class PredictionResponse(BaseModel):
    predicted_price_per_day: float
    currency: str
    confidence_interval: dict


class BatchPredictionResponse(BaseModel):
    predictions: list[dict]
    count: int


# ─── Helpers ──────────────────────────────────────────────────────────────────
def car_to_dataframe(car: CarFeatures) -> pd.DataFrame:
    """Convert a CarFeatures Pydantic model to a pandas DataFrame row."""
    data = {
        "model_key": [car.model_key],
        "mileage": [float(car.mileage)],
        "engine_power": [float(car.engine_power)],
        "fuel": [car.fuel],
        "paint_color": [car.paint_color],
        "car_type": [car.car_type],
        "private_parking_available": [int(car.private_parking_available)],
        "has_gps": [int(car.has_gps)],
        "has_air_conditioning": [int(car.has_air_conditioning)],
        "automatic_car": [int(car.automatic_car)],
        "has_getaround_connect": [int(car.has_getaround_connect)],
        "has_speed_regulator": [int(car.has_speed_regulator)],
        "winter_tires": [int(car.winter_tires)],
    }
    cols = ["mileage", "engine_power", "model_key", "fuel", "paint_color", "car_type",
            "private_parking_available", "has_gps", "has_air_conditioning",
            "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires"]
    return pd.DataFrame(data)[cols]


def build_confidence_interval(price: float) -> dict:
    """
    Estimate a confidence interval around the prediction.
    We use ±15% as a practical approximation for a RandomForest model
    (actual prediction intervals would require quantile regression).
    """
    margin = price * 0.15
    return {
        "low": round(max(0.0, price - margin), 2),
        "high": round(price + margin, 2),
        "method": "±15% empirical margin",
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", summary="Welcome", tags=["General"])
def root():
    """Welcome message and API description."""
    return {
        "message": "Welcome to the GetAround Pricing API 🚗",
        "description": "Predict the optimal daily rental price for any car.",
        "endpoints": {
            "GET  /":           "This welcome message",
            "GET  /health":     "API health check",
            "GET  /cars/stats": "Dataset statistics",
            "POST /predict":    "Predict daily rental price (single car)",
            "POST /predict/batch": "Predict for multiple cars at once",
            "GET  /docs":       "Full Swagger UI documentation",
        },
        "version": "1.0.0",
    }


@app.get("/health", summary="Health Check", tags=["General"])
def health():
    """Check that the API and model are operational."""
    model_loaded = model is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "api_version": "1.0.0",
    }


@app.get("/cars/stats", summary="Dataset Statistics", tags=["Analytics"])
def cars_stats():
    """
    Return descriptive statistics about the training dataset.
    Useful to contextualize a price prediction.
    """
    if stats:
        return stats
    return JSONResponse(
        status_code=503,
        content={"error": "Statistics not available. Run the notebook first to generate stats.json."},
    )


@app.post("/predict", response_model=PredictionResponse,
          summary="Predict Daily Rental Price", tags=["Prediction"])
def predict(car: CarFeatures):
    """
    Predict the optimal daily rental price for a single car.

    **Input:** Car characteristics (model, mileage, engine power, equipment...)

    **Output:**
    - `predicted_price_per_day`: Recommended price in EUR
    - `currency`: Always "EUR"
    - `confidence_interval`: Estimated low/high bounds (±15%)
    """
    df_input = car_to_dataframe(car)
    raw_price = float(model.predict(df_input)[0])
    price = round(raw_price, 2)

    return PredictionResponse(
        predicted_price_per_day=price,
        currency="EUR",
        confidence_interval=build_confidence_interval(price),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse,
          summary="Batch Price Prediction", tags=["Prediction"])
def predict_batch(payload: BatchCarFeatures):
    """
    Predict the optimal daily rental price for **multiple cars** in one request.

    **Input:** List of cars with their characteristics.

    **Output:** List of predictions with confidence intervals, plus total count.
    """
    results = []
    for car in payload.cars:
        df_input = car_to_dataframe(car)
        raw_price = float(model.predict(df_input)[0])
        price = round(raw_price, 2)
        results.append({
            "model_key": car.model_key,
            "car_type": car.car_type,
            "predicted_price_per_day": price,
            "currency": "EUR",
            "confidence_interval": build_confidence_interval(price),
        })

    return BatchPredictionResponse(predictions=results, count=len(results))
