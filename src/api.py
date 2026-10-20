
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.predict import load_model, predict_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

REQUESTS = Counter(
    "prediction_requests_total",
    "Number of API requests",
    ["method", "endpoint", "status"],
)

LATENCY = Histogram(
    "prediction_request_duration_seconds",
    "API request latency in seconds",
)

MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL

    model_path = Path("artifacts/heart_disease_model.joblib")

    if model_path.exists():
        MODEL = load_model(model_path)
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model file is missing")

    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start

        REQUESTS.labels(
            request.method,
            request.url.path,
            str(status),
        ).inc()

        LATENCY.observe(elapsed)

        # Do not log patient feature values.
        logger.info(
            "method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            status,
            elapsed * 1000,
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict")
async def predict(request: Request):
    # Return 503 before validating input if the model is unavailable.
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Model is not loaded"},
        )

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=422,
            content={"detail": "Request body must be valid JSON"},
        )

    if not isinstance(payload, dict) or not payload:
        return JSONResponse(
            status_code=422,
            content={"detail": "A non-empty JSON object is required"},
        )

    try:
        result = predict_one(payload, model=MODEL)
        return result
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )
