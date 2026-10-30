"""
FastAPI application — Credit Risk Prediction API.

Endpoints:
    GET  /health     → health check + model status
    POST /predict    → single-application credit-risk prediction
    GET  /metrics    → latest model evaluation metrics

Start:
    uvicorn api.main:app --reload --port 8000

Or via script:
    python scripts/run_api.py
"""
from __future__ import annotations

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))

from credit_risk.config import MODEL_METADATA_PATH, MODEL_PIPELINE_PATH, METRICS_DIR
from credit_risk.logger import get_logger
from api.schemas import (
    ApplicationInput,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    classify_risk_segment,
)

logger = get_logger("api.main")

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Analytics API",
    description=(
        "Prototype credit-risk scoring API built on the Home Credit Default Risk dataset. "
        "NOT for production lending decisions or regulatory use."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Model state ───────────────────────────────────────────────────────────────
_pipeline = None
_threshold = 0.5
_model_version = "not_loaded"

PREDICTION_LOG_PATH = METRICS_DIR / "prediction_log.jsonl"


def _load_model_on_startup():
    global _pipeline, _threshold, _model_version
    try:
        from credit_risk.model import load_model
        _pipeline, _threshold = load_model()
        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH) as f:
                meta = json.load(f)
            _model_version = meta.get("winner", "unknown") + "@" + meta.get("trained_at", "?")[:10]
        logger.info("Model loaded successfully (threshold=%.4f).", _threshold)
    except FileNotFoundError as e:
        logger.warning("Model not available at startup: %s", e)
        logger.warning("Run scripts/04_train.py first, then restart the API.")


@app.on_event("startup")
async def startup_event():
    _load_model_on_startup()


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail, error_type="HTTPException").model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error.", error_type=type(exc).__name__
        ).model_dump(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """Health check — confirms the API is running and whether the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: ApplicationInput):
    """
    Score a single loan application for credit default risk.

    Returns the predicted default probability, binary classification,
    and risk segment. All inputs are validated server-side.

    **Disclaimer**: Research prototype only. Not a regulatory credit decision.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Run scripts/04_train.py and restart the API."
            ),
        )

    # Convert input to DataFrame (single row)
    input_dict = application.model_dump()

    # Add derived features that the pipeline expects
    try:
        input_df = pd.DataFrame([input_dict])

        # Replace DAYS_EMPLOYED anomaly before feature engineering
        if "DAYS_EMPLOYED" in input_df.columns and input_df["DAYS_EMPLOYED"].iloc[0] == 365243:
            input_df["DAYS_EMPLOYED"] = float("nan")

        # Reproduce the exact same feature engineering as training
        from credit_risk.features import build_app_features
        input_df = build_app_features(input_df)

        # Align with model input features (reindexing fills missing historical features with NaN)
        if hasattr(_pipeline, "feature_names_in_"):
            input_df = input_df.reindex(columns=_pipeline.feature_names_in_)

        prob = float(_pipeline.predict_proba(input_df)[0, 1])
        pred = int(prob >= _threshold)
        risk_segment = classify_risk_segment(prob)

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    # ── Prediction log (no personal identifiers) ──────────────────────────────
    log_entry = {
        "ts": datetime.utcnow().isoformat(),
        "default_probability": round(prob, 4),
        "prediction": pred,
        "risk_segment": risk_segment,
        "name_contract_type": input_dict.get("NAME_CONTRACT_TYPE"),
        "code_gender": input_dict.get("CODE_GENDER"),
        "name_income_type": input_dict.get("NAME_INCOME_TYPE"),
    }
    try:
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTION_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as log_exc:
        logger.warning("Could not write prediction log: %s", log_exc)

    return PredictionResponse(
        default_probability=round(prob, 4),
        prediction=pred,
        risk_segment=risk_segment,
        threshold=round(_threshold, 4),
    )


@app.get("/metrics", tags=["Operations"])
async def get_metrics():
    """Return the latest model evaluation metrics from disk."""
    metrics_path = METRICS_DIR / "model_evaluation.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model evaluation metrics not found. Run scripts/04_train.py first.",
        )
    with open(metrics_path) as f:
        return json.load(f)
