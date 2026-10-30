"""
Tests for the FastAPI application.
Uses TestClient to test all endpoints without a running server.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Try importing fastapi and httpx ──────────────────────────────────────────
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
httpx = pytest.importorskip("httpx", reason="httpx not installed")

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """A complete, valid application payload."""
    return {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 1.0,
        "DAYS_BIRTH": -9461,           # ~26 years
        "DAYS_EMPLOYED": -637,
        "DAYS_REGISTRATION": -3648.0,
        "DAYS_ID_PUBLISH": -2120,
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000.0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_POPULATION_RELATIVE": 0.018801,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "EXT_SOURCE_1": 0.083,
        "EXT_SOURCE_2": 0.263,
        "EXT_SOURCE_3": 0.139,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_PHONE": 1,
        "FLAG_EMAIL": 0,
    }


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "model_version" in data

    def test_health_response_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["model_version"], str)


class TestPredictEndpoint:
    def test_predict_without_model_returns_503(self, client, valid_payload):
        """When model is not loaded, /predict should return 503."""
        import api.main as main_module
        original_pipeline = main_module._pipeline
        main_module._pipeline = None
        try:
            resp = client.post("/predict", json=valid_payload)
            assert resp.status_code == 503
        finally:
            main_module._pipeline = original_pipeline

    def test_predict_invalid_gender_rejected(self, client, valid_payload):
        """Invalid CODE_GENDER should return 422."""
        payload = valid_payload.copy()
        payload["CODE_GENDER"] = "Z"
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_contract_type_rejected(self, client, valid_payload):
        """Invalid NAME_CONTRACT_TYPE should return 422."""
        payload = valid_payload.copy()
        payload["NAME_CONTRACT_TYPE"] = "Bad type"
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_underage_rejected(self, client, valid_payload):
        """Applicant age < 18 should return 422."""
        payload = valid_payload.copy()
        payload["DAYS_BIRTH"] = -100  # < 1 year old
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_negative_income_rejected(self, client, valid_payload):
        """Negative income should fail validation."""
        payload = valid_payload.copy()
        payload["AMT_INCOME_TOTAL"] = -1000
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_with_mock_model(self, client, valid_payload):
        """With a mock model, /predict should return 200 with correct schema."""
        import api.main as main_module

        mock_pipe = MagicMock()
        mock_pipe.predict_proba.return_value = np.array([[0.75, 0.25]])

        original_pipeline = main_module._pipeline
        original_threshold = main_module._threshold
        main_module._pipeline = mock_pipe
        main_module._threshold = 0.5
        try:
            resp = client.post("/predict", json=valid_payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "default_probability" in data
            assert "prediction" in data
            assert "risk_segment" in data
            assert "threshold" in data
            assert 0.0 <= data["default_probability"] <= 1.0
            assert data["prediction"] in [0, 1]
            assert data["risk_segment"] in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
        finally:
            main_module._pipeline = original_pipeline
            main_module._threshold = original_threshold

    def test_predict_with_real_trained_pipeline(self, client, valid_payload, tiny_app_df):
        """Verify end-to-end prediction with a real fitted scikit-learn pipeline."""
        import api.main as main_module
        from credit_risk.features import build_app_features
        from credit_risk.model import build_lr_pipeline, _select_features

        df = tiny_app_df.copy()
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, float("nan"))
        df = build_app_features(df)
        num_cols, cat_cols = _select_features(df)
        X = df.drop(columns=["TARGET", "SK_ID_CURR"])
        y = df["TARGET"]

        real_pipe = build_lr_pipeline(num_cols, cat_cols)
        real_pipe.fit(X, y)

        original_pipeline = main_module._pipeline
        original_threshold = main_module._threshold
        main_module._pipeline = real_pipe
        main_module._threshold = 0.5
        try:
            resp = client.post("/predict", json=valid_payload)
            assert resp.status_code == 200
            data = resp.json()
            assert 0.0 <= data["default_probability"] <= 1.0
            assert data["prediction"] in (0, 1)
            assert data["risk_segment"] in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")
        finally:
            main_module._pipeline = original_pipeline
            main_module._threshold = original_threshold


class TestInputSchemaValidation:
    def test_missing_required_field(self, client, valid_payload):
        """Omitting a required field should return 422."""
        payload = valid_payload.copy()
        del payload["AMT_INCOME_TOTAL"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_cnt_children_negative_rejected(self, client, valid_payload):
        payload = valid_payload.copy()
        payload["CNT_CHILDREN"] = -1
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_when_absent_returns_404(self, client):
        """If model_evaluation.json doesn't exist, return 404."""
        metrics_path = Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "model_evaluation.json"
        if not metrics_path.exists():
            resp = client.get("/metrics")
            assert resp.status_code == 404

    def test_metrics_when_present_returns_200(self, client):
        """If model_evaluation.json exists, return 200 with JSON."""
        metrics_path = Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "model_evaluation.json"
        if metrics_path.exists():
            resp = client.get("/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, dict)
