"""
Tests for the model module (preprocessing + training logic).
Uses tiny synthetic data only — does NOT require raw CSV files.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from credit_risk.features import build_app_features
from credit_risk.model import (
    _select_features,
    build_lr_pipeline,
    build_lgbm_pipeline,
    evaluate_model,
    find_best_threshold,
)


@pytest.fixture
def feature_df(tiny_app_df):
    """Tiny app DataFrame with engineered features."""
    df = tiny_app_df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, float("nan"))
    return build_app_features(df)


class TestFeatureSelection:
    def test_excludes_target(self, feature_df):
        num_cols, cat_cols = _select_features(feature_df)
        assert "TARGET" not in num_cols
        assert "TARGET" not in cat_cols

    def test_excludes_id(self, feature_df):
        num_cols, cat_cols = _select_features(feature_df)
        assert "SK_ID_CURR" not in num_cols
        assert "SK_ID_CURR" not in cat_cols

    def test_returns_lists(self, feature_df):
        num_cols, cat_cols = _select_features(feature_df)
        assert isinstance(num_cols, list)
        assert isinstance(cat_cols, list)
        assert len(num_cols) > 0


class TestLRPipeline:
    def test_fit_predict(self, feature_df):
        num_cols, cat_cols = _select_features(feature_df)
        X = feature_df.drop(columns=["TARGET", "SK_ID_CURR"])
        y = feature_df["TARGET"]
        pipe = build_lr_pipeline(num_cols, cat_cols)
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)
        assert probas.shape == (len(X), 2)
        assert (probas >= 0).all()
        assert (probas <= 1).all()

    def test_probas_sum_to_one(self, feature_df):
        num_cols, cat_cols = _select_features(feature_df)
        X = feature_df.drop(columns=["TARGET", "SK_ID_CURR"])
        y = feature_df["TARGET"]
        pipe = build_lr_pipeline(num_cols, cat_cols)
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


class TestLGBMPipeline:
    def test_fit_predict(self, feature_df):
        pytest.importorskip("lightgbm")
        num_cols, cat_cols = _select_features(feature_df)
        X = feature_df.drop(columns=["TARGET", "SK_ID_CURR"])
        y = feature_df["TARGET"]
        pipe = build_lgbm_pipeline(num_cols, cat_cols)
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)
        assert probas.shape == (len(X), 2)


class TestThresholdOptimisation:
    def test_threshold_in_range(self):
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.random.uniform(0, 1, len(y_true))
        threshold = find_best_threshold(y_true, y_prob)
        assert 0.0 <= threshold <= 1.0

    def test_threshold_type(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        threshold = find_best_threshold(y_true, y_prob)
        assert isinstance(threshold, float)


class TestEvaluateModel:
    def test_metrics_keys_present(self, feature_df):
        num_cols, cat_cols = _select_features(feature_df)
        X = feature_df.drop(columns=["TARGET", "SK_ID_CURR"])
        y = feature_df["TARGET"]
        pipe = build_lr_pipeline(num_cols, cat_cols)
        pipe.fit(X, y)
        metrics = evaluate_model("TestLR", pipe, X, y)
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "confusion_matrix"]:
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_in_valid_range(self, feature_df):
        num_cols, cat_cols = _select_features(feature_df)
        X = feature_df.drop(columns=["TARGET", "SK_ID_CURR"])
        y = feature_df["TARGET"]
        pipe = build_lr_pipeline(num_cols, cat_cols)
        pipe.fit(X, y)
        metrics = evaluate_model("TestLR", pipe, X, y)
        for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]:
            val = metrics[metric]
            assert 0.0 <= val <= 1.0, f"{metric}={val} out of [0, 1]"
