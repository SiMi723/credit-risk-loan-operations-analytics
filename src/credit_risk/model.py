"""
Preprocessing pipeline and model training.

Design decisions
----------------
- Logistic Regression baseline: simple, interpretable, fast.
- LightGBM: handles missing values natively, good for tabular credit data.
- class_weight / scale_pos_weight: address ~8% default-rate imbalance.
- StandardScaler only for LR (LightGBM is invariant to scale).
- SimpleImputer: median for numeric, most-frequent for categorical.
- OrdinalEncoder for LightGBM (supports integer-coded categoricals).
- OneHotEncoder for LR (needs linear separability).
- Threshold optimised on validation F1 (not hard-coded at 0.5).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_risk.config import (
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    ID_COL,
    TEST_SIZE,
)
from credit_risk.logger import get_logger

logger = get_logger("credit_risk.model")

# ── Feature selection ─────────────────────────────────────────────────────────
# Columns to exclude from the feature matrix
EXCLUDE_COLS = {
    TARGET_COL,
    ID_COL,
    # Leakage-prone or purely administrative
    "SK_ID_BUREAU",
    "SK_ID_PREV",
}

# Known high-cardinality / free-text columns that are not useful for modelling
DROP_COLS = [
    "ORGANIZATION_TYPE",  # 58 categories — keep only if needed; OHE explodes size
]

# Numeric columns with special semantics
NUMERIC_SPECIAL: dict[str, float | None] = {
    "DAYS_EMPLOYED": None,   # 365243 anomaly already replaced before this stage
}


def _select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) lists from the DataFrame."""
    all_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in DROP_COLS]
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def build_lr_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Logistic Regression pipeline with StandardScaler + OneHotEncoder."""
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ], remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            C=0.1,
            random_state=RANDOM_STATE,
        )),
    ])


def build_lgbm_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """LightGBM pipeline with OrdinalEncoder (LightGBM handles missing natively)."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ], remainder="drop")

    # Class weight via scale_pos_weight — set after seeing data
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=-1,
        class_weight="balanced",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold that maximises F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) == 0, 0,
        2 * precision * recall / (precision + recall + 1e-10),
    )
    best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
    return float(thresholds[best_idx])


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute and return a full evaluation metrics dictionary."""
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    if threshold is None:
        threshold = find_best_threshold(y_val.values, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_name": name,
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "val_size": int(len(y_val)),
        "val_positive_rate": round(float(y_val.mean()), 4),
    }
    logger.info(
        "%s | ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | threshold=%.4f",
        name, roc_auc, pr_auc, metrics["f1"], threshold,
    )
    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """
    Full training pipeline:
    1. Stratified 80/20 split.
    2. Train LR baseline and LightGBM.
    3. Evaluate both on the same validation fold.
    4. Select winner by ROC-AUC (tie-break: PR-AUC).
    5. Persist the winning pipeline + metadata.
    6. Return the full evaluation report.
    """
    if MODEL_PIPELINE_PATH.exists() and not force_retrain:
        logger.info("Model already exists at %s — skipping training.", MODEL_PIPELINE_PATH.name)
        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH) as f:
                return json.load(f)

    logger.info("Starting training pipeline …")

    # ── Split ────────────────────────────────────────────────────────────────
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns] +
                        [c for c in DROP_COLS if c in df.columns])

    num_cols, cat_cols = _select_features(df)
    logger.info("Features: %d numeric, %d categorical", len(num_cols), len(cat_cols))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows | Val: %d rows", len(X_train), len(X_val))

    results = {}

    # ── Logistic Regression baseline ─────────────────────────────────────────
    logger.info("Training Logistic Regression baseline …")
    t0 = time.time()
    lr_pipe = build_lr_pipeline(num_cols, cat_cols)
    lr_pipe.fit(X_train, y_train)
    lr_time = time.time() - t0
    logger.info("LR training time: %.1f s", lr_time)
    lr_metrics = evaluate_model("LogisticRegression", lr_pipe, X_val, y_val)
    lr_metrics["train_time_seconds"] = round(lr_time, 2)
    results["LogisticRegression"] = lr_metrics

    # ── LightGBM ─────────────────────────────────────────────────────────────
    logger.info("Training LightGBM …")
    t0 = time.time()
    lgbm_pipe = build_lgbm_pipeline(num_cols, cat_cols)
    lgbm_pipe.fit(X_train, y_train)
    lgbm_time = time.time() - t0
    logger.info("LightGBM training time: %.1f s", lgbm_time)
    lgbm_metrics = evaluate_model("LightGBM", lgbm_pipe, X_val, y_val)
    lgbm_metrics["train_time_seconds"] = round(lgbm_time, 2)
    results["LightGBM"] = lgbm_metrics

    # ── Model selection ───────────────────────────────────────────────────────
    winner_name = max(
        results,
        key=lambda k: (results[k]["roc_auc"], results[k]["pr_auc"]),
    )
    winner_pipe = lgbm_pipe if winner_name == "LightGBM" else lr_pipe
    winner_threshold = results[winner_name]["threshold"]
    logger.info("Winner: %s (ROC-AUC=%.4f)", winner_name, results[winner_name]["roc_auc"])

    # ── Persist ───────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_pipe, MODEL_PIPELINE_PATH)
    logger.info("Model pipeline saved → %s", MODEL_PIPELINE_PATH.name)

    metadata = {
        "winner": winner_name,
        "threshold": winner_threshold,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "num_features": len(num_cols),
        "cat_features": len(cat_cols),
        "random_state": RANDOM_STATE,
        "all_results": results,
        "selection_criterion": "ROC-AUC (tie-break PR-AUC)",
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    eval_path = METRICS_DIR / "model_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Evaluation metrics saved → %s", eval_path.name)

    return metadata


def load_model() -> tuple[Pipeline, float]:
    """Load the saved model pipeline and threshold from disk."""
    if not MODEL_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PIPELINE_PATH}. "
            "Run scripts/04_train.py first."
        )
    pipeline = joblib.load(MODEL_PIPELINE_PATH)
    threshold = 0.5  # fallback
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH) as f:
            meta = json.load(f)
        threshold = float(meta.get("threshold", 0.5))
    logger.info("Model loaded from %s (threshold=%.4f)", MODEL_PIPELINE_PATH.name, threshold)
    return pipeline, threshold
