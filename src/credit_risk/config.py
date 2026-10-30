"""
Centralised configuration for the Credit Risk platform.
All paths are relative to the project root so the project is portable.
"""
from pathlib import Path

# ── Project root (two levels up from this file: src/credit_risk/config.py) ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data directories ─────────────────────────────────────────────────────────
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SYNTHETIC = PROJECT_ROOT / "data" / "synthetic"

# ── Raw CSV files ─────────────────────────────────────────────────────────────
APP_TRAIN_CSV = DATA_RAW / "application_train.csv"
BUREAU_CSV = DATA_RAW / "bureau.csv"
PREV_APP_CSV = DATA_RAW / "previous_application.csv"
INSTALLMENTS_CSV = DATA_RAW / "installments_payments.csv"
COLUMNS_DESC_CSV = DATA_RAW / "HomeCredit_columns_description.csv"

# Optional files (may not exist)
APP_TEST_CSV = DATA_RAW / "application_test.csv"
BUREAU_BALANCE_CSV = DATA_RAW / "bureau_balance.csv"
POS_CASH_CSV = DATA_RAW / "POS_CASH_balance.csv"
CREDIT_CARD_CSV = DATA_RAW / "credit_card_balance.csv"

# ── Processed Parquet outputs ─────────────────────────────────────────────────
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
BUREAU_FEATS_PARQUET = DATA_PROCESSED / "bureau_features.parquet"
PREV_FEATS_PARQUET = DATA_PROCESSED / "prev_app_features.parquet"
INSTALMENT_FEATS_PARQUET = DATA_PROCESSED / "instalment_features.parquet"

# ── Model artifacts ───────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "model_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# ── Output directories ────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# ── Synthetic operational data ────────────────────────────────────────────────
SYNTHETIC_OPS_CSV = DATA_SYNTHETIC / "operational_records.csv"

# ── SQL directory ─────────────────────────────────────────────────────────────
SQL_DIR = PROJECT_ROOT / "sql"

# ── Modelling parameters ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# DuckDB memory limit (conservative for 1.8 GB available)
DUCKDB_MEMORY_LIMIT = "1.2GB"
DUCKDB_THREADS = 2

# Encoding used by Home Credit CSVs (some have latin-1 bytes)
CSV_ENCODING = "latin-1"
