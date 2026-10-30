# Credit Risk & Loan Operations Analytics Platform

A complete, locally-runnable, well-tested portfolio project demonstrating an end-to-end credit-risk analytics pipeline built on the **Home Credit Default Risk** dataset (Kaggle, 2018).

---

## ⚠️ Important Disclaimer

> This project is a **portfolio research prototype** only.  
> - The dataset is from **Home Credit**, a consumer-lending company. It is **not** an Indian SME/MSME dataset.  
> - This model was **not** developed for, trained on, or validated by **Protium Finance**.  
> - This platform does **not** constitute a regulatory-approved credit decision system, production lending tool, or compliance framework.  
> - All synthetic operational records are clearly labelled and are not real loan data.

---

## Dataset Provenance

| File | Source | Rows | Description |
|------|--------|------|-------------|
| `application_train.csv` | [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) | 307,511 | Main application data + TARGET |
| `bureau.csv` | Same | 1,716,428 | Credit bureau records per applicant |
| `previous_application.csv` | Same | 1,670,214 | Previous Home Credit applications |
| `installments_payments.csv` | Same | 13,605,401 | Payment history per installment |
| `HomeCredit_columns_description.csv` | Same | — | Data dictionary |

**Absent optional files**: `bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `application_test.csv` — not included; their absence is logged and handled gracefully.

The dataset is licensed under the [Home Credit Group competition terms](https://www.kaggle.com/c/home-credit-default-risk/rules). Raw data files are excluded from version control via `.gitignore`.

---

## Architecture

```
data/raw/           ← Raw CSVs (gitignored)
   │
scripts/01_ingest.py   ← Schema validation, file inventory, key overlaps
scripts/02_eda.py      ← Missing values, target distribution, correlations
scripts/03_features.py ← DuckDB SQL aggregations → Parquet feature tables
scripts/04_train.py    ← Stratified split, LR + LightGBM, evaluation
scripts/05_monitor.py  ← Operational exceptions, distribution monitoring
scripts/06_benchmark.py ← Inference latency measurement
   │
api/main.py         ← FastAPI: /health, /predict, /metrics
dashboard/app.py    ← Streamlit interactive dashboard
tests/              ← pytest unit tests (synthetic fixtures only)
```

**Memory strategy**: The 13.6M-row `installments_payments.csv` is never loaded into Python memory. DuckDB SQL aggregations run directly against the raw CSV files, producing compact Parquet files at the `SK_ID_CURR` grain.

---

## Repository Structure

```
protium-credit-risk/
├── api/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app
│   └── schemas.py       ← Pydantic input/output models
├── config/
├── dashboard/
│   └── app.py           ← Streamlit dashboard
├── data/
│   ├── raw/             ← Raw CSVs (gitignored)
│   ├── processed/       ← Parquet features (gitignored)
│   └── synthetic/       ← Labelled synthetic operational records
├── models/              ← Saved model artifacts (gitignored)
├── outputs/
│   ├── metrics/         ← Evaluation JSONs (gitignored at runtime)
│   └── reports/         ← EDA and quality reports
├── scripts/
│   ├── 01_ingest.py
│   ├── 02_eda.py
│   ├── 03_features.py
│   ├── 04_train.py
│   ├── 05_monitor.py
│   ├── 06_benchmark.py
│   └── run_api.py
├── sql/
│   ├── bureau_features.sql
│   ├── instalment_features.sql
│   ├── prev_app_features.sql
│   └── schema_validation.sql
├── src/credit_risk/
│   ├── __init__.py
│   ├── config.py        ← Centralised paths & settings
│   ├── data_io.py       ← Memory-aware CSV/Parquet I/O
│   ├── data_quality.py  ← Rule-based validation engine
│   ├── exceptions.py    ← Operational exception checker
│   ├── features.py      ← Feature engineering pipeline
│   ├── logger.py        ← Centralised logging
│   └── model.py         ← Preprocessing + LR + LightGBM
├── tests/
│   ├── conftest.py      ← Synthetic fixtures
│   ├── test_api.py
│   ├── test_data_quality.py
│   ├── test_exceptions.py
│   ├── test_features.py
│   └── test_model.py
├── .gitignore
├── PLAN.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## Setup (Windows)

### Prerequisites
- Python 3.10+ (tested on 3.12.6)
- ~2 GB free RAM (DuckDB keeps memory bounded to 1.2 GB)
- ~2 GB free disk for processed Parquet files + model artifacts
- Home Credit dataset CSV files in `data/raw/`

### Installation

```powershell
# Clone / navigate to project directory
cd c:\Users\simi\Desktop\protium-credit-risk

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Execution — Step by Step

### 1. Data Inventory & Schema Validation
```powershell
python scripts/01_ingest.py
# → outputs/reports/data_inventory.json
# → outputs/reports/data_quality_report.json
```

### 2. Exploratory Data Analysis
```powershell
python scripts/02_eda.py
# → outputs/reports/eda_report.json
# → outputs/reports/missing_values.csv
```

### 3. Feature Engineering
```powershell
python scripts/03_features.py
# → data/processed/bureau_features.parquet
# → data/processed/prev_app_features.parquet
# → data/processed/instalment_features.parquet
# → data/processed/features.parquet  (main feature table)
# ⏱ ~5–10 min (installments aggregation: 13.6M rows via DuckDB)
```

### 4. Model Training & Evaluation
```powershell
python scripts/04_train.py
# → models/model_pipeline.joblib
# → models/model_metadata.json
# → outputs/metrics/model_evaluation.json
# ⏱ ~5–15 min depending on hardware
```

### 5. Operational Monitoring
```powershell
python scripts/05_monitor.py
# → data/synthetic/operational_records.csv
# → outputs/reports/exception_report.json
```

### 6. API Server
```powershell
# Option A — via script
python scripts/run_api.py

# Option B — via uvicorn directly
uvicorn api.main:app --port 8000 --reload

# Test the API
# Health:   http://localhost:8000/health
# Docs:     http://localhost:8000/docs
# Metrics:  http://localhost:8000/metrics
```

### 7. Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

### 8. Latency Benchmark
```powershell
python scripts/06_benchmark.py
# → outputs/metrics/latency_benchmark.json
```

### 9. Run Tests
```powershell
pytest tests/ -v
```

---

## Feature Definitions

### Application-level derived features
| Feature | Formula | Missing handling |
|---------|---------|-----------------|
| `credit_income_ratio` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Income=0 → NaN |
| `annuity_income_ratio` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Income=0 → NaN |
| `age_years` | `-DAYS_BIRTH / 365` | No missing |
| `employed_years` | `-DAYS_EMPLOYED / 365` | `365243` → NaN → median imputed |
| `ext_source_mean` | mean(EXT_SOURCE_1, 2, 3) | Per-source median |

### Bureau aggregation (per SK_ID_CURR)
| Feature | Definition |
|---------|-----------|
| `bureau_credit_count` | Count of bureau records |
| `bureau_active_count` | Count where CREDIT_ACTIVE='Active' |
| `bureau_overdue_max` | Max CREDIT_DAY_OVERDUE |
| `bureau_amt_credit_sum` | Total external credit balance |
| `bureau_days_credit_min` | Most recent bureau entry |

### Installment payment features
| Feature | Definition |
|---------|-----------|
| `instalment_late_count` | Payments where `DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT` |
| `instalment_late_ratio` | Late count / total (proportion of late payments) |
| `instalment_dpd_mean` | Mean days past due (floored at 0) |
| `instalment_amt_diff_mean` | Mean(actual payment − scheduled payment) |
| `instalment_underpay_count` | Payments short by more than 1 unit |

---

## Model Architecture

### Class imbalance
Home Credit has ~8% default rate. Both models address this:
- **Logistic Regression**: `class_weight='balanced'`
- **LightGBM**: `class_weight='balanced'`

### Threshold selection
The decision threshold is optimised on the validation set to maximise F1-score (not hardcoded at 0.5).

### Model selection criterion
Winner = highest validation ROC-AUC; tie-break: PR-AUC.

### Evaluation metrics (actual measured on validation set)
See `outputs/metrics/model_evaluation.json` after running `scripts/04_train.py`.
Key metrics: **ROC-AUC**, **PR-AUC**, **Precision**, **Recall**, **F1**, Confusion Matrix.
Accuracy is reported as a supplementary metric only due to class imbalance.

---

## API Reference

### `GET /health`
Returns API status and model load status.

### `POST /predict`
Accepts a JSON application payload, returns:
```json
{
  "default_probability": 0.2341,
  "prediction": 0,
  "risk_segment": "MEDIUM",
  "threshold": 0.3125,
  "model_version": "LightGBM@2024-...",
  "disclaimer": "Research prototype only..."
}
```

### `GET /metrics`
Returns the full model evaluation JSON from the last training run.

Full interactive docs at `/docs` (Swagger) or `/redoc`.

---

## Data Quality Rules

The `DataQualityChecker` applies these deterministic rules:
- **Missing required fields** (ERROR)
- **Duplicate SK_ID_CURR** (ERROR)
- **Invalid TARGET values** (ERROR, must be 0 or 1)
- **Negative AMT_* amounts** (WARNING)
- **Invalid CODE_GENDER** (WARNING)
- **DAYS_BIRTH outside age 18–100** (WARNING)
- **DAYS_EMPLOYED = 365243** (WARNING — known retiree/unemployed placeholder)
- **EXT_SOURCE values outside [0, 1]** (WARNING)
- **No documents submitted** (WARNING)
- **Credit/income ratio > 20×** (WARNING)

---

## Limitations

1. **Dataset scope**: Home Credit data covers consumer loans in multiple countries. It does not represent Indian MSME/SME credit risk profiles.
2. **Missing optional tables**: `bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `application_test.csv` are absent. This reduces feature richness compared to the full Kaggle competition setup.
3. **Memory**: With ~1.8 GB free RAM, the full DataFrame is tight. DuckDB SQL aggregations mitigate this, but the merged feature table (~300 MB+ in RAM) may be close to limits.
4. **Not production-ready**: Single-process FastAPI server; no authentication, rate limiting, database backend, or monitoring infrastructure.
5. **Local latency only**: Inference latency measurements are local in-process results and do not represent production API response times.
6. **No time-series split**: A random stratified split is used. For production models, a temporal split should be used to prevent future-data leakage.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `PermissionError` on `data/raw/` | Run as Administrator or fix ACL with `icacls data /grant Users:F /T` |
| `MemoryError` during feature build | Reduce DuckDB memory limit in `src/credit_risk/config.py`: `DUCKDB_MEMORY_LIMIT = "800MB"` |
| `FileNotFoundError: Model not found` | Run `scripts/04_train.py` first |
| Feature table missing | Run `scripts/03_features.py` first |
| LightGBM import error | `pip install lightgbm` |
| `uvicorn` not found | `pip install "uvicorn[standard]"` |

---

## Suggested Resume Bullets

*(Use only after running the pipeline and verifying actual results)*

- Built an end-to-end credit-risk analytics platform on the Home Credit Default Risk dataset (307K applications, 13.6M installment records) using Python, DuckDB, and Parquet, keeping peak memory usage under 1.5 GB on commodity hardware.
- Engineered 30+ features from four raw tables by aggregating 13.6M installment payment records to customer grain via DuckDB SQL, deriving payment delinquency rates, days past due, and credit utilisation metrics.
- Trained and evaluated Logistic Regression and LightGBM models on an 80/20 stratified split; selected winner by validation ROC-AUC; stored all evaluation metrics in machine-readable JSON (no fabricated results).
- Built a FastAPI prediction microservice with Pydantic input validation, prediction logging, and a `/metrics` endpoint surfacing live model evaluation data.
- Implemented a deterministic rules-based exception-checking engine flagging 9 operational risk categories (invalid income, underage applicants, extreme credit ratios, anomalous employment codes, etc.).
- Delivered an interactive Streamlit dashboard showing risk-score distributions, model performance metrics, data-quality exceptions, and a live API prediction log.
- Wrote a pytest test suite (data quality, feature engineering, model evaluation, API endpoints) using synthetic fixtures that run in < 30 seconds.
