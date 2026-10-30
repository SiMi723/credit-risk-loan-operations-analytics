# Credit Risk & Loan Operations Analytics Platform

An end-to-end credit risk analytics platform built using **Python, SQL, DuckDB, LightGBM, FastAPI, and Streamlit**. The project demonstrates large-scale data processing, feature engineering, predictive risk modelling, operational exception monitoring, and model serving using the Home Credit Default Risk dataset.

## Project Overview

This project develops a complete credit-risk analytics workflow, from raw loan application and repayment data to predictive modelling and API-based inference.

It processes customer application records, external credit bureau information, previous loan applications, and installment payment histories to derive customer-level risk indicators and evaluate the likelihood of loan default.

The platform also includes a rules-based operational exception monitoring engine, a FastAPI prediction service, and an interactive Streamlit dashboard.

**Key capabilities:**
- Large-scale data ingestion and processing using DuckDB and Parquet.
- Customer-level feature engineering from multiple financial datasets.
- Credit default prediction using Logistic Regression and LightGBM.
- Model evaluation using ROC-AUC, PR-AUC, Precision, Recall, and F1-score.
- REST API for model inference with input validation and prediction logging.
- Operational exception detection and data-quality monitoring.
- Interactive dashboard for risk analysis and model performance reporting.

---

## Important Disclaimer

This project is a **portfolio research prototype**, not a production lending system.

- The dataset comes from Home Credit, a consumer-lending company. It does not represent Indian SME/MSME lending data.
- The platform is not a regulatory-approved credit decision system or a production credit underwriting tool.
- Synthetic operational records are used for testing the exception-monitoring workflow and are not real customer or loan records.
- Model performance and inference benchmarks are based on local experimentation and validation.

---

## Dataset & Data Provenance

The project uses the publicly available [Home Credit Default Risk dataset on Kaggle](https://www.kaggle.com/c/home-credit-default-risk).

| Dataset | Records | Description |
|---|---:|---|
| `application_train.csv` | 307,511 | Loan application data with default labels |
| `bureau.csv` | 1,716,428 | External credit bureau records |
| `previous_application.csv` | 1,670,214 | Previous Home Credit applications |
| `installments_payments.csv` | 13,605,401 | Historical installment payment records |
| `HomeCredit_columns_description.csv` | — | Dataset data dictionary |

The target variable is `TARGET`, representing whether an applicant experienced loan repayment difficulties.

The dataset contains approximately **8.07% default cases**, making it an imbalanced classification problem.

**Optional datasets not included:** `bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, and `application_test.csv`. Their absence is handled by the ingestion pipeline.

### Data Storage

Raw datasets, processed feature tables, trained models, and generated outputs are excluded from version control through `.gitignore`.

To reproduce the pipeline, download the dataset separately from Kaggle and place the required CSV files inside `data/raw/`.

---

## System Architecture

```text
                 HOME CREDIT DATASET
                         |
                         v
                 DATA INGESTION
               Schema & Quality Checks
                         |
                         v
              EXPLORATORY DATA ANALYSIS
               Missing Values & Patterns
                         |
                         v
                FEATURE ENGINEERING
                DuckDB SQL Aggregations
                  Parquet Generation
                         |
                         v
                  MODEL TRAINING
               Logistic Regression
                     LightGBM
                         |
                         v
                MODEL EVALUATION
              ROC-AUC / PR-AUC / F1
                         |
                +--------+--------+
                |                 |
                v                 v
           FASTAPI SERVICE   EXCEPTION MONITOR
             /predict        Data Quality Rules
             /metrics        Anomaly Detection
                |                 |
                +--------+--------+
                         |
                         v
                 STREAMLIT DASHBOARD
             Risk & Operational Monitoring
```

### Memory-Efficient Processing

The 13.6-million-row installment payment dataset is processed through DuckDB SQL aggregations rather than loading the entire raw table into Python memory.

Aggregated features are generated at the customer level (`SK_ID_CURR`) and stored in Parquet format.

The DuckDB execution environment uses a configured memory limit of approximately 1.2 GB.

---

## Repository Structure

```text
credit-risk-loan-operations-analytics/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py
│
├── scripts/
│   ├── 01_ingest.py
│   ├── 02_eda.py
│   ├── 03_features.py
│   ├── 04_train.py
│   ├── 05_monitor.py
│   ├── 06_benchmark.py
│   ├── _fix_unicode.py
│   └── run_api.py
│
├── sql/
│   ├── bureau_features.sql
│   ├── instalment_features.sql
│   ├── prev_app_features.sql
│   └── schema_validation.sql
│
├── src/
│   └── credit_risk/
│       ├── __init__.py
│       ├── config.py
│       ├── data_io.py
│       ├── data_quality.py
│       ├── exceptions.py
│       ├── features.py
│       ├── logger.py
│       └── model.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data_quality.py
│   ├── test_exceptions.py
│   ├── test_features.py
│   └── test_model.py
│
├── .gitignore
├── PLAN.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

**Generated locally and excluded from Git:**

- `data/raw/` — original CSV datasets.
- `data/processed/` — engineered Parquet feature tables.
- `data/synthetic/` — synthetic operational records.
- `models/` — trained model artifacts.
- `outputs/` — evaluation metrics
