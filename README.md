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
- `outputs/` — evaluation metrics, reports, and logs.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Data Processing | DuckDB, Parquet |
| Database & Querying | SQL |
| Machine Learning | Scikit-learn, LightGBM |
| Model Evaluation | ROC-AUC, PR-AUC, Precision, Recall, F1 |
| API Development | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Testing | Pytest |
| Development Tools | Git, GitHub |

---

## Setup & Installation

### Prerequisites

- Python 3.10 or above (tested with Python 3.12.6).
- Approximately 2 GB of available RAM, subject to dataset size and processing configuration.
- Approximately 2 GB of free disk space for generated feature tables and model artifacts.
- Home Credit dataset CSV files downloaded separately.

### 1. Clone the repository

```bash
git clone https://github.com/SiMi723/credit-risk-loan-operations-analytics.git

cd credit-risk-loan-operations-analytics
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv

venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

If LightGBM or Uvicorn is unavailable in the environment:

```bash
pip install lightgbm "uvicorn[standard]"
```

### 4. Prepare the dataset

Download the Home Credit Default Risk dataset from Kaggle and place the required CSV files inside:

```text
data/
└── raw/
    ├── application_train.csv
    ├── bureau.csv
    ├── previous_application.csv
    ├── installments_payments.csv
    └── HomeCredit_columns_description.csv
```

---

## Execution Pipeline

Run the following scripts in sequence.

### 1. Data Ingestion & Schema Validation

```bash
python scripts/01_ingest.py
```

Performs dataset inventory, schema validation, and key-overlap checks.

Generates data inventory and quality reports under `outputs/reports/`.

### 2. Exploratory Data Analysis

```bash
python scripts/02_eda.py
```

Analyses:
- Missing-value distributions.
- Target-class distribution.
- Feature correlations.
- Dataset-level statistics.

Generates EDA and missing-value reports.

### 3. Feature Engineering

```bash
python scripts/03_features.py
```

Uses DuckDB SQL aggregations to generate customer-level features from application, bureau, previous application, and installment payment data.

Generated feature tables are stored in `data/processed/`.

### 4. Model Training & Evaluation

```bash
python scripts/04_train.py
```

Trains and evaluates:
- Logistic Regression.
- LightGBM.

Performs stratified train-validation splitting, preprocessing, model evaluation, and validation-based threshold selection.

Stores trained models and evaluation metrics locally.

### 5. Operational Exception Monitoring

```bash
python scripts/05_monitor.py
```

Generates synthetic operational records and evaluates the rules-based exception-monitoring engine.

Produces exception reports under `outputs/reports/`.

### 6. Run the FastAPI Service

```bash
python scripts/run_api.py
```

Alternatively:

```bash
uvicorn api.main:app --port 8000 --reload
```

Available endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | API and model health status |
| `POST /predict` | Generate a default-risk prediction |
| `GET /metrics` | Retrieve model evaluation metrics |
| `/docs` | Interactive Swagger API documentation |

### 7. Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard provides interactive views of risk scores, model performance, operational exceptions, and data-quality information.

### 8. Run Inference Latency Benchmark

```bash
python scripts/06_benchmark.py
```

Measures local model inference latency and generates a benchmark report.

### 9. Run Automated Tests

```bash
pytest tests/ -v
```

The test suite covers API functionality, data quality, exception detection, feature engineering, and model behaviour.

---

## Feature Engineering

The pipeline generates customer-level features from multiple datasets.

### Application-Level Features

| Feature | Definition |
|---|---|
| `credit_income_ratio` | Credit amount / total income |
| `annuity_income_ratio` | Annuity amount / total income |
| `age_years` | Applicant age derived from `DAYS_BIRTH` |
| `employed_years` | Employment duration derived from `DAYS_EMPLOYED` |
| `ext_source_mean` | Mean of available external source scores |

### Bureau-Level Aggregations

| Feature | Definition |
|---|---|
| `bureau_credit_count` | Number of external credit records |
| `bureau_active_count` | Number of active credit records |
| `bureau_overdue_max` | Maximum days of credit overdue |
| `bureau_amt_credit_sum` | Aggregated external credit amount |
| `bureau_days_credit_min` | Most recent bureau record indicator |

### Installment Payment Features

| Feature | Definition |
|---|---|
| `instalment_late_count` | Number of late installment payments |
| `instalment_late_ratio` | Proportion of late payments |
| `instalment_dpd_mean` | Mean days past due, floored at zero |
| `instalment_amt_diff_mean` | Mean difference between actual and scheduled payments |
| `instalment_underpay_count` | Number of payments short by more than one unit |

The final engineered dataset contains **307,511 customer records and 173 columns**.

---

## Model Development & Evaluation

### Class Imbalance Handling

The dataset has an approximately 8.07% default rate.

To account for class imbalance, both models use class-weight balancing:

- Logistic Regression: `class_weight='balanced'`
- LightGBM: `class_weight='balanced'`

The training-validation split uses an 80/20 stratified random split with seed 42. Preprocessing is fitted on the training data only.

### Model Selection

Both models are evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

The final model is selected using validation ROC-AUC, with PR-AUC as a tie-breaker.

The classification threshold is selected using validation F1-score optimisation rather than assuming a fixed threshold of 0.5.

### Validation Performance

| Metric | Logistic Regression | LightGBM |
|---|---:|---:|
| ROC-AUC | 0.7655 | **0.7805** |
| PR-AUC | 0.2490 | **0.2763** |
| Precision | 0.2420 | **0.2830** |
| Recall | **0.4429** | 0.4077 |
| F1-score | 0.3130 | **0.3341** |
| Accuracy | 0.8430 | **0.8688** |
| Selected threshold | 0.6630 | **0.6765** |

LightGBM achieved a ROC-AUC improvement of **0.0150** and a PR-AUC improvement of **0.0273** over the Logistic Regression baseline.

Based on the validation results, LightGBM was selected as the final model.

*These are locally measured validation results, not production performance figures.*

---

## API & Inference

The FastAPI service provides a REST interface for model predictions.

### `GET /health`

Checks API availability and model loading status.

### `POST /predict`

Accepts an application-level JSON payload and returns a predicted default probability, classification, and risk segment.

Example response from a locally tested inference request:

```json
{
  "default_probability": 0.7730,
  "prediction": 1,
  "risk_segment": "VERY_HIGH"
}
```

The actual response includes fields defined by the API response schema.

### `GET /metrics`

Exposes the model evaluation metrics from the latest training run.

Interactive API documentation is available through:

```text
http://localhost:8000/docs
```

---

## Operational Exception Monitoring

A deterministic rules-based exception engine identifies potential data-quality and operational issues.

The monitoring workflow includes checks for:

- Missing required application fields.
- Invalid income values.
- Negative financial amounts.
- Invalid gender codes.
- Applicants outside defined age boundaries.
- Employment placeholder values.
- Missing submitted documents.
- Unusually high credit-to-income ratios.
- Other invalid or inconsistent application attributes.

The exception monitor was evaluated using **200 labelled synthetic operational records**, consisting of 150 valid records and 50 anomalous records.

The test generated **114 exception flags across 9 operational rule categories**.

These results demonstrate the functioning of the monitoring rules on synthetic test data; they do not represent real-world fraud or default detection performance.

---

## Inference Latency & Testing

### Local Inference Benchmark

The model was benchmarked using 1,000 local inference requests.

| Latency Metric | Result |
|---|---:|
| Mean | 5.52 ms |
| Median | 5.44 ms |
| 75th percentile | 6.65 ms |
| 95th percentile | 9.39 ms |
| 99th percentile | 12.76 ms |

These measurements represent local inference performance and should not be interpreted as production API response times.

### Automated Testing

The project includes automated tests covering:

- API endpoints and input validation.
- Data-quality checks.
- Exception-monitoring logic.
- Feature engineering.
- Model evaluation and prediction behaviour.

**Test results: 61 passed, 0 failed, 0 skipped.**

---

## Limitations & Future Improvements

1. **Dataset scope:** The Home Credit dataset represents consumer lending across multiple countries and does not directly represent Indian SME/MSME credit risk profiles.

2. **Missing optional tables:** Additional datasets such as `bureau_balance.csv`, `POS_CASH_balance.csv`, and `credit_card_balance.csv` are not included, limiting the potential feature set.

3. **Temporal validation:** The current model uses a stratified random train-validation split. A time-based out-of-time validation strategy would be more appropriate for assessing performance on future loan applications.

4. **Production deployment:** The current FastAPI service is a local prototype and does not include production-grade authentication, rate limiting, persistent database infrastructure, or distributed monitoring.

5. **Latency measurement:** Benchmarks are based on local inference and do not represent real-world cloud deployment latency.

6. **SME-specific adaptation:** Applying this approach to Indian SME lending would require suitable business financial statements, GST data, banking transaction histories, and other SME-specific financial indicators.

### Potential Future Enhancements

- Out-of-time validation and model drift monitoring.
- Explainable AI using SHAP for feature-level risk interpretation.
- Cost-sensitive threshold optimisation based on business objectives.
- Integration of SME-specific financial and transaction datasets.
- Production deployment with authentication, database persistence, and monitoring infrastructure.

---

## Author

**Simi**

B.Tech Biotechnology, Motilal Nehru National Institute of Technology (MNNIT), Allahabad.

GitHub: [SiMi723](https://github.com/SiMi723)

---

*This project was developed as an applied portfolio project exploring data engineering, predictive analytics, credit-risk modelling, and operational monitoring.*
