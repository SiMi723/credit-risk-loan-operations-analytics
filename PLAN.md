# Credit Risk & Loan Operations Analytics Platform — Architecture Plan

## Overview

This platform demonstrates an end-to-end credit-risk analytics pipeline using the
**Home Credit Default Risk** dataset (Kaggle, 2018). It is a portfolio project built
for a Data/ML Analyst role at a technology-driven lending company.

> **Disclaimer**: The dataset is from Home Credit, a consumer-lending company operating
> in multiple countries. It is **not** an Indian SME/MSME dataset and this model was
> **not** developed for, trained on, or validated by Protium Finance.

---

## Data Inventory

| File | Size | Rows | Grain |
|------|------|------|-------|
| `application_train.csv` | 166 MB | 307,511 | One row per loan application |
| `bureau.csv` | 170 MB | 1,716,428 | One row per external credit bureau record |
| `previous_application.csv` | 405 MB | 1,670,214 | One row per previous HC application |
| `installments_payments.csv` | 723 MB | 13,605,401 | One row per installment payment |
| `HomeCredit_columns_description.csv` | 37 KB | — | Data dictionary |

**Missing optional files**: `bureau_balance.csv`, `POS_CASH_balance.csv`,
`credit_card_balance.csv`, `application_test.csv` — not present in workspace.
Fallback: these tables are omitted from feature engineering without approximation.

---

## Table Relationships

```
application_train (307,511 rows)
    │  SK_ID_CURR  (PK)
    ├──► bureau (1.7M rows)          SK_ID_CURR → many bureau records
    │        └──► bureau_balance     SK_ID_BUREAU → many (ABSENT)
    ├──► previous_application (1.7M) SK_ID_CURR → many prev applications
    │        ├──► installments_payments (13.6M)  SK_ID_PREV → many
    │        ├──► POS_CASH_balance   SK_ID_PREV → many (ABSENT)
    │        └──► credit_card_balance SK_ID_PREV → many (ABSENT)
    └── TARGET (binary: 1=default, 0=no default)
```

**Class imbalance**: ~8% default rate (TARGET=1), ~92% non-default.

---

## Architecture

```
data/raw/          ← Original CSV files (gitignored)
   │
   ▼
scripts/01_ingest.py    ← Schema validation, row counts, encoding check
   │
   ▼
scripts/02_eda.py       ← Missing values, distributions, correlations
   │
   ▼
scripts/03_features.py  ← DuckDB SQL aggregations → Parquet feature tables
   │
   ▼
scripts/04_train.py     ← Stratified split, LR baseline + GBM, evaluation
   │
   ▼
models/                 ← Saved pipeline artifacts (joblib)
outputs/metrics/        ← JSON evaluation results
   │
   ├──► api/main.py          ← FastAPI prediction + health endpoints
   ├──► dashboard/app.py     ← Streamlit operational dashboard
   └──► scripts/05_monitor.py ← Exception checking + distribution monitoring
```

---

## Implementation Phases

### Phase 1 — Infrastructure & Data Audit
- [ ] Virtual environment, `requirements.txt`
- [ ] `src/credit_risk/` package: config, logger, schema, data_io
- [ ] `scripts/01_ingest.py`: inventory, schema check, encoding detection
- [ ] SQL files for raw table inspection

### Phase 2 — EDA & Data Quality
- [ ] `scripts/02_eda.py`: missing value report, target distribution, correlations
- [ ] `src/credit_risk/data_quality.py`: rule-based validation engine
- [ ] Outputs: `outputs/reports/eda_report.json`

### Phase 3 — Feature Engineering
- [ ] `sql/bureau_features.sql`: bureau aggregation per SK_ID_CURR
- [ ] `sql/prev_app_features.sql`: previous application aggregation
- [ ] `sql/installment_features.sql`: payment behavior features
- [ ] `scripts/03_features.py`: DuckDB pipeline → `data/processed/features.parquet`

### Phase 4 — Modelling
- [ ] `src/credit_risk/model.py`: preprocessing pipeline, LR + LightGBM
- [ ] `scripts/04_train.py`: stratified split, train, evaluate, persist
- [ ] `outputs/metrics/model_evaluation.json`: real measured metrics

### Phase 5 — API
- [ ] `api/main.py`: FastAPI app with `/health`, `/predict`, `/metrics`
- [ ] `api/schemas.py`: Pydantic input/output models
- [ ] `scripts/06_benchmark.py`: latency measurement (1000 predictions)

### Phase 6 — Monitoring & Dashboard
- [ ] `src/credit_risk/exceptions.py`: operational exception rules
- [ ] `data/synthetic/operational_records.csv`: labelled synthetic data
- [ ] `scripts/05_monitor.py`: exception flagging, distribution checks
- [ ] `dashboard/app.py`: Streamlit interactive dashboard

### Phase 7 — Testing
- [ ] `tests/test_data_quality.py`
- [ ] `tests/test_features.py`
- [ ] `tests/test_model.py`
- [ ] `tests/test_api.py`
- [ ] `tests/test_exceptions.py`

---

## Resource Constraints & Mitigations

| Constraint | Value | Mitigation |
|------------|-------|------------|
| Available RAM | ~1.8 GB free | DuckDB lazy eval; chunked CSV reading; Parquet |
| Disk free (C:) | 106 GB | Sufficient for Parquet + models |
| Python | 3.12.6 | Compatible with all dependencies |
| installments_payments | 13.6M rows | Aggregate in SQL before loading to Python |

**Memory strategy**: Never load full `installments_payments.csv` into pandas.
Use DuckDB to aggregate to SK_ID_CURR grain in SQL, then read the result only.

---

## Feature Definitions

### Application-level features (selected)
| Feature | Definition | Missing handling |
|---------|------------|-----------------|
| `CREDIT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Impute income median |
| `ANNUITY_INCOME_RATIO` | AMT_ANNUITY / AMT_INCOME_TOTAL | Impute annuity median |
| `AGE_YEARS` | -DAYS_BIRTH / 365 | No missing |
| `EMPLOYED_YEARS` | -DAYS_EMPLOYED / 365 (capped at 0) | 365243 → NaN → median |
| `EXT_SOURCE_MEAN` | Mean of EXT_SOURCE_1,2,3 | Per-source median |

### Bureau aggregation features
| Feature | Definition |
|---------|------------|
| `BUREAU_CREDIT_COUNT` | Count of bureau records |
| `BUREAU_ACTIVE_COUNT` | Count where CREDIT_ACTIVE='Active' |
| `BUREAU_OVERDUE_MAX` | Max CREDIT_DAY_OVERDUE |
| `BUREAU_AMT_CREDIT_SUM` | Sum of AMT_CREDIT_SUM |
| `BUREAU_DAYS_CREDIT_MIN` | Min DAYS_CREDIT (most recent) |

### Previous application features
| Feature | Definition |
|---------|------------|
| `PREV_APP_COUNT` | Total previous applications |
| `PREV_APPROVED_COUNT` | Count where NAME_CONTRACT_STATUS='Approved' |
| `PREV_REFUSED_COUNT` | Count where NAME_CONTRACT_STATUS='Refused' |
| `PREV_AMT_CREDIT_MEAN` | Mean credit amount across prev apps |
| `PREV_ANNUITY_MEAN` | Mean annuity across prev apps |

### Installment payment features
| Feature | Definition |
|---------|------------|
| `INSTALMENT_COUNT` | Total installment records |
| `INSTALMENT_LATE_COUNT` | Count where DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT |
| `INSTALMENT_LATE_RATIO` | INSTALMENT_LATE_COUNT / INSTALMENT_COUNT |
| `INSTALMENT_AMT_DIFF_MEAN` | Mean (AMT_PAYMENT - AMT_INSTALMENT) |
| `INSTALMENT_DPD_MEAN` | Mean days past due (max 0 floored) |

---

## Validation Plan

1. **Key integrity**: Verify SK_ID_CURR overlaps between tables before joining.
2. **No target leakage**: Remove TARGET, SK_ID_CURR, and any bureau/installment
   columns that encode the outcome.
3. **Train/validation split**: Stratified 80/20 on TARGET; fixed `random_state=42`.
4. **Preprocessing**: Fit only on training fold; apply same transform to validation.
5. **Class imbalance**: Use `class_weight='balanced'` for LR;
   `scale_pos_weight` for LightGBM.
6. **Threshold selection**: Optimise F1 on validation set predictions.
7. **Evaluation metrics**: ROC-AUC, PR-AUC, Precision, Recall, F1, Confusion Matrix.
8. **Model selection criterion**: Higher ROC-AUC wins; tie-break on PR-AUC.
9. **Persistence**: Save sklearn Pipeline (preprocessor + model) via joblib.
10. **No fabricated metrics**: All reported numbers come from actual validation-set
    predictions stored in `outputs/metrics/model_evaluation.json`.
