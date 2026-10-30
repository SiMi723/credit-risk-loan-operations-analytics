"""
Shared test fixtures for the credit-risk test suite.

All fixtures use small synthetic DataFrames so tests run in milliseconds
without touching the raw CSV files.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_app_df():
    """Minimal application_train-like DataFrame with 50 rows."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "SK_ID_CURR": range(100000, 100000 + n),
        "TARGET": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER": np.random.choice(["M", "F", "XNA"], n, p=[0.5, 0.45, 0.05]),
        "FLAG_OWN_CAR": np.random.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], n),
        "CNT_CHILDREN": np.random.randint(0, 4, n),
        "AMT_INCOME_TOTAL": np.random.uniform(50000, 500000, n),
        "AMT_CREDIT": np.random.uniform(100000, 1000000, n),
        "AMT_ANNUITY": np.random.uniform(5000, 50000, n),
        "AMT_GOODS_PRICE": np.random.uniform(80000, 900000, n),
        "NAME_INCOME_TYPE": np.random.choice(["Working", "State servant", "Commercial associate"], n),
        "NAME_EDUCATION_TYPE": np.random.choice([
            "Secondary / secondary special", "Higher education", "Incomplete higher"
        ], n),
        "NAME_FAMILY_STATUS": np.random.choice(["Single / not married", "Married", "Civil marriage"], n),
        "NAME_HOUSING_TYPE": np.random.choice(["House / apartment", "Rented apartment"], n),
        "REGION_POPULATION_RELATIVE": np.random.uniform(0.001, 0.07, n),
        "DAYS_BIRTH": np.random.randint(-25000, -6570, n),   # ages 18–68
        "DAYS_EMPLOYED": np.random.randint(-5000, -30, n),
        "DAYS_REGISTRATION": np.random.uniform(-20000, -100, n),
        "DAYS_ID_PUBLISH": np.random.randint(-5000, -100, n),
        "OWN_CAR_AGE": np.where(np.random.rand(n) > 0.5, np.random.uniform(0, 20, n), np.nan),
        "FLAG_MOBIL": np.ones(n, dtype=int),
        "FLAG_EMP_PHONE": np.random.randint(0, 2, n),
        "FLAG_WORK_PHONE": np.random.randint(0, 2, n),
        "FLAG_CONT_MOBILE": np.random.randint(0, 2, n),
        "FLAG_PHONE": np.random.randint(0, 2, n),
        "FLAG_EMAIL": np.random.randint(0, 2, n),
        "CNT_FAM_MEMBERS": np.random.choice([1, 2, 3, 4], n).astype(float),
        "REGION_RATING_CLIENT": np.random.randint(1, 4, n),
        "REGION_RATING_CLIENT_W_CITY": np.random.randint(1, 4, n),
        "EXT_SOURCE_1": np.where(np.random.rand(n) > 0.3, np.random.uniform(0, 1, n), np.nan),
        "EXT_SOURCE_2": np.random.uniform(0, 1, n),
        "EXT_SOURCE_3": np.where(np.random.rand(n) > 0.2, np.random.uniform(0, 1, n), np.nan),
        "FLAG_DOCUMENT_3": np.random.randint(0, 2, n),
        "FLAG_DOCUMENT_6": np.random.randint(0, 2, n),
        "DAYS_LAST_PHONE_CHANGE": np.random.uniform(-3000, -10, n),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.random.uniform(0, 5, n),
        "AMT_REQ_CREDIT_BUREAU_MON": np.random.uniform(0, 3, n),
        "AMT_REQ_CREDIT_BUREAU_QRT": np.random.uniform(0, 2, n),
        "AMT_REQ_CREDIT_BUREAU_WEEK": np.zeros(n),
        "AMT_REQ_CREDIT_BUREAU_DAY": np.zeros(n),
        "AMT_REQ_CREDIT_BUREAU_HOUR": np.zeros(n),
        "OBS_30_CNT_SOCIAL_CIRCLE": np.random.uniform(0, 5, n),
        "DEF_30_CNT_SOCIAL_CIRCLE": np.zeros(n),
        "OBS_60_CNT_SOCIAL_CIRCLE": np.random.uniform(0, 3, n),
        "DEF_60_CNT_SOCIAL_CIRCLE": np.zeros(n),
    })


@pytest.fixture
def tiny_bureau_df():
    """Small bureau-like DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "SK_ID_CURR": np.random.choice(range(100000, 100050), n),
        "SK_ID_BUREAU": range(500000, 500000 + n),
        "CREDIT_ACTIVE": np.random.choice(["Active", "Closed", "Sold"], n),
        "CREDIT_DAY_OVERDUE": np.random.randint(0, 30, n),
        "AMT_CREDIT_SUM": np.random.uniform(10000, 500000, n),
        "AMT_CREDIT_SUM_DEBT": np.random.uniform(0, 200000, n),
        "DAYS_CREDIT": np.random.randint(-2000, -30, n),
        "CREDIT_TYPE": np.random.choice(["Consumer credit", "Credit card", "Mortgage"], n),
        "AMT_ANNUITY": np.random.uniform(1000, 30000, n),
    })


@pytest.fixture
def tiny_instalment_df():
    """Small installments-like DataFrame."""
    np.random.seed(42)
    n = 500
    sched_days = np.random.randint(-1000, -10, n)
    return pd.DataFrame({
        "SK_ID_CURR": np.random.choice(range(100000, 100050), n),
        "SK_ID_PREV": np.random.randint(1000000, 1001000, n),
        "NUM_INSTALMENT_VERSION": np.ones(n),
        "NUM_INSTALMENT_NUMBER": np.random.randint(1, 24, n),
        "DAYS_INSTALMENT": sched_days,
        "DAYS_ENTRY_PAYMENT": sched_days + np.random.randint(-5, 10, n),
        "AMT_INSTALMENT": np.random.uniform(1000, 20000, n),
        "AMT_PAYMENT": np.random.uniform(500, 21000, n),
    })
