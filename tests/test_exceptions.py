"""
Tests for the operational exception-checking module.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from credit_risk.exceptions import (
    ExceptionSeverity,
    ExceptionFlag,
    OperationalExceptionChecker,
    generate_synthetic_operational_records,
)


@pytest.fixture
def clean_df():
    """A clean set of synthetic operational records with no exceptions."""
    return pd.DataFrame({
        "SK_ID_CURR": range(1, 11),
        "AMT_INCOME_TOTAL": [100000.0] * 10,
        "AMT_CREDIT": [300000.0] * 10,
        "AMT_ANNUITY": [15000.0] * 10,
        "DAYS_BIRTH": [-13000] * 10,  # ~36 years
        "DAYS_EMPLOYED": [-1000] * 10,
        "CODE_GENDER": ["M"] * 5 + ["F"] * 5,
        "FLAG_DOCUMENT_3": [1] * 10,
        "FLAG_DOCUMENT_6": [0] * 10,
    })


@pytest.fixture
def problematic_df():
    """Records with planted exceptions."""
    return pd.DataFrame({
        "SK_ID_CURR": range(1, 7),
        "AMT_INCOME_TOTAL": [0, float("nan"), 100000, 50000, 200000, 80000],
        "AMT_CREDIT": [300000, 200000, 5000000, 400000, 300000, 200000],
        "AMT_ANNUITY": [15000, 10000, 20000, -5000, 12000, 9000],
        "DAYS_BIRTH": [-13000, -13000, -13000, -13000, -4000, -13000],  # row 4: underage
        "DAYS_EMPLOYED": [-1000, -500, 365243, -800, -300, -200],
        "CODE_GENDER": ["M", "F", "U", "M", "F", "M"],  # row 2: invalid gender
        "FLAG_DOCUMENT_3": [0, 0, 0, 0, 0, 0],
        "FLAG_DOCUMENT_6": [0, 0, 0, 0, 0, 0],
    })


class TestCleanData:
    def test_no_income_flags_on_clean(self, clean_df):
        checker = OperationalExceptionChecker(clean_df)
        checker.rule_invalid_income()
        income_flags = [f for f in checker.flags if f.rule == "invalid_income"]
        assert len(income_flags) == 0

    def test_no_duplicate_flags_on_clean(self, clean_df):
        checker = OperationalExceptionChecker(clean_df)
        checker.rule_duplicate_id()
        assert len(checker.flags) == 0

    def test_run_all_no_critical_errors_on_clean(self, clean_df):
        checker = OperationalExceptionChecker(clean_df)
        checker.run_all()
        errors = [f for f in checker.flags if f.severity == ExceptionSeverity.ERROR]
        # No income=0, no duplicates, no underage, no invalid gender, no negative annuity
        assert len(errors) == 0


class TestProblematicData:
    def test_zero_income_flagged(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.rule_invalid_income()
        assert any(f.rule == "invalid_income" for f in checker.flags)

    def test_missing_income_flagged(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.rule_missing_required_fields()
        assert any(f.rule == "missing_required_field" and f.field == "AMT_INCOME_TOTAL"
                   for f in checker.flags)

    def test_underage_flagged(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.rule_underage_applicant()
        assert any(f.rule == "underage_applicant" for f in checker.flags)

    def test_invalid_gender_flagged(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.rule_inconsistent_gender()
        assert any(f.rule == "invalid_gender_code" for f in checker.flags)

    def test_negative_annuity_flagged(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.rule_negative_annuity()
        assert any(f.rule == "negative_annuity" for f in checker.flags)

    def test_anomalous_employment_flagged(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.rule_anomalous_employment()
        assert any(f.rule == "anomalous_days_employed" for f in checker.flags)

    def test_extreme_ratio_flagged(self):
        df = pd.DataFrame({
            "SK_ID_CURR": [1],
            "AMT_INCOME_TOTAL": [10000.0],
            "AMT_CREDIT": [5000000.0],  # 500× income
            "AMT_ANNUITY": [15000.0],
            "DAYS_BIRTH": [-13000],
            "DAYS_EMPLOYED": [-500],
            "CODE_GENDER": ["M"],
        })
        checker = OperationalExceptionChecker(df)
        checker.rule_extreme_credit_income_ratio()
        assert any(f.rule == "extreme_credit_income_ratio" for f in checker.flags)


class TestSummary:
    def test_summary_structure(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.run_all()
        summary = checker.summary()
        assert "total_records" in summary
        assert "total_flags" in summary
        assert "by_severity" in summary
        assert "by_rule" in summary

    def test_flag_to_dict(self, problematic_df):
        checker = OperationalExceptionChecker(problematic_df)
        checker.run_all()
        for flag in checker.flags:
            d = flag.to_dict()
            assert "rule" in d
            assert "severity" in d
            assert "detail" in d


class TestSyntheticGeneration:
    def test_generates_correct_count(self):
        df = generate_synthetic_operational_records(n=100)
        assert len(df) == 100

    def test_has_data_source_column(self):
        df = generate_synthetic_operational_records(n=20)
        assert "data_source" in df.columns
        assert (df["data_source"] == "SYNTHETIC — NOT real loan data").all()

    def test_has_both_record_types(self):
        df = generate_synthetic_operational_records(n=100)
        assert "synthetic_valid" in df["record_type"].values
        assert "synthetic_exception" in df["record_type"].values
