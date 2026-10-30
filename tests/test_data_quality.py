"""
Tests for the data quality checker.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from credit_risk.data_quality import DataQualityChecker, Issue


class TestRequiredColumns:
    def test_all_present(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        issues = checker.check_required_columns()
        assert len(issues) == 0

    def test_missing_column(self, tiny_app_df):
        df = tiny_app_df.drop(columns=["AMT_INCOME_TOTAL"])
        checker = DataQualityChecker(df)
        issues = checker.check_required_columns()
        assert any(i.column == "AMT_INCOME_TOTAL" for i in issues)
        assert any(i.severity == "ERROR" for i in issues)


class TestDuplicateIds:
    def test_no_duplicates(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        issues = checker.check_duplicate_ids()
        assert len(issues) == 0

    def test_with_duplicates(self, tiny_app_df):
        df = pd.concat([tiny_app_df, tiny_app_df.head(5)], ignore_index=True)
        checker = DataQualityChecker(df)
        issues = checker.check_duplicate_ids()
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].row_count == 5


class TestTargetValues:
    def test_valid_target(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        issues = checker.check_target_values()
        assert len(issues) == 0

    def test_invalid_target(self, tiny_app_df):
        df = tiny_app_df.copy()
        df.loc[0, "TARGET"] = 99  # invalid
        checker = DataQualityChecker(df)
        issues = checker.check_target_values()
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"


class TestNegativeAmounts:
    def test_no_negative(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        issues = checker.check_negative_amounts()
        assert all(i.check == "negative_amount" for i in issues)  # only negative-amount issues

    def test_with_negative(self, tiny_app_df):
        df = tiny_app_df.copy()
        df.loc[0, "AMT_CREDIT"] = -500
        checker = DataQualityChecker(df)
        issues = checker.check_negative_amounts()
        neg_issues = [i for i in issues if i.column == "AMT_CREDIT"]
        assert len(neg_issues) == 1


class TestDaysEmployedAnomaly:
    def test_anomaly_detected(self, tiny_app_df):
        df = tiny_app_df.copy()
        df.loc[0, "DAYS_EMPLOYED"] = 365243
        checker = DataQualityChecker(df)
        issues = checker.check_days_employed_anomaly()
        assert len(issues) == 1
        assert issues[0].row_count == 1

    def test_no_anomaly(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        issues = checker.check_days_employed_anomaly()
        assert len(issues) == 0


class TestGenderCodes:
    def test_valid_genders(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        issues = checker.check_gender_codes()
        assert len(issues) == 0

    def test_invalid_gender(self, tiny_app_df):
        df = tiny_app_df.copy()
        df.loc[0, "CODE_GENDER"] = "U"
        checker = DataQualityChecker(df)
        issues = checker.check_gender_codes()
        assert len(issues) == 1


class TestRunAll:
    def test_run_all_returns_list(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        issues = checker.run_all()
        assert isinstance(issues, list)

    def test_summary_structure(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        checker.run_all()
        summary = checker.summary()
        assert "total_rows" in summary
        assert "error_count" in summary
        assert "warning_count" in summary
        assert "issues" in summary

    def test_issue_serialisable(self, tiny_app_df):
        checker = DataQualityChecker(tiny_app_df)
        checker.run_all()
        for issue in checker.issues:
            d = issue.to_dict()
            assert isinstance(d, dict)
            assert "check" in d
            assert "severity" in d
