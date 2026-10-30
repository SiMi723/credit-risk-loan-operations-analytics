"""
Tests for feature engineering functions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from credit_risk.features import (
    APP_FEATURE_DEFS,
    build_app_features,
    merge_features,
)


class TestAppFeatures:
    def test_credit_income_ratio(self, tiny_app_df):
        df = build_app_features(tiny_app_df.copy())
        assert "credit_income_ratio" in df.columns
        valid = df.loc[df["AMT_INCOME_TOTAL"] > 0, "credit_income_ratio"]
        assert (valid > 0).all()

    def test_age_years(self, tiny_app_df):
        df = build_app_features(tiny_app_df.copy())
        assert "age_years" in df.columns
        # All should be positive (DAYS_BIRTH is negative)
        assert (df["age_years"] > 0).all()

    def test_ext_source_mean(self, tiny_app_df):
        df = build_app_features(tiny_app_df.copy())
        assert "ext_source_mean" in df.columns
        # Mean should be in [0, 1] where defined
        valid = df["ext_source_mean"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_days_employed_anomaly_replaced(self):
        """DAYS_EMPLOYED=365243 should already be replaced before feature engineering."""
        df = pd.DataFrame({
            "DAYS_BIRTH": [-15000],
            "DAYS_EMPLOYED": [float("nan")],  # already replaced
            "AMT_CREDIT": [300000.0],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_ANNUITY": [15000.0],
            "AMT_GOODS_PRICE": [280000.0],
            "CNT_FAM_MEMBERS": [2.0],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.4],
            "DAYS_ID_PUBLISH": [-1000],
            "DAYS_REGISTRATION": [-5000.0],
            "DAYS_LAST_PHONE_CHANGE": [-200.0],
        })
        result = build_app_features(df)
        # employed_years should be NaN because DAYS_EMPLOYED is NaN
        assert pd.isna(result["employed_years"].iloc[0])

    def test_no_row_multiplication(self, tiny_app_df):
        """build_app_features should not change the row count."""
        n_before = len(tiny_app_df)
        result = build_app_features(tiny_app_df.copy())
        assert len(result) == n_before


class TestMergeFeatures:
    def test_merge_preserves_rows(self, tiny_app_df):
        """Left join should never multiply rows."""
        bureau_agg = pd.DataFrame({
            "SK_ID_CURR": list(range(100000, 100030)),
            "bureau_credit_count": np.random.randint(1, 10, 30),
        })
        prev_agg = pd.DataFrame({
            "SK_ID_CURR": list(range(100000, 100040)),
            "prev_app_count": np.random.randint(0, 5, 40),
        })
        inst_agg = pd.DataFrame({
            "SK_ID_CURR": list(range(100000, 100025)),
            "instalment_count": np.random.randint(5, 50, 25),
        })
        n_before = len(tiny_app_df)
        merged = merge_features(tiny_app_df, bureau_agg, prev_agg, inst_agg)
        assert len(merged) == n_before

    def test_merge_adds_columns(self, tiny_app_df):
        """Merged DataFrame should include auxiliary feature columns."""
        bureau_agg = pd.DataFrame({
            "SK_ID_CURR": list(range(100000, 100050)),
            "bureau_credit_count": np.ones(50),
        })
        prev_agg = pd.DataFrame({
            "SK_ID_CURR": list(range(100000, 100050)),
            "prev_app_count": np.ones(50),
        })
        inst_agg = pd.DataFrame({
            "SK_ID_CURR": list(range(100000, 100050)),
            "instalment_count": np.ones(50),
        })
        merged = merge_features(tiny_app_df, bureau_agg, prev_agg, inst_agg)
        assert "bureau_credit_count" in merged.columns
        assert "prev_app_count" in merged.columns
        assert "instalment_count" in merged.columns

    def test_merge_unmatched_rows_get_nan(self, tiny_app_df):
        """Rows in app_df with no match in auxiliary tables should get NaN."""
        # No overlap: bureau only covers IDs not in tiny_app_df
        bureau_agg = pd.DataFrame({
            "SK_ID_CURR": [999999],
            "bureau_credit_count": [5],
        })
        prev_agg = pd.DataFrame({"SK_ID_CURR": [], "prev_app_count": []})
        inst_agg = pd.DataFrame({"SK_ID_CURR": [], "instalment_count": []})
        merged = merge_features(tiny_app_df, bureau_agg, prev_agg, inst_agg)
        assert merged["bureau_credit_count"].isna().all()
