"""
Data quality validation engine.

Rules are deterministic and transparent — no ML involved.
Each check returns a list of ``Issue`` objects, which are plain dataclasses.

Usage
-----
    from credit_risk.data_quality import DataQualityChecker

    checker = DataQualityChecker(df)
    issues = checker.run_all()
    report = checker.summary()
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.logger import get_logger

logger = get_logger("credit_risk.data_quality")


@dataclasses.dataclass
class Issue:
    """Represents a single data-quality flag."""
    check: str          # Name of the rule that fired
    severity: str       # "ERROR" | "WARNING" | "INFO"
    column: str | None  # Affected column, or None for row-level issues
    detail: str         # Human-readable description
    row_count: int = 0  # Number of affected rows (0 if not applicable)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class DataQualityChecker:
    """Run a battery of deterministic data-quality rules against a DataFrame."""

    REQUIRED_APP_COLS = [
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "CODE_GENDER",
        "NAME_CONTRACT_TYPE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]

    def __init__(self, df: pd.DataFrame, table_name: str = "application_train"):
        self.df = df
        self.table_name = table_name
        self.issues: list[Issue] = []

    # ── Individual checks ────────────────────────────────────────────────────

    def check_required_columns(self) -> list[Issue]:
        """Flag any required columns that are entirely absent."""
        issues = []
        for col in self.REQUIRED_APP_COLS:
            if col not in self.df.columns:
                issues.append(Issue(
                    check="required_column_missing",
                    severity="ERROR",
                    column=col,
                    detail=f"Required column '{col}' is absent from {self.table_name}.",
                ))
        return issues

    def check_missing_values(self, threshold: float = 0.50) -> list[Issue]:
        """Flag columns with missing-value rates above threshold."""
        issues = []
        total = len(self.df)
        if total == 0:
            return issues
        for col in self.df.columns:
            rate = self.df[col].isna().sum() / total
            if rate > threshold:
                issues.append(Issue(
                    check="high_missing_rate",
                    severity="WARNING",
                    column=col,
                    detail=f"{rate:.1%} missing in '{col}' (threshold {threshold:.0%}).",
                    row_count=int(self.df[col].isna().sum()),
                ))
        return issues

    def check_duplicate_ids(self, id_col: str = "SK_ID_CURR") -> list[Issue]:
        """Flag duplicate primary-key values."""
        issues = []
        if id_col not in self.df.columns:
            return issues
        dupes = self.df[id_col].duplicated().sum()
        if dupes > 0:
            issues.append(Issue(
                check="duplicate_id",
                severity="ERROR",
                column=id_col,
                detail=f"{dupes:,} duplicate rows on '{id_col}'.",
                row_count=int(dupes),
            ))
        return issues

    def check_target_values(self, target_col: str = "TARGET") -> list[Issue]:
        """Ensure TARGET contains only 0 and 1."""
        issues = []
        if target_col not in self.df.columns:
            return issues
        invalid = ~self.df[target_col].isin([0, 1, np.nan])
        n_invalid = invalid.sum()
        if n_invalid > 0:
            bad_vals = self.df.loc[invalid, target_col].unique()[:5].tolist()
            issues.append(Issue(
                check="invalid_target_value",
                severity="ERROR",
                column=target_col,
                detail=f"{n_invalid:,} rows with invalid TARGET values: {bad_vals}.",
                row_count=int(n_invalid),
            ))
        return issues

    def check_negative_amounts(self) -> list[Issue]:
        """Flag negative values in amount columns (should be ≥ 0)."""
        issues = []
        amount_cols = [c for c in self.df.columns if c.startswith("AMT_")]
        for col in amount_cols:
            if col not in self.df.columns:
                continue
            neg = (self.df[col] < 0).sum()
            if neg > 0:
                issues.append(Issue(
                    check="negative_amount",
                    severity="WARNING",
                    column=col,
                    detail=f"{neg:,} negative values in '{col}'.",
                    row_count=int(neg),
                ))
        return issues

    def check_gender_codes(self, col: str = "CODE_GENDER") -> list[Issue]:
        """Ensure gender codes are in the expected set."""
        issues = []
        if col not in self.df.columns:
            return issues
        valid = {"M", "F", "XNA"}
        invalid_mask = ~self.df[col].isin(valid) & self.df[col].notna()
        n = invalid_mask.sum()
        if n > 0:
            issues.append(Issue(
                check="invalid_gender_code",
                severity="WARNING",
                column=col,
                detail=f"{n:,} rows with unexpected CODE_GENDER values.",
                row_count=int(n),
            ))
        return issues

    def check_days_birth_range(self, col: str = "DAYS_BIRTH") -> list[Issue]:
        """DAYS_BIRTH should be negative and correspond to ages 18–100."""
        issues = []
        if col not in self.df.columns:
            return issues
        age_years = -self.df[col] / 365.0
        out_of_range = ((age_years < 18) | (age_years > 100)) & self.df[col].notna()
        n = out_of_range.sum()
        if n > 0:
            issues.append(Issue(
                check="age_out_of_range",
                severity="WARNING",
                column=col,
                detail=f"{n:,} applicants with age outside 18–100 years.",
                row_count=int(n),
            ))
        return issues

    def check_days_employed_anomaly(self, col: str = "DAYS_EMPLOYED") -> list[Issue]:
        """Flag the known anomalous 365243 value in DAYS_EMPLOYED."""
        issues = []
        if col not in self.df.columns:
            return issues
        n_anomaly = (self.df[col] == 365243).sum()
        if n_anomaly > 0:
            issues.append(Issue(
                check="days_employed_anomaly",
                severity="WARNING",
                column=col,
                detail=(
                    f"{n_anomaly:,} rows have DAYS_EMPLOYED=365243 "
                    "(known placeholder for retirees/unemployed; treat as NaN)."
                ),
                row_count=int(n_anomaly),
            ))
        return issues

    def check_ext_source_range(self) -> list[Issue]:
        """EXT_SOURCE_* should be in [0, 1]."""
        issues = []
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if col not in self.df.columns:
                continue
            out = ((self.df[col] < 0) | (self.df[col] > 1)) & self.df[col].notna()
            n = out.sum()
            if n > 0:
                issues.append(Issue(
                    check="ext_source_out_of_range",
                    severity="WARNING",
                    column=col,
                    detail=f"{n:,} values outside [0, 1] in '{col}'.",
                    row_count=int(n),
                ))
        return issues

    # ── Operational record checks ────────────────────────────────────────────

    def check_missing_documents(self) -> list[Issue]:
        """Flag applications where none of the FLAG_DOCUMENT_* columns is set."""
        issues = []
        doc_cols = [c for c in self.df.columns if c.startswith("FLAG_DOCUMENT_")]
        if not doc_cols:
            return issues
        all_zero = (self.df[doc_cols].fillna(0).sum(axis=1) == 0)
        n = all_zero.sum()
        if n > 0:
            issues.append(Issue(
                check="no_documents_submitted",
                severity="WARNING",
                column=None,
                detail=f"{n:,} applications with no FLAG_DOCUMENT_* set to 1.",
                row_count=int(n),
            ))
        return issues

    def check_income_credit_ratio(self) -> list[Issue]:
        """Flag extreme credit-to-income ratios (> 20× income)."""
        issues = []
        if "AMT_CREDIT" not in self.df.columns or "AMT_INCOME_TOTAL" not in self.df.columns:
            return issues
        valid = self.df["AMT_INCOME_TOTAL"] > 0
        ratio = self.df.loc[valid, "AMT_CREDIT"] / self.df.loc[valid, "AMT_INCOME_TOTAL"]
        extreme = (ratio > 20).sum()
        if extreme > 0:
            issues.append(Issue(
                check="extreme_credit_income_ratio",
                severity="WARNING",
                column="AMT_CREDIT",
                detail=f"{extreme:,} applications with credit > 20× annual income.",
                row_count=int(extreme),
            ))
        return issues

    # ── Run all ──────────────────────────────────────────────────────────────

    def run_all(self) -> list[Issue]:
        """Execute every registered check and return the combined issue list."""
        self.issues = []
        checks = [
            self.check_required_columns,
            self.check_missing_values,
            self.check_duplicate_ids,
            self.check_target_values,
            self.check_negative_amounts,
            self.check_gender_codes,
            self.check_days_birth_range,
            self.check_days_employed_anomaly,
            self.check_ext_source_range,
            self.check_missing_documents,
            self.check_income_credit_ratio,
        ]
        for check_fn in checks:
            try:
                found = check_fn()
                self.issues.extend(found)
                for iss in found:
                    log_level = logging.WARNING if iss.severity == "WARNING" else logging.ERROR
                    logger.log(log_level, "[%s] %s — %s", iss.severity, iss.check, iss.detail)
            except Exception as exc:
                logger.exception("Check %s raised an error: %s", check_fn.__name__, exc)
        return self.issues

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of check results."""
        errors = [i for i in self.issues if i.severity == "ERROR"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]
        return {
            "table": self.table_name,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


import logging  # noqa: E402 — needed for logger.log calls above
