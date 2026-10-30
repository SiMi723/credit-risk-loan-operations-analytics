"""
Operational exception-checking module.

This module implements transparent, deterministic rules for flagging
potential data quality and operational issues in application records.

IMPORTANT:  These checks are illustrative rule-based validations only.
They do NOT constitute lending decisions, regulatory compliance checks,
or production risk controls.

Where actual operational data is unavailable, clearly labelled synthetic
records are used (stored in data/synthetic/operational_records.csv).
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

import pandas as pd

from credit_risk.logger import get_logger

logger = get_logger("credit_risk.exceptions")


class ExceptionSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclasses.dataclass
class ExceptionFlag:
    """Represents a single operational exception."""
    application_id: int | str
    rule: str
    severity: ExceptionSeverity
    detail: str
    field: str | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["severity"] = self.severity.value
        return d


class OperationalExceptionChecker:
    """
    Apply a battery of deterministic operational rules to a DataFrame
    of loan applications.

    Parameters
    ----------
    df         : DataFrame with application records.
    id_col     : Column name for the application/customer identifier.
    """

    def __init__(self, df: pd.DataFrame, id_col: str = "SK_ID_CURR"):
        self.df = df.copy()
        self.id_col = id_col
        self.flags: list[ExceptionFlag] = []

    # ── Rule helpers ─────────────────────────────────────────────────────────

    def _flag(self, mask: pd.Series, rule: str, severity: ExceptionSeverity,
              detail_fn, field: str | None = None) -> None:
        """Apply a boolean mask and add flags for matching rows."""
        for idx in self.df[mask].index:
            app_id = self.df.at[idx, self.id_col] if self.id_col in self.df.columns else idx
            self.flags.append(ExceptionFlag(
                application_id=app_id,
                rule=rule,
                severity=severity,
                detail=detail_fn(self.df.loc[idx]),
                field=field,
            ))

    # ── Rules ────────────────────────────────────────────────────────────────

    def rule_missing_required_fields(self) -> None:
        """Flag applications missing any mandatory field."""
        required = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "CODE_GENDER"]
        for col in required:
            if col not in self.df.columns:
                continue
            mask = self.df[col].isna()
            self._flag(
                mask,
                rule="missing_required_field",
                severity=ExceptionSeverity.ERROR,
                detail_fn=lambda row, c=col: f"Required field '{c}' is missing.",
                field=col,
            )

    def rule_invalid_income(self) -> None:
        """Flag zero or negative income values."""
        if "AMT_INCOME_TOTAL" not in self.df.columns:
            return
        mask = self.df["AMT_INCOME_TOTAL"].notna() & (self.df["AMT_INCOME_TOTAL"] <= 0)
        self._flag(
            mask,
            rule="invalid_income",
            severity=ExceptionSeverity.ERROR,
            detail_fn=lambda row: f"AMT_INCOME_TOTAL={row['AMT_INCOME_TOTAL']:.0f} is invalid.",
            field="AMT_INCOME_TOTAL",
        )

    def rule_extreme_credit_income_ratio(self, threshold: float = 20.0) -> None:
        """Flag applications where credit > threshold × income."""
        if not {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(self.df.columns):
            return
        valid = self.df["AMT_INCOME_TOTAL"] > 0
        ratio = self.df["AMT_CREDIT"] / self.df["AMT_INCOME_TOTAL"].replace(0, float("nan"))
        mask = valid & (ratio > threshold)
        self._flag(
            mask,
            rule="extreme_credit_income_ratio",
            severity=ExceptionSeverity.WARNING,
            detail_fn=lambda row: (
                f"Credit/income ratio={row['AMT_CREDIT']/row['AMT_INCOME_TOTAL']:.1f}× "
                f"(threshold {threshold}×)."
            ),
            field="AMT_CREDIT",
        )

    def rule_underage_applicant(self) -> None:
        """Flag applicants younger than 18 years."""
        if "DAYS_BIRTH" not in self.df.columns:
            return
        age_years = -self.df["DAYS_BIRTH"] / 365.0
        mask = self.df["DAYS_BIRTH"].notna() & (age_years < 18)
        self._flag(
            mask,
            rule="underage_applicant",
            severity=ExceptionSeverity.ERROR,
            detail_fn=lambda row: f"Applicant age={-row['DAYS_BIRTH']/365:.1f} yr (< 18).",
            field="DAYS_BIRTH",
        )

    def rule_anomalous_employment(self) -> None:
        """Flag the 365243 DAYS_EMPLOYED placeholder."""
        if "DAYS_EMPLOYED" not in self.df.columns:
            return
        mask = self.df["DAYS_EMPLOYED"] == 365243
        self._flag(
            mask,
            rule="anomalous_days_employed",
            severity=ExceptionSeverity.INFO,
            detail_fn=lambda row: "DAYS_EMPLOYED=365243 (known retiree/unemployed placeholder).",
            field="DAYS_EMPLOYED",
        )

    def rule_duplicate_id(self) -> None:
        """Flag duplicate application IDs."""
        if self.id_col not in self.df.columns:
            return
        dupes = self.df[self.id_col].duplicated(keep=False)
        self._flag(
            dupes,
            rule="duplicate_application_id",
            severity=ExceptionSeverity.ERROR,
            detail_fn=lambda row: f"Duplicate {self.id_col}={row[self.id_col]}.",
            field=self.id_col,
        )

    def rule_missing_documents(self) -> None:
        """Flag applications where no document flags are set."""
        doc_cols = [c for c in self.df.columns if c.startswith("FLAG_DOCUMENT_")]
        if not doc_cols:
            return
        no_docs = self.df[doc_cols].fillna(0).sum(axis=1) == 0
        self._flag(
            no_docs,
            rule="no_documents_submitted",
            severity=ExceptionSeverity.WARNING,
            detail_fn=lambda row: "No FLAG_DOCUMENT_* columns set to 1.",
        )

    def rule_inconsistent_gender(self) -> None:
        """Flag unexpected gender codes."""
        if "CODE_GENDER" not in self.df.columns:
            return
        valid_genders = {"M", "F", "XNA"}
        mask = self.df["CODE_GENDER"].notna() & ~self.df["CODE_GENDER"].isin(valid_genders)
        self._flag(
            mask,
            rule="invalid_gender_code",
            severity=ExceptionSeverity.WARNING,
            detail_fn=lambda row: f"CODE_GENDER='{row['CODE_GENDER']}' is not in {valid_genders}.",
            field="CODE_GENDER",
        )

    def rule_negative_annuity(self) -> None:
        """Flag negative loan annuity values."""
        if "AMT_ANNUITY" not in self.df.columns:
            return
        mask = self.df["AMT_ANNUITY"].notna() & (self.df["AMT_ANNUITY"] < 0)
        self._flag(
            mask,
            rule="negative_annuity",
            severity=ExceptionSeverity.ERROR,
            detail_fn=lambda row: f"AMT_ANNUITY={row['AMT_ANNUITY']:.2f} is negative.",
            field="AMT_ANNUITY",
        )

    # ── Run all ──────────────────────────────────────────────────────────────

    def run_all(self) -> list[ExceptionFlag]:
        """Execute all registered rules and return the combined flag list."""
        self.flags = []
        rules = [
            self.rule_missing_required_fields,
            self.rule_invalid_income,
            self.rule_extreme_credit_income_ratio,
            self.rule_underage_applicant,
            self.rule_anomalous_employment,
            self.rule_duplicate_id,
            self.rule_missing_documents,
            self.rule_inconsistent_gender,
            self.rule_negative_annuity,
        ]
        for rule_fn in rules:
            try:
                rule_fn()
            except Exception as exc:
                logger.exception("Rule %s raised: %s", rule_fn.__name__, exc)
        logger.info("Exception checking complete — %d flags raised.", len(self.flags))
        return self.flags

    def summary(self) -> dict[str, Any]:
        """Summarise flags by severity."""
        from collections import Counter
        severity_counts = Counter(f.severity.value for f in self.flags)
        rule_counts = Counter(f.rule for f in self.flags)
        return {
            "total_records": len(self.df),
            "total_flags": len(self.flags),
            "by_severity": dict(severity_counts),
            "by_rule": dict(rule_counts),
            "flags": [f.to_dict() for f in self.flags[:500]],  # cap at 500 for JSON size
        }


def generate_synthetic_operational_records(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate clearly labelled synthetic operational records for demonstration.

    These records are NOT from the Home Credit dataset and are NOT real loans.
    They are used only to demonstrate the exception-checking module where
    actual operational data is unavailable.
    """
    rng = pd.np.random if hasattr(pd, 'np') else __import__('numpy').random.default_rng(seed)
    import numpy as np
    rng = np.random.default_rng(seed)

    n_valid = int(n * 0.75)
    n_bad = n - n_valid

    # ── Valid records ─────────────────────────────────────────────────────────
    valid = pd.DataFrame({
        "SK_ID_CURR": range(900000, 900000 + n_valid),
        "AMT_INCOME_TOTAL": rng.uniform(50000, 500000, n_valid),
        "AMT_CREDIT": rng.uniform(100000, 1000000, n_valid),
        "AMT_ANNUITY": rng.uniform(5000, 50000, n_valid),
        "DAYS_BIRTH": rng.integers(-25000, -6570, n_valid),   # ages 18–68
        "DAYS_EMPLOYED": rng.integers(-5000, -30, n_valid),
        "CODE_GENDER": rng.choice(["M", "F"], n_valid),
        "FLAG_DOCUMENT_3": rng.integers(0, 2, n_valid),
        "FLAG_DOCUMENT_6": rng.integers(0, 2, n_valid),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n_valid),
        "record_type": "synthetic_valid",
    })

    # ── Bad records with planted exceptions ──────────────────────────────────
    bad_records = []
    exc_types = [
        # missing income
        {"AMT_INCOME_TOTAL": float("nan"), "AMT_CREDIT": 300000, "DAYS_BIRTH": -12000,
         "CODE_GENDER": "F", "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_6": 0,
         "AMT_ANNUITY": 15000, "DAYS_EMPLOYED": -500, "exception_planted": "missing_income"},
        # zero income
        {"AMT_INCOME_TOTAL": 0, "AMT_CREDIT": 500000, "DAYS_BIRTH": -14000,
         "CODE_GENDER": "M", "FLAG_DOCUMENT_3": 0, "FLAG_DOCUMENT_6": 0,
         "AMT_ANNUITY": 20000, "DAYS_EMPLOYED": -200, "exception_planted": "zero_income"},
        # underage
        {"AMT_INCOME_TOTAL": 120000, "AMT_CREDIT": 200000, "DAYS_BIRTH": -5000,
         "CODE_GENDER": "M", "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_6": 0,
         "AMT_ANNUITY": 10000, "DAYS_EMPLOYED": -100, "exception_planted": "underage"},
        # extreme credit ratio
        {"AMT_INCOME_TOTAL": 10000, "AMT_CREDIT": 5000000, "DAYS_BIRTH": -14000,
         "CODE_GENDER": "F", "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_6": 0,
         "AMT_ANNUITY": 50000, "DAYS_EMPLOYED": -800, "exception_planted": "extreme_ratio"},
        # invalid gender
        {"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 300000, "DAYS_BIRTH": -13000,
         "CODE_GENDER": "U", "FLAG_DOCUMENT_3": 0, "FLAG_DOCUMENT_6": 0,
         "AMT_ANNUITY": 15000, "DAYS_EMPLOYED": -400, "exception_planted": "invalid_gender"},
        # negative annuity
        {"AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 250000, "DAYS_BIRTH": -11000,
         "CODE_GENDER": "M", "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_6": 1,
         "AMT_ANNUITY": -5000, "DAYS_EMPLOYED": -300, "exception_planted": "negative_annuity"},
        # anomalous employment
        {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 200000, "DAYS_BIRTH": -15000,
         "CODE_GENDER": "F", "FLAG_DOCUMENT_3": 0, "FLAG_DOCUMENT_6": 0,
         "AMT_ANNUITY": 12000, "DAYS_EMPLOYED": 365243, "exception_planted": "anomalous_employment"},
        # no documents
        {"AMT_INCOME_TOTAL": 90000, "AMT_CREDIT": 180000, "DAYS_BIRTH": -13500,
         "CODE_GENDER": "M", "FLAG_DOCUMENT_3": 0, "FLAG_DOCUMENT_6": 0,
         "AMT_ANNUITY": 9000, "DAYS_EMPLOYED": -600, "exception_planted": "no_documents"},
    ]

    # Repeat to fill n_bad rows
    rows = []
    for i in range(n_bad):
        row = exc_types[i % len(exc_types)].copy()
        row["SK_ID_CURR"] = 990000 + i
        row["NAME_CONTRACT_TYPE"] = "Cash loans"
        row["record_type"] = "synthetic_exception"
        rows.append(row)
    bad = pd.DataFrame(rows)

    df = pd.concat([valid, bad], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["data_source"] = "SYNTHETIC — NOT real loan data"
    return df
