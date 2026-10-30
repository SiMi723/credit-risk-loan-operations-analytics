"""
Feature engineering pipeline.

All aggregations against large tables (bureau, previous_application,
installments_payments) are executed via DuckDB SQL on the raw CSV files,
so they never fully load into Python memory.

The output is a single merged Parquet file (features.parquet) at the
SK_ID_CURR grain, ready for modelling.
"""
from __future__ import annotations

import duckdb
import pandas as pd
from pathlib import Path

from credit_risk.config import (
    APP_TRAIN_CSV,
    BUREAU_CSV,
    PREV_APP_CSV,
    INSTALLMENTS_CSV,
    FEATURES_PARQUET,
    BUREAU_FEATS_PARQUET,
    PREV_FEATS_PARQUET,
    INSTALMENT_FEATS_PARQUET,
    CSV_ENCODING,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    TARGET_COL,
    ID_COL,
)
from credit_risk.logger import get_logger

logger = get_logger("credit_risk.features")

# ── SQL templates ─────────────────────────────────────────────────────────────

BUREAU_SQL = """
SELECT
    SK_ID_CURR,
    COUNT(*)                                                AS bureau_credit_count,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)  AS bureau_active_count,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)  AS bureau_closed_count,
    MAX(CREDIT_DAY_OVERDUE)                                AS bureau_overdue_max,
    AVG(CREDIT_DAY_OVERDUE)                                AS bureau_overdue_mean,
    SUM(AMT_CREDIT_SUM)                                    AS bureau_amt_credit_sum,
    AVG(AMT_CREDIT_SUM)                                    AS bureau_amt_credit_mean,
    MIN(DAYS_CREDIT)                                       AS bureau_days_credit_min,
    MAX(DAYS_CREDIT)                                       AS bureau_days_credit_max,
    AVG(DAYS_CREDIT)                                       AS bureau_days_credit_mean,
    SUM(AMT_CREDIT_SUM_DEBT)                               AS bureau_debt_sum,
    AVG(AMT_CREDIT_SUM_DEBT)                               AS bureau_debt_mean,
    SUM(AMT_ANNUITY)                                       AS bureau_annuity_sum,
    COUNT(DISTINCT CREDIT_TYPE)                            AS bureau_credit_type_count
FROM read_csv_auto('{csv_path}', ignore_errors=true)
GROUP BY SK_ID_CURR
"""

PREV_APP_SQL = """
SELECT
    SK_ID_CURR,
    COUNT(*)                                                          AS prev_app_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)   AS prev_approved_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END)   AS prev_refused_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END)   AS prev_canceled_count,
    AVG(AMT_CREDIT)                                                   AS prev_amt_credit_mean,
    MAX(AMT_CREDIT)                                                   AS prev_amt_credit_max,
    AVG(AMT_ANNUITY)                                                  AS prev_annuity_mean,
    AVG(AMT_APPLICATION)                                              AS prev_amt_application_mean,
    AVG(DAYS_DECISION)                                                AS prev_days_decision_mean,
    MIN(DAYS_DECISION)                                                AS prev_days_decision_min,
    AVG(CASE WHEN AMT_APPLICATION > 0
             THEN AMT_CREDIT / AMT_APPLICATION ELSE NULL END)         AS prev_credit_to_app_ratio,
    COUNT(DISTINCT NAME_CONTRACT_TYPE)                                AS prev_contract_type_count,
    SUM(CASE WHEN NAME_YIELD_GROUP = 'high'  THEN 1 ELSE 0 END)       AS prev_high_yield_count,
    SUM(CASE WHEN NAME_YIELD_GROUP = 'XNA'   THEN 1 ELSE 0 END)       AS prev_xna_yield_count
FROM read_csv_auto('{csv_path}', ignore_errors=true)
GROUP BY SK_ID_CURR
"""

INSTALMENT_SQL = """
WITH base AS (
    SELECT
        SK_ID_CURR,
        DAYS_INSTALMENT,
        DAYS_ENTRY_PAYMENT,
        AMT_INSTALMENT,
        AMT_PAYMENT,
        -- Days past due: positive = late, negative = early
        (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)            AS dpd,
        -- Payment shortfall: negative = underpaid, positive = overpaid
        (AMT_PAYMENT - AMT_INSTALMENT)                    AS amt_diff
    FROM read_csv_auto('{csv_path}', ignore_errors=true)
    WHERE AMT_INSTALMENT IS NOT NULL
      AND AMT_PAYMENT    IS NOT NULL
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                           AS instalment_count,
    SUM(CASE WHEN dpd > 0 THEN 1 ELSE 0 END)           AS instalment_late_count,
    AVG(GREATEST(dpd, 0))                              AS instalment_dpd_mean,
    MAX(GREATEST(dpd, 0))                              AS instalment_dpd_max,
    -- Ratio of late payments
    SUM(CASE WHEN dpd > 0 THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(*), 0)                          AS instalment_late_ratio,
    AVG(amt_diff)                                      AS instalment_amt_diff_mean,
    MIN(amt_diff)                                      AS instalment_amt_diff_min,
    -- Under-payment count
    SUM(CASE WHEN amt_diff < -1 THEN 1 ELSE 0 END)     AS instalment_underpay_count,
    AVG(AMT_PAYMENT)                                   AS instalment_amt_payment_mean,
    AVG(AMT_INSTALMENT)                                AS instalment_amt_scheduled_mean
FROM base
GROUP BY SK_ID_CURR
"""

# Application-level feature engineering (runs in pandas on the loaded df)
APP_FEATURE_DEFS = {
    "credit_income_ratio": lambda df: pd.to_numeric(df["AMT_CREDIT"], errors="coerce") / pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce").replace(0, float("nan")),
    "annuity_income_ratio": lambda df: pd.to_numeric(df["AMT_ANNUITY"], errors="coerce") / pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce").replace(0, float("nan")),
    "credit_goods_ratio": lambda df: pd.to_numeric(df["AMT_CREDIT"], errors="coerce") / pd.to_numeric(df["AMT_GOODS_PRICE"], errors="coerce").replace(0, float("nan")),
    "age_years": lambda df: -pd.to_numeric(df["DAYS_BIRTH"], errors="coerce") / 365.0,
    "employed_years": lambda df: (-pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce").replace(365243, float("nan"))) / 365.0,
    "id_publish_years": lambda df: -pd.to_numeric(df["DAYS_ID_PUBLISH"], errors="coerce") / 365.0,
    "ext_source_mean": lambda df: df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].apply(pd.to_numeric, errors="coerce").mean(axis=1),
    "ext_source_min": lambda df: df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].apply(pd.to_numeric, errors="coerce").min(axis=1),
    "ext_source_max": lambda df: df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].apply(pd.to_numeric, errors="coerce").max(axis=1),
    "income_per_family_member": lambda df: pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce") / pd.to_numeric(df["CNT_FAM_MEMBERS"], errors="coerce").replace(0, float("nan")),
    "annuity_credit_ratio": lambda df: pd.to_numeric(df["AMT_ANNUITY"], errors="coerce") / pd.to_numeric(df["AMT_CREDIT"], errors="coerce").replace(0, float("nan")),
    "phone_change_years": lambda df: -pd.to_numeric(df.get("DAYS_LAST_PHONE_CHANGE", pd.Series([float("nan")] * len(df))), errors="coerce") / 365.0,
    "registration_years": lambda df: -pd.to_numeric(df.get("DAYS_REGISTRATION", pd.Series([float("nan")] * len(df))), errors="coerce") / 365.0,
}


def build_bureau_features(csv_path: Path = BUREAU_CSV, output: Path = BUREAU_FEATS_PARQUET) -> pd.DataFrame:
    """Aggregate bureau.csv to SK_ID_CURR grain via DuckDB."""
    logger.info("Building bureau features …")
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    csv_str = str(csv_path).replace("\\", "/")
    sql = BUREAU_SQL.format(csv_path=csv_str)
    df = con.execute(sql).df()
    con.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Bureau features: %d rows × %d cols → %s", len(df), df.shape[1], output.name)
    return df


def build_prev_app_features(csv_path: Path = PREV_APP_CSV, output: Path = PREV_FEATS_PARQUET) -> pd.DataFrame:
    """Aggregate previous_application.csv to SK_ID_CURR grain via DuckDB."""
    logger.info("Building previous-application features …")
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    csv_str = str(csv_path).replace("\\", "/")
    sql = PREV_APP_SQL.format(csv_path=csv_str)
    df = con.execute(sql).df()
    con.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Prev-app features: %d rows × %d cols → %s", len(df), df.shape[1], output.name)
    return df


def build_instalment_features(csv_path: Path = INSTALLMENTS_CSV, output: Path = INSTALMENT_FEATS_PARQUET) -> pd.DataFrame:
    """Aggregate installments_payments.csv to SK_ID_CURR grain via DuckDB."""
    logger.info("Building instalment payment features (13.6M rows — this may take 2–3 min) …")
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    csv_str = str(csv_path).replace("\\", "/")
    sql = INSTALMENT_SQL.format(csv_path=csv_str)
    df = con.execute(sql).df()
    con.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Instalment features: %d rows × %d cols → %s", len(df), df.shape[1], output.name)
    return df


def build_app_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to the application DataFrame (in-place)."""
    logger.info("Engineering application-level features …")
    for name, fn in APP_FEATURE_DEFS.items():
        try:
            df[name] = fn(df)
        except Exception as exc:
            logger.warning("Could not compute feature '%s': %s", name, exc)
    return df


def load_or_build_auxiliary_features(
    bureau_csv: Path = BUREAU_CSV,
    prev_csv: Path = PREV_APP_CSV,
    inst_csv: Path = INSTALLMENTS_CSV,
    bureau_out: Path = BUREAU_FEATS_PARQUET,
    prev_out: Path = PREV_FEATS_PARQUET,
    inst_out: Path = INSTALMENT_FEATS_PARQUET,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load auxiliary feature tables from Parquet if they exist and force_rebuild
    is False; otherwise build and cache them.

    Returns (bureau_df, prev_df, inst_df).
    """

    def _load_or_build(parquet_path: Path, build_fn, *args):
        if parquet_path.exists() and not force_rebuild:
            logger.info("Loading cached features from %s", parquet_path.name)
            return pd.read_parquet(parquet_path)
        return build_fn(*args, output=parquet_path)

    bureau_df = _load_or_build(bureau_out, build_bureau_features, bureau_csv)
    prev_df = _load_or_build(prev_out, build_prev_app_features, prev_csv)
    inst_df = _load_or_build(inst_out, build_instalment_features, inst_csv)
    return bureau_df, prev_df, inst_df


def merge_features(
    app_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    inst_df: pd.DataFrame,
    id_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """
    Left-join auxiliary feature tables onto the application DataFrame.
    The join is on SK_ID_CURR.  Missing rows in auxiliary tables get NaN,
    which downstream imputation handles.
    """
    logger.info("Merging feature tables …")
    n_before = len(app_df)

    df = app_df.merge(bureau_df, on=id_col, how="left")
    df = df.merge(prev_df, on=id_col, how="left")
    df = df.merge(inst_df, on=id_col, how="left")

    assert len(df) == n_before, (
        f"Row count changed after merge: {n_before} → {len(df)}. "
        "Possible duplicate key in auxiliary table."
    )
    logger.info("Merged feature table: %d rows × %d cols", len(df), df.shape[1])
    return df


def build_full_feature_table(
    app_csv: Path = APP_TRAIN_CSV,
    output_parquet: Path = FEATURES_PARQUET,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    End-to-end pipeline: load application CSV, engineer features,
    join auxiliary aggregations, and write the result to Parquet.

    Returns the final feature DataFrame (includes TARGET for train set).
    """
    if output_parquet.exists() and not force_rebuild:
        logger.info("Feature table already exists — loading from %s", output_parquet.name)
        return pd.read_parquet(output_parquet)

    from credit_risk.data_io import read_csv_full
    logger.info("Loading application CSV …")
    app_df = read_csv_full(app_csv)

    # Replace DAYS_EMPLOYED anomaly before feature engineering
    app_df["DAYS_EMPLOYED"] = app_df["DAYS_EMPLOYED"].replace(365243, float("nan"))

    app_df = build_app_features(app_df)

    bureau_df, prev_df, inst_df = load_or_build_auxiliary_features(force_rebuild=force_rebuild)

    merged = merge_features(app_df, bureau_df, prev_df, inst_df)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    logger.info("Full feature table written: %d rows × %d cols → %s",
                len(merged), merged.shape[1], output_parquet.name)
    return merged
