"""
Script 01 -- Dataset inventory and schema validation.

Run:
    python scripts/01_ingest.py

Outputs:
    outputs/reports/data_inventory.json
    outputs/reports/data_quality_report.json
"""
import sys
import os

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path

import pandas as pd
import duckdb

from credit_risk.config import (
    DATA_RAW, APP_TRAIN_CSV, BUREAU_CSV, PREV_APP_CSV,
    INSTALLMENTS_CSV, COLUMNS_DESC_CSV, REPORTS_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS, CSV_ENCODING,
)
from credit_risk.data_quality import DataQualityChecker
from credit_risk.logger import get_logger

logger = get_logger("script.01_ingest")


EXPECTED_FILES = {
    "application_train.csv": {"required": True},
    "bureau.csv": {"required": True},
    "previous_application.csv": {"required": True},
    "installments_payments.csv": {"required": True},
    "HomeCredit_columns_description.csv": {"required": True},
    "bureau_balance.csv": {"required": False},
    "POS_CASH_balance.csv": {"required": False},
    "credit_card_balance.csv": {"required": False},
    "application_test.csv": {"required": False},
}


def inventory_files() -> dict:
    """Return a file inventory dict with sizes, presence and status."""
    inventory = {}
    for fname, meta in EXPECTED_FILES.items():
        path = DATA_RAW / fname
        exists = path.exists()
        entry = {
            "filename": fname,
            "required": meta["required"],
            "exists": exists,
            "size_mb": round(path.stat().st_size / 1e6, 2) if exists else None,
            "status": "OK" if exists else ("MISSING_REQUIRED" if meta["required"] else "ABSENT_OPTIONAL"),
        }
        inventory[fname] = entry
        status_str = "[OK]" if exists else ("✗ REQUIRED" if meta["required"] else "- optional")
        logger.info("  %-45s %s", fname, status_str)
    return inventory


def get_csv_stats(path: Path, sample_rows: int = 5000) -> dict:
    """
    Return lightweight schema stats for a CSV file using DuckDB on a sample.
    Never loads the full file into memory for large tables.
    """
    csv_str = str(path).replace("\\", "/")
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")

    try:
        # Total row count
        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{csv_str}', ignore_errors=true)"
        ).fetchone()[0]

        # Column info from sample
        sample_sql = f"""
            SELECT * FROM read_csv_auto('{csv_str}', ignore_errors=true)
            LIMIT {sample_rows}
        """
        sample_df = con.execute(sample_sql).df()
        total = len(sample_df)

        col_stats = {}
        for col in sample_df.columns:
            s = sample_df[col]
            null_count = int(s.isna().sum())
            col_stats[col] = {
                "dtype": str(s.dtype),
                "null_rate_sample": round(null_count / total, 3) if total > 0 else None,
                "n_unique_sample": int(s.nunique()),
                "sample_values": [str(v) for v in s.dropna().unique()[:5]],
            }

        return {
            "n_rows": int(n_rows),
            "n_cols": len(sample_df.columns),
            "columns": col_stats,
        }
    finally:
        con.close()


def check_key_overlap() -> dict:
    """Check SK_ID_CURR overlap between application_train and auxiliary tables."""
    results = {}
    app_csv = str(APP_TRAIN_CSV).replace("\\", "/")
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")

    try:
        app_ids = con.execute(
            f"SELECT COUNT(DISTINCT SK_ID_CURR) FROM read_csv_auto('{app_csv}', ignore_errors=true)"
        ).fetchone()[0]

        for fname, csv_path in [
            ("bureau", BUREAU_CSV),
            ("previous_application", PREV_APP_CSV),
            ("installments_payments", INSTALLMENTS_CSV),
        ]:
            if not csv_path.exists():
                results[fname] = {"status": "file_absent"}
                continue
            other_csv = str(csv_path).replace("\\", "/")
            overlap = con.execute(f"""
                SELECT COUNT(DISTINCT b.SK_ID_CURR)
                FROM read_csv_auto('{other_csv}', ignore_errors=true) b
                INNER JOIN (
                    SELECT DISTINCT SK_ID_CURR
                    FROM read_csv_auto('{app_csv}', ignore_errors=true)
                ) a ON b.SK_ID_CURR = a.SK_ID_CURR
            """).fetchone()[0]
            other_unique = con.execute(
                f"SELECT COUNT(DISTINCT SK_ID_CURR) FROM read_csv_auto('{other_csv}', ignore_errors=true)"
            ).fetchone()[0]
            results[fname] = {
                "app_unique_ids": int(app_ids),
                "table_unique_ids": int(other_unique),
                "overlap_ids": int(overlap),
                "overlap_pct": round(overlap / app_ids * 100, 1),
            }
            logger.info(
                "Key overlap %s ↔ app_train: %d / %d (%.1f%%)",
                fname, overlap, app_ids, overlap / app_ids * 100,
            )
    finally:
        con.close()
    return results


def run_data_quality_check() -> dict:
    """Run DataQualityChecker on a sample of application_train."""
    logger.info("Running data-quality checks on application_train (5000-row sample) …")
    sample = pd.read_csv(APP_TRAIN_CSV, nrows=5000, encoding=CSV_ENCODING, low_memory=False)
    checker = DataQualityChecker(sample, table_name="application_train")
    checker.run_all()
    return checker.summary()


def main():
    logger.info("=" * 60)
    logger.info("Script 01 -- Dataset Inventory & Schema Validation")
    logger.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── File inventory ─────────────────────────────────────────────────────
    logger.info("\n── File inventory ──")
    inventory = inventory_files()

    missing_required = [k for k, v in inventory.items() if not v["exists"] and v["required"]]
    if missing_required:
        logger.error("MISSING REQUIRED FILES: %s", missing_required)
        logger.error("Cannot proceed with training without these files.")
    else:
        logger.info("All required files present.")

    # ── Schema stats ───────────────────────────────────────────────────────
    schema_stats = {}
    for fname, meta in inventory.items():
        if not meta["exists"]:
            continue
        path = DATA_RAW / fname
        if fname.endswith(".csv") and fname != "HomeCredit_columns_description.csv":
            logger.info("\n── Schema: %s ──", fname)
            try:
                stats = get_csv_stats(path)
                schema_stats[fname] = stats
                logger.info("  Rows: %s | Cols: %d", f"{stats['n_rows']:,}", stats['n_cols'])
            except Exception as exc:
                logger.error("Could not read %s: %s", fname, exc)
                schema_stats[fname] = {"error": str(exc)}

    # ── Key overlap ────────────────────────────────────────────────────────
    logger.info("\n── Key overlap checks ──")
    key_overlap = check_key_overlap()

    # ── Data quality ───────────────────────────────────────────────────────
    logger.info("\n── Data quality (application_train sample) ──")
    dq_report = run_data_quality_check()
    logger.info(
        "Quality check: %d errors, %d warnings",
        dq_report["error_count"], dq_report["warning_count"],
    )

    # ── Write reports ──────────────────────────────────────────────────────
    inventory_report = {
        "inventory": inventory,
        "missing_required": missing_required,
        "schema_stats": schema_stats,
        "key_overlap": key_overlap,
    }
    inv_path = REPORTS_DIR / "data_inventory.json"
    with open(inv_path, "w") as f:
        json.dump(inventory_report, f, indent=2, default=str)
    logger.info("\nInventory report -> %s", inv_path)

    dq_path = REPORTS_DIR / "data_quality_report.json"
    with open(dq_path, "w") as f:
        json.dump(dq_report, f, indent=2, default=str)
    logger.info("Data quality report -> %s", dq_path)

    logger.info("\n[OK] Script 01 complete.")
    return inventory_report, dq_report


if __name__ == "__main__":
    main()
