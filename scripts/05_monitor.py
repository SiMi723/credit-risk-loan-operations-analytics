"""
Script 05 -- Operational monitoring and exception flagging.

Run:
    python scripts/05_monitor.py

Outputs:
    outputs/reports/exception_report.json
    data/synthetic/operational_records.csv  (created if absent)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from datetime import datetime

import pandas as pd

from credit_risk.config import REPORTS_DIR, DATA_SYNTHETIC, SYNTHETIC_OPS_CSV
from credit_risk.exceptions import OperationalExceptionChecker, generate_synthetic_operational_records
from credit_risk.logger import get_logger

logger = get_logger("script.05_monitor")


def run_monitoring(df: pd.DataFrame, dataset_label: str) -> dict:
    """Run exception checks and return summary."""
    checker = OperationalExceptionChecker(df)
    checker.run_all()
    summary = checker.summary()
    summary["dataset_label"] = dataset_label
    summary["checked_at"] = datetime.now().isoformat()
    return summary


def check_distribution(df: pd.DataFrame, col: str, ref_mean: float, ref_std: float) -> dict:
    """
    Simple z-score distribution check: flag if column mean deviates > 3σ
    from a reference.  This is a minimal PSI-like indicator, not a full
    Population Stability Index calculation.
    """
    if col not in df.columns:
        return {"status": "column_absent"}
    vals = df[col].dropna()
    if len(vals) == 0:
        return {"status": "all_null"}
    current_mean = float(vals.mean())
    z = abs(current_mean - ref_mean) / max(ref_std, 1e-6)
    return {
        "column": col,
        "current_mean": round(current_mean, 4),
        "reference_mean": round(ref_mean, 4),
        "z_score": round(z, 3),
        "alert": z > 3.0,
        "note": "Distribution shift detected (|z| > 3)" if z > 3.0 else "Within expected range",
    }


def main():
    logger.info("=" * 60)
    logger.info("Script 05 -- Operational Monitoring & Exception Flagging")
    logger.info("=" * 60)
    logger.info("NOTE: Using labelled synthetic records for demonstration.")
    logger.info("These are NOT real loan applications.\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SYNTHETIC.mkdir(parents=True, exist_ok=True)

    # ── Generate or load synthetic records ────────────────────────────────────
    if SYNTHETIC_OPS_CSV.exists():
        logger.info("Loading synthetic records from %s …", SYNTHETIC_OPS_CSV.name)
        ops_df = pd.read_csv(SYNTHETIC_OPS_CSV)
    else:
        logger.info("Generating 200 synthetic operational records …")
        ops_df = generate_synthetic_operational_records(n=200)
        ops_df.to_csv(SYNTHETIC_OPS_CSV, index=False)
        logger.info("Synthetic records -> %s", SYNTHETIC_OPS_CSV)

    logger.info("Records: %d total (%d valid, %d exception)",
                len(ops_df),
                (ops_df.get("record_type", pd.Series()) == "synthetic_valid").sum(),
                (ops_df.get("record_type", pd.Series()) == "synthetic_exception").sum())

    # ── Exception checking ────────────────────────────────────────────────────
    logger.info("\nRunning operational exception checks …")
    exc_summary = run_monitoring(ops_df, "synthetic_operational_records")
    logger.info("  Total flags: %d", exc_summary["total_flags"])
    for severity, count in exc_summary.get("by_severity", {}).items():
        logger.info("    %s: %d", severity, count)
    logger.info("  By rule:")
    for rule, count in exc_summary.get("by_rule", {}).items():
        logger.info("    %-40s %d", rule, count)

    # ── Distribution monitoring ───────────────────────────────────────────────
    logger.info("\nRunning distribution checks …")
    # Reference values approximate Home Credit population means
    dist_checks = [
        check_distribution(ops_df, "AMT_INCOME_TOTAL", ref_mean=168797.9, ref_std=237123.1),
        check_distribution(ops_df, "AMT_CREDIT", ref_mean=599025.9, ref_std=402490.8),
        check_distribution(ops_df, "AMT_ANNUITY", ref_mean=27108.6, ref_std=14493.7),
    ]
    for chk in dist_checks:
        alert = "⚠ ALERT" if chk.get("alert") else "  OK   "
        logger.info("  %s | %s: current_mean=%.1f, z=%.2f",
                    alert, chk.get("column", "?"),
                    chk.get("current_mean", 0), chk.get("z_score", 0))

    # ── Compile report ────────────────────────────────────────────────────────
    report = {
        "monitoring_run_at": datetime.now().isoformat(),
        "dataset_label": "synthetic_operational_records",
        "important_disclaimer": (
            "All records are SYNTHETIC. This is a demonstration of rule-based "
            "exception checking and is NOT a production compliance system."
        ),
        "exception_summary": exc_summary,
        "distribution_checks": dist_checks,
    }

    report_path = REPORTS_DIR / "exception_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("\nException report -> %s", report_path)

    logger.info("\n[OK] Script 05 complete.")
    return report


if __name__ == "__main__":
    main()
