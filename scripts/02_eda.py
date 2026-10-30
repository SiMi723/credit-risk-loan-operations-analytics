"""
Script 02 -- Exploratory Data Analysis.

Run:
    python scripts/02_eda.py

Outputs:
    outputs/reports/eda_report.json
    outputs/reports/missing_values.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import APP_TRAIN_CSV, REPORTS_DIR, CSV_ENCODING, TARGET_COL
from credit_risk.logger import get_logger

logger = get_logger("script.02_eda")


def analyse_target(df: pd.DataFrame) -> dict:
    """Summarise the target variable distribution."""
    vc = df[TARGET_COL].value_counts()
    return {
        "n_rows": len(df),
        "default_count": int(vc.get(1, 0)),
        "non_default_count": int(vc.get(0, 0)),
        "default_rate": round(float(vc.get(1, 0)) / len(df), 4),
        "class_imbalance_ratio": round(float(vc.get(0, 0)) / max(vc.get(1, 1), 1), 2),
    }


def analyse_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame ranking columns by missing-value rate."""
    total = len(df)
    missing = df.isna().sum()
    mv = pd.DataFrame({
        "column": missing.index,
        "n_missing": missing.values,
        "missing_rate": (missing.values / total).round(4),
        "dtype": df.dtypes.values.astype(str),
    }).sort_values("missing_rate", ascending=False)
    return mv[mv["n_missing"] > 0].reset_index(drop=True)


def analyse_numeric_features(df: pd.DataFrame) -> dict:
    """Basic statistics for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats = stats.round(4)
    return stats.to_dict(orient="index")


def analyse_categorical_features(df: pd.DataFrame) -> dict:
    """Value counts for categorical columns."""
    cat_df = df.select_dtypes(exclude=[np.number])
    result = {}
    for col in cat_df.columns:
        vc = cat_df[col].value_counts(dropna=False)
        result[col] = {
            "n_unique": int(cat_df[col].nunique()),
            "top_values": vc.head(10).to_dict(),
            "null_count": int(cat_df[col].isna().sum()),
        }
    return result


def analyse_correlations_with_target(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Spearman correlation of numeric features with TARGET."""
    num_df = df.select_dtypes(include=[np.number])
    if TARGET_COL not in num_df.columns:
        return {}
    correlations = num_df.corr(method="spearman")[TARGET_COL].drop(TARGET_COL)
    top = correlations.abs().nlargest(top_n)
    return {
        col: round(float(correlations[col]), 4)
        for col in top.index
    }


def analyse_ext_sources(df: pd.DataFrame) -> dict:
    """Deep-dive on the external credit score columns."""
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    result = {}
    for col in ext_cols:
        if col not in df.columns:
            continue
        s = df[col]
        result[col] = {
            "missing_rate": round(s.isna().mean(), 4),
            "mean": round(float(s.mean()), 4) if not s.isna().all() else None,
            "std": round(float(s.std()), 4) if not s.isna().all() else None,
            "mean_by_target": {
                str(k): round(float(v), 4)
                for k, v in df.groupby(TARGET_COL)[col].mean().items()
            } if TARGET_COL in df.columns else {},
        }
    return result


def main():
    logger.info("=" * 60)
    logger.info("Script 02 -- Exploratory Data Analysis")
    logger.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading application_train.csv …")
    df = pd.read_csv(APP_TRAIN_CSV, encoding=CSV_ENCODING, low_memory=False)
    logger.info("Loaded: %d rows × %d cols", len(df), df.shape[1])

    logger.info("Analysing TARGET distribution …")
    target_stats = analyse_target(df)
    logger.info(
        "  Default rate: %.2f%% (%d defaults / %d total)",
        target_stats["default_rate"] * 100,
        target_stats["default_count"],
        target_stats["n_rows"],
    )

    logger.info("Analysing missing values …")
    mv_df = analyse_missing_values(df)
    mv_path = REPORTS_DIR / "missing_values.csv"
    mv_df.to_csv(mv_path, index=False)
    logger.info("  %d columns with missing values -> %s", len(mv_df), mv_path.name)

    logger.info("Analysing numeric feature statistics …")
    num_stats = analyse_numeric_features(df)

    logger.info("Analysing categorical features …")
    cat_stats = analyse_categorical_features(df)

    logger.info("Computing Spearman correlations with TARGET …")
    correlations = analyse_correlations_with_target(df)
    logger.info("  Top 5: %s", list(correlations.items())[:5])

    logger.info("Analysing EXT_SOURCE columns …")
    ext_stats = analyse_ext_sources(df)
    for col, stats in ext_stats.items():
        logger.info("  %s: missing=%.1f%%, mean=%.3f",
                    col, stats["missing_rate"] * 100, stats.get("mean") or 0)

    # Compile full report
    report = {
        "target_distribution": target_stats,
        "missing_value_summary": {
            "n_columns_with_missing": len(mv_df),
            "columns_above_50pct_missing": mv_df[mv_df["missing_rate"] > 0.5]["column"].tolist(),
            "top_10_missing": mv_df.head(10).to_dict(orient="records"),
        },
        "numeric_feature_stats": num_stats,
        "categorical_feature_stats": cat_stats,
        "spearman_correlations_with_target": correlations,
        "ext_source_analysis": ext_stats,
    }

    report_path = REPORTS_DIR / "eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("\nEDA report -> %s", report_path)

    logger.info("\n[OK] Script 02 complete.")
    return report


if __name__ == "__main__":
    main()
