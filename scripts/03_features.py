"""
Script 03 -- Feature engineering pipeline.

Run:
    python scripts/03_features.py

Outputs (all in data/processed/):
    bureau_features.parquet
    prev_app_features.parquet
    instalment_features.parquet
    features.parquet  <- final merged feature table
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse

from credit_risk.config import FEATURES_PARQUET
from credit_risk.features import build_full_feature_table
from credit_risk.logger import get_logger

logger = get_logger("script.03_features")


def main():
    parser = argparse.ArgumentParser(description="Build feature table from raw CSVs.")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cached Parquet files exist.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Script 03 -- Feature Engineering")
    logger.info("=" * 60)

    df = build_full_feature_table(force_rebuild=args.force)

    logger.info("\nFinal feature table summary:")
    logger.info("  Shape: %d rows × %d cols", df.shape[0], df.shape[1])
    logger.info("  Memory usage: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)
    logger.info("  Missing values (any col): %d rows", df.isna().any(axis=1).sum())
    logger.info("  Target distribution:\n%s", df["TARGET"].value_counts().to_string())

    logger.info("\n[OK] Script 03 complete. Feature table -> %s", FEATURES_PARQUET)


if __name__ == "__main__":
    main()
