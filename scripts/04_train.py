"""
Script 04 -- Model training and evaluation.

Run:
    python scripts/04_train.py              # train from feature table
    python scripts/04_train.py --force      # force retrain even if model exists

Prerequisites:
    scripts/03_features.py must have run first (or it will be triggered here).

Outputs:
    models/model_pipeline.joblib
    models/model_metadata.json
    outputs/metrics/model_evaluation.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json

import pandas as pd

from credit_risk.config import FEATURES_PARQUET, METRICS_DIR
from credit_risk.features import build_full_feature_table
from credit_risk.model import train_and_evaluate
from credit_risk.logger import get_logger

logger = get_logger("script.04_train")


def main():
    parser = argparse.ArgumentParser(description="Train credit-risk models.")
    parser.add_argument("--force", action="store_true", help="Force retrain.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Script 04 -- Model Training & Evaluation")
    logger.info("=" * 60)

    # Load or build feature table
    if FEATURES_PARQUET.exists() and not args.force:
        logger.info("Loading feature table from Parquet …")
        df = pd.read_parquet(FEATURES_PARQUET)
    else:
        logger.info("Building feature table …")
        df = build_full_feature_table(force_rebuild=args.force)

    logger.info("Feature table: %d rows × %d cols", df.shape[0], df.shape[1])

    # Train and evaluate
    results = train_and_evaluate(df, force_retrain=args.force)

    # Pretty-print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    winner = results["winner"]
    for model_name, metrics in results["all_results"].items():
        marker = " <- SELECTED" if model_name == winner else ""
        logger.info("\n%s%s", model_name, marker)
        logger.info("  ROC-AUC:   %.4f", metrics["roc_auc"])
        logger.info("  PR-AUC:    %.4f", metrics["pr_auc"])
        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall:    %.4f", metrics["recall"])
        logger.info("  F1:        %.4f", metrics["f1"])
        logger.info("  Accuracy:  %.4f (supplementary)", metrics["accuracy"])
        logger.info("  Threshold: %.4f", metrics["threshold"])
        cm = metrics["confusion_matrix"]
        logger.info("  Confusion matrix: TN=%d FP=%d FN=%d TP=%d",
                    cm["TN"], cm["FP"], cm["FN"], cm["TP"])
        logger.info("  Train time: %.1f s", metrics["train_time_seconds"])

    logger.info("\nSelection criterion: %s", results["selection_criterion"])
    logger.info("Trained at: %s", results["trained_at"])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("\n[OK] Script 04 complete.")
    logger.info("  Model pipeline -> models/model_pipeline.joblib")
    logger.info("  Metrics       -> outputs/metrics/model_evaluation.json")
    return results


if __name__ == "__main__":
    main()
