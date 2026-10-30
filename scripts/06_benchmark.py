"""
Script 06 -- API inference latency benchmark.

Measures prediction latency by sending repeated requests to the
trained model pipeline locally (without HTTP overhead).

Run:
    python scripts/06_benchmark.py

Outputs:
    outputs/metrics/latency_benchmark.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time

import numpy as np
import pandas as pd

from credit_risk.config import FEATURES_PARQUET, METRICS_DIR, MODEL_PIPELINE_PATH
from credit_risk.model import load_model
from credit_risk.logger import get_logger

logger = get_logger("script.06_benchmark")

N_WARMUP = 10
N_SAMPLES = 1000


def main():
    logger.info("=" * 60)
    logger.info("Script 06 -- Inference Latency Benchmark")
    logger.info("=" * 60)
    logger.info("Conditions: local in-process inference, no HTTP overhead.")
    logger.info("Hardware: measured on the machine running this script.\n")

    if not MODEL_PIPELINE_PATH.exists():
        logger.error("Model not found. Run scripts/04_train.py first.")
        sys.exit(1)

    pipeline, threshold = load_model()

    if not FEATURES_PARQUET.exists():
        logger.error("Feature table not found. Run scripts/03_features.py first.")
        sys.exit(1)

    # Load a small sample of real feature rows for benchmarking
    df = pd.read_parquet(FEATURES_PARQUET)
    # Remove target and id columns
    drop_cols = [c for c in ["TARGET", "SK_ID_CURR"] if c in df.columns]
    X = df.drop(columns=drop_cols).head(100)

    # ── Warmup ────────────────────────────────────────────────────────────────
    logger.info("Warming up (%d predictions) …", N_WARMUP)
    for _ in range(N_WARMUP):
        row = X.sample(1, random_state=0)
        pipeline.predict_proba(row)

    # ── Benchmark: single-row inference ──────────────────────────────────────
    logger.info("Benchmarking single-row inference (%d samples) …", N_SAMPLES)
    latencies_ms = []
    for i in range(N_SAMPLES):
        row = X.sample(1, random_state=i)
        t0 = time.perf_counter()
        proba = pipeline.predict_proba(row)[0, 1]
        pred = int(proba >= threshold)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    latencies = np.array(latencies_ms)
    results = {
        "test_type": "single_row_local_inference",
        "n_warmup": N_WARMUP,
        "n_samples": N_SAMPLES,
        "hardware_note": "Local machine; no HTTP overhead; results vary by hardware.",
        "latency_ms": {
            "min": round(float(latencies.min()), 4),
            "median": round(float(np.median(latencies)), 4),
            "p75": round(float(np.percentile(latencies, 75)), 4),
            "p95": round(float(np.percentile(latencies, 95)), 4),
            "p99": round(float(np.percentile(latencies, 99)), 4),
            "max": round(float(latencies.max()), 4),
            "mean": round(float(latencies.mean()), 4),
            "std": round(float(latencies.std()), 4),
        },
        "disclaimer": (
            "Local prototype measurements only. "
            "Do not extrapolate to production throughput or SLA."
        ),
    }

    logger.info("\nLatency results (ms):")
    for k, v in results["latency_ms"].items():
        logger.info("  %-8s %.4f ms", k, v)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "latency_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nLatency benchmark -> %s", out_path)
    logger.info("\n[OK] Script 06 complete.")
    return results


if __name__ == "__main__":
    main()
