"""
Streamlit Dashboard — Credit Risk & Loan Operations Analytics.

Run:
    streamlit run dashboard/app.py

Features:
- Validation-set risk-score distribution
- Risk segment breakdown
- Model performance metrics (from outputs/metrics/)
- Data quality & operational exception summary
- Processing log
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from credit_risk.config import (
    FEATURES_PARQUET,
    METRICS_DIR,
    REPORTS_DIR,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_metrics() -> dict | None:
    path = METRICS_DIR / "model_evaluation.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_exception_report() -> dict | None:
    path = REPORTS_DIR / "exception_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_dq_report() -> dict | None:
    path = REPORTS_DIR / "data_quality_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=600)
def load_risk_scores() -> pd.DataFrame | None:
    """Load feature table and compute risk scores using saved model."""
    if not FEATURES_PARQUET.exists() or not MODEL_PIPELINE_PATH.exists():
        return None
    try:
        from credit_risk.model import load_model
        pipeline, threshold = load_model()
        df = pd.read_parquet(FEATURES_PARQUET)
        drop_cols = [c for c in ["TARGET", "SK_ID_CURR"] if c in df.columns]
        y = df["TARGET"].values if "TARGET" in df.columns else None
        X = df.drop(columns=drop_cols)
        # Use a random sample of 5000 to keep dashboard fast
        sample_idx = X.sample(n=min(5000, len(X)), random_state=42).index
        X_sample = X.loc[sample_idx]
        probs = pipeline.predict_proba(X_sample)[:, 1]
        result = pd.DataFrame({
            "default_probability": probs,
            "prediction": (probs >= threshold).astype(int),
        })
        if y is not None:
            result["actual"] = y[sample_idx]
        result["risk_segment"] = result["default_probability"].apply(_segment)
        return result
    except Exception as exc:
        st.warning(f"Could not load risk scores: {exc}")
        return None


def _segment(p: float) -> str:
    if p < 0.10:
        return "LOW"
    elif p < 0.25:
        return "MEDIUM"
    elif p < 0.50:
        return "HIGH"
    else:
        return "VERY_HIGH"


def _load_prediction_log() -> pd.DataFrame | None:
    log_path = METRICS_DIR / "prediction_log.jsonl"
    if not log_path.exists():
        return None
    rows = []
    with open(log_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows) if rows else None


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Credit Risk Analytics")
st.sidebar.markdown(
    """
**Dataset:** Home Credit Default Risk  
**Source:** Kaggle (2018)  
⚠️ *Not affiliated with Protium Finance.*  
"""
)
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Risk Scores", "🔍 Model Performance", "⚠️ Data Quality", "🔔 Exceptions", "📋 Monitoring Log"],
)

# ── Overview ──────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Credit Risk & Loan Operations Analytics Platform")
    st.markdown(
        """
        This dashboard visualises results from an end-to-end credit-risk pipeline
        built on the **Home Credit Default Risk** dataset.

        > **Disclaimer:** This is a portfolio research project. The model was trained on
        > publicly available Home Credit data and is **not** a regulatory-approved
        > credit decision tool. It was not developed for or validated by Protium Finance.
        """
    )

    # Status tiles
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model_ok = MODEL_PIPELINE_PATH.exists()
        st.metric("Model Status", "✅ Loaded" if model_ok else "❌ Not Trained")
    with col2:
        features_ok = FEATURES_PARQUET.exists()
        st.metric("Feature Table", "✅ Ready" if features_ok else "⚠️ Not Built")
    with col3:
        metrics = load_metrics()
        if metrics:
            winner = metrics.get("winner", "—")
            roc = metrics["all_results"][winner]["roc_auc"] if "all_results" in metrics else "—"
            st.metric("Best ROC-AUC", f"{roc:.4f}" if isinstance(roc, float) else roc)
        else:
            st.metric("Best ROC-AUC", "—")
    with col4:
        exc_report = load_exception_report()
        if exc_report:
            n_flags = exc_report.get("exception_summary", {}).get("total_flags", 0)
            st.metric("Exception Flags", n_flags)
        else:
            st.metric("Exception Flags", "—")

    st.divider()
    st.subheader("Quick Start")
    st.code(
        """# 1. Build feature table
python scripts/03_features.py

# 2. Train models
python scripts/04_train.py

# 3. Run monitoring
python scripts/05_monitor.py

# 4. Start API
uvicorn api.main:app --port 8000

# 5. Run tests
pytest tests/ -v""",
        language="bash",
    )


# ── Risk Scores ───────────────────────────────────────────────────────────────
elif page == "📈 Risk Scores":
    st.title("Risk Score Distribution")

    scores_df = load_risk_scores()
    if scores_df is None:
        st.warning("Risk scores not available. Train the model and build the feature table first.")
    else:
        st.info(f"Showing {len(scores_df):,} sample predictions from the validation set.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Default Probability Distribution")
            import numpy as np
            bins = pd.cut(scores_df["default_probability"], bins=20)
            hist = scores_df.groupby(bins)["default_probability"].count()
            st.bar_chart(hist)

        with col2:
            st.subheader("Risk Segment Breakdown")
            seg_counts = scores_df["risk_segment"].value_counts()
            st.bar_chart(seg_counts)

        # Risk segment table
        st.subheader("Risk Segment Counts")
        seg_table = scores_df.groupby("risk_segment").agg(
            count=("default_probability", "count"),
            mean_prob=("default_probability", "mean"),
            median_prob=("default_probability", "median"),
        ).round(4).reset_index()
        st.dataframe(seg_table, use_container_width=True)

        if "actual" in scores_df.columns:
            st.subheader("Actual Default Rate by Risk Segment")
            actual_rate = scores_df.groupby("risk_segment")["actual"].mean().round(4)
            st.bar_chart(actual_rate)


# ── Model Performance ─────────────────────────────────────────────────────────
elif page == "🔍 Model Performance":
    st.title("Model Performance Metrics")
    metrics = load_metrics()

    if metrics is None:
        st.warning("Model evaluation metrics not found. Run `python scripts/04_train.py` first.")
    else:
        winner = metrics.get("winner", "—")
        st.success(f"**Selected model:** {winner}")
        st.caption(f"Trained at: {metrics.get('trained_at', '—')}")
        st.caption(f"Selection criterion: {metrics.get('selection_criterion', '—')}")

        for model_name, m in metrics.get("all_results", {}).items():
            label = f"🏆 {model_name} (SELECTED)" if model_name == winner else model_name
            with st.expander(label, expanded=(model_name == winner)):
                c1, c2, c3 = st.columns(3)
                c1.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
                c2.metric("PR-AUC", f"{m['pr_auc']:.4f}")
                c3.metric("F1", f"{m['f1']:.4f}")

                c4, c5, c6 = st.columns(3)
                c4.metric("Precision", f"{m['precision']:.4f}")
                c5.metric("Recall", f"{m['recall']:.4f}")
                c6.metric("Threshold", f"{m['threshold']:.4f}")

                st.markdown("**Confusion Matrix**")
                cm = m["confusion_matrix"]
                cm_df = pd.DataFrame(
                    [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"],
                )
                st.dataframe(cm_df)

                st.caption(
                    f"Accuracy (supplementary): {m['accuracy']:.4f} | "
                    f"Val set: {m['val_size']:,} rows | "
                    f"Train time: {m['train_time_seconds']:.1f}s"
                )

        # Latency benchmark
        lat_path = METRICS_DIR / "latency_benchmark.json"
        if lat_path.exists():
            with open(lat_path) as f:
                lat = json.load(f)
            st.subheader("Inference Latency (local, single-row)")
            lm = lat["latency_ms"]
            l1, l2, l3 = st.columns(3)
            l1.metric("Median (ms)", f"{lm['median']:.3f}")
            l2.metric("P95 (ms)", f"{lm['p95']:.3f}")
            l3.metric("P99 (ms)", f"{lm['p99']:.3f}")
            st.caption(lat.get("disclaimer", ""))


# ── Data Quality ──────────────────────────────────────────────────────────────
elif page == "⚠️ Data Quality":
    st.title("Data Quality Report")
    dq = load_dq_report()

    if dq is None:
        st.warning("Data quality report not found. Run `python scripts/01_ingest.py` first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows (sample)", f"{dq.get('total_rows', 0):,}")
        c2.metric("Errors", dq.get("error_count", 0))
        c3.metric("Warnings", dq.get("warning_count", 0))

        issues = dq.get("issues", [])
        if issues:
            st.subheader("Issues Detected")
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df, use_container_width=True)
        else:
            st.success("No issues detected in the sample.")

    # Missing values table
    mv_path = REPORTS_DIR / "missing_values.csv"
    if mv_path.exists():
        st.subheader("Missing Value Rates (Top 30)")
        mv_df = pd.read_csv(mv_path).head(30)
        st.dataframe(mv_df, use_container_width=True)


# ── Exceptions ────────────────────────────────────────────────────────────────
elif page == "🔔 Exceptions":
    st.title("Operational Exceptions")
    exc_report = load_exception_report()

    st.info(
        "Exception checks run on **synthetic labelled records** (not real loan applications). "
        "This demonstrates the rule-based exception-flagging module only."
    )

    if exc_report is None:
        st.warning("Exception report not found. Run `python scripts/05_monitor.py` first.")
    else:
        exc_sum = exc_report.get("exception_summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", exc_sum.get("total_records", 0))
        c2.metric("Total Flags", exc_sum.get("total_flags", 0))
        errors = exc_sum.get("by_severity", {}).get("ERROR", 0)
        c3.metric("Errors", errors)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Flags by Severity")
            bsev = exc_sum.get("by_severity", {})
            if bsev:
                st.bar_chart(pd.Series(bsev))
        with col2:
            st.subheader("Flags by Rule")
            brule = exc_sum.get("by_rule", {})
            if brule:
                rule_df = pd.Series(brule).sort_values(ascending=False)
                st.bar_chart(rule_df)

        flags = exc_sum.get("flags", [])
        if flags:
            st.subheader("Flag Detail (first 50)")
            flags_df = pd.DataFrame(flags).head(50)
            st.dataframe(flags_df, use_container_width=True)

        # Distribution checks
        dist_checks = exc_report.get("distribution_checks", [])
        if dist_checks:
            st.subheader("Distribution Monitoring")
            st.caption("Simple z-score check against Home Credit population reference means.")
            dc_df = pd.DataFrame(dist_checks)
            st.dataframe(dc_df, use_container_width=True)


# ── Monitoring Log ────────────────────────────────────────────────────────────
elif page == "📋 Monitoring Log":
    st.title("API Prediction Log")
    log_df = _load_prediction_log()

    if log_df is None or len(log_df) == 0:
        st.info("No predictions logged yet. Start the API and make /predict calls.")
    else:
        st.metric("Total Predictions Logged", len(log_df))
        if "ts" in log_df.columns:
            log_df["ts"] = pd.to_datetime(log_df["ts"])
        st.dataframe(log_df.tail(100), use_container_width=True)

        if "default_probability" in log_df.columns:
            st.subheader("Recent Risk Score Distribution")
            st.bar_chart(log_df["default_probability"].tail(200).value_counts(bins=10, sort=False))

    st.divider()
    st.subheader("Platform Status")
    st.json({
        "features_ready": FEATURES_PARQUET.exists(),
        "model_ready": MODEL_PIPELINE_PATH.exists(),
        "metrics_ready": (METRICS_DIR / "model_evaluation.json").exists(),
        "exception_report_ready": (REPORTS_DIR / "exception_report.json").exists(),
        "dashboard_time": datetime.now().isoformat(),
    })
