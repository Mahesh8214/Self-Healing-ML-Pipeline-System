import streamlit as st
import json
import pandas as pd
import os
import sys
import threading
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.job_manager import JobManager
from src.registry.model_registry import ModelRegistry
from src.pipelines.monitoring_pipeline import MonitoringPipeline
from src.pipelines.training_pipeline import run_training_pipeline

st.set_page_config(
    page_title="Drift Monitoring & Self-Healing Center",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# Header & Navigation
# ---------------------------------------------------
st.title("📊 Drift Monitoring & Self-Healing Center")
st.caption("Real-Time Data Drift Diagnostics, Feature Breakdown, Background Training & Model Governance")

# ---------------------------------------------------
# Data Loaders (Safely Top-Level Initialized)
# ---------------------------------------------------
report_path = "artifacts/reports/drift_report.json"
reference_path = "artifacts/data/reference_data.csv"
batch_folder = "data/production_batches"
monitor_log_path = "artifacts/monitoring/monitoring_log.json"

report = None
feat_results = {}
if os.path.exists(report_path):
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
            feat_results = report.get("feature_results", {})
    except Exception as e:
        st.error(f"Error reading drift report: {e}")

latest_job = JobManager.get_latest_job()
active_job = JobManager.has_active_job()
settings = JobManager.get_settings()

registry = ModelRegistry()
try:
    latest_model_path = registry.get_latest_model()
    reg_data = registry.load_registry()
    version_count = len(reg_data.get("versions", []))
    latest_version = reg_data.get("versions", [{}])[-1].get("version", f"v{version_count}") if reg_data.get("versions") else "v1"
except Exception:
    latest_version = "v1"

# Determine System Status
drift_flag = report.get("drift_detected", False) if report else False

if active_job:
    sys_status = "RETRAINING"
    sys_badge = "🔄 RETRAINING IN PROGRESS"
    sys_color = "orange"
elif drift_flag and settings.get("auto_healing", True):
    sys_status = "RECOVERED"
    sys_badge = "⚡ DRIFT DETECTED — AUTO-HEALED"
    sys_color = "green"
elif drift_flag and not settings.get("auto_healing", True):
    sys_status = "ACTION REQUIRED"
    sys_badge = "⚠️ DRIFT DETECTED — RETRAINING RECOMMENDED"
    sys_color = "red"
else:
    sys_status = "HEALTHY"
    sys_badge = "✅ SYSTEM HEALTHY"
    sys_color = "green"

# ---------------------------------------------------
# SECTION A: SYSTEM STATUS HEADER
# ---------------------------------------------------
st.subheader("A. System Status Header")

col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)

col_a1.metric("Overall Status", sys_status)
col_a2.metric("Production Model", latest_version)

last_mon_time = report.get("timestamp", "N/A") if report else "No runs yet"
col_a3.metric("Last Monitoring", last_mon_time.split(" ")[-1] if " " in last_mon_time else last_mon_time)

batches = sorted(os.listdir(batch_folder)) if os.path.exists(batch_folder) else []
latest_batch_name = batches[-1] if batches else "None"
col_a4.metric("Active Batch", latest_batch_name)

current_pipeline_state = latest_job.get("status") if active_job else ("DRIFT DETECTED" if drift_flag else "MONITORING IDLE")
col_a5.metric("Pipeline State", current_pipeline_state)

# Operations Bar
st.write("---")
btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])

with btn_col1:
    if st.button("▶ Run Monitoring Pipeline", use_container_width=True):
        with st.spinner("Executing monitoring pipeline..."):
            mp = MonitoringPipeline()
            mp.run_monitoring()
        st.success("Monitoring complete!")
        st.rerun()

with btn_col2:
    if st.button("⚡ Trigger Retraining Job", type="primary", disabled=active_job, use_container_width=True):
        if not JobManager.has_active_job():
            job = JobManager.create_job(trigger_reason="manual_dashboard_trigger")
            if job:
                t = threading.Thread(
                    target=run_training_pipeline,
                    kwargs={"job_id": job["job_id"], "demo_mode": settings.get("demo_mode", True)},
                    daemon=True
                )
                t.start()
                st.success(f"Retraining job #{job['job_id']} started!")
                st.rerun()

with btn_col3:
    if active_job:
        st.warning(f"🔄 Training Job #{latest_job['job_id']} is running ({latest_job.get('current_stage')}). Auto-refreshing...")

st.divider()

# ---------------------------------------------------
# SECTION B: DRIFT SUMMARY
# ---------------------------------------------------
st.subheader("B. Data Drift Summary Metrics")

if not report:
    st.warning("No drift report found. Please run the monitoring pipeline first.")
else:
    total_feat = report.get("total_features", 0)
    drifted_count = report.get("drifted_count", len(report.get("drifted_features", [])))
    drift_pct = report.get("drift_percentage", 0.0)
    retrain_req = report.get("retraining_required", False)
    threshold_str = f"{report.get('drift_threshold', 0.20)} PSI / 0.05 KS"

    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    b_col1.metric("Total Monitored Features", total_feat)
    b_col2.metric("Drifted Features Count", drifted_count)
    b_col3.metric("Drift Percentage", f"{drift_pct}%")
    b_col4.metric("Overall Drift Status", "DETECTED" if drift_flag else "NONE")
    b_col5.metric("Retraining Required", "YES" if retrain_req else "NO")

    # ---------------------------------------------------
    # SECTION C: FEATURE-LEVEL DRIFT TABLE
    # ---------------------------------------------------
    st.subheader("C. Feature-Level Drift Diagnostics")

    if feat_results:
        table_rows = []
        for feat, val in feat_results.items():
            table_rows.append({
                "Feature": feat,
                "Test / Metric": val.get("metric", "KS / PSI"),
                "KS p-value": val.get("ks_p_value", "N/A"),
                "PSI Score": val.get("psi_score", 0.0),
                "Threshold": val.get("threshold", 0.20),
                "Status": val.get("status", "Drifted" if val.get("drift_detected") else "Stable")
            })

        df_feat = pd.DataFrame(table_rows)

        def style_status(val):
            if val == "Drifted":
                return "background-color: #ffcccc; color: #990000; font-weight: bold;"
            elif val == "Warning":
                return "background-color: #fff3cd; color: #856404; font-weight: bold;"
            else:
                return "background-color: #d4edda; color: #155724; font-weight: bold;"

        st.dataframe(
            df_feat.style.applymap(style_status, subset=["Status"]),
            use_container_width=True
        )
    else:
        st.info("No detailed feature drift statistics recorded.")

st.divider()

# ---------------------------------------------------
# SECTION D: FEATURE DRIFT EXPLORER
# ---------------------------------------------------
st.subheader("D. Feature Drift Explorer & Distribution Comparison")

if os.path.exists(reference_path) and os.path.exists(batch_folder) and batches:
    ref_df = pd.read_csv(reference_path)
    latest_batch_df = pd.read_csv(os.path.join(batch_folder, latest_batch_name))

    # All available features
    all_features = [col for col in ref_df.columns if col in latest_batch_df.columns and col != "price"]

    # Prioritize drifted features at top of select list
    drifted_list = report.get("drifted_features", []) if report else []
    ordered_features = [f for f in drifted_list if f in all_features] + [f for f in all_features if f not in drifted_list]

    selected_feature = st.selectbox(
        "Select Monitored Feature for Distribution Comparison (Drifted Features Prioritized):",
        options=ordered_features
    )

    if selected_feature:
        # Display Feature Metadata Card
        feat_meta = feat_results.get(selected_feature, {}) if feat_results else {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Feature Name", selected_feature)
        m2.metric("PSI Score", feat_meta.get("psi_score", "N/A"))
        m3.metric("KS p-value", feat_meta.get("ks_p_value", "N/A"))
        m4.metric("Status", feat_meta.get("status", "N/A"))

        # Plot Distributions
        if pd.api.types.is_numeric_dtype(ref_df[selected_feature]):
            chart_df = pd.DataFrame({
                "Reference Distribution": ref_df[selected_feature],
                f"Current Production Batch ({latest_batch_name})": latest_batch_df[selected_feature]
            }).melt(var_name="Dataset", value_name="Value")

            fig = px.histogram(
                chart_df,
                x="Value",
                color="Dataset",
                opacity=0.6,
                barmode="overlay",
                histnorm="probability density",
                title=f"Numerical Distribution Comparison for '{selected_feature}'"
            )
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            ref_counts = ref_df[selected_feature].value_counts(normalize=True).reset_index()
            ref_counts.columns = ["Category", "Proportion"]
            ref_counts["Dataset"] = "Reference Distribution"

            batch_counts = latest_batch_df[selected_feature].value_counts(normalize=True).reset_index()
            batch_counts.columns = ["Category", "Proportion"]
            batch_counts["Dataset"] = f"Current Production Batch ({latest_batch_name})"

            cat_df = pd.concat([ref_counts, batch_counts])

            fig = px.bar(
                cat_df,
                x="Category",
                y="Proportion",
                color="Dataset",
                barmode="group",
                title=f"Categorical Proportion Comparison for '{selected_feature}'"
            )
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **MLOps Note**: Retraining updates the candidate model to adapt to shifted data distributions. Retraining does NOT alter the input data distribution itself.")
else:
    st.warning("Reference dataset or production batch files missing.")

st.divider()

# ---------------------------------------------------
# SECTION E: SELF-HEALING PIPELINE STATUS & JOB UI
# ---------------------------------------------------
st.subheader("E. Self-Healing Pipeline Execution & Retraining Job Status")

if not latest_job:
    st.info("No training jobs recorded yet. Click 'Trigger Retraining Job' above to start a background training run.")
else:
    job_id = latest_job.get("job_id")
    job_status = latest_job.get("status")
    curr_stage = latest_job.get("current_stage")
    stages = latest_job.get("stages", {})

    j_col1, j_col2, j_col3, j_col4 = st.columns(4)
    j_col1.metric("Training Job ID", job_id)
    j_col2.metric("Job Status", job_status)
    j_col3.metric("Current Pipeline Stage", curr_stage)
    j_col4.metric("Trigger Reason", latest_job.get("trigger_reason", "manual"))

    st.write("#### ⏳ Stage Progress Stepper")
    stage_cols = st.columns(len(stages))
    for idx, (s_name, s_status) in enumerate(stages.items()):
        with stage_cols[idx]:
            if s_status == "COMPLETED":
                st.success(f"✓ {s_name}")
            elif s_status == "RUNNING":
                st.warning(f"⏳ {s_name}")
            elif s_status == "FAILED":
                st.error(f"❌ {s_name}")
            else:
                st.info(f"○ {s_name}")

    if latest_job.get("error_message"):
        st.error(f"Job Error Traceback: {latest_job['error_message']}")

st.divider()

# ---------------------------------------------------
# SECTION F: CHAMPION VS CHALLENGER COMPARISON
# ---------------------------------------------------
st.subheader("F. Champion vs Challenger Quality Gate")

if latest_job and latest_job.get("status") == "COMPLETED" and latest_job.get("challenger_metrics"):
    champ = latest_job.get("champion_metrics") or {}
    chall = latest_job.get("challenger_metrics") or {}
    decision = latest_job.get("promotion_decision", "UNKNOWN")
    reason = latest_job.get("promotion_reason", "No details available.")

    st.write(f"### Outcome: **:{'green' if decision=='PROMOTED' else 'red'}[{decision}]**")
    st.info(f"📋 **Quality Gate Decision Explanation**: {reason}")

    comp_rows = [
        {
            "Metric": "Model Version / Name",
            "Champion (Production)": champ.get("model_version", "vOld"),
            "Challenger (Candidate)": chall.get("model_name", "DecisionTree"),
            "Improvement (Delta)": "-"
        },
        {
            "Metric": "R² Score (Primary)",
            "Champion (Production)": f"{champ.get('r2', 0.0):.4f}" if champ.get("r2") is not None else "N/A",
            "Challenger (Candidate)": f"{chall.get('r2', 0.0):.4f}",
            "Improvement (Delta)": f"{chall.get('r2', 0.0) - champ.get('r2', 0.0):+.4f}" if champ.get("r2") is not None else "N/A"
        },
        {
            "Metric": "Mean Absolute Error (MAE)",
            "Champion (Production)": f"${champ.get('mae', 0.0):,.2f}" if champ.get("mae") is not None else "N/A",
            "Challenger (Candidate)": f"${chall.get('mae', 0.0):,.2f}",
            "Improvement (Delta)": f"${chall.get('mae', 0.0) - champ.get('mae', 0.0):+,.2f}" if champ.get("mae") is not None else "N/A"
        },
        {
            "Metric": "Root Mean Squared Error (RMSE)",
            "Champion (Production)": f"${champ.get('rmse', 0.0):,.2f}" if champ.get("rmse") is not None else "N/A",
            "Challenger (Candidate)": f"${chall.get('rmse', 0.0):,.2f}",
            "Improvement (Delta)": f"${chall.get('rmse', 0.0) - champ.get('rmse', 0.0):+,.2f}" if champ.get("rmse") is not None else "N/A"
        }
    ]

    st.table(pd.DataFrame(comp_rows))
else:
    st.info("No recent completed retraining job evaluation data available.")

st.divider()

# ---------------------------------------------------
# SECTION G: MODEL PERFORMANCE RECOVERY
# ---------------------------------------------------
st.subheader("G. Model Performance Recovery Trend")

if os.path.exists(monitor_log_path):
    try:
        with open(monitor_log_path, "r") as f:
            logs = json.load(f)

        if logs:
            df_perf = pd.DataFrame(logs)
            fig_perf = px.line(
                df_perf,
                x="batch",
                y="r2_score",
                markers=True,
                color="drift_detected",
                title="Production Model R² Performance Across Batches (Highlighting Drift & Retraining Recovery)"
            )
            fig_perf.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_perf, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering performance recovery chart: {e}")
else:
    st.info("No monitoring history logs found.")

st.divider()

# ---------------------------------------------------
# SECTION H: SELF-HEALING EVENT TIMELINE
# ---------------------------------------------------
st.subheader("H. Self-Healing Event Timeline")

events = JobManager.get_timeline_events()
if events:
    df_events = pd.DataFrame(events[::-1])  # Reverse chronological order
    st.dataframe(df_events[["timestamp", "event_type", "description"]], use_container_width=True)
else:
    st.info("No self-healing timeline events recorded yet.")
