import streamlit as st
import sys
import os
import threading
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.job_manager import JobManager
from src.pipelines.monitoring_pipeline import MonitoringPipeline
from src.pipelines.training_pipeline import run_training_pipeline

st.set_page_config(
    page_title="Monitoring Control & System Operations",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Monitoring Control & Operations")
st.caption("Manual Pipeline Execution, Batch Management & Job Operations Control")

# ---------------------------------------------------
# System Control Panel
# ---------------------------------------------------
col1, col2 = st.columns(2)

active_job = JobManager.has_active_job()
settings = JobManager.get_settings()

with col1:
    st.subheader("📊 Monitoring Operations")
    st.markdown("Run data validation, feature drift analysis, and batch model evaluation.")
    if st.button("▶ Run Monitoring Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running monitoring pipeline across production batches..."):
            mp = MonitoringPipeline()
            mp.run_monitoring()
        st.success("Monitoring pipeline executed successfully!")
        st.rerun()

with col2:
    st.subheader("⚡ Retraining Operations")
    st.markdown("Trigger asynchronous background retraining, model evaluation, and promotion.")
    if st.button("⚡ Trigger Retraining Pipeline", disabled=active_job, use_container_width=True):
        job = JobManager.create_job(trigger_reason="manual_monitoring_control")
        if job:
            t = threading.Thread(
                target=run_training_pipeline,
                kwargs={"job_id": job["job_id"], "demo_mode": settings.get("demo_mode", True)},
                daemon=True
            )
            t.start()
            st.success(f"Retraining job #{job['job_id']} queued!")
            st.rerun()

st.divider()

# ---------------------------------------------------
# Navigation Shortcut
# ---------------------------------------------------
nav_col1, nav_col2 = st.columns([3, 1])
with nav_col1:
    st.info("💡 Once monitoring completes, navigate to the Drift Monitoring & Self-Healing Center for visual feature diagnostics.")
with nav_col2:
    if st.button("📊 Open Drift Dashboard", use_container_width=True):
        st.switch_page("pages/drift_dashboard.py")

st.divider()

# ---------------------------------------------------
# Production Batches List
# ---------------------------------------------------
st.subheader("📦 Available Production Batches")
batch_folder = "data/production_batches"
if os.path.exists(batch_folder):
    batches = sorted(os.listdir(batch_folder))
    from src.logger import load_batch_log
    batch_log = load_batch_log()
    processed = batch_log.get("processed_batches", [])

    b_data = []
    for b in batches:
        b_path = os.path.join(batch_folder, b)
        b_size = os.path.getsize(b_path)
        b_data.append({
            "Batch File": b,
            "Size (bytes)": b_size,
            "Status": "Processed ✅" if b in processed else "Pending ⏳"
        })

    st.table(pd.DataFrame(b_data))
else:
    st.warning("No production batches folder found.")

st.divider()

# ---------------------------------------------------
# Training Job History
# ---------------------------------------------------
st.subheader("📋 Background Job History")
jobs = JobManager.load_jobs()
if jobs:
    df_jobs = pd.DataFrame(jobs[::-1])
    cols_to_show = [c for c in ["job_id", "status", "created_at", "completed_at", "trigger_reason", "promotion_decision"] if c in df_jobs.columns]
    st.dataframe(df_jobs[cols_to_show], use_container_width=True)
else:
    st.info("No background training jobs recorded yet.")