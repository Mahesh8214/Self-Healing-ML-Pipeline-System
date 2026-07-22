import streamlit as st
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.job_manager import JobManager
from src.registry.model_registry import ModelRegistry

st.set_page_config(
    page_title="Self-Healing ML Pipeline",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# Sidebar Controls & System Settings
# ---------------------------------------------------
st.sidebar.title("⚙️ System Control Center")

current_settings = JobManager.get_settings()

auto_healing_val = st.sidebar.toggle(
    "🤖 Auto-Healing System Mode",
    value=current_settings.get("auto_healing", True),
    help="When ON, detected data drift automatically launches background retraining."
)

demo_mode_val = st.sidebar.toggle(
    "⚡ Demo Mode (Fast Subsampling)",
    value=current_settings.get("demo_mode", True),
    help="When ON, training uses sample subsets for fast ~20s evaluation cycles."
)

if (auto_healing_val != current_settings.get("auto_healing")) or (demo_mode_val != current_settings.get("demo_mode")):
    JobManager.update_settings(auto_healing=auto_healing_val, demo_mode=demo_mode_val)
    st.sidebar.success("Settings updated successfully!")

st.sidebar.divider()
st.sidebar.subheader("📌 Quick Navigation")
if st.sidebar.button("📊 Launch Drift Dashboard", use_container_width=True):
    st.switch_page("pages/drift_dashboard.py")

if st.sidebar.button("⚙️ Open Monitoring Control", use_container_width=True):
    st.switch_page("pages/monitoring.py")

if st.sidebar.button("💎 Run Diamond Price Predictor", use_container_width=True):
    st.switch_page("pages/prediction.py")

if st.sidebar.button("📜 View Model Registry", use_container_width=True):
    st.switch_page("pages/model_registry.py")

# ---------------------------------------------------
# Header & System Status Determination
# ---------------------------------------------------
st.title("🛡️ Self-Healing ML Pipeline Platform")
st.caption("Continuous Data Drift Monitoring, Non-Blocking Auto-Retraining, Champion/Challenger Quality Gates & Model Lifecycle Governance")

# Determine system status dynamically from reports and job state
report_path = "artifacts/reports/drift_report.json"
drift_detected = False
if os.path.exists(report_path):
    try:
        with open(report_path, "r") as f:
            rep = json.load(f)
            drift_detected = rep.get("drift_detected", False)
    except Exception:
        pass

latest_job = JobManager.get_latest_job()
active_job = JobManager.has_active_job()

if active_job:
    status_label = "RETRAINING"
    status_color = "orange"
    status_msg = "🔄 Background Model Retraining in Progress..."
elif drift_detected and auto_healing_val:
    status_label = "RECOVERED / ACTION IN PROGRESS"
    status_color = "blue"
    status_msg = "⚡ Data Drift Detected — Retraining Triggered & Managed"
elif drift_detected and not auto_healing_val:
    status_label = "ACTION REQUIRED"
    status_color = "red"
    status_msg = "⚠️ Data Drift Detected — Retraining Recommended (Auto-Healing OFF)"
else:
    status_label = "HEALTHY"
    status_color = "green"
    status_msg = "✅ Pipeline Operational — All Monitored Features Stable"

col_status, col_btn = st.columns([3, 1])

with col_status:
    st.subheader(f"System Status: :{status_color}[{status_label}]")
    st.info(status_msg)

with col_btn:
    st.write("###")
    if st.button("▶ Run Monitor Now", type="primary", use_container_width=True):
        st.switch_page("pages/drift_dashboard.py")

st.divider()

# ---------------------------------------------------
# Core System Key Metrics
# ---------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

registry = ModelRegistry()
try:
    latest_model_path = registry.get_latest_model()
    reg_data = registry.load_registry()
    version_count = len(reg_data.get("versions", []))
    latest_version = reg_data.get("versions", [{}])[-1].get("version", f"v{version_count}")
except Exception:
    latest_version = "v1"

m_col1.metric("Active Production Model", latest_version)
m_col2.metric("Auto-Healing Mode", "ENABLED" if auto_healing_val else "DISABLED")
m_col3.metric("Execution Engine", "Demo (~20s)" if demo_mode_val else "Production (Full)")

latest_job_status = latest_job.get("status") if latest_job else "None"
m_col4.metric("Latest Training Job", latest_job_status)

st.divider()

# ---------------------------------------------------
# Interactive Self-Healing Lifecycle Overview
# ---------------------------------------------------
st.subheader("🔄 Automated Self-Healing Architecture")

st.markdown("""
```
 ┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │  Production Data Batch  ├─────►│ Continuous Drift Monitor ├─────►│ Feature-Level Diagnosis  │
 └─────────────────────────┘      └─────────────┬────────────┘      └─────────────┬────────────┘
                                                │                                 │
                                                ▼                                 ▼
 ┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │ Active Model Promoted   │◄─────┤ Champion vs Challenger   │◄─────┤ Async Background Job     │
 │ & Monitoring Continues  │      │ Quality Gate Validation  │      │ Retraining Pipeline      │
 └─────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
```
""")

st.markdown("""
### Key Features of this Platform:
1. **Non-Blocking UI Execution**: Training pipeline runs asynchronously in the background. The Streamlit interface remains fully responsive.
2. **Feature-Level Drift Diagnostics**: Statistical Kolmogorov-Smirnov (KS) & Population Stability Index (PSI) testing for all input features.
3. **Champion vs Challenger Quality Gate**: Evaluates new models ($R^2$, MAE, MSE, RMSE) against active production models before promotion.
4. **Resilient Persistence**: Job metadata, monitoring logs, and event timelines are persisted to survive browser refreshes.
""")
