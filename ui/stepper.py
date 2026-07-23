import streamlit as st
import time
import textwrap
from datetime import datetime

PIPELINE_STAGES_INFO = [
    {
        "name": "Data Validation",
        "icon": "📥",
        "desc": "Validating input schema, dataset integrity, and missing values.",
        "weight": 12
    },
    {
        "name": "Drift Detection",
        "icon": "📊",
        "desc": "Calculating KS-Test & PSI drift metrics across production features.",
        "weight": 25
    },
    {
        "name": "Retraining Trigger",
        "icon": "⚡",
        "desc": "Verifying dual-condition gates (drift AND performance drop).",
        "weight": 37
    },
    {
        "name": "Model Training",
        "icon": "🤖",
        "desc": "Training candidate ensemble models (Linear, Lasso, Ridge, ElasticNet, DecisionTree).",
        "weight": 50
    },
    {
        "name": "Model Evaluation",
        "icon": "📈",
        "desc": "Computing test set regression metrics (R², MAE, MSE, RMSE).",
        "weight": 62
    },
    {
        "name": "Champion vs Challenger",
        "icon": "⚖️",
        "desc": "Benchmarking candidate model metrics against active production champion.",
        "weight": 75
    },
    {
        "name": "Quality Gate",
        "icon": "🛡️",
        "desc": "Evaluating minimum R² improvement threshold requirements.",
        "weight": 87
    },
    {
        "name": "Model Promotion",
        "icon": "🏆",
        "desc": "Updating model registry pointer and deploying new production artifact.",
        "weight": 100
    }
]


def inject_stepper_css():
    css = textwrap.dedent("""
    <style>
    /* Responsive metric value styling to prevent truncation */
    [data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
        word-break: break-word !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        line-height: 1.3 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        white-space: normal !important;
        color: #94a3b8 !important;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px 14px;
    }

    .stepper-card {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.85), rgba(36, 59, 85, 0.85));
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        color: #ffffff;
    }
    
    .stepper-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .stepper-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #00d2ff;
        display: flex;
        align-items: center;
        gap: 10px;
        word-break: break-word;
    }

    .stepper-timer {
        background: rgba(0, 210, 255, 0.15);
        border: 1px solid #00d2ff;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #e0f7fa;
    }

    .progress-bar-bg {
        width: 100%;
        height: 14px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 7px;
        overflow: hidden;
        margin-bottom: 15px;
        position: relative;
    }

    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        border-radius: 7px;
        transition: width 0.4s ease-in-out;
        box-shadow: 0 0 12px #00c6ff;
    }

    .step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-top: 15px;
    }

    .step-item {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        position: relative;
        transition: all 0.3s ease;
    }

    .step-item.completed {
        border-color: rgba(46, 204, 113, 0.6);
        background: rgba(46, 204, 113, 0.08);
    }

    .step-item.running {
        border-color: #00d2ff;
        background: rgba(0, 210, 255, 0.15);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
        animation: pulse-border 1.5s infinite alternate;
    }

    .step-item.waiting {
        opacity: 0.6;
    }

    .step-item.failed {
        border-color: #e74c3c;
        background: rgba(231, 76, 60, 0.15);
    }

    @keyframes pulse-border {
        0% { box-shadow: 0 0 5px rgba(0, 210, 255, 0.3); }
        100% { box-shadow: 0 0 20px rgba(0, 210, 255, 0.8); }
    }

    .step-badge {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        float: right;
    }

    .badge-completed { background: #2ecc71; color: #fff; }
    .badge-running { background: #00d2ff; color: #000; }
    .badge-waiting { background: rgba(255,255,255,0.2); color: #ccc; }
    .badge-failed { background: #e74c3c; color: #fff; }

    .step-name {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
        color: #ffffff;
    }

    .step-desc {
        font-size: 0.78rem;
        color: #b0bec5;
        line-height: 1.25;
    }
    </style>
    """)
    st.markdown(css, unsafe_allow_html=True)


def render_retraining_stepper(job):
    if not job:
        return

    inject_stepper_css()

    status = job.get("status", "QUEUED")
    current_stage = job.get("current_stage", "Data Validation")
    stages = job.get("stages", {})

    # Calculate elapsed time
    started_at_str = job.get("started_at") or job.get("created_at")
    elapsed_sec = 0.0
    if started_at_str:
        try:
            start_dt = datetime.strptime(started_at_str, "%Y-%m-%d %H:%M:%S")
            if status in ["QUEUED", "RUNNING"]:
                elapsed_sec = (datetime.now() - start_dt).total_seconds()
            elif job.get("completed_at"):
                end_dt = datetime.strptime(job["completed_at"], "%Y-%m-%d %H:%M:%S")
                elapsed_sec = (end_dt - start_dt).total_seconds()
        except Exception:
            elapsed_sec = 0.0

    # Determine progress percentage
    pct = 0
    if status == "COMPLETED":
        pct = 100
    elif status == "QUEUED":
        pct = 5
    else:
        for idx, info in enumerate(PIPELINE_STAGES_INFO):
            s_name = info["name"]
            st_val = stages.get(s_name)
            if s_name == current_stage or st_val == "RUNNING":
                pct = info["weight"]
                break
            elif st_val == "COMPLETED":
                pct = info["weight"]

    status_icon = "🔄" if status in ["QUEUED", "RUNNING"] else ("✅" if status == "COMPLETED" else "❌")
    status_label = "AUTONOMOUS RETRAINING IN PROGRESS" if status in ["QUEUED", "RUNNING"] else (
        "RETRAINING COMPLETE" if status == "COMPLETED" else "RETRAINING FAILED"
    )

    # Render Streamlit native progress bar as reliable visual fallback/component
    st.progress(max(0.0, min(1.0, float(pct) / 100.0)))

    # Render individual stage cards
    cards_html = ""
    for info in PIPELINE_STAGES_INFO:
        s_name = info["name"]
        st_val = stages.get(s_name, "WAITING")
        if status == "COMPLETED":
            st_val = "COMPLETED"

        css_class = st_val.lower()
        badge_class = f"badge-{st_val.lower()}"
        badge_text = st_val

        if st_val == "COMPLETED":
            node_icon = "✅"
        elif st_val == "RUNNING":
            node_icon = "🔄"
        elif st_val == "FAILED":
            node_icon = "❌"
        else:
            node_icon = "⏳"

        cards_html += (
            f'<div class="step-item {css_class}">'
            f'<span class="step-badge {badge_class}">{badge_text}</span>'
            f'<div class="step-name">{node_icon} {info["icon"]} {s_name}</div>'
            f'<div class="step-desc">{info["desc"]}</div>'
            f'</div>'
        )

    full_stepper_html = (
        f'<div class="stepper-card">'
        f'<div class="stepper-header">'
        f'<div class="stepper-title">'
        f'<span>{status_icon} {status_label}</span>'
        f'<span style="font-size:0.85rem; color:#b0bec5; font-weight:normal;">(Job #{job["job_id"]})</span>'
        f'</div>'
        f'<div class="stepper-timer">'
        f'⏱️ Elapsed: {elapsed_sec:.1f}s / Target: &lt;30.0s'
        f'</div>'
        f'</div>'
        f'<div class="progress-bar-bg">'
        f'<div class="progress-bar-fill" style="width: {pct}%;"></div>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#80deea; margin-top:-5px; margin-bottom:15px;">'
        f'<span>Current Stage: <strong>{current_stage}</strong></span>'
        f'<span>Overall Progress: <strong>{pct}%</strong></span>'
        f'</div>'
        f'<div class="step-grid">{cards_html}</div>'
        f'</div>'
    )

    st.markdown(full_stepper_html, unsafe_allow_html=True)

    # Auto rerun if job is still active
    if status in ["QUEUED", "RUNNING"]:
        time.sleep(0.5)
        st.rerun()

