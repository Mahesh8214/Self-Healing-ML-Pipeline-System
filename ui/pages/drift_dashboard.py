import streamlit as st
import json
import pandas as pd
import os
import plotly.express as px

st.title("Drift Dashboard")

# ---------------------------------------------------
# Load Drift Report
# ---------------------------------------------------

report_path = "artifacts/reports/drift_report.json"

if not os.path.exists(report_path):

    st.warning("No drift report found. Run monitoring pipeline first.")

else:

    with open(report_path, "r") as f:
        report = json.load(f)

    st.subheader("Drift Summary")

    drift_flag = report.get("drift_report", False)

    col1, col2 = st.columns(2)

    if drift_flag:
        col1.error("⚠ Data Drift Detected")
    else:
        col1.success("✅ No Data Drift Detected")

    feature_results = report.get("feature_results", {})

    if feature_results:
        col2.metric("Total Features Monitored", len(feature_results))

    # ---------------------------------------------------
    # Feature Drift Table
    # ---------------------------------------------------

    st.subheader("Feature Drift Details")

    if feature_results:

        df = pd.DataFrame(feature_results).T

        st.dataframe(df)

        # Drifted feature list
        st.subheader("Drifted Features")

        drifted = df[df["drift_detected"] == True]

        if len(drifted) > 0:

            st.error("Drift detected in the following features:")

            st.write(drifted.index.tolist())

        else:

            st.success("No features drifted")

    else:

        st.warning("No feature drift information available")

# ---------------------------------------------------
# Feature Distribution Visualization
# ---------------------------------------------------

st.subheader("Feature Distribution Comparison")

reference_path = "artifacts/data/reference_data.csv"
batch_folder = "data/production_batches"

if os.path.exists(reference_path) and os.path.exists(batch_folder):

    ref_df = pd.read_csv(reference_path)

    batches = os.listdir(batch_folder)

    if len(batches) > 0:

        latest_batch = sorted(batches)[-1]

        batch_df = pd.read_csv(os.path.join(batch_folder, latest_batch))

        feature = st.selectbox(
            "Select Feature",
            ["carat", "depth", "table", "x", "y", "z"]
        )

        chart_df = pd.DataFrame({
            "Reference": ref_df[feature],
            "Production": batch_df[feature]
        })

        chart_df = chart_df.melt(
            var_name="Dataset",
            value_name="Value"
        )

        fig = px.histogram(
            chart_df,
            x="Value",
            color="Dataset",
            opacity=0.6,
            barmode="overlay",
            title=f"{feature} Distribution Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.warning("No production batches found")

else:

    st.warning("Reference or production data missing")

# ---------------------------------------------------
# PSI Visualization (if PSI exists in drift report)
# ---------------------------------------------------

if "df" in locals() and "psi_score" in df.columns:

    st.subheader("Population Stability Index (PSI)")

    psi_fig = px.bar(
        df,
        x="psi_score",
        y=df.index,
        orientation="h",
        title="PSI Score by Feature"
    )

    st.plotly_chart(psi_fig, use_container_width=True)

# ---------------------------------------------------
# Model Performance Trend
# ---------------------------------------------------

st.subheader("Model Performance Trend")

monitor_log = "artifacts/monitoring/monitoring_log.json"

if os.path.exists(monitor_log):
    with open(monitor_log, "r") as f:
        logs = json.load(f)
    perf_df = pd.DataFrame(logs)

    if "r2_score" in perf_df.columns:
        perf_fig = px.line(
            perf_df,
            x="batch",
            y="r2_score",
            markers=True,
            title="Model Performance Over Batches"
        )
        st.plotly_chart(perf_fig, use_container_width=True)

    else:
        st.warning("Performance metrics not found in monitoring log")

else:
    st.warning("No monitoring logs found")


st.subheader("Monitoring Timeline")
log_path = "artifacts/monitoring/monitoring_log.json"

if os.path.exists(log_path):
    with open(log_path) as f:
        logs = json.load(f)
    df = pd.DataFrame(logs)
    st.dataframe(df)
    st.subheader("Model Performance Over Time")
    import plotly.express as px
    fig = px.line(
        df,
        x="batch",
        y="r2_score",
        markers=True,
        title="Model Performance Across Batches"
    )
    st.plotly_chart(fig)
    st.subheader("Retraining Events")
    retrained = df[df["retraining_triggered"] == True]
    if len(retrained) > 0:
        st.error("Automatic Retraining Triggered")
        st.write(retrained)
else:
    st.warning("No monitoring logs found. Run monitoring pipeline first.")