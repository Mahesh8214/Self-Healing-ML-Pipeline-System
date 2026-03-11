import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.pipelines.monitoring_pipeline import MonitoringPipeline
from src.pipelines.training_pipeline import run_training_pipeline


st.title("Monitoring Control")
st.subheader("System Operations")

if st.button("Run Monitoring Pipeline"):
    monitor = MonitoringPipeline()
    monitor.run_monitoring()
    st.success("Monitoring pipeline executed")


if st.button("Run Training Pipeline"):
    run_training_pipeline()
    st.success("Training pipeline executed")