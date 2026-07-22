import streamlit as st
import json
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.registry.model_registry import ModelRegistry

st.set_page_config(
    page_title="Model Registry & Governance",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Production Model Registry & Version History")
st.caption("Model Lineage, Registration Reasons, Deployment Timestamps & Production Artifact Tracking")

registry = ModelRegistry()
path = "artifacts/metadata/model_registry.json"

if not os.path.exists(path):
    st.warning("Model registry metadata file not found at artifacts/metadata/model_registry.json")
else:
    try:
        reg_data = registry.load_registry()
        latest_model = reg_data.get("latest_model", "None")
        versions = reg_data.get("versions", [])

        st.success(f"🏆 Active Production Model Path: `{latest_model}`")

        if versions:
            df = pd.DataFrame(versions[::-1])  # Latest versions at top
            st.subheader("Registered Model Versions")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No registered model versions recorded.")
    except Exception as e:
        st.error(f"Error loading model registry: {e}")