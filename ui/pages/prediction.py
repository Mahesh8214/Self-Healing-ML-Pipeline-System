import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.pipelines.prediction_pipeline import PredictPipeline
from src.registry.model_registry import ModelRegistry

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Production Diamond Price Prediction")

registry = ModelRegistry()
try:
    latest_model_path = registry.get_latest_model()
    reg_data = registry.load_registry()
    version_count = len(reg_data.get("versions", []))
    latest_version = reg_data.get("versions", [{}])[-1].get("version", f"v{version_count}") if reg_data.get("versions") else "v1"
    st.info(f"Serving predictions with active production model version **{latest_version}** (`{latest_model_path}`)")
except Exception:
    st.info("Serving predictions with active production model.")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", value=0.7, step=0.01)
    depth = st.number_input("Depth", value=62.0, step=0.1)
    table = st.number_input("Table", value=55.0, step=0.1)
    x = st.number_input("X (Length in mm)", value=5.8, step=0.01)
    y = st.number_input("Y (Width in mm)", value=5.8, step=0.01)
    z = st.number_input("Z (Depth in mm)", value=3.6, step=0.01)

with col2:
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

if st.button("🔮 Predict Diamond Price", type="primary"):
    data = pd.DataFrame({
        "carat": [carat],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity]
    })

    pipeline = PredictPipeline()
    prediction = pipeline.predict(data)
    st.success(f"Estimated Price: **${prediction[0]:,.2f} USD**")