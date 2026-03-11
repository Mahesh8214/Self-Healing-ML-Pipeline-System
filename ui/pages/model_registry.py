import streamlit as st
import json
import pandas as pd
import os

st.title("Model Registry")
path = "artifacts/metadata/model_registry.json"

if not os.path.exists(path):
    st.warning("Model registry not found")

else:

    with open(path,"r") as f:
        registry = json.load(f)
    df = pd.DataFrame(registry["versions"])
    st.dataframe(df)
    st.success(f"Active Model: {registry['latest_model']}")