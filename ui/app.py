import streamlit as st

st.set_page_config(
    page_title="Self-Healing ML Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("Self-Healing ML Pipeline Dashboard")

st.markdown("""
Welcome to the **ML Monitoring Dashboard**

Use the sidebar to navigate:

• Prediction  
• Monitoring Control  
• Drift Dashboard  
• Model Registry
""")