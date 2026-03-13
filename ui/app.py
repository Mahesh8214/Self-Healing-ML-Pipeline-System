import streamlit as st

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Self-Healing ML Pipeline",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("Self-Healing ML Pipeline System")
st.subheader("Data Drift Detection & Automatic Model Retraining")

# ---------------------------------------------------
# Introduction
# ---------------------------------------------------

st.markdown("""
### Introduction

This application demonstrates a **Self-Healing Machine Learning System** that can automatically detect **data drift** and retrain the model when performance drops.

In real-world ML systems, data changes over time.  
When the data distribution changes, the model trained earlier may no longer perform well.  
This problem is known as **Data Drift**.

This project shows how a system can:

• Monitor incoming production data  
• Detect distribution changes  
• Check model performance  
• Automatically retrain the model when needed  

The goal is to keep the machine learning model **reliable and accurate over time**.
""")

# ---------------------------------------------------
# Problem Statement
# ---------------------------------------------------

st.markdown("""
### Problem This Project Solves

Traditional machine learning workflows usually follow this pattern:

**Train → Deploy → Ignore**

Once deployed, the model is rarely monitored.  
But in real applications:

• Data keeps changing  
• Model accuracy decreases  
• Systems fail silently  

This project builds a **monitoring pipeline** that detects these issues early and fixes them automatically.
""")

# ---------------------------------------------------
# How To Use The Application
# ---------------------------------------------------

st.markdown("""
### How To Use This Application

Follow these steps to explore the system.

#### Step 1 — Generate Production Data

Before running the application, generate production batches using:

This script splits the dataset and creates **simulated production batches**.

---

#### Step 2 — Train the Initial Model

Go to the **Training Page** in the sidebar and click:

**Run Training Pipeline**

The system trains the baseline machine learning model.

⏱ Expected time: **2–3 minute**

---

#### Step 3 — Run Monitoring

Next, go to the **Monitoring Page** and click:

**Run Monitoring Pipeline**

The system will process the production batches and:

• Detect data drift  
• Evaluate model performance  
• Retrain the model automatically if performance drops  

⏱ Expected time: **2–3 seconds**

---

#### Step 4 — View Results

Finally, open the **Drift Dashboard**.

You will see:

• Drift detection results  
• Feature drift details  
• Model performance graphs  
• Retraining events
""")

# ---------------------------------------------------
# Drift Metrics Explanation
# ---------------------------------------------------

st.markdown("""
### Understanding Drift Metrics

#### KS Test (Kolmogorov–Smirnov Test)

This statistical test compares two data distributions.

It checks whether **production data looks different from training data**.

A small p-value means the distributions are different → **possible drift**.

---

#### PSI (Population Stability Index)

PSI measures **how much the distribution has changed**.

| PSI Value | Meaning |
|-----------|---------|
| < 0.1 | No drift |
| 0.1 – 0.2 | Moderate drift |
| > 0.2 | Significant drift |

---

### What Happens in This Demo

This project includes **controlled drift simulation**, so one of the production batches will contain drift.

When drift appears:

1. The system detects it  
2. Model performance decreases  
3. Retraining is triggered automatically  
4. Model performance improves again

This demonstrates a **Self-Healing Machine Learning System**.
""")

# ---------------------------------------------------
# Technologies Used
# ---------------------------------------------------

st.markdown("""
### Technologies Used

• Python  
• Scikit-Learn  
• Streamlit  
• Pandas  
• Plotly  

This project demonstrates **ML Engineering + Monitoring + Retraining pipelines**.
""")

# ---------------------------------------------------
# Author
# ---------------------------------------------------

st.markdown("""
---

### Author

**Mahesh Singh**  
Artificial Intelligence & Data Science
""")
