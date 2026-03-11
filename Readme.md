# Self-Healing ML Pipeline System
### Data Drift Detection & Automatic Model Retraining

<p align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/MachineLearning-ScikitLearn-orange)
![Monitoring](https://img.shields.io/badge/DataDrift-Detection-green)
![Dashboard](https://img.shields.io/badge/UI-Streamlit-red)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-purple)

</p>

---

# Overview

Machine Learning models deployed in production often **degrade silently** as real-world data evolves over time.

This phenomenon is known as **Data Drift**.

Traditional ML pipelines follow this lifecycle:

Train → Deploy → Ignore


Once deployed, models are rarely monitored systematically, which leads to:

• Performance degradation  
• Incorrect predictions  
• Business decision failures  

This project implements a **Self-Healing Machine Learning Pipeline** capable of:

• Detecting **data drift automatically**  
• Monitoring **model performance continuously**  
• Triggering **automatic model retraining**  
• Maintaining **model version history**  
• Providing a **real-time monitoring dashboard**

The system demonstrates **production-oriented ML engineering practices** including monitoring, retraining automation, and modular pipelines.

---

# Problem Statement

In real-world ML systems:

• Data distributions change over time  
• Models trained on historical data become outdated  
• Performance deteriorates silently in production  

The key challenge is:

> How can we automatically detect data drift in deployed ML models and trigger corrective actions without human intervention?

This project provides a **self-healing solution** that detects drift, evaluates performance, and retrains models when necessary.

---

# Key Features

• Data Drift Detection using **Kolmogorov-Smirnov Test**  
• Drift magnitude measurement using **Population Stability Index (PSI)**  
• Feature-wise drift analysis  
• Model performance monitoring using **R² score**  
• Automatic retraining when drift + performance degradation occurs  
• Model version tracking with **Model Registry**  
• Batch simulation of production data  
• Visualization dashboard using **Streamlit**  
• Feature distribution comparison (Reference vs Production)  
• Modular ML pipeline architecture

---

# System Architecture

                Reference Dataset
                        │
                        ▼
                Training Pipeline
                        │
                        ▼
                  Model Registry
                        │
                        ▼
                    Prediction API
                        │
                        ▼
            Production Data Batches
                        │
                        ▼
               Monitoring Pipeline
                        │
    ┌───────────────────┴───────────────────┐
    │                                       │
    ▼                                       ▼

Data Drift Detection Performance Monitoring
│ │
└───────────────┬───────────────────────┘
▼
Retraining Trigger
│
▼
Updated Model Version


The system continuously monitors production data and maintains **model reliability automatically**.

---

# Project Folder Structure

```bash
Self-Healing-ML-Pipeline-System
│
├── src
│ ├── components
│ │ ├── data_validation.py
│ │ ├── drift_detector.py
│ │ ├── performance_monitor.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ │
│ ├── pipelines
│ │ ├── training_pipeline.py
│ │ ├── monitoring_pipeline.py
│ │ └── prediction_pipeline.py
│ │
│ ├── registry
│ │ └── model_registry.py
│ │
│ └── utils.py
│
├── ui
│ ├── app.py
│ └── pages
│ ├── prediction.py
│ ├── monitoring.py
│ ├── drift_dashboard.py
│ └── model_registry.py
│
├── artifacts
│ ├── models
│ ├── reports
│ └── monitoring
│
├── data
│ ├── reference_data.csv
│ └── production_batches
│
├── test_drift_data_maker.py
├── requirements.txt
└── README.md
'''

The project follows a **modular ML engineering architecture** separating pipelines, components, monitoring, and UI.

---

# Data Drift Detection

## Kolmogorov-Smirnov Test

The KS test compares the probability distributions of two datasets.


D = sup |F1(x) − F2(x)|


Where

• F1(x) = Reference distribution  
• F2(x) = Production distribution  

A small p-value indicates **statistically significant drift**.

---

## Population Stability Index (PSI)

PSI measures how much the distribution has shifted.


PSI = Σ (Actual − Expected) × ln(Actual / Expected)


### PSI Interpretation

| PSI Value | Meaning |
|-----------|---------|
| < 0.1 | Stable |
| 0.1 – 0.2 | Moderate Drift |
| > 0.2 | Significant Drift |

---

# Monitoring Strategy

The monitoring pipeline performs the following steps:

1️⃣ Load latest production batch  
2️⃣ Validate incoming data  
3️⃣ Detect feature-wise drift  
4️⃣ Evaluate model performance  
5️⃣ Trigger retraining if necessary  

Retraining condition:


Drift detected AND Performance degraded


This ensures retraining occurs **only when required**, avoiding unnecessary computation.

---

# Dashboard Overview

The project includes a **Streamlit monitoring dashboard** for real-time interaction.

### Prediction Interface
Predict diamond prices using input features.

### Monitoring Control
Trigger training or monitoring pipelines directly from the UI.

### Drift Dashboard
Visualize distribution changes between reference and production data.

### Model Registry
Track model versions, timestamps, and retraining history.

---

# Running the Project

## 1 Install Dependencies

```bash
pip install -r requirements.txt
---


## 2 Generate Production Data Batches

Run the batch generator script:
```bash
python test_drift_data_maker.py
```

This splits the dataset into simulated production batches inside:


data/production_batches


---

## 3 Train Initial Model

```bash
python -m src.pipelines.training_pipeline
---

## 4 Launch Monitoring Dashboard

```bash
streamlit run ui/app.py
---

## 5 Trigger Monitoring

Inside the dashboard:
```bash
Monitoring Page → Run Monitoring Pipeline
---

# Challenges Faced

## Histogram Bin Errors in PSI

Error encountered during development:


ValueError: bins must increase monotonically


Cause:

• constant feature values  
• NaN values  
• improper histogram bin generation  

Solution:

• cleaned dataset before PSI calculation  
• generated safe bins using `numpy.linspace`

---

## Model Registry Synchronization

Occasionally the registry referenced model files that were removed.

Solution:

• validation checks before loading models  
• fallback behavior added to monitoring pipeline

---

# Technologies Used
```bash
| Category | Technology |
|--------|-------------|
| Language | Python |
| Machine Learning | Scikit-Learn |
| Monitoring | Custom Drift Detection |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Data Processing | Pandas |
| Model Versioning | Custom Model Registry |
---

# Future Improvements

Potential extensions include:

• MLflow experiment tracking  
• real-time drift monitoring with streaming data  
• CI/CD pipeline for automated retraining  
• feature importance drift analysis  
• cloud deployment (AWS / GCP)

---

# Conclusion

This project demonstrates a **production-oriented machine learning monitoring system** capable of:

• detecting data drift  
• evaluating model performance  
• automatically retraining models  

The architecture emphasizes:

• modular pipeline design  
• automated monitoring  
• robust drift detection  
• reproducible ML workflows

---

# Author

Mahesh Singh  
Artificial Intelligence & Data Science

---

# License

This project is intended for educational and research purposes.
