🚨 Enterprise FWA Intelligence Platform
AI/ML Fraud, Waste & Abuse Detection System for Health Insurance
📌 Overview

This repository implements a multi-layered Enterprise Fraud Monitoring System for health insurance claims.

The system combines:

✅ Rule-Based Fraud Detection

✅ Supervised Machine Learning (XGBoost)

✅ Unsupervised Anomaly Detection (Isolation Forest)

✅ Network Intelligence & Graph Analytics

✅ Fraud Contagion Simulation

✅ Case Management Console

✅ Model Drift Monitoring

✅ SHAP Explainability

✅ Business-Weighted Risk Aggregation

The architecture is designed to simulate real-world insurance FWA control systems with enterprise-ready modularity.

🏗 Enterprise Architecture
Multi-Layer Detection Pipeline

Claim Input
↓
Rule Engine
↓
ML Model (XGBoost)
↓
Anomaly Detection (Isolation Forest)
↓
Weighted Risk Aggregation
↓
Risk Tier Classification
↓
Case Management & Network Intelligence

📂 Project Structure
fwa/

├── app.py                         # Streamlit Enterprise Console
├── data/                          # Data generation & SQLite DB
├── models/                        # ML & anomaly training
├── rules/                         # Configurable rule engine
├── services/                      # Aggregation, velocity, network, explainability
├── config/                        # Rule configuration
├── create_case_table.py           # Case management schema
├── Dockerfile                     # Deployment container
├── render.yaml                    # Server deployment config
└── requirements.txt

🔍 Fraud Detection Layers
1️⃣ Rule Engine

Configurable fraud triggers

Early-policy abuse detection

Disease cost deviation logic

Aggregation-based exposure detection

Fully explainable rule hits

2️⃣ Supervised ML (XGBoost)

Class imbalance handling

Feature expansion (ratios, velocity, exposure metrics)

Calibrated probability output

SHAP explainability

3️⃣ Anomaly Detection (Isolation Forest)

Detects structural outliers

Independent of fraud labels

Contributes to final composite score

4️⃣ Unified Risk Scoring

Final Risk Score:

(Rule Weight × Rule Score)
+ (ML Weight × ML Score)
+ (Anomaly Weight × Anomaly Score)


Includes:

Risk tier classification (LOW / REVIEW / INVESTIGATE / CRITICAL)

Component-level contribution visibility

📊 Enterprise Intelligence Modules
🔥 Risk Heatmap Dashboard

Hospital vs Doctor fraud concentration

Financial exposure heatmap

🧠 Model Drift Monitoring

30-day fraud rate comparison

Automated drift alerting

Operational stability signal

🕸 Fraud Network Graph

Doctor-Hospital relationship mapping

Node size = exposure

Node color = fraud rate

Centrality analysis

Bridge detection

Network Risk Score

🧨 Fraud Contagion Simulation

1-hop risk propagation

Systemic exposure estimation

Network-level fraud escalation detection

🗂 Case Management Console

Auto-save scored claims

Priority scoring logic

Analyst assignment

Case status workflow

Investigation notes tracking

Enterprise-style queue management

📊 Fraud Score Decomposition Panel

Displays:

Rule Engine contribution

ML Model contribution

Anomaly Model contribution

Provides transparent model governance.

🧠 Explainable AI

SHAP feature attribution

Top drivers per claim

Component-level visibility

Supports audit & compliance requirements

🚀 How to Run (Local)
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Generate Data
python -m fwa.data.generate_data

3️⃣ Load to SQLite
python -m fwa.data.load_to_db

4️⃣ Train Models
python -m fwa.models.train_model
python -m fwa.models.train_anomaly

5️⃣ Launch Enterprise Console
streamlit run fwa/app.py

🌍 Deployment

Supports:

Docker containerization

Render server deployment

Persistent SQLite disk storage

Production-ready configuration

🎯 Key Capabilities

Multi-layer fraud detection

Network-based fraud intelligence

Behavioral & aggregation analytics

Enterprise case workflow

Explainable AI

Drift monitoring

Deployment-ready architecture

🔮 Future Enhancements

Graph Neural Networks

Real-time FastAPI integration

Automated retraining pipelines

Production monitoring stack

PostgreSQL upgrade

Role-based access control

👨‍💻 Author

Developed as an Enterprise FWA Intelligence System
Focused on algorithmic rigor, modular architecture, and production readiness.

⚠ Disclaimer

This is a simulation-based enterprise prototype designed for architectural and fraud detection research purposes.
