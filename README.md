🚨 FWA Monitoring System – AI/ML Fraud Detection Engine

📌 Overview



This repository implements a multi-layered Fraud, Waste and Abuse (FWA) Monitoring System for health insurance claims.



The system is designed to simulate and evaluate fraud detection logic using:



Rule-Based Detection



Supervised Machine Learning (XGBoost)



Unsupervised Anomaly Detection (Isolation Forest)



Unified Weighted Risk Scoring



Threshold Optimization (Business-Cost Based)



SHAP Explainability



The repository follows a mono-repo structure and is modular for future expansion (e.g., Graph Analytics, APIs, Dashboard integration).



🏗 Architecture

Diagram:

Claim Input

&nbsp;  ↓

Rule Engine

&nbsp;  ↓

ML Model (XGBoost)

&nbsp;  ↓

Anomaly Detection (Isolation Forest)

&nbsp;  ↓

Weighted Risk Aggregation

&nbsp;  ↓

Threshold Optimization

&nbsp;  ↓

Risk Classification (LOW/MEDIUM/HIGH)





The FWA module (fwa/) is structured as:



fwa/

│

├── data/         → Data simulation \& SQLite storage

├── models/       → ML and anomaly model training

├── rules/        → Configurable rule engine

├── services/     → Evaluation, scoring, explainability

├── main.py       → Unified scoring pipeline

└── config.py





The existing implementation is preserved under:



claim\_automation/



🔍 Fraud Detection Layers

1️⃣ Rule Engine



Weighted deterministic fraud triggers



Configurable thresholds



Explainable outputs



2️⃣ Supervised ML Model



XGBoost classifier



Class imbalance handling (scale\_pos\_weight)



ROC-AUC evaluation



3️⃣ Anomaly Detection



Isolation Forest



Outlier detection independent of fraud labels



4️⃣ Unified Risk Scoring



Final Score =

(Rule Weight × Rule Score) + (ML Weight × ML Score) + (Anomaly Weight × Anomaly Score)



Dynamic threshold tuning optimizes fraud recall while balancing operational cost.



5️⃣ Explainability



SHAP feature attribution



Top feature drivers for fraud decision



📊 Evaluation



The system supports:



Confusion Matrix



Precision / Recall / F1 Score



ROC-AUC



Threshold Optimization (Business-weighted scoring)



🚀 How to Run

1️⃣ Install Dependencies

pip install -r requirements.txt



2️⃣ Generate Data

python -m fwa.data.generate\_data



3️⃣ Load to SQLite

python -m fwa.data.load\_to\_db



4️⃣ Train Models

python -m fwa.models.train\_model

python -m fwa.models.train\_anomaly



5️⃣ Evaluate Full Pipeline

python -m fwa.services.evaluate\_pipeline



6️⃣ Score a New Claim

python -m fwa.services.score\_new\_claim



🎯 Key Highlights



Multi-layer fraud detection architecture



Business-aware threshold tuning



Modular mono-repo design



SQLite-based simulation



Explainable AI integration



Ready for graph-based extension



📌 Future Enhancements



Graph Neural Network integration



Real-time API layer (FastAPI)



Dashboard analytics



Model drift monitoring



Deployment containerization



👨‍💻 Author



Developed as part of FWA Monitoring System implementation with deep algorithmic focus.



🎯 Important



This is a simulation-based prototype for architectural and algorithmic exploration.

