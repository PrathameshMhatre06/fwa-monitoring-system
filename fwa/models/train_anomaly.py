import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from datetime import timedelta
import joblib

print("=" * 60)
print("  PRAHARI - Isolation Forest Anomaly Detector Trainer")
print("=" * 60)

# =====================================================
# CONNECT TO DATABASE
# =====================================================
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")
data = pd.read_sql("SELECT * FROM claims", engine)
data["claim_date"] = pd.to_datetime(data["claim_date"], format="mixed")
data = data.sort_values("claim_date").reset_index(drop=True)

print(f"\nLoaded {len(data):,} claims from database")

# =====================================================
# TIME WINDOW FEATURES (must match train_model.py)
# =====================================================
print("\nEngineering time-window features...")

hospital_30, patient_60, velocity_7, doctor_30 = [], [], [], []

for i, row in data.iterrows():
    current_date = row["claim_date"]
    past_data = data[data["claim_date"] < current_date]

    hospital_window = past_data[
        (past_data["hospital_id"] == row["hospital_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=30))
    ]
    hospital_30.append(hospital_window["claim_amount"].sum())

    patient_window = past_data[
        (past_data["patient_id"] == row["patient_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=60))
    ]
    patient_60.append(len(patient_window))

    velocity_window = past_data[
        (past_data["patient_id"] == row["patient_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=7))
    ]
    velocity_7.append(len(velocity_window))

    doctor_window = past_data[
        (past_data["doctor_id"] == row["doctor_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=30))
    ]
    doctor_30.append(doctor_window["claim_amount"].sum())

data["hospital_30day_total"] = hospital_30
data["patient_60day_count"] = patient_60
data["patient_7day_count"] = velocity_7
data["doctor_30day_total"] = doctor_30

# =====================================================
# BEHAVIORAL FEATURES (must match train_model.py)
# =====================================================
print("Engineering behavioral deviation features...")

patient_avg = data.groupby("patient_id")["claim_amount"].transform("mean")
hospital_avg = data.groupby("hospital_id")["claim_amount"].transform("mean")
disease_avg = data.groupby("disease_code")["claim_amount"].transform("mean")

data["patient_deviation_ratio"] = np.where(patient_avg > 0, data["claim_amount"] / patient_avg, 0)
data["hospital_deviation_ratio"] = np.where(hospital_avg > 0, data["claim_amount"] / hospital_avg, 0)
data["disease_deviation_ratio"] = np.where(disease_avg > 0, data["claim_amount"] / disease_avg, 0)

data["patient_deviation_ratio"] = np.log1p(data["patient_deviation_ratio"])
data["hospital_deviation_ratio"] = np.log1p(data["hospital_deviation_ratio"])
data["disease_deviation_ratio"] = np.log1p(data["disease_deviation_ratio"])

# =====================================================
# FINANCIAL FEATURES
# =====================================================
data["cost_per_day"] = data["claim_amount"] / data["length_of_stay"].replace(0, 1)
data["claim_policy_ratio"] = data["claim_amount"] / data["policy_age_days"].replace(0, 1)

# =====================================================
# ADVANCED FEATURES (must match train_model.py)
# =====================================================
print("Engineering advanced fraud signal features...")

data["early_claim_flag"] = (data["policy_age_days"] < 90).astype(int)
data["claim_burst_score"] = data["patient_7day_count"] * data["patient_60day_count"]
data["disease_amount_percentile"] = data.groupby("disease_code")["claim_amount"].rank(pct=True)

disease_avg_stay = data.groupby("disease_code")["length_of_stay"].transform("mean")
data["stay_vs_disease_norm"] = np.where(
    disease_avg_stay > 0,
    data["length_of_stay"] / disease_avg_stay,
    1.0
)

hospital_fraud_rate = data.groupby("hospital_id")["is_fraud"].transform("mean")
data["hospital_fraud_concentration"] = hospital_fraud_rate

doctor_avg_amount = data.groupby("doctor_id")["claim_amount"].transform("mean")
data["doctor_billing_intensity"] = np.log1p(doctor_avg_amount)

data["combined_risk_index"] = (
    data["early_claim_flag"] * 0.3 +
    data["disease_amount_percentile"] * 0.3 +
    data["stay_vs_disease_norm"] * 0.2 +
    data["hospital_fraud_concentration"] * 0.2
)

data = data.fillna(0)

# =====================================================
# SAME FEATURE LIST AS train_model.py
# =====================================================
feature_columns = [
    "claim_amount", "disease_code", "length_of_stay",
    "policy_age_days", "previous_claims_count",
    "cost_per_day", "claim_policy_ratio",
    "hospital_30day_total", "patient_60day_count",
    "patient_7day_count", "doctor_30day_total",
    "patient_deviation_ratio", "hospital_deviation_ratio",
    "disease_deviation_ratio",
    "early_claim_flag", "claim_burst_score",
    "disease_amount_percentile", "stay_vs_disease_norm",
    "hospital_fraud_concentration", "doctor_billing_intensity",
    "combined_risk_index"
]

X = data[feature_columns]

print(f"\nFeatures: {len(feature_columns)} (14 base + 7 advanced)")
print(f"Training on: {len(X):,} claims")

# =====================================================
# ISOLATION FOREST
# Contamination = actual fraud rate in dataset
# =====================================================
fraud_rate = data["is_fraud"].mean()
print(f"\nFraud rate (contamination): {fraud_rate:.3f}")

anomaly_model = IsolationForest(
    n_estimators=300,
    contamination=fraud_rate,
    max_samples="auto",
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=42
)

print("Training Isolation Forest...")
anomaly_model.fit(X)

# =====================================================
# QUICK EVALUATION
# =====================================================
scores = anomaly_model.decision_function(X)
predictions = anomaly_model.predict(X)  # -1 = anomaly, 1 = normal

# Convert to 0/1
pred_labels = (predictions == -1).astype(int)

from sklearn.metrics import classification_report, roc_auc_score
print("\nAnomaly Detection Report:")
print(classification_report(data["is_fraud"], pred_labels, target_names=["Clean", "Anomaly"]))

try:
    auc = roc_auc_score(data["is_fraud"], -scores)  # negate: lower score = more anomalous
    print(f"ROC-AUC (anomaly scores): {auc:.4f}")
except:
    pass

# =====================================================
# SAVE
# =====================================================
joblib.dump(anomaly_model, "fwa/models/isolation_forest.pkl")

print("\n" + "=" * 60)
print("Isolation Forest Saved!")
print(f"  Model path  : fwa/models/isolation_forest.pkl")
print(f"  Features    : {len(feature_columns)} (matches train_model.py exactly)")
print(f"  Estimators  : 300")
print(f"  Contamination: {fraud_rate:.3f}")
print("=" * 60)