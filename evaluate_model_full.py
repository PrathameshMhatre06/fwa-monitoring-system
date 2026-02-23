import sys
import os

# --------------------------------------------------
# FIX PYTHON PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FWA_PATH = os.path.join(PROJECT_ROOT, "fwa")
sys.path.append(FWA_PATH)

import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from services.aggregation_service import get_aggregation_metrics
from services.velocity_service import get_velocity_metrics
from services.network_service import get_doctor_network_metrics
from services.behavioral_service import get_behavioral_metrics
from services.benchmark_service import get_disease_deviation


print("\n🔎 Loading dataset...")

DATA_PATH = os.path.join(PROJECT_ROOT, "fwa", "data", "claims_data.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "fwa", "models", "xgb_fraud_model.pkl")

df = pd.read_csv(DATA_PATH)

print("Total records:", len(df))

# --------------------------------------------------
# GENERATE FAKE CLAIM DATES (IMPORTANT FIX)
# --------------------------------------------------
print("⚙ Generating synthetic claim dates...")

start_date = datetime(2023, 1, 1)

df["claim_date"] = [
    start_date + timedelta(days=int(i / 10))
    for i in range(len(df))
]

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load(MODEL_PATH)

feature_rows = []
y_true = []

print("⚙ Rebuilding features using pipeline...")

for _, row in df.iterrows():

    claim_date = row["claim_date"]

    aggregation = get_aggregation_metrics(
        row["patient_id"],
        row["hospital_id"],
        claim_date
    )

    velocity = get_velocity_metrics(
        row["patient_id"],
        claim_date
    )

    network = get_doctor_network_metrics(
        row["doctor_id"],
        claim_date
    )

    behavioral = get_behavioral_metrics(
        row["patient_id"],
        row["hospital_id"],
        row["claim_amount"],
        claim_date
    )

    disease_deviation_ratio = get_disease_deviation(
        row["disease_code"],
        row["claim_amount"]
    )

    feature_rows.append({
        "claim_amount": row["claim_amount"],
        "disease_code": row["disease_code"],
        "length_of_stay": row["length_of_stay"],
        "policy_age_days": row["policy_age_days"],
        "previous_claims_count": row["previous_claims_count"],
        "cost_per_day": row["claim_amount"] / max(row["length_of_stay"], 1),
        "claim_policy_ratio": row["claim_amount"] / max(row["policy_age_days"], 1),
        "hospital_30day_total": aggregation["hospital_30day_total"],
        "patient_60day_count": aggregation["patient_60day_count"],
        "patient_7day_count": velocity["patient_7day_count"],
        "doctor_30day_total": network["doctor_30day_total"],
        "patient_deviation_ratio": behavioral["patient_deviation_ratio"],
        "hospital_deviation_ratio": behavioral["hospital_deviation_ratio"],
        "disease_deviation_ratio": disease_deviation_ratio
    })

    y_true.append(int(row["is_fraud"]))

# --------------------------------------------------
# CREATE FEATURE MATRIX
# --------------------------------------------------
X = pd.DataFrame(feature_rows)
X = X[model.feature_names_in_]

print("🤖 Running predictions...")

y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_proba)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
fpr = fp / (fp + tn)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
print("\n==============================")
print("📊 MODEL PERFORMANCE")
print("==============================")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"ROC-AUC: {auc_score:.4f}")

print("\nConfusion Matrix:")
print(f"TP: {tp} | FP: {fp}")
print(f"FN: {fn} | TN: {tn}")

print("\n✅ Evaluation Complete.")
