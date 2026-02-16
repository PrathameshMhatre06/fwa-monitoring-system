import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from datetime import timedelta
import joblib

# =====================================================
# CONNECT TO DATABASE
# =====================================================

engine = create_engine("sqlite:///fwa/data/fwa_claims.db")
data = pd.read_sql("SELECT * FROM claims", engine)

data["claim_date"] = pd.to_datetime(data["claim_date"])
data = data.sort_values("claim_date").reset_index(drop=True)

# =====================================================
# TIME WINDOW FEATURES (Same as ML)
# =====================================================

hospital_30 = []
patient_60 = []
velocity_7 = []
doctor_30 = []

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
# BEHAVIORAL FEATURES (Same as ML)
# =====================================================

patient_avg = data.groupby("patient_id")["claim_amount"].transform("mean")
hospital_avg = data.groupby("hospital_id")["claim_amount"].transform("mean")
disease_avg = data.groupby("disease_code")["claim_amount"].transform("mean")

data["patient_deviation_ratio"] = np.where(patient_avg > 0, data["claim_amount"] / patient_avg, 0)
data["hospital_deviation_ratio"] = np.where(hospital_avg > 0, data["claim_amount"] / hospital_avg, 0)
data["disease_deviation_ratio"] = np.where(disease_avg > 0, data["claim_amount"] / disease_avg, 0)

# Log normalize
data["patient_deviation_ratio"] = np.log1p(data["patient_deviation_ratio"])
data["hospital_deviation_ratio"] = np.log1p(data["hospital_deviation_ratio"])
data["disease_deviation_ratio"] = np.log1p(data["disease_deviation_ratio"])

# =====================================================
# FINANCIAL FEATURES
# =====================================================

data["cost_per_day"] = data["claim_amount"] / data["length_of_stay"].replace(0, 1)
data["claim_policy_ratio"] = data["claim_amount"] / data["policy_age_days"].replace(0, 1)

data = data.fillna(0)

# =====================================================
# FINAL FEATURE LIST (MUST MATCH ML EXACTLY)
# =====================================================

feature_columns = [
    "claim_amount",
    "disease_code",
    "length_of_stay",
    "policy_age_days",
    "previous_claims_count",
    "cost_per_day",
    "claim_policy_ratio",
    "hospital_30day_total",
    "patient_60day_count",
    "patient_7day_count",
    "doctor_30day_total",
    "patient_deviation_ratio",
    "hospital_deviation_ratio",
    "disease_deviation_ratio"
]

X = data[feature_columns]

# =====================================================
# TRAIN ISOLATION FOREST
# =====================================================

iso_model = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42
)

iso_model.fit(X)

joblib.dump(iso_model, "fwa/models/isolation_forest.pkl")

print("✅ Behavior-Driven Isolation Forest Saved Successfully!")
