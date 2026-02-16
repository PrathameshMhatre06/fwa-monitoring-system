import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from datetime import timedelta

# =====================================================
# CONNECT TO DATABASE
# =====================================================

engine = create_engine("sqlite:///fwa/data/fwa_claims.db")
data = pd.read_sql("SELECT * FROM claims", engine)

data["claim_date"] = pd.to_datetime(data["claim_date"])
data = data.sort_values("claim_date").reset_index(drop=True)

# =====================================================
# TIME WINDOW FEATURES
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
# BEHAVIORAL FEATURES
# =====================================================

patient_avg = data.groupby("patient_id")["claim_amount"].transform("mean")
hospital_avg = data.groupby("hospital_id")["claim_amount"].transform("mean")
disease_avg = data.groupby("disease_code")["claim_amount"].transform("mean")

data["patient_deviation_ratio"] = np.where(patient_avg > 0, data["claim_amount"] / patient_avg, 0)
data["hospital_deviation_ratio"] = np.where(hospital_avg > 0, data["claim_amount"] / hospital_avg, 0)
data["disease_deviation_ratio"] = np.where(disease_avg > 0, data["claim_amount"] / disease_avg, 0)

# Log normalize deviations (VERY IMPORTANT)
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
# FINAL FEATURE LIST (NO RAW IDs)
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
y = data["is_fraud"]

# =====================================================
# TIME SPLIT
# =====================================================

split_index = int(len(data) * 0.8)

X_train_full = X.iloc[:split_index]
y_train_full = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full, test_size=0.2, shuffle=False
)

# =====================================================
# MODEL
# =====================================================

base_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss"
)

base_model.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_calib, y_calib)

y_proba = calibrated_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, calibrated_model.predict(X_test)))
print("\nROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

joblib.dump(calibrated_model, "fwa/models/xgb_fraud_model.pkl")

print("\n✅ Behavior-Driven ML Model Saved Successfully!")
