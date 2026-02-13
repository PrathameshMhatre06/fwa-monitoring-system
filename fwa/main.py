import pandas as pd
import joblib
from sqlalchemy import create_engine
from fwa.rules.rule_engine import evaluate_rules


# -----------------------------
# CONNECT TO DATABASE
# -----------------------------
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")

# -----------------------------
# LOAD TRAINED MODELS
# -----------------------------
ml_model = joblib.load("fwa/models/xgb_fraud_model.pkl")
anomaly_model = joblib.load("fwa/models/isolation_forest.pkl")

# -----------------------------
# LOAD SAMPLE CLAIMS
# -----------------------------
claims = pd.read_sql("SELECT * FROM claims LIMIT 10", engine)

# -----------------------------
# WEIGHT CONFIGURATION
# -----------------------------
RULE_WEIGHT = 0.3
ML_WEIGHT = 0.5
ANOMALY_WEIGHT = 0.2

for _, row in claims.iterrows():

    # =============================
    # 1️⃣ RULE ENGINE
    # =============================
    rule_score, triggered_rules = evaluate_rules(row)
    rule_score_normalized = min(rule_score, 100)

    # =============================
    # 2️⃣ ML MODEL SCORING
    # =============================
    features = row.drop(["claim_id", "patient_id", "is_fraud"])
    features_df = pd.DataFrame([features])

    ml_proba = ml_model.predict_proba(features_df)[0][1]
    ml_score = ml_proba * 100

    # =============================
    # 3️⃣ ANOMALY DETECTION
    # =============================
    anomaly_score_raw = anomaly_model.decision_function(features_df)[0]

    # Convert anomaly score to positive scale
    anomaly_score = (1 - anomaly_score_raw) * 50

    if anomaly_score < 0:
        anomaly_score = 0

    # =============================
    # 4️⃣ FINAL WEIGHTED RISK SCORE
    # =============================
    final_score = (
        RULE_WEIGHT * rule_score_normalized +
        ML_WEIGHT * ml_score +
        ANOMALY_WEIGHT * anomaly_score
    )

    # =============================
    # 5️⃣ RISK CLASSIFICATION
    # =============================
    if final_score < 30:
        risk_level = "LOW"
    elif final_score < 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # =============================
    # 6️⃣ PRINT OUTPUT
    # =============================
    print(f"Claim ID: {row['claim_id']}")
    print(f"Rule Score: {rule_score_normalized}")
    print(f"ML Score: {ml_score:.2f}")
    print(f"Anomaly Score: {anomaly_score:.2f}")
    print(f"Final Risk Score: {final_score:.2f}")
    print(f"Risk Level: {risk_level}")
    print(f"Triggered Rules: {triggered_rules}")
    print("--------------------------------------------------")
