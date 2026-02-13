import pandas as pd
import joblib
from sqlalchemy import create_engine
from fwa.rules.rule_engine import evaluate_rules

# Load models
ml_model = joblib.load("fwa/models/xgb_fraud_model.pkl")
anomaly_model = joblib.load("fwa/models/isolation_forest.pkl")

# Weights
RULE_WEIGHT = 0.3
ML_WEIGHT = 0.5
ANOMALY_WEIGHT = 0.2

def score_claim(claim_dict):

    claim_df = pd.DataFrame([claim_dict])

    # Rule Score
    rule_score, triggered = evaluate_rules(claim_df.iloc[0])
    rule_score = min(rule_score, 100)

    # ML Score
    features_df = claim_df.drop(columns=["claim_id", "patient_id"])
    ml_proba = ml_model.predict_proba(features_df)[0][1]
    ml_score = ml_proba * 100

    # Anomaly Score
    anomaly_raw = anomaly_model.decision_function(features_df)[0]
    anomaly_score = (1 - anomaly_raw) * 50
    if anomaly_score < 0:
        anomaly_score = 0

    # Final Score
    final_score = (
        RULE_WEIGHT * rule_score +
        ML_WEIGHT * ml_score +
        ANOMALY_WEIGHT * anomaly_score
    )

    # Classification
    if final_score < 30:
        risk = "LOW"
    elif final_score < 60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "final_score": round(final_score, 2),
        "risk_level": risk,
        "triggered_rules": triggered
    }


# ----------------------------
# DEMO CLAIM
# ----------------------------
if __name__ == "__main__":

    sample_claim = {
        "claim_id": 9999,
        "patient_id": 1500,
        "hospital_id": 12,
        "doctor_id": 25,
        "claim_amount": 150000,
        "disease_code": 3,
        "length_of_stay": 12,
        "policy_age_days": 90,
        "previous_claims_count": 8,
        "hospital_fraud_history_score": 0.9
    }

    result = score_claim(sample_claim)

    print("\n----- Real-Time Claim Scoring -----")
    print("Final Risk Score:", result["final_score"])
    print("Risk Level:", result["risk_level"])
    print("Triggered Rules:", result["triggered_rules"])
