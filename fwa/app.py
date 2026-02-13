import streamlit as st
import pandas as pd
import joblib
import os
from rules.rule_engine import evaluate_rules

# ----------------------------
# Get current file directory
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_fraud_model.pkl")
ANOMALY_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")

# ----------------------------
# Load Models
# ----------------------------
ml_model = joblib.load(MODEL_PATH)
anomaly_model = joblib.load(ANOMALY_PATH)

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

    if final_score < 30:
        risk = "LOW"
    elif final_score < 60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return final_score, risk, triggered


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("FWA Monitoring System - Claim Risk Scoring")

st.sidebar.header("Enter Claim Details")

claim_id = st.sidebar.number_input("Claim ID", value=1001)
patient_id = st.sidebar.number_input("Patient ID", value=1500)
hospital_id = st.sidebar.number_input("Hospital ID", value=10)
doctor_id = st.sidebar.number_input("Doctor ID", value=20)
claim_amount = st.sidebar.number_input("Claim Amount", value=50000)
disease_code = st.sidebar.number_input("Disease Code", value=3)
length_of_stay = st.sidebar.number_input("Length of Stay (days)", value=5)
policy_age_days = st.sidebar.number_input("Policy Age (days)", value=300)
previous_claims_count = st.sidebar.number_input("Previous Claims Count", value=2)
hospital_fraud_history_score = st.sidebar.slider(
    "Hospital Fraud History Score", 0.0, 1.0, 0.2
)

if st.sidebar.button("Score Claim"):

    claim_data = {
        "claim_id": claim_id,
        "patient_id": patient_id,
        "hospital_id": hospital_id,
        "doctor_id": doctor_id,
        "claim_amount": claim_amount,
        "disease_code": disease_code,
        "length_of_stay": length_of_stay,
        "policy_age_days": policy_age_days,
        "previous_claims_count": previous_claims_count,
        "hospital_fraud_history_score": hospital_fraud_history_score,
    }

    final_score, risk_level, triggered_rules = score_claim(claim_data)

    st.subheader("Scoring Result")

    st.metric("Final Risk Score", f"{final_score:.2f}")
    st.metric("Risk Level", risk_level)

    st.write("Triggered Rules:")
    if triggered_rules:
        for rule in triggered_rules:
            st.write(f"- {rule}")
    else:
        st.write("None")
