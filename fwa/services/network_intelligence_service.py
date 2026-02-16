import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta

# ==========================================================
# PROJECT PATH
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fwa_claims.db")


# ==========================================================
# NETWORK INTELLIGENCE ENGINE
# ==========================================================
def get_network_intelligence(patient_id, doctor_id, hospital_id, claim_date):
    """
    Enterprise Network Risk Detection

    Detects:
    - Doctor–Hospital dependency ratio
    - Patient–Doctor repeat dependency
    - Potential collusion signals
    """

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()

    if df.empty:
        return {
            "doctor_hospital_dependency_ratio": 0,
            "patient_doctor_repeat_ratio": 0,
            "network_collusion_score": 0
        }

    df["claim_date"] = pd.to_datetime(df["claim_date"])
    claim_date = pd.to_datetime(claim_date)

    # Use 180-day window for relationship analysis
    window_start = claim_date - timedelta(days=180)
    df = df[df["claim_date"] >= window_start]

    # ======================================================
    # 1️⃣ Doctor-Hospital Dependency Ratio
    # ======================================================
    doctor_claims = df[df["doctor_id"] == doctor_id]

    if len(doctor_claims) > 0:
        hospital_claims_for_doctor = doctor_claims[
            doctor_claims["hospital_id"] == hospital_id
        ]
        doctor_hospital_dependency_ratio = (
            len(hospital_claims_for_doctor) / len(doctor_claims)
        )
    else:
        doctor_hospital_dependency_ratio = 0

    # ======================================================
    # 2️⃣ Patient-Doctor Repeat Ratio
    # ======================================================
    patient_claims = df[df["patient_id"] == patient_id]

    if len(patient_claims) > 0:
        repeat_with_same_doctor = patient_claims[
            patient_claims["doctor_id"] == doctor_id
        ]
        patient_doctor_repeat_ratio = (
            len(repeat_with_same_doctor) / len(patient_claims)
        )
    else:
        patient_doctor_repeat_ratio = 0

    # ======================================================
    # 3️⃣ Network Collusion Score
    # ======================================================
    collusion_score = 0

    # High doctor-hospital exclusivity
    if doctor_hospital_dependency_ratio > 0.75:
        collusion_score += 40

    # High patient-doctor repeat pattern
    if patient_doctor_repeat_ratio > 0.70:
        collusion_score += 30

    # Combined amplification
    if (
        doctor_hospital_dependency_ratio > 0.75
        and patient_doctor_repeat_ratio > 0.70
    ):
        collusion_score += 30

    collusion_score = min(collusion_score, 100)

    return {
        "doctor_hospital_dependency_ratio": round(
            doctor_hospital_dependency_ratio, 2
        ),
        "patient_doctor_repeat_ratio": round(
            patient_doctor_repeat_ratio, 2
        ),
        "network_collusion_score": collusion_score
    }
