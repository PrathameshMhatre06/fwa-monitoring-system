import sqlite3
import os
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fwa_claims.db")


def get_behavioral_metrics(patient_id, hospital_id, current_amount, claim_date):
    """
    Calculates behavioral deviation metrics using historical data
    BEFORE the current claim date (prevents leakage).
    """

    # ------------------------------------------------
    # Normalize claim_date (handle str OR date)
    # ------------------------------------------------
    if isinstance(claim_date, str):
        claim_date = datetime.strptime(claim_date, "%Y-%m-%d")
    elif hasattr(claim_date, "strftime"):
        claim_date = datetime.combine(claim_date, datetime.min.time())

    claim_date_str = claim_date.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ----------------------------------------
    # PATIENT HISTORY (only past claims)
    # ----------------------------------------
    cursor.execute("""
        SELECT claim_amount
        FROM claims
        WHERE patient_id = ?
        AND claim_date < ?
    """, (patient_id, claim_date_str))

    patient_claims = [row[0] for row in cursor.fetchall()]

    if len(patient_claims) > 1:
        patient_avg = np.mean(patient_claims)
        patient_std = np.std(patient_claims)
    else:
        patient_avg = 0
        patient_std = 0

    # ----------------------------------------
    # HOSPITAL HISTORY (only past claims)
    # ----------------------------------------
    cursor.execute("""
        SELECT claim_amount
        FROM claims
        WHERE hospital_id = ?
        AND claim_date < ?
    """, (hospital_id, claim_date_str))

    hospital_claims = [row[0] for row in cursor.fetchall()]

    if len(hospital_claims) > 1:
        hospital_avg = np.mean(hospital_claims)
        hospital_std = np.std(hospital_claims)
    else:
        hospital_avg = 0
        hospital_std = 0

    conn.close()

    # ----------------------------------------
    # Deviation Ratios
    # ----------------------------------------
    patient_deviation_ratio = (
        current_amount / patient_avg if patient_avg > 0 else 0
    )

    hospital_deviation_ratio = (
        current_amount / hospital_avg if hospital_avg > 0 else 0
    )

    return {
        "patient_avg_claim": patient_avg,
        "patient_std_claim": patient_std,
        "patient_deviation_ratio": patient_deviation_ratio,
        "hospital_avg_claim": hospital_avg,
        "hospital_std_claim": hospital_std,
        "hospital_deviation_ratio": hospital_deviation_ratio
    }
