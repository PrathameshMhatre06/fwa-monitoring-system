import sqlite3
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fwa_claims.db")


def get_aggregation_metrics(patient_id, hospital_id, claim_date):

    # ------------------------------------------------
    # Normalize claim_date (accept str OR datetime)
    # ------------------------------------------------
    if isinstance(claim_date, str):
        claim_date = datetime.strptime(claim_date, "%Y-%m-%d")
    elif hasattr(claim_date, "strftime"):
        claim_date = datetime.combine(claim_date, datetime.min.time())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    date_30 = (claim_date - timedelta(days=30)).strftime("%Y-%m-%d")
    date_60 = (claim_date - timedelta(days=60)).strftime("%Y-%m-%d")

    # Hospital 30-day total
    cursor.execute("""
        SELECT SUM(claim_amount), COUNT(*)
        FROM claims
        WHERE hospital_id = ?
        AND claim_date >= ?
    """, (hospital_id, date_30))

    hospital_total, hospital_count = cursor.fetchone()
    hospital_total = hospital_total or 0
    hospital_count = hospital_count or 0

    # Patient 60-day total
    cursor.execute("""
        SELECT SUM(claim_amount), COUNT(*)
        FROM claims
        WHERE patient_id = ?
        AND claim_date >= ?
    """, (patient_id, date_60))

    patient_total, patient_count = cursor.fetchone()
    patient_total = patient_total or 0
    patient_count = patient_count or 0

    conn.close()

    return {
        "hospital_30day_total": hospital_total,
        "hospital_30day_count": hospital_count,
        "patient_60day_total": patient_total,
        "patient_60day_count": patient_count
    }
