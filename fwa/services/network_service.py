import sqlite3
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fwa_claims.db")


def get_doctor_network_metrics(doctor_id, claim_date):

    # ------------------------------------------------
    # Normalize claim_date (handle str OR date)
    # ------------------------------------------------
    if isinstance(claim_date, str):
        claim_date = datetime.strptime(claim_date, "%Y-%m-%d")
    elif hasattr(claim_date, "strftime"):
        claim_date = datetime.combine(claim_date, datetime.min.time())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    date_30 = (claim_date - timedelta(days=30)).strftime("%Y-%m-%d")

    # Doctor 30-day total billing & count
    cursor.execute("""
        SELECT SUM(claim_amount), COUNT(*)
        FROM claims
        WHERE doctor_id = ?
        AND claim_date >= ?
    """, (doctor_id, date_30))

    total_billing, claim_count = cursor.fetchone()
    total_billing = total_billing or 0
    claim_count = claim_count or 0

    # Unique hospitals doctor works with
    cursor.execute("""
        SELECT COUNT(DISTINCT hospital_id)
        FROM claims
        WHERE doctor_id = ?
        AND claim_date >= ?
    """, (doctor_id, date_30))

    unique_hospitals = cursor.fetchone()[0] or 0

    conn.close()

    return {
        "doctor_30day_total": total_billing,
        "doctor_30day_count": claim_count,
        "doctor_unique_hospitals": unique_hospitals
    }
