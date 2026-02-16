import sqlite3
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fwa_claims.db")


def get_velocity_metrics(patient_id, claim_date):

    # ------------------------------------------------
    # Normalize claim_date (handle str OR date)
    # ------------------------------------------------
    if isinstance(claim_date, str):
        claim_date = datetime.strptime(claim_date, "%Y-%m-%d")
    elif hasattr(claim_date, "strftime"):
        claim_date = datetime.combine(claim_date, datetime.min.time())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    date_7 = (claim_date - timedelta(days=7)).strftime("%Y-%m-%d")
    date_15 = (claim_date - timedelta(days=15)).strftime("%Y-%m-%d")

    # Claims in last 7 days
    cursor.execute("""
        SELECT COUNT(*), SUM(claim_amount)
        FROM claims
        WHERE patient_id = ?
        AND claim_date >= ?
    """, (patient_id, date_7))

    count_7, total_7 = cursor.fetchone()
    count_7 = count_7 or 0
    total_7 = total_7 or 0

    # Claims in last 15 days
    cursor.execute("""
        SELECT COUNT(*)
        FROM claims
        WHERE patient_id = ?
        AND claim_date >= ?
    """, (patient_id, date_15))

    count_15 = cursor.fetchone()[0] or 0

    conn.close()

    return {
        "patient_7day_count": count_7,
        "patient_7day_total": total_7,
        "patient_15day_count": count_15
    }
