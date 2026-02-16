import sqlite3
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fwa_claims.db")


def get_disease_deviation(disease_code, claim_amount):
    """
    Calculates how much this claim deviates from
    historical average for the same disease.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT claim_amount
        FROM claims
        WHERE disease_code = ?
    """, (disease_code,))

    amounts = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(amounts) > 1:
        avg_amount = np.mean(amounts)
    else:
        avg_amount = 0

    if avg_amount > 0:
        deviation_ratio = claim_amount / avg_amount
    else:
        deviation_ratio = 0

    return deviation_ratio
