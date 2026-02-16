import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fwa_claims.db")


def calculate_entity_risk(entity_type, entity_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if entity_type == "hospital":
        query = """
            SELECT COUNT(*),
                   SUM(is_fraud),
                   SUM(claim_amount)
            FROM claims
            WHERE hospital_id = ?
        """

    elif entity_type == "doctor":
        query = """
            SELECT COUNT(*),
                   SUM(is_fraud),
                   SUM(claim_amount)
            FROM claims
            WHERE doctor_id = ?
        """

    elif entity_type == "patient":
        query = """
            SELECT COUNT(*),
                   SUM(is_fraud),
                   SUM(claim_amount)
            FROM claims
            WHERE patient_id = ?
        """
    else:
        conn.close()
        return None

    cursor.execute(query, (entity_id,))
    result = cursor.fetchone()

    conn.close()

    total_claims = result[0] if result[0] else 0
    fraud_count = result[1] if result[1] else 0
    total_amount = result[2] if result[2] else 0

    fraud_ratio = (fraud_count / total_claims) if total_claims > 0 else 0

    # Risk Index Calculation (Enterprise Style)
    risk_index = (
        (fraud_ratio * 60) + 
        (min(total_claims / 50, 1) * 20) +
        (min(total_amount / 2000000, 1) * 20)
    )

    if risk_index < 30:
        risk_band = "LOW"
    elif risk_index < 60:
        risk_band = "MEDIUM"
    else:
        risk_band = "HIGH"

    return {
        "total_claims": total_claims,
        "fraud_count": fraud_count,
        "fraud_ratio": round(fraud_ratio, 2),
        "total_amount": total_amount,
        "risk_index": round(risk_index, 2),
        "risk_band": risk_band
    }
