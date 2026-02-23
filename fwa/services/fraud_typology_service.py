# ==========================================================
# FRAUD TYPOLOGY CLASSIFICATION ENGINE
# ==========================================================

def classify_fraud_typology(
    claim_dict,
    aggregation,
    velocity,
    behavioral,
    disease_deviation_ratio,
    network_metrics
):
    """
    Classifies suspected fraud type based on claim behavior patterns.
    Returns list of detected fraud typologies.
    """

    fraud_types = []

    # ------------------------------------------------------
    # 1️⃣ Early Policy Exploitation
    # ------------------------------------------------------
    if claim_dict["policy_age_days"] < 15 and claim_dict["claim_amount"] > 200000:
        fraud_types.append("Early Policy Exploitation")

    # ------------------------------------------------------
    # 2️⃣ High Frequency Abuse
    # ------------------------------------------------------
    if velocity["patient_7day_count"] > 3:
        fraud_types.append("High Frequency Abuse")

    # ------------------------------------------------------
    # 3️⃣ Inflated Length of Stay
    # ------------------------------------------------------
    if claim_dict["length_of_stay"] > 10 and behavioral["hospital_deviation_ratio"] > 1.5:
        fraud_types.append("Inflated Length of Stay")

    # ------------------------------------------------------
    # 4️⃣ Upcoding Suspicion
    # ------------------------------------------------------
    if disease_deviation_ratio > 1.7:
        fraud_types.append("Upcoding Suspicion")

    # ------------------------------------------------------
    # 5️⃣ Phantom Billing Pattern
    # ------------------------------------------------------
    if claim_dict["claim_amount"] > 400000 and claim_dict["length_of_stay"] <= 2:
        fraud_types.append("Phantom Billing Pattern")

    # ------------------------------------------------------
    # 6️⃣ Collusion Risk (Network Based)
    # ------------------------------------------------------
    if network_metrics["doctor_30day_total"] > 10:
        fraud_types.append("Collusion Risk")

    # ------------------------------------------------------
    # Default Case
    # ------------------------------------------------------
    if not fraud_types:
        fraud_types.append("Unspecified Risk Pattern")

    return fraud_types
