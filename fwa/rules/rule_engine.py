from .rule_config import RULE_WEIGHTS, THRESHOLDS

def evaluate_rules(claim_row):
    score = 0
    triggered_rules = []

    # Rule 1: High Claim Amount
    if claim_row["claim_amount"] > THRESHOLDS["claim_amount"]:
        score += RULE_WEIGHTS["high_claim_amount"]
        triggered_rules.append("high_claim_amount")

    # Rule 2: High Hospital Fraud Risk
    if claim_row["hospital_fraud_history_score"] > THRESHOLDS["hospital_risk"]:
        score += RULE_WEIGHTS["high_hospital_risk"]
        triggered_rules.append("high_hospital_risk")

    # Rule 3: Frequent Claims
    if claim_row["previous_claims_count"] > THRESHOLDS["previous_claims"]:
        score += RULE_WEIGHTS["frequent_claims"]
        triggered_rules.append("frequent_claims")

    # Rule 4: Long Stay Anomaly
    if claim_row["length_of_stay"] > THRESHOLDS["length_of_stay"]:
        score += RULE_WEIGHTS["long_stay_anomaly"]
        triggered_rules.append("long_stay_anomaly")

    return score, triggered_rules
