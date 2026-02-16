import json
import os

# ==========================================================
# LOAD CONFIG
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "rule_config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)


# ==========================================================
# ENTERPRISE RULE ENGINE
# ==========================================================
def evaluate_rules(
    claim_row,
    aggregation,
    velocity,
    network,
    behavioral,
    disease_deviation_ratio,
    network_intel=None  # NEW
):

    score = 0
    triggered = []

    # ======================================================
    # 1️⃣ Aggregation Rules
    # ======================================================
    agg_cfg = CONFIG["aggregation_rules"]

    if aggregation["hospital_30day_total"] > agg_cfg["hospital_30day_total_threshold"]:
        score += 20
        triggered.append("hospital_aggregation_high")

    if aggregation["patient_60day_count"] > agg_cfg["patient_60day_count_threshold"]:
        score += 20
        triggered.append("patient_frequency_high")

    # ======================================================
    # 2️⃣ Velocity Rules
    # ======================================================
    vel_cfg = CONFIG["velocity_rules"]

    if velocity["patient_7day_count"] > vel_cfg["patient_7day_count_threshold"]:
        score += 25
        triggered.append("patient_velocity_spike")

    # ======================================================
    # 3️⃣ Network Exposure Rules
    # ======================================================
    net_cfg = CONFIG["network_rules"]

    if network["doctor_30day_total"] > net_cfg["doctor_30day_total_threshold"]:
        score += 20
        triggered.append("doctor_high_exposure")

    if network["doctor_unique_hospitals"] > net_cfg["doctor_unique_hospital_threshold"]:
        score += 15
        triggered.append("doctor_network_anomaly")

    # ======================================================
    # 4️⃣ Behavioral Rules
    # ======================================================
    beh_cfg = CONFIG["behavioral_rules"]

    if behavioral["patient_deviation_ratio"] > beh_cfg["critical_deviation_threshold"]:
        score += 40
        triggered.append("critical_patient_behavioral_spike")

    elif behavioral["patient_deviation_ratio"] > beh_cfg["strong_deviation_threshold"]:
        score += 25
        triggered.append("strong_patient_behavioral_spike")

    # ======================================================
    # 5️⃣ Disease Deviation Rules
    # ======================================================
    dis_cfg = CONFIG["disease_rules"]

    if disease_deviation_ratio > dis_cfg["critical_deviation_threshold"]:
        score += 35
        triggered.append("critical_disease_cost_spike")

    elif disease_deviation_ratio > dis_cfg["strong_deviation_threshold"]:
        score += 20
        triggered.append("strong_disease_cost_spike")

    # ======================================================
    # 6️⃣ Early Policy Abuse Rules
    # ======================================================
    early_cfg = CONFIG["early_policy_rules"]

    claim_policy_ratio = (
        claim_row["claim_amount"] /
        max(claim_row["policy_age_days"], 1)
    )

    if (
        claim_row["policy_age_days"] < early_cfg["critical_policy_age_days"]
        and claim_row["claim_amount"] > early_cfg["early_claim_amount_threshold"]
    ):
        score += 30
        triggered.append("critical_early_large_claim")

    if claim_policy_ratio > early_cfg.get("early_claim_policy_ratio_threshold", 999999):
        score += 25
        triggered.append("early_policy_ratio_spike")

    # ======================================================
    # 7️⃣ NEW — Network Intelligence Rules
    # ======================================================
    if network_intel:

        if network_intel["doctor_hospital_dependency_ratio"] > 0.75:
            score += 25
            triggered.append("doctor_hospital_dependency_high")

        if network_intel["patient_doctor_repeat_ratio"] > 0.70:
            score += 20
            triggered.append("patient_doctor_repeat_pattern")

        if network_intel["network_collusion_score"] >= 60:
            score += 40
            triggered.append("network_collusion_detected")

    return min(score, 100), triggered
