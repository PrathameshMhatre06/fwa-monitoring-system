import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from fwa.rules.rule_engine import evaluate_rules


# -----------------------------
# CONNECT TO DATABASE
# -----------------------------
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")

# -----------------------------
# LOAD MODELS
# -----------------------------
ml_model = joblib.load("fwa/models/xgb_fraud_model.pkl")
anomaly_model = joblib.load("fwa/models/isolation_forest.pkl")

# -----------------------------
# LOAD FULL DATASET
# -----------------------------
data = pd.read_sql("SELECT * FROM claims", engine)

# -----------------------------
# WEIGHTS
# -----------------------------
RULE_WEIGHT = 0.3
ML_WEIGHT = 0.5
ANOMALY_WEIGHT = 0.2

final_scores = []
predicted_labels = []

for _, row in data.iterrows():

    # RULE SCORE
    rule_score, _ = evaluate_rules(row)
    rule_score = min(rule_score, 100)

    # ML SCORE
    features = row.drop(["claim_id", "patient_id", "is_fraud"])
    features_df = pd.DataFrame([features])
    ml_proba = ml_model.predict_proba(features_df)[0][1]
    ml_score = ml_proba * 100

    # ANOMALY SCORE
    anomaly_raw = anomaly_model.decision_function(features_df)[0]
    anomaly_score = (1 - anomaly_raw) * 50
    if anomaly_score < 0:
        anomaly_score = 0

    # FINAL SCORE
    final_score = (
        RULE_WEIGHT * rule_score +
        ML_WEIGHT * ml_score +
        ANOMALY_WEIGHT * anomaly_score
    )

    final_scores.append(final_score)

    # Convert to fraud prediction
    if final_score >= 60:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

# -----------------------------
# EVALUATION METRICS
# -----------------------------
y_true = data["is_fraud"]

print("Confusion Matrix:")
print(confusion_matrix(y_true, predicted_labels))

print("\nClassification Report:")
print(classification_report(y_true, predicted_labels))

print("\nROC-AUC Score:")
print(roc_auc_score(y_true, final_scores))

import numpy as np
from sklearn.metrics import precision_score, recall_score

print("\n------ Threshold Optimization ------")

thresholds = np.arange(10, 90, 5)

best_threshold = 0
best_score = 0

for threshold in thresholds:

    temp_preds = []

    for score in final_scores:
        if score >= threshold:
            temp_preds.append(1)
        else:
            temp_preds.append(0)

    precision = precision_score(y_true, temp_preds)
    recall = recall_score(y_true, temp_preds)

    # Business scoring logic:
    # We value recall more than precision (fraud detection priority)
    business_score = (0.7 * recall) + (0.3 * precision)

    print(f"Threshold: {threshold} | Precision: {precision:.3f} | Recall: {recall:.3f} | Business Score: {business_score:.3f}")

    if business_score > best_score:
        best_score = business_score
        best_threshold = threshold

print("\nOptimal Threshold Based on Business Score:", best_threshold)
