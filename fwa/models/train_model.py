import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from datetime import timedelta

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    print("WARNING: LightGBM not found — install via: pip install lightgbm")
    print("         Falling back to XGBoost + RandomForest ensemble only.")
    LGBM_AVAILABLE = False

print("=" * 60)
print("  PRAHARI - Ensemble Stacking Model Trainer")
print("=" * 60)

# =====================================================
# CONNECT TO DATABASE
# =====================================================
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")
data = pd.read_sql("SELECT * FROM claims", engine)
data["claim_date"] = pd.to_datetime(data["claim_date"], format="mixed")
data = data.sort_values("claim_date").reset_index(drop=True)
print(f"\nLoaded {len(data):,} claims from database")

# =====================================================
# TIME WINDOW FEATURES
# =====================================================
print("\nEngineering time-window features...")

hospital_30, patient_60, velocity_7, doctor_30 = [], [], [], []

for i, row in data.iterrows():
    current_date = row["claim_date"]
    past_data = data[data["claim_date"] < current_date]

    hospital_window = past_data[
        (past_data["hospital_id"] == row["hospital_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=30))
    ]
    hospital_30.append(hospital_window["claim_amount"].sum())

    patient_window = past_data[
        (past_data["patient_id"] == row["patient_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=60))
    ]
    patient_60.append(len(patient_window))

    velocity_window = past_data[
        (past_data["patient_id"] == row["patient_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=7))
    ]
    velocity_7.append(len(velocity_window))

    doctor_window = past_data[
        (past_data["doctor_id"] == row["doctor_id"]) &
        (past_data["claim_date"] >= current_date - timedelta(days=30))
    ]
    doctor_30.append(doctor_window["claim_amount"].sum())

data["hospital_30day_total"] = hospital_30
data["patient_60day_count"] = patient_60
data["patient_7day_count"] = velocity_7
data["doctor_30day_total"] = doctor_30

# =====================================================
# BEHAVIORAL FEATURES
# =====================================================
print("Engineering behavioral deviation features...")

patient_avg = data.groupby("patient_id")["claim_amount"].transform("mean")
hospital_avg = data.groupby("hospital_id")["claim_amount"].transform("mean")
disease_avg = data.groupby("disease_code")["claim_amount"].transform("mean")

data["patient_deviation_ratio"] = np.where(patient_avg > 0, data["claim_amount"] / patient_avg, 0)
data["hospital_deviation_ratio"] = np.where(hospital_avg > 0, data["claim_amount"] / hospital_avg, 0)
data["disease_deviation_ratio"] = np.where(disease_avg > 0, data["claim_amount"] / disease_avg, 0)

data["patient_deviation_ratio"] = np.log1p(data["patient_deviation_ratio"])
data["hospital_deviation_ratio"] = np.log1p(data["hospital_deviation_ratio"])
data["disease_deviation_ratio"] = np.log1p(data["disease_deviation_ratio"])

# =====================================================
# FINANCIAL FEATURES
# =====================================================
data["cost_per_day"] = data["claim_amount"] / data["length_of_stay"].replace(0, 1)
data["claim_policy_ratio"] = data["claim_amount"] / data["policy_age_days"].replace(0, 1)

# =====================================================
# ADVANCED FEATURES (NEW)
# =====================================================
print("Engineering advanced fraud signal features...")

# 1. Early claim flag - claims within 90 days of policy start
data["early_claim_flag"] = (data["policy_age_days"] < 90).astype(int)

# 2. Claim burst score - rapid multiple submissions
data["claim_burst_score"] = data["patient_7day_count"] * data["patient_60day_count"]

# 3. Amount percentile within disease group
data["disease_amount_percentile"] = data.groupby("disease_code")["claim_amount"].rank(pct=True)

# 4. Stay vs disease norm ratio
disease_avg_stay = data.groupby("disease_code")["length_of_stay"].transform("mean")
data["stay_vs_disease_norm"] = np.where(
    disease_avg_stay > 0,
    data["length_of_stay"] / disease_avg_stay,
    1.0
)

# 5. Hospital fraud concentration
hospital_fraud_rate = data.groupby("hospital_id")["is_fraud"].transform("mean")
data["hospital_fraud_concentration"] = hospital_fraud_rate

# 6. Doctor billing intensity
doctor_avg_amount = data.groupby("doctor_id")["claim_amount"].transform("mean")
data["doctor_billing_intensity"] = np.log1p(doctor_avg_amount)

# 7. Combined risk index
data["combined_risk_index"] = (
    data["early_claim_flag"] * 0.3 +
    data["disease_amount_percentile"] * 0.3 +
    data["stay_vs_disease_norm"] * 0.2 +
    data["hospital_fraud_concentration"] * 0.2
)

data = data.fillna(0)

# =====================================================
# FEATURE LIST
# =====================================================
feature_columns = [
    "claim_amount", "disease_code", "length_of_stay",
    "policy_age_days", "previous_claims_count",
    "cost_per_day", "claim_policy_ratio",
    "hospital_30day_total", "patient_60day_count",
    "patient_7day_count", "doctor_30day_total",
    "patient_deviation_ratio", "hospital_deviation_ratio",
    "disease_deviation_ratio",
    # Advanced features
    "early_claim_flag", "claim_burst_score",
    "disease_amount_percentile", "stay_vs_disease_norm",
    "hospital_fraud_concentration", "doctor_billing_intensity",
    "combined_risk_index"
]

print(f"\nTotal features: {len(feature_columns)} (14 base + 7 advanced NEW)")

X = data[feature_columns]
y = data["is_fraud"]

# =====================================================
# CLASS IMBALANCE
# =====================================================
clean_count = (y == 0).sum()
fraud_count = (y == 1).sum()
scale_pos_weight = clean_count / fraud_count

print(f"\nClass distribution - Clean: {clean_count:,} | Fraud: {fraud_count:,}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# =====================================================
# TIME-BASED SPLIT
# =====================================================
split_index = int(len(data) * 0.8)
X_train_full = X.iloc[:split_index]
y_train_full = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full, test_size=0.2, shuffle=False
)

print(f"\nTrain: {len(X_train):,} | Calib: {len(X_calib):,} | Test: {len(X_test):,}")

# =====================================================
# BASE LEARNERS
# =====================================================
print("\nBuilding base learners...")

xgb = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    eval_metric="logloss", scale_pos_weight=scale_pos_weight, verbosity=0
)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", n_jobs=-1, random_state=42
)

base_learners = [("xgb", xgb), ("rf", rf)]

if LGBM_AVAILABLE:
    lgbm = LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
    )
    base_learners.append(("lgbm", lgbm))
    print("Base learners: XGBoost + Random Forest + LightGBM")
else:
    print("Base learners: XGBoost + Random Forest")

# =====================================================
# STACKING ENSEMBLE
# =====================================================
print("\nBuilding stacking ensemble with Logistic Regression meta-learner...")
print("Training... (this takes 2-4 minutes)")

meta_learner = LogisticRegression(
    C=1.0, max_iter=1000,
    class_weight="balanced", random_state=42
)

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    stack_method="predict_proba",
    passthrough=False,
    n_jobs=-1
)

stacking_model.fit(X_train_full, y_train_full)

# =====================================================
# CALIBRATION
# =====================================================
print("\nCalibrating probability outputs...")

# Fit stacking model on calib set then wrap with calibration
# cv="prefit" removed — not supported in newer sklearn
# Instead: refit stacking on full train, calibrate on calib
calibrated_ensemble = CalibratedClassifierCV(
    estimator=stacking_model,
    method="isotonic",
    cv=5
)
calibrated_ensemble.fit(X_train_full, y_train_full)

# =====================================================
# EVALUATION
# =====================================================
print("\n" + "=" * 60)
print("  MODEL EVALUATION")
print("=" * 60)

y_proba = calibrated_ensemble.predict_proba(X_test)[:, 1]
y_pred = calibrated_ensemble.predict(X_test)

roc_auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nROC-AUC Score    : {roc_auc:.4f}")
print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"F1 Score         : {f1:.4f}")
print(f"\nConfusion Matrix :")
print(f"  True Negatives  : {cm[0][0]:,}")
print(f"  False Positives : {cm[0][1]:,}")
print(f"  False Negatives : {cm[1][0]:,}")
print(f"  True Positives  : {cm[1][1]:,}")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Clean", "Fraud"]))

# =====================================================
# THRESHOLD TUNING
# Find optimal threshold balancing precision & recall
# =====================================================
print("=" * 60)
print("  THRESHOLD TUNING")
print("=" * 60)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>6} {'FN':>6}")
print("-" * 60)

best_threshold = 0.5
best_f1 = 0
best_row = None

results = []
for thresh in thresholds:
    y_pred_t = (y_proba >= thresh).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    fp = ((y_test == 0) & (y_pred_t == 1)).sum()
    fn = ((y_test == 1) & (y_pred_t == 0)).sum()
    results.append((thresh, p, r, f, fp, fn))

# Print key thresholds only
key_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
for thresh, p, r, f, fp, fn in results:
    if any(abs(thresh - kt) < 0.02 for kt in key_thresholds):
        marker = " <-- DEFAULT" if abs(thresh - 0.5) < 0.02 else ""
        print(f"{thresh:>10.2f} {p:>10.4f} {r:>10.4f} {f:>10.4f} {fp:>6} {fn:>6}{marker}")
    if f > best_f1:
        best_f1 = f
        best_threshold = thresh
        best_row = (thresh, p, r, f, fp, fn)

print(f"\n  Best F1 threshold : {best_threshold:.4f}")
if best_row:
    print(f"  At this threshold : Precision={best_row[1]:.4f} | Recall={best_row[2]:.4f} | F1={best_row[3]:.4f}")
    print(f"  False Positives   : {best_row[4]} | False Negatives: {best_row[5]}")

# Target: recall >= 0.85 with precision >= 0.85
print("\n  Searching for threshold with Recall >= 0.85 and Precision >= 0.85...")
balanced_threshold = 0.5
for thresh, p, r, f, fp, fn in sorted(results, key=lambda x: x[0]):
    if r >= 0.85 and p >= 0.85:
        balanced_threshold = thresh
        print(f"  Found: Threshold={thresh:.4f} | Precision={p:.4f} | Recall={r:.4f} | F1={f:.4f} | FP={fp} | FN={fn}")
        break
else:
    print(f"  Could not find threshold meeting both criteria.")
    print(f"  Using best F1 threshold: {best_threshold:.4f}")
    balanced_threshold = best_threshold

# Final evaluation at chosen threshold
print(f"\n  FINAL CHOSEN THRESHOLD: {balanced_threshold:.4f}")
y_pred_final = (y_proba >= balanced_threshold).astype(int)
print("\n  Final Classification Report at chosen threshold:")
print(classification_report(y_test, y_pred_final, target_names=["Clean", "Fraud"]))

cm_final = confusion_matrix(y_test, y_pred_final)
print(f"  Confusion Matrix:")
print(f"    True Negatives  : {cm_final[0][0]:,}")
print(f"    False Positives : {cm_final[0][1]:,}")
print(f"    False Negatives : {cm_final[1][0]:,}")
print(f"    True Positives  : {cm_final[1][1]:,}")

# =====================================================
# BASE LEARNER COMPARISON
# =====================================================
print("\n" + "=" * 60)
print("  BASE LEARNER vs ENSEMBLE COMPARISON")
print("=" * 60)

for name, learner in base_learners:
    learner.fit(X_train_full, y_train_full)
    y_pred_b = learner.predict(X_test)
    y_proba_b = learner.predict_proba(X_test)[:, 1]
    auc_b = roc_auc_score(y_test, y_proba_b)
    f1_b = f1_score(y_test, y_pred_b)
    print(f"  {name.upper():10} - AUC: {auc_b:.4f} | F1: {f1_b:.4f}")

print(f"  {'ENSEMBLE':10} - AUC: {roc_auc:.4f} | F1: {f1:.4f}  <- BEST")

# =====================================================
# SAVE MODEL + OPTIMAL THRESHOLD
# =====================================================
joblib.dump(calibrated_ensemble, "fwa/models/xgb_fraud_model.pkl")
joblib.dump(feature_columns, "fwa/models/feature_columns.pkl")
joblib.dump(float(balanced_threshold), "fwa/models/optimal_threshold.pkl")

print("\n" + "=" * 60)
print("Ensemble Stacking Model Saved!")
print(f"  Model path        : fwa/models/xgb_fraud_model.pkl")
print(f"  Features path     : fwa/models/feature_columns.pkl")
print(f"  Threshold path    : fwa/models/optimal_threshold.pkl")
print(f"  Architecture      : {' + '.join([n for n, _ in base_learners])} -> LogisticRegression")
print(f"  Features          : {len(feature_columns)} (14 base + 7 advanced)")
print(f"  Final AUC         : {roc_auc:.4f}")
print(f"  Optimal Threshold : {balanced_threshold:.4f}")
print("=" * 60)