import pandas as pd
import joblib
import shap
from sqlalchemy import create_engine
import numpy as np

# Connect to DB
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")

# Load model
model = joblib.load("fwa/models/xgb_fraud_model.pkl")

# Load one sample claim
data = pd.read_sql("SELECT * FROM claims LIMIT 1", engine)

X = data.drop(columns=["claim_id", "patient_id", "is_fraud"])

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

feature_names = X.columns
shap_vals = shap_values[0]

# Combine feature + value
feature_impact = list(zip(feature_names, shap_vals))

# Sort by absolute impact
feature_impact_sorted = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)

print("Top 3 Feature Drivers for This Claim:\n")

for feature, value in feature_impact_sorted[:3]:
    direction = "Increased Fraud Risk" if value > 0 else "Decreased Fraud Risk"
    print(f"{feature} → {value:.4f} ({direction})")
