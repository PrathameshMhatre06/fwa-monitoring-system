import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
import joblib

# Connect to DB
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")

# Load data
data = pd.read_sql("SELECT * FROM claims", engine)

# Use only feature columns (drop IDs and label)
X = data.drop(columns=["claim_id", "patient_id", "is_fraud"])

# Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X)

# Save model
joblib.dump(model, "fwa/models/isolation_forest.pkl")

print("Anomaly model trained and saved successfully!")
