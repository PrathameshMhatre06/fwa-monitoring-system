import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# Connect to SQLite DB
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")

# Load data
data = pd.read_sql("SELECT * FROM claims", engine)

# Drop IDs (not useful for training)
X = data.drop(columns=["claim_id", "patient_id", "is_fraud"])
y = data["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # handle imbalance
    use_label_encoder=False,
    eval_metric="logloss"
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(model, "fwa/models/xgb_fraud_model.pkl")

print("Model trained and saved successfully!")
