import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000

data = pd.DataFrame({
    "claim_id": range(1, N + 1),
    "patient_id": np.random.randint(1000, 2000, N),
    "hospital_id": np.random.randint(1, 50, N),
    "doctor_id": np.random.randint(1, 100, N),
    "claim_amount": np.random.normal(50000, 15000, N).clip(5000, 200000),
    "disease_code": np.random.randint(1, 20, N),
    "length_of_stay": np.random.randint(1, 15, N),
    "policy_age_days": np.random.randint(30, 2000, N),
    "previous_claims_count": np.random.randint(0, 10, N),
    "hospital_fraud_history_score": np.random.uniform(0, 1, N)
})

# Simulate fraud logic
fraud_condition = (
    (data["claim_amount"] > 100000) |
    (data["hospital_fraud_history_score"] > 0.8) |
    (data["previous_claims_count"] > 7)
)

data["is_fraud"] = fraud_condition.astype(int)

data.to_csv("fwa/data/claims_data.csv", index=False)

print("Data generated successfully!")
