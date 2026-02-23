import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000

# =====================================================
# REALISTIC INDIAN HEALTH INSURANCE DATA
# Fraud rate: ~10% (industry estimate 8-15%)
# Claim amounts: realistic INR ranges
# Policy ages: realistic spread
# =====================================================

all_dates = pd.date_range(start="2022-01-01", periods=N, freq="8h")
shuffled_dates = pd.Series(all_dates).sample(frac=1, random_state=42).values

# =====================================================
# CLEAN CLAIMS — 90% of total
# Realistic: routine hospitalizations, day surgeries
# =====================================================
clean_n = 4500
clean = pd.DataFrame({
    "claim_id": range(1, clean_n + 1),
    "patient_id": np.random.randint(1000, 5000, clean_n),
    "hospital_id": np.random.randint(1, 50, clean_n),
    "doctor_id": np.random.randint(1, 100, clean_n),
    # Realistic claim amounts — ₹15k to ₹3L (routine hospitalization)
    "claim_amount": np.random.lognormal(
        mean=np.log(60000), sigma=0.7, size=clean_n
    ).clip(15000, 300000),
    "disease_code": np.random.randint(1, 20, clean_n),
    # Short stays — routine treatment
    "length_of_stay": np.random.choice(
        [1, 2, 3, 4, 5, 6, 7], clean_n,
        p=[0.25, 0.25, 0.20, 0.15, 0.08, 0.04, 0.03]
    ),
    # Mature policies — most policyholders have held policy 1-5 years
    "policy_age_days": np.random.randint(180, 3650, clean_n),
    "previous_claims_count": np.random.choice(
        [0, 1, 2, 3], clean_n, p=[0.55, 0.28, 0.12, 0.05]
    ),
    "claim_date": shuffled_dates[:clean_n],
    "is_fraud": 0
})

# =====================================================
# FRAUD CLAIMS — 10% of total
# Inflated amounts, early policy claims, long stays
# =====================================================
fraud_n = N - clean_n  # 500
fraud = pd.DataFrame({
    "claim_id": range(clean_n + 1, N + 1),
    "patient_id": np.random.randint(1000, 5000, fraud_n),
    "hospital_id": np.random.randint(1, 50, fraud_n),
    "doctor_id": np.random.randint(1, 100, fraud_n),
    # Inflated amounts — ₹2L to ₹8L (upcoded, ghost claims)
    "claim_amount": np.random.lognormal(
        mean=np.log(350000), sigma=0.5, size=fraud_n
    ).clip(150000, 800000),
    "disease_code": np.random.randint(1, 20, fraud_n),
    # Unusually long stays — padding
    "length_of_stay": np.random.randint(10, 25, fraud_n),
    # Early policy claims — red flag (within 6 months of policy)
    "policy_age_days": np.random.randint(15, 180, fraud_n),
    # High previous claims — repeat offenders
    "previous_claims_count": np.random.randint(4, 10, fraud_n),
    "claim_date": shuffled_dates[clean_n:],
    "is_fraud": 1
})

# =====================================================
# COMBINE, SHUFFLE AND ADD NOISE
# =====================================================
data = pd.concat([clean, fraud], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data["claim_id"] = range(1, N + 1)

# Realistic noise — flip ~3% labels (not 5% — real world has cleaner labels)
flip_clean = data[data["is_fraud"] == 0].sample(frac=0.03, random_state=42).index
flip_fraud = data[data["is_fraud"] == 1].sample(frac=0.03, random_state=42).index
data.loc[flip_clean, "is_fraud"] = 1
data.loc[flip_fraud, "is_fraud"] = 0

# Sort by date for time-window feature engineering
data = data.sort_values("claim_date").reset_index(drop=True)
data["claim_id"] = range(1, N + 1)

# =====================================================
# SAVE
# =====================================================
data.to_csv("fwa/data/claims_data.csv", index=False)

print(f"✅ Realistic data generated successfully!")
print(f"Total claims  : {N}")
print(f"Fraud cases   : {data['is_fraud'].sum()} ({data['is_fraud'].mean()*100:.1f}%)")
print(f"Clean cases   : {(data['is_fraud'] == 0).sum()}")
print(f"Avg clean amt : ₹{data[data['is_fraud']==0]['claim_amount'].mean():,.0f}")
print(f"Avg fraud amt : ₹{data[data['is_fraud']==1]['claim_amount'].mean():,.0f}")
print(f"Avg clean stay: {data[data['is_fraud']==0]['length_of_stay'].mean():.1f} days")
print(f"Avg fraud stay: {data[data['is_fraud']==1]['length_of_stay'].mean():.1f} days")

# Verify fraud distribution across time splits
split = int(N * 0.8)
train_fraud_rate = data.iloc[:split]["is_fraud"].mean() * 100
test_fraud_rate = data.iloc[split:]["is_fraud"].mean() * 100
print(f"Train fraud rate : {train_fraud_rate:.1f}%")
print(f"Test fraud rate  : {test_fraud_rate:.1f}%")