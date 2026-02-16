import sqlite3
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "fwa", "data", "fwa_claims.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS fraud_cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    risk_score REAL,
    risk_tier TEXT,
    status TEXT,
    assigned_to TEXT,
    created_date TEXT,
    closed_date TEXT
)
""")

conn.commit()
conn.close()

print("✅ Fraud Case Table Created Successfully!")
