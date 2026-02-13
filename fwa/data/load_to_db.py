import pandas as pd
from sqlalchemy import create_engine

# Load CSV
data = pd.read_csv("fwa/data/claims_data.csv")

# Create SQLite engine
engine = create_engine("sqlite:///fwa/data/fwa_claims.db")

# Push to database
data.to_sql("claims", con=engine, if_exists="replace", index=False)

print("Data successfully loaded into SQLite database!")
