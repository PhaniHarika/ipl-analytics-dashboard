import pandas as pd
from sqlalchemy import create_engine

# ⚠️ Replace 'yourpassword' with your MySQL root password
engine = create_engine('mysql+pymysql://root:Ammulu%40123@localhost/ipl_analytics')

# Load CSVs
print("Loading matches...")
matches = pd.read_csv("C:/Users/PhaniHarikaSoma24/ipl-match-predictor/data/matches.csv")
print(f"  Found {len(matches)} matches")

print("Loading deliveries...")
deliveries = pd.read_csv("C:/Users/PhaniHarikaSoma24/ipl-match-predictor/data/deliveries.csv")
print(f"  Found {len(deliveries)} deliveries")

# Fix column name if needed
if 'over' in deliveries.columns:
    deliveries = deliveries.rename(columns={'over': 'over_num'})

# Push to MySQL
print("Pushing to MySQL...")
matches.to_sql('matches', engine, if_exists='replace', index=False)
deliveries.to_sql('deliveries', engine, if_exists='replace', index=False)

print("✅ Done! Data loaded successfully.")
print(f"   Matches table: {len(matches)} rows")
print(f"   Deliveries table: {len(deliveries)} rows")