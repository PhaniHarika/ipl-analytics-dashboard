import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:Ammulu%40123@localhost/ipl_analytics')

matches = pd.read_sql("SELECT * FROM matches", engine)

# Check 1: What does result column contain?
print("=== RESULT column unique values ===")
print(matches['result'].value_counts())

# Check 2: What does season look like?
print("\n=== SEASON column sample ===")
print(matches['season'].head(10))
print("Season dtype:", matches['season'].dtype)

# Check 3: Sample of winners
print("\n=== Sample rows ===")
print(matches[['team1','team2','winner','result','result_margin','toss_winner','toss_decision']].head(10).to_string())

# Check 4: How many matches per season?
print("\n=== Matches per season ===")
print(matches['season'].value_counts().sort_index())

# Check 5: Any nulls?
print("\n=== Null counts ===")
print(matches[['winner','result','toss_winner','toss_decision']].isnull().sum())