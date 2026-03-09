import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

engine = create_engine('mysql+pymysql://root:Ammulu%40123@localhost/ipl_analytics')

print("📥 Loading data from MySQL...")
matches = pd.read_sql("SELECT * FROM matches ORDER BY date", engine)
matches['date'] = pd.to_datetime(matches['date'])
matches = matches.sort_values('date').reset_index(drop=True)

# ── FIX 1: Standardize team names ───────────────────────
team_name_map = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Pune Warriors': 'Rising Pune Supergiant',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
}

for col in ['team1', 'team2', 'winner', 'toss_winner']:
    matches[col] = matches[col].replace(team_name_map)

# ── FIX 2: Clean result ──────────────────────────────────
matches = matches[matches['winner'].notna()]
matches = matches[matches['winner'] != '']
matches = matches[~matches['result'].isin(['no result', 'tie'])]
matches = matches.reset_index(drop=True)
print(f"   {len(matches)} valid matches after cleaning")

# ── FIX 3: Season as numeric ─────────────────────────────
def parse_season(s):
    return int(str(s)[:4])

matches['season_year'] = matches['season'].apply(parse_season)

# ── FEATURE ENGINEERING ──────────────────────────────────
print("\n⚙️  Building features (this takes ~2 mins)...")

rows = []
for idx, row in matches.iterrows():
    t1, t2 = row['team1'], row['team2']
    past = matches.iloc[:idx]

    def win_pct(team, data):
        played = data[(data['team1']==team) | (data['team2']==team)]
        if len(played) < 3:
            return 0.5
        return round((played['winner']==team).sum() / len(played), 4)

    def recent_form(team, data, n=5):
        played = data[(data['team1']==team) | (data['team2']==team)].tail(n)
        if len(played) < 2:
            return 0.5
        return round((played['winner']==team).sum() / len(played), 4)

    def h2h(t1, t2, data):
        played = data[
            ((data['team1']==t1) & (data['team2']==t2)) |
            ((data['team1']==t2) & (data['team2']==t1))
        ]
        if len(played) < 2:
            return 0.5
        return round((played['winner']==t1).sum() / len(played), 4)

    def venue_bat_pct(venue, data):
        v = data[data['venue']==venue]
        if len(v) < 3:
            return 0.5
        return round((v['result']=='runs').sum() / len(v), 4)

    t1_wp  = win_pct(t1, past)
    t2_wp  = win_pct(t2, past)
    t1_f   = recent_form(t1, past)
    t2_f   = recent_form(t2, past)
    h2h_p  = h2h(t1, t2, past)
    vbp    = venue_bat_pct(row['venue'], past)
    toss1  = int(row['toss_winner'] == t1)
    bat    = int(row['toss_decision'] == 'bat')

    rows.append({
        't1_win_pct'   : t1_wp,
        't2_win_pct'   : t2_wp,
        't1_form'      : t1_f,
        't2_form'      : t2_f,
        'h2h_pct'      : h2h_p,
        'venue_bat_pct': vbp,
        'toss_team1'   : toss1,
        'chose_bat'    : bat,
        'win_pct_diff' : round(t1_wp - t2_wp, 4),
        'form_diff'    : round(t1_f  - t2_f,  4),
        'season_year'  : row['season_year'],
        'target'       : int(row['winner'] == t1)
    })

    if idx % 150 == 0 and idx > 0:
        print(f"   {idx}/{len(matches)} matches processed...")

df = pd.DataFrame(rows)
print(f"   Done! {len(df)} feature rows built")

# ── CHECK TARGET BALANCE ─────────────────────────────────
print(f"\n   Target balance — Team1 wins: {df['target'].mean():.1%} | Team2 wins: {(1-df['target']).mean():.1%}")

feature_cols = [
    't1_win_pct','t2_win_pct',
    't1_form','t2_form',
    'h2h_pct','venue_bat_pct',
    'toss_team1','chose_bat',
    'win_pct_diff','form_diff',
    'season_year'
]

X = df[feature_cols]
y = df['target']

# Time-based split — train on old, test on recent
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"\n🔀 Train: {len(X_train)} | Test: {len(X_test)}")

# ── TRAIN ────────────────────────────────────────────────
print("\n🚀 Training XGBoost...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# ── EVALUATE ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred, target_names=['Team2 wins','Team1 wins']))

# ── FEATURE IMPORTANCE ───────────────────────────────────
imp = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n📈 Feature Importance:")
print(imp.to_string(index=False))

# ── SAVE EVERYTHING ──────────────────────────────────────
print("\n💾 Saving model and stats...")

with open('model/ipl_model.pkl','wb') as f:
    pickle.dump(model, f)
with open('model/feature_cols.pkl','wb') as f:
    pickle.dump(feature_cols, f)

# Team stats for app
team_stats = {}
all_teams = set(matches['team1'].tolist() + matches['team2'].tolist())
for team in all_teams:
    played = matches[(matches['team1']==team)|(matches['team2']==team)]
    if len(played) == 0:
        continue
    wins = (played['winner']==team).sum()
    recent = played.tail(5)
    recent_wins = (recent['winner']==team).sum()
    team_stats[team] = {
        'win_pct': round(wins/len(played), 4),
        'form'   : round(recent_wins/len(recent), 4) if len(recent)>0 else 0.5
    }

with open('model/team_stats.pkl','wb') as f:
    pickle.dump(team_stats, f)

# Venue stats for app
venue_stats = {}
for venue in matches['venue'].unique():
    v = matches[matches['venue']==venue]
    venue_stats[venue] = round((v['result']=='runs').sum()/len(v), 4)

with open('model/venue_stats.pkl','wb') as f:
    pickle.dump(venue_stats, f)

# Active teams list for app dropdown
active_teams = sorted([
    'Mumbai Indians', 'Chennai Super Kings',
    'Royal Challengers Bengaluru', 'Kolkata Knight Riders',
    'Delhi Capitals', 'Punjab Kings', 'Rajasthan Royals',
    'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
])
with open('model/active_teams.pkl','wb') as f:
    pickle.dump(active_teams, f)

print("✅ Everything saved! Ready to build app.py")