# 🏏 IPL Analytics Dashboard

An interactive data analytics dashboard built on 16 years of IPL data (2008–2024), featuring match insights, venue analysis, toss impact trends, head-to-head stats, and ML-powered win probability.

## 🔴 Live Demo
👉 [Click here to view the app](https://ipl-analytics-dashboard-y6umjr4v7vskctjki5dbdp.streamlit.app/)

## 📊 Features
- **Team Stats** — Win leaderboard, season champions, bar chart comparison
- **Venue Analysis** — Bat first vs chase win % across all stadiums
- **Toss Impact** — Season-wise trend of toss winner winning the match
- **Head to Head** — Any two teams, all-time record + last 10 matches
- **Match Insights** — XGBoost ML model predicting win probability based on historical data

## 🛠️ Tech Stack
- **Database:** MySQL (1,095 matches • 260,920 deliveries)
- **ML Model:** XGBoost with feature engineering
- **Frontend:** Streamlit (5-tab dashboard)
- **Language:** Python

## 📁 Project Structure
```
ipl-analytics-dashboard/
├── app_deploy.py      # Main Streamlit app (cloud version)
├── app.py             # Local version with MySQL connection
├── train.py           # XGBoost model training
├── load_data.py       # MySQL data loading script
├── data/              # IPL dataset (CSV)
├── model/             # Saved ML model files
└── requirements.txt
```

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app_deploy.py
```

## 📈 Key Insights Found
- Mumbai Indians lead all-time with highest win count
- Chinnaswamy Stadium, Bengaluru = highest bat-first win %
- Toss advantage hovers around 50% — less impactful than popularly believed

## 👩‍💻 Built By
**Phani Harika Soma** — B.Tech CSE-DS, Sridevi Women's Engineering College  
[LinkedIn](https://www.linkedin.com/in/phaniharika-soma-551a32326/) • [GitHub](https://github.com/PhaniHarika)
