import streamlit as st
import pandas as pd
import pickle
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide"
)

# ── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .title-text {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #ffd200;
        border-left: 4px solid #f7971e;
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── DB CONNECTION (cached) ───────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine('mysql+pymysql://root:Ammulu%40123@localhost/ipl_analytics')

@st.cache_data
def load_matches():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM matches", engine)
    # Standardize team names
    name_map = {
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Kings XI Punjab': 'Punjab Kings',
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Pune Warriors': 'Rising Pune Supergiant',
        'Rising Pune Supergiants': 'Rising Pune Supergiant',
    }
    for col in ['team1','team2','winner','toss_winner']:
        df[col] = df[col].replace(name_map)
    df = df[df['winner'].notna() & (df['winner'] != '')]
    df = df[~df['result'].isin(['no result'])]
    df['date'] = pd.to_datetime(df['date'])
    df['season_year'] = df['season'].apply(lambda x: int(str(x)[:4]))
    return df

@st.cache_resource
def load_model():
    with open('model/ipl_model.pkl','rb') as f:
        model = pickle.load(f)
    with open('model/team_stats.pkl','rb') as f:
        team_stats = pickle.load(f)
    with open('model/venue_stats.pkl','rb') as f:
        venue_stats = pickle.load(f)
    with open('model/active_teams.pkl','rb') as f:
        active_teams = pickle.load(f)
    return model, team_stats, venue_stats, active_teams

matches = load_matches()
model, team_stats, venue_stats, active_teams = load_model()

# ── HEADER ───────────────────────────────────────────────
st.markdown('<div class="title-text">🏏 IPL Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#aaa;">16 Years of IPL Data • 2008–2024 • Built by Phani Harika</p>', unsafe_allow_html=True)
st.markdown("---")

# ── NAVIGATION ───────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Team Stats",
    "🏟️ Venue Analysis",
    "🪙 Toss Impact",
    "⚔️ Head to Head",
    "🔮 Match Insights"
])

# ════════════════════════════════════════════════════════
# TAB 1 — TEAM STATS
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Overall Team Performance</div>', unsafe_allow_html=True)

    # Top metrics
    total_matches = len(matches)
    total_seasons = matches['season_year'].nunique()
    top_team = matches['winner'].value_counts().index[0]
    top_wins  = matches['winner'].value_counts().iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches", f"{total_matches:,}")
    c2.metric("Seasons", total_seasons)
    c3.metric("Most Successful Team", top_team)
    c4.metric("Most Wins", top_wins)

    st.markdown("---")

    # Win leaderboard
    st.markdown('<div class="section-header">🏆 Win Leaderboard</div>', unsafe_allow_html=True)

    all_teams_list = set(matches['team1'].tolist() + matches['team2'].tolist())
    leaderboard = []
    for team in all_teams_list:
        played = matches[(matches['team1']==team)|(matches['team2']==team)]
        wins   = (matches['winner']==team).sum()
        recent = played.tail(10)
        recent_wins = (recent['winner']==team).sum()
        leaderboard.append({
            'Team': team,
            'Matches Played': len(played),
            'Wins': int(wins),
            'Win %': f"{wins/len(played)*100:.1f}%" if len(played)>0 else "0%",
            'Last 10 Form': f"{recent_wins}/10"
        })

    lb_df = pd.DataFrame(leaderboard).sort_values('Wins', ascending=False).reset_index(drop=True)
    lb_df.index += 1
    st.dataframe(lb_df, use_container_width=True)

    # Bar chart
    st.markdown('<div class="section-header">📊 Wins by Team</div>', unsafe_allow_html=True)
    top10 = lb_df.head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#ffd200' if i == 0 else '#f7971e' if i < 3 else '#4a9eda'
              for i in range(len(top10))]
    bars = ax.barh(top10['Team'], top10['Wins'], color=colors, edgecolor='none')
    ax.set_xlabel('Total Wins', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, top10['Wins']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Season wise wins
    st.markdown('<div class="section-header">📈 Season-wise Champions</div>', unsafe_allow_html=True)
    season_winners = matches.groupby('season_year')['winner'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    season_winners.columns = ['Season', 'Champion']
    st.dataframe(season_winners, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 2 — VENUE ANALYSIS
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🏟️ Venue Statistics</div>', unsafe_allow_html=True)

    venue_data = []
    for venue in matches['venue'].unique():
        v = matches[matches['venue']==venue]
        bat_wins   = (v['result']=='runs').sum()
        chase_wins = (v['result']=='wickets').sum()
        total = len(v)
        if total < 5:
            continue
        venue_data.append({
            'Venue': venue,
            'Matches': total,
            'Bat First Wins': int(bat_wins),
            'Chase Wins': int(chase_wins),
            'Bat First Win %': f"{bat_wins/total*100:.1f}%",
            'Favours': '🏏 Batting' if bat_wins > chase_wins else '🎯 Chasing'
        })

    v_df = pd.DataFrame(venue_data).sort_values('Matches', ascending=False).reset_index(drop=True)
    v_df.index += 1

    # Filter
    col1, col2 = st.columns([1, 3])
    with col1:
        favour_filter = st.radio("Filter by", ["All", "Batting Friendly", "Chasing Friendly"])
    if favour_filter == "Batting Friendly":
        v_df = v_df[v_df['Favours']=='🏏 Batting']
    elif favour_filter == "Chasing Friendly":
        v_df = v_df[v_df['Favours']=='🎯 Chasing']

    st.dataframe(v_df, use_container_width=True)

    # Top 10 venues chart
    st.markdown('<div class="section-header">Top 10 Venues — Batting vs Chasing</div>', unsafe_allow_html=True)
    top_v = pd.DataFrame(venue_data).sort_values('Matches', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    x = range(len(top_v))
    w = 0.35
    ax.bar([i - w/2 for i in x], top_v['Bat First Wins'], w, label='Bat First Wins', color='#f7971e')
    ax.bar([i + w/2 for i in x], top_v['Chase Wins'],     w, label='Chase Wins',     color='#4a9eda')
    ax.set_xticks(list(x))
    ax.set_xticklabels([v[:15]+'...' if len(v)>15 else v for v in top_v['Venue']],
                       rotation=45, ha='right', color='white', fontsize=8)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════
# TAB 3 — TOSS IMPACT
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🪙 Does the Toss Really Matter?</div>', unsafe_allow_html=True)

    toss_won_match = (matches['toss_winner'] == matches['winner']).sum()
    total = len(matches)
    toss_pct = toss_won_match / total * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Toss Winner Also Won Match", f"{toss_won_match}")
    c2.metric("Total Matches", f"{total}")
    c3.metric("Toss Winner Win %", f"{toss_pct:.1f}%",
              delta="Above 50% = toss matters" if toss_pct > 50 else "Below 50% = toss doesn't matter")

    # By decision
    st.markdown('<div class="section-header">By Toss Decision</div>', unsafe_allow_html=True)
    toss_df = matches.groupby('toss_decision').apply(
        lambda x: pd.Series({
            'Total Matches': len(x),
            'Toss Winner Won': (x['toss_winner']==x['winner']).sum(),
            'Win %': f"{(x['toss_winner']==x['winner']).mean()*100:.1f}%"
        })
    ).reset_index()
    st.dataframe(toss_df, use_container_width=True, hide_index=True)

    # Season wise toss impact
    st.markdown('<div class="section-header">📈 Toss Impact by Season</div>', unsafe_allow_html=True)
    season_toss = matches.groupby('season_year').apply(
        lambda x: round((x['toss_winner']==x['winner']).mean()*100, 1)
    ).reset_index()
    season_toss.columns = ['Season', 'Toss Winner Win %']

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.plot(season_toss['Season'], season_toss['Toss Winner Win %'],
            color='#ffd200', linewidth=2.5, marker='o', markersize=6)
    ax.axhline(y=50, color='#ff4444', linestyle='--', alpha=0.7, label='50% baseline')
    ax.fill_between(season_toss['Season'], season_toss['Toss Winner Win %'], 50,
                    where=season_toss['Toss Winner Win %']>50,
                    alpha=0.2, color='#ffd200')
    ax.set_xlabel('Season', color='white')
    ax.set_ylabel('Toss Winner Win %', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════
# TAB 4 — HEAD TO HEAD
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">⚔️ Head to Head Record</div>', unsafe_allow_html=True)

    all_teams_sorted = sorted(set(matches['team1'].tolist() + matches['team2'].tolist()))
    col1, col2 = st.columns(2)
    with col1:
        t1_sel = st.selectbox("Team 1", all_teams_sorted, index=0)
    with col2:
        t2_sel = st.selectbox("Team 2", [t for t in all_teams_sorted if t != t1_sel], index=0)

    h2h = matches[
        ((matches['team1']==t1_sel) & (matches['team2']==t2_sel)) |
        ((matches['team1']==t2_sel) & (matches['team2']==t1_sel))
    ]

    if len(h2h) == 0:
        st.warning("No matches found between these teams!")
    else:
        t1_wins = (h2h['winner']==t1_sel).sum()
        t2_wins = (h2h['winner']==t2_sel).sum()
        total_h2h = len(h2h)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"🔵 {t1_sel} Wins", t1_wins)
        c2.metric("Total Matches", total_h2h)
        c3.metric(f"🔴 {t2_sel} Wins", t2_wins)

        # Progress bar style win comparison
        t1_pct = t1_wins / total_h2h
        st.markdown(f"**{t1_sel}** dominates by **{t1_pct:.0%}**" if t1_pct > 0.5
                    else f"**{t2_sel}** dominates by **{1-t1_pct:.0%}**")
        st.progress(float(t1_pct))
        st.caption(f"← {t1_sel} ({t1_pct:.0%}) | {t2_sel} ({1-t1_pct:.0%}) →")

        # Recent matches
        st.markdown('<div class="section-header">Recent Matches</div>', unsafe_allow_html=True)
        recent_h2h = h2h[['date','venue','toss_winner','toss_decision','winner','result','result_margin']]\
                        .sort_values('date', ascending=False).head(10)
        st.dataframe(recent_h2h, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 5 — MATCH INSIGHTS
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🔮 Match Insights — Historical Win Probability</div>', unsafe_allow_html=True)
    st.info("💡 This shows win probability based on historical performance, head-to-head record, recent form, and venue stats — not a live predictor.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("🔵 Select Team 1", active_teams, key='t1')
    with col2:
        team2 = st.selectbox("🔴 Select Team 2",
                             [t for t in active_teams if t != team1], key='t2')

    venue_sel  = st.selectbox("🏟️ Venue", sorted(venue_stats.keys()))
    toss_winner = st.radio("🪙 Toss Winner", [team1, team2])
    toss_dec    = st.radio("📋 Toss Decision", ["bat", "field"])

    if st.button("🔮 Get Match Insights", type="primary"):

        # Build feature vector
        t1s = team_stats.get(team1, {'win_pct':0.5,'form':0.5})
        t2s = team_stats.get(team2, {'win_pct':0.5,'form':0.5})
        vbp = venue_stats.get(venue_sel, 0.5)

        # H2H from data
        h2h_data = matches[
            ((matches['team1']==team1)&(matches['team2']==team2))|
            ((matches['team1']==team2)&(matches['team2']==team1))
        ]
        if len(h2h_data) > 0:
            h2h_val = round((h2h_data['winner']==team1).sum()/len(h2h_data), 4)
        else:
            h2h_val = 0.5

        import numpy as np
        features = pd.DataFrame([{
            't1_win_pct'   : t1s['win_pct'],
            't2_win_pct'   : t2s['win_pct'],
            't1_form'      : t1s['form'],
            't2_form'      : t2s['form'],
            'h2h_pct'      : h2h_val,
            'venue_bat_pct': vbp,
            'toss_team1'   : int(toss_winner==team1),
            'chose_bat'    : int(toss_dec=='bat'),
            'win_pct_diff' : round(t1s['win_pct']-t2s['win_pct'],4),
            'form_diff'    : round(t1s['form']-t2s['form'],4),
            'season_year'  : 2024
        }])

        prob = model.predict_proba(features)[0]
        t1_prob = prob[1]
        t2_prob = prob[0]

        # Results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 🔵 {team1}")
            st.metric("Win Probability", f"{t1_prob*100:.1f}%")
            st.metric("Overall Win %", f"{t1s['win_pct']*100:.1f}%")
            st.metric("Recent Form (last 5)", f"{t1s['form']*100:.0f}%")
        with col2:
            st.markdown(f"### 🔴 {team2}")
            st.metric("Win Probability", f"{t2_prob*100:.1f}%")
            st.metric("Overall Win %", f"{t2s['win_pct']*100:.1f}%")
            st.metric("Recent Form (last 5)", f"{t2s['form']*100:.0f}%")

        # Win prob bar
        st.markdown("### 📊 Win Probability Comparison")
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.barh([''], [t1_prob], color='#4a9eda', label=team1)
        ax.barh([''], [t2_prob], left=[t1_prob], color='#f7971e', label=team2)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.5)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', loc='upper right')
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%','25%','50%','75%','100%'], color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # H2H summary
        st.markdown(f"### ⚔️ Historical H2H: {team1} vs {team2}")
        if len(h2h_data) > 0:
            t1w = (h2h_data['winner']==team1).sum()
            t2w = (h2h_data['winner']==team2).sum()
            st.markdown(f"**{team1}** won **{t1w}** out of **{len(h2h_data)}** matches against **{team2}** ({t2w} wins)")
        else:
            st.markdown("No previous meetings between these teams!")

        # Venue info
        st.markdown(f"### 🏟️ Venue Insight: {venue_sel[:40]}")
        bat_pct = vbp * 100
        st.markdown(f"Teams batting first win **{bat_pct:.1f}%** of matches at this venue — "
                    f"{'favours batting first 🏏' if bat_pct > 50 else 'favours chasing 🎯'}")

        # Confidence note
        diff = abs(t1_prob - t2_prob)
        if diff < 0.1:
            st.warning("⚠️ Very evenly matched! Historical data suggests this could go either way.")
        elif diff < 0.2:
            st.info("📊 Slight edge based on historical data, but cricket is unpredictable!")
        else:
            winner = team1 if t1_prob > t2_prob else team2
            st.success(f"📈 Historical data gives **{winner}** a clear edge in this matchup!")