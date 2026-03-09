# This version uses CSV files instead of MySQL (for cloud deployment)
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<style>
    .title-text {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #ffd200;
        border-left: 4px solid #f7971e;
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/matches.csv')
    name_map = {
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Kings XI Punjab': 'Punjab Kings',
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Pune Warriors': 'Rising Pune Supergiant',
        'Rising Pune Supergiants': 'Rising Pune Supergiant',
    }
    for col in ['team1','team2','winner','toss_winner']:
        if col in df.columns:
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

matches = load_data()
model, team_stats, venue_stats, active_teams = load_model()

# ── HEADER ───────────────────────────────────────────────
st.markdown('<div class="title-text">🏏 IPL Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#aaa;">16 Years of IPL Data • 2008–2024 • Built by Phani Harika Soma</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Team Stats",
    "🏟️ Venue Analysis",
    "🪙 Toss Impact",
    "⚔️ Head to Head",
    "🔮 Match Insights"
])

# ── TAB 1 ────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Overall Team Performance</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Matches", len(matches))
    c2.metric("Seasons", matches['season_year'].nunique())
    c3.metric("Most Wins", matches['winner'].value_counts().index[0])
    c4.metric("Win Count", int(matches['winner'].value_counts().iloc[0]))

    st.markdown('<div class="section-header">🏆 Win Leaderboard</div>', unsafe_allow_html=True)
    teams = set(matches['team1'].tolist()+matches['team2'].tolist())
    lb = []
    for team in teams:
        played = matches[(matches['team1']==team)|(matches['team2']==team)]
        wins = (matches['winner']==team).sum()
        recent = played.tail(10)
        lb.append({
            'Team': team,
            'Played': len(played),
            'Wins': int(wins),
            'Win %': f"{wins/len(played)*100:.1f}%" if len(played)>0 else "0%",
            'Last 10': f"{int((recent['winner']==team).sum())}/10"
        })
    lb_df = pd.DataFrame(lb).sort_values('Wins', ascending=False).reset_index(drop=True)
    lb_df.index += 1
    st.dataframe(lb_df, use_container_width=True)

    st.markdown('<div class="section-header">📊 Wins by Team</div>', unsafe_allow_html=True)
    top10 = lb_df.head(10)
    fig, ax = plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#ffd200' if i==0 else '#f7971e' if i<3 else '#4a9eda' for i in range(len(top10))]
    bars = ax.barh(top10['Team'], top10['Wins'], color=colors)
    ax.tick_params(colors='white')
    ax.set_xlabel('Total Wins', color='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, top10['Wins']):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                str(val), va='center', color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 2 ────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🏟️ Venue Statistics</div>', unsafe_allow_html=True)
    vdata = []
    for v in matches['venue'].unique():
        vm = matches[matches['venue']==v]
        if len(vm) < 5: continue
        bat = (vm['result']=='runs').sum()
        chase = (vm['result']=='wickets').sum()
        vdata.append({
            'Venue': v,
            'Matches': len(vm),
            'Bat First Wins': int(bat),
            'Chase Wins': int(chase),
            'Bat First Win %': f"{bat/len(vm)*100:.1f}%",
            'Favours': '🏏 Batting' if bat>chase else '🎯 Chasing'
        })
    v_df = pd.DataFrame(vdata).sort_values('Matches', ascending=False).reset_index(drop=True)
    v_df.index += 1
    st.dataframe(v_df, use_container_width=True)

# ── TAB 3 ────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🪙 Does Winning the Toss Matter?</div>', unsafe_allow_html=True)
    toss_won = (matches['toss_winner']==matches['winner']).sum()
    toss_pct = toss_won/len(matches)*100
    c1,c2,c3 = st.columns(3)
    c1.metric("Toss Winner Won Match", int(toss_won))
    c2.metric("Total Matches", len(matches))
    c3.metric("Toss Win %", f"{toss_pct:.1f}%")

    toss_df = matches.groupby('toss_decision').apply(
        lambda x: pd.Series({
            'Matches': len(x),
            'Toss Winner Won': int((x['toss_winner']==x['winner']).sum()),
            'Win %': f"{(x['toss_winner']==x['winner']).mean()*100:.1f}%"
        })
    ).reset_index()
    st.dataframe(toss_df, use_container_width=True, hide_index=True)

    season_toss = matches.groupby('season_year').apply(
        lambda x: round((x['toss_winner']==x['winner']).mean()*100,1)
    ).reset_index()
    season_toss.columns = ['Season','Toss Win %']
    fig,ax = plt.subplots(figsize=(12,4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.plot(season_toss['Season'], season_toss['Toss Win %'],
            color='#ffd200', linewidth=2.5, marker='o')
    ax.axhline(y=50, color='#ff4444', linestyle='--', alpha=0.7)
    ax.tick_params(colors='white')
    ax.set_xlabel('Season', color='white')
    ax.set_ylabel('Toss Winner Win %', color='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 4 ────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">⚔️ Head to Head</div>', unsafe_allow_html=True)
    all_t = sorted(set(matches['team1'].tolist()+matches['team2'].tolist()))
    c1,c2 = st.columns(2)
    with c1:
        t1s = st.selectbox("Team 1", all_t)
    with c2:
        t2s = st.selectbox("Team 2", [t for t in all_t if t!=t1s])

    h2h = matches[
        ((matches['team1']==t1s)&(matches['team2']==t2s))|
        ((matches['team1']==t2s)&(matches['team2']==t1s))
    ]
    if len(h2h)==0:
        st.warning("No matches found!")
    else:
        t1w = int((h2h['winner']==t1s).sum())
        t2w = int((h2h['winner']==t2s).sum())
        c1,c2,c3 = st.columns(3)
        c1.metric(f"🔵 {t1s}", t1w)
        c2.metric("Total", len(h2h))
        c3.metric(f"🔴 {t2s}", t2w)
        st.progress(float(t1w/len(h2h)))
        st.caption(f"← {t1s} ({t1w/len(h2h):.0%}) | {t2s} ({t2w/len(h2h):.0%}) →")
        st.dataframe(
            h2h[['date','venue','winner','result','result_margin']]\
               .sort_values('date', ascending=False).head(10),
            use_container_width=True, hide_index=True
        )

# ── TAB 5 ────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔮 Match Insights</div>', unsafe_allow_html=True)
    st.info("💡 Win probability based on historical performance, head-to-head, recent form and venue stats.")

    c1,c2 = st.columns(2)
    with c1:
        team1 = st.selectbox("🔵 Team 1", active_teams, key='t1')
    with c2:
        team2 = st.selectbox("🔴 Team 2", [t for t in active_teams if t!=team1], key='t2')

    venue_sel   = st.selectbox("🏟️ Venue", sorted(venue_stats.keys()))
    toss_winner = st.radio("🪙 Toss Winner", [team1, team2])
    toss_dec    = st.radio("📋 Toss Decision", ["bat","field"])

    if st.button("🔮 Get Match Insights", type="primary"):
        t1s = team_stats.get(team1, {'win_pct':0.5,'form':0.5})
        t2s = team_stats.get(team2, {'win_pct':0.5,'form':0.5})
        vbp = venue_stats.get(venue_sel, 0.5)
        h2h_data = matches[
            ((matches['team1']==team1)&(matches['team2']==team2))|
            ((matches['team1']==team2)&(matches['team2']==team1))
        ]
        h2h_val = round((h2h_data['winner']==team1).sum()/len(h2h_data),4) if len(h2h_data)>0 else 0.5

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

        prob   = model.predict_proba(features)[0]
        t1_prob = prob[1]
        t2_prob = prob[0]

        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"### 🔵 {team1}")
            st.metric("Win Probability", f"{t1_prob*100:.1f}%")
            st.metric("Overall Win %",   f"{t1s['win_pct']*100:.1f}%")
            st.metric("Recent Form",     f"{t1s['form']*100:.0f}%")
        with c2:
            st.markdown(f"### 🔴 {team2}")
            st.metric("Win Probability", f"{t2_prob*100:.1f}%")
            st.metric("Overall Win %",   f"{t2s['win_pct']*100:.1f}%")
            st.metric("Recent Form",     f"{t2s['form']*100:.0f}%")

        fig,ax = plt.subplots(figsize=(10,2))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.barh([''], [t1_prob], color='#4a9eda', label=team1)
        ax.barh([''], [t2_prob], left=[t1_prob], color='#f7971e', label=team2)
        ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.5)
        ax.set_xlim(0,1)
        ax.set_xticks([0,0.25,0.5,0.75,1.0])
        ax.set_xticklabels(['0%','25%','50%','75%','100%'], color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if len(h2h_data)>0:
            t1w = int((h2h_data['winner']==team1).sum())
            t2w = int((h2h_data['winner']==team2).sum())
            st.markdown(f"### ⚔️ H2H: **{team1}** won **{t1w}** | **{team2}** won **{t2w}** out of **{len(h2h_data)}** matches")

        bat_pct = vbp*100
        st.markdown(f"### 🏟️ {venue_sel[:50]} — Batting first wins **{bat_pct:.1f}%** here {'🏏' if bat_pct>50 else '🎯'}")

        diff = abs(t1_prob-t2_prob)
        if diff < 0.1:
            st.warning("⚠️ Very evenly matched — could go either way!")
        elif diff < 0.2:
            st.info("📊 Slight historical edge — cricket is always unpredictable!")
        else:
            winner = team1 if t1_prob>t2_prob else team2
            st.success(f"📈 **{winner}** has a clear historical edge in this matchup!")