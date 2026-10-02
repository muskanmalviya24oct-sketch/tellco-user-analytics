"""TellCo User Analytics Dashboard.

Run with:
    streamlit run dashboard/app.py

On first run (or whenever the source Excel file changes), use the sidebar
"Rebuild from Excel" option to compute and cache all features into the
feature store. After that, the dashboard loads instantly from the feature
store on every subsequent run.
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from telco_analytics.cleaning import load_and_clean, APPS
from telco_analytics import overview, engagement, experience, satisfaction
from telco_analytics.feature_store import FeatureStore

st.set_page_config(page_title='TellCo User Analytics', layout='wide')
sns.set_theme(style='whitegrid')

FS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'feature_store')
fs = FeatureStore(FS_ROOT)


# ---------------------------------------------------------------------------
# Data loading / caching
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_all_features(excel_path: str):
    df = load_and_clean(excel_path)

    user = overview.user_aggregates(df)

    eng = engagement.engagement_table(user)
    eng_c, eng_scaler, km_eng, X_eng = engagement.cluster_engagement(eng, k=3)
    least = engagement.least_engaged_cluster(km_eng)
    inertias = engagement.elbow_inertias(X_eng, ks=range(1, 11))

    exp = experience.experience_table(df)
    exp_c, exp_scaler, km_exp, X_exp, worst = experience.cluster_experience(exp, k=3)

    es = satisfaction.engagement_score(X_eng, km_eng, least)
    xs = satisfaction.experience_score(X_exp, km_exp, worst)
    scores = satisfaction.build_scores(eng_c.index, exp_c.index, es, xs)
    scored, km_sat = satisfaction.cluster_scores(scores, k=2)

    return df, user, eng_c, exp_c, scored, inertias


def save_to_feature_store(df, user, eng_c, exp_c, scored):
    fs.save('user_overview', user, description='Per-user session/app aggregates (Task 1)')
    fs.save('engagement_scores', eng_c, description='Per-user engagement metrics + cluster (Task 2)')
    fs.save('experience_scores', exp_c, description='Per-user experience metrics + cluster (Task 3)')
    fs.save('satisfaction_scores', scored, description='Per-user engagement/experience/satisfaction scores (Task 4)')


def load_from_feature_store():
    user = fs.load('user_overview')
    eng_c = fs.load('engagement_scores')
    exp_c = fs.load('experience_scores')
    scored = fs.load('satisfaction_scores')
    return user, eng_c, exp_c, scored


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title('TellCo Dashboard')
st.sidebar.caption('Nexthikes IT Solutions — User Analytics in the Telecommunications Industry')

excel_path = st.sidebar.text_input('Raw Excel file path', value='telcom_data.xlsx')
rebuild = st.sidebar.button('Rebuild from Excel (slow, ~1-2 min)')

have_cache = fs.exists('satisfaction_scores')

if rebuild:
    if not os.path.exists(excel_path):
        st.sidebar.error(f"File not found: {excel_path}")
        st.stop()
    df, user, eng_c, exp_c, scored, inertias = build_all_features(excel_path)
    save_to_feature_store(df, user, eng_c, exp_c, scored)
    st.sidebar.success('Rebuilt and saved to feature store.')
elif have_cache:
    user, eng_c, exp_c, scored = load_from_feature_store()
    df = None
    inertias = None
else:
    st.sidebar.warning('No cached features yet — click "Rebuild from Excel" to compute them.')
    st.stop()

page = st.sidebar.radio('Section', ['Overview', 'Engagement', 'Experience', 'Satisfaction'])

st.sidebar.markdown('---')
st.sidebar.caption(f'Users in feature store: {user.shape[0]:,}')


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == 'Overview':
    st.header('Task 1 — User Overview')

    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Top 10 handsets')
            st.bar_chart(overview.top_handsets(df))
        with col2:
            st.subheader('Top 3 manufacturers')
            st.bar_chart(overview.top_manufacturers(df))
    else:
        st.info('Rebuild from Excel to see handset breakdowns (not cached in the feature store).')

    st.subheader('Per-user aggregates')
    st.dataframe(user.head(20))

    st.subheader('Total data volume per application')
    app_cols = [f'{a}_bytes' for a in APPS]
    app_totals = user[app_cols].sum().sort_values(ascending=False)
    app_totals.index = [c.replace('_bytes', '') for c in app_totals.index]
    st.bar_chart(app_totals)

    st.subheader('Correlation between applications')
    corr = overview.app_correlation(user)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

elif page == 'Engagement':
    st.header('Task 2 — User Engagement')

    st.subheader('Engagement clusters (k=3)')
    cluster_means = eng_c.groupby('cluster')[engagement.ENGAGEMENT_METRICS].mean()
    st.dataframe(cluster_means)
    st.bar_chart(cluster_means)

    st.subheader('Top 10 users by total data')
    st.dataframe(eng_c['total_data'].nlargest(10))

    if inertias is not None:
        st.subheader('Elbow method (k = 1 to 10)')
        st.line_chart(pd.Series(inertias, index=range(1, 11), name='inertia'))

elif page == 'Experience':
    st.header('Task 3 — User Experience')

    st.subheader('Experience clusters (k=3)')
    cluster_means = exp_c.groupby('cluster')[experience.EXPERIENCE_FEATURES].mean()
    st.dataframe(cluster_means)
    st.bar_chart(cluster_means)

    st.subheader('Average throughput by handset (top 10 handsets)')
    summary = experience.per_handset_summary(exp_c)
    st.bar_chart(summary['mean_tput'].sort_values(ascending=False))

    st.subheader('Average TCP retransmission by handset (top 10 handsets)')
    st.bar_chart(summary['mean_tcp'].sort_values(ascending=False))

else:  # Satisfaction
    st.header('Task 4 — Satisfaction Analysis')

    st.subheader('Top 10 satisfied customers')
    st.dataframe(satisfaction.top_satisfied(scored, n=10))

    st.subheader('Average scores per satisfaction cluster')
    st.dataframe(satisfaction.cluster_averages(scored))

    st.subheader('Engagement vs experience score')
    st.scatter_chart(scored.sample(min(3000, len(scored)), random_state=1),
                      x='engagement_score', y='experience_score', color='sat_cluster')
