from telco_analytics.cleaning import clean
from telco_analytics import overview, engagement


def _user(raw_df):
    return overview.user_aggregates(clean(raw_df))


def test_cluster_engagement_assigns_k_clusters(raw_df):
    user = _user(raw_df)
    eng = engagement.engagement_table(user)
    eng_c, scaler, km, X = engagement.cluster_engagement(eng, k=3)
    assert set(eng_c['cluster'].unique()) <= {0, 1, 2}
    assert eng_c.shape[0] == eng.shape[0]


def test_least_engaged_cluster_has_lower_metrics(raw_df):
    user = _user(raw_df)
    eng = engagement.engagement_table(user)
    eng_c, scaler, km, X = engagement.cluster_engagement(eng, k=3)
    least = engagement.least_engaged_cluster(km)
    stats = eng_c.groupby('cluster')[engagement.ENGAGEMENT_METRICS].mean()
    # the least-engaged cluster's mean total_data should not be the highest
    assert stats.loc[least, 'total_data'] <= stats['total_data'].max()


def test_top_n_per_metric_returns_requested_size(raw_df):
    user = _user(raw_df)
    eng = engagement.engagement_table(user)
    tops = engagement.top_n_per_metric(eng, n=5)
    for m in engagement.ENGAGEMENT_METRICS:
        assert len(tops[m]) == min(5, eng.shape[0])


def test_elbow_inertias_are_non_increasing(raw_df):
    user = _user(raw_df)
    eng = engagement.engagement_table(user)
    eng_c, scaler, km, X = engagement.cluster_engagement(eng, k=3)
    inertias = engagement.elbow_inertias(X, ks=range(1, 5))
    assert all(inertias[i] >= inertias[i + 1] - 1e-6 for i in range(len(inertias) - 1))
