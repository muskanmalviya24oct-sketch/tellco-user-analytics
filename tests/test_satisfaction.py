from telco_analytics.cleaning import clean
from telco_analytics import overview, engagement, experience, satisfaction


def _scores(raw_df):
    df = clean(raw_df)
    user = overview.user_aggregates(df)
    eng = engagement.engagement_table(user)
    eng_c, eng_scaler, km_eng, X_eng = engagement.cluster_engagement(eng, k=3)
    least = engagement.least_engaged_cluster(km_eng)

    exp = experience.experience_table(df)
    exp_c, exp_scaler, km_exp, X_exp, worst = experience.cluster_experience(exp, k=3)

    es = satisfaction.engagement_score(X_eng, km_eng, least)
    xs = satisfaction.experience_score(X_exp, km_exp, worst)
    return satisfaction.build_scores(eng_c.index, exp_c.index, es, xs)


def test_satisfaction_score_is_average_of_components(raw_df):
    scores = _scores(raw_df)
    expected = (scores['engagement_score'] + scores['experience_score']) / 2
    assert (scores['satisfaction_score'] - expected).abs().max() < 1e-9


def test_scores_are_non_negative(raw_df):
    scores = _scores(raw_df)
    assert (scores['engagement_score'] >= 0).all()
    assert (scores['experience_score'] >= 0).all()


def test_top_satisfied_is_sorted_descending(raw_df):
    scores = _scores(raw_df)
    top = satisfaction.top_satisfied(scores, n=5)
    assert list(top['satisfaction_score']) == sorted(top['satisfaction_score'], reverse=True)


def test_cluster_scores_assigns_two_clusters(raw_df):
    scores = _scores(raw_df)
    scored, km = satisfaction.cluster_scores(scores, k=2)
    assert set(scored['sat_cluster'].unique()) <= {0, 1}
