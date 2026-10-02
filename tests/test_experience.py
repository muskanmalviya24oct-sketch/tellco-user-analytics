from telco_analytics.cleaning import clean
from telco_analytics import experience


def _exp(raw_df):
    return experience.experience_table(clean(raw_df))


def test_experience_table_one_row_per_user(raw_df):
    df = clean(raw_df)
    exp = experience.experience_table(df)
    assert exp.index.is_unique
    assert exp.shape[0] == df['MSISDN'].nunique()


def test_experience_table_has_expected_columns(raw_df):
    exp = _exp(raw_df)
    for col in ['avg_tcp', 'avg_rtt', 'avg_tput', 'handset']:
        assert col in exp.columns


def test_cluster_experience_returns_valid_worst_index(raw_df):
    exp = _exp(raw_df)
    exp_c, scaler, km, X, worst = experience.cluster_experience(exp, k=3)
    assert worst in {0, 1, 2}
    assert exp_c.shape[0] == exp.shape[0]


def test_top_bottom_frequent_shape(raw_df):
    exp = _exp(raw_df)
    out = experience.top_bottom_frequent(exp['avg_tcp'], n=5)
    assert out.shape[0] == 5
    assert set(out.columns) == {'top', 'bottom', 'most_frequent_value', 'frequency'}
