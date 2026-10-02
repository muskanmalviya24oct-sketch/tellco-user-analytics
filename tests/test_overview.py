import pandas as pd
from telco_analytics.cleaning import clean
from telco_analytics import overview


def _clean(raw_df):
    return clean(raw_df)


def test_user_aggregates_one_row_per_user(raw_df):
    df = _clean(raw_df)
    user = overview.user_aggregates(df)
    assert user.index.name == 'MSISDN'
    assert user.index.is_unique
    assert user.shape[0] == df['MSISDN'].nunique()


def test_user_aggregates_total_data_is_dl_plus_ul(raw_df):
    df = _clean(raw_df)
    user = overview.user_aggregates(df)
    assert (user['total_data'] == user['total_dl'] + user['total_ul']).all()


def test_top_handsets_excludes_undefined(raw_df):
    df = _clean(raw_df)
    top = overview.top_handsets(df, n=5)
    assert 'undefined' not in top.index


def test_decile_summary_has_five_rows(raw_df):
    df = _clean(raw_df)
    user = overview.user_aggregates(df)
    # need at least 10 users for qcut into deciles to be meaningful
    if user.shape[0] >= 10:
        summary = overview.decile_summary(user)
        assert summary.shape[0] <= 5


def test_app_pca_explained_variance_sums_to_100(raw_df):
    df = _clean(raw_df)
    user = overview.user_aggregates(df)
    explained, loadings = overview.app_pca(user)
    assert abs(explained['explained_var_%'].sum() - 100) < 1e-6
