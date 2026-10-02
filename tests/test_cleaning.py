import numpy as np
from telco_analytics.cleaning import clean, NUM_COLS


def test_drops_trailing_summary_row(raw_df):
    df = clean(raw_df)
    assert df.shape[0] == raw_df.shape[0] - 1


def test_no_missing_in_numeric_cols(raw_df):
    df = clean(raw_df)
    present = [c for c in NUM_COLS if c in df.columns]
    assert df[present].isna().sum().sum() == 0


def test_undefined_fill_for_handset(raw_df):
    df = clean(raw_df)
    assert (df['Handset Type'] == 'undefined').sum() >= 1
    assert df['Handset Type'].isna().sum() == 0


def test_outlier_is_capped_by_mean_replacement(raw_df):
    df = clean(raw_df)
    # the injected 1e9 ms outlier should no longer be present
    assert df['Dur. (ms)'].max() < 1e9


def test_msisdn_is_integer(raw_df):
    df = clean(raw_df)
    assert df['MSISDN'].dtype.kind in 'iu'


def test_app_totals_created(raw_df):
    df = clean(raw_df)
    assert 'Social Media Total' in df.columns
    assert np.isclose(
        df['Social Media Total'].iloc[0],
        df['Social Media DL (Bytes)'].iloc[0] + df['Social Media UL (Bytes)'].iloc[0],
    )
