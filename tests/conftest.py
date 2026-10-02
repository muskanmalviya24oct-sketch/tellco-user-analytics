import numpy as np
import pandas as pd
import pytest

from telco_analytics.cleaning import APPS

N = 60  # small synthetic session set


@pytest.fixture
def raw_df():
    """A small synthetic frame shaped like the real TellCo export, including
    a trailing summary row and some missing values/outliers to clean."""
    rng = np.random.default_rng(42)
    n_users = 12
    msisdn = rng.choice(range(1000, 1000 + n_users), size=N)

    data = {
        'Bearer Id': np.arange(N),
        'Start': pd.date_range('2019-04-01', periods=N, freq='h'),
        'MSISDN/Number': msisdn.astype(float),
        'Dur. (ms)': rng.normal(1000, 50, N),
        'Total DL (Bytes)': rng.normal(1e8, 1e7, N),
        'Total UL (Bytes)': rng.normal(1e7, 1e6, N),
        'Avg RTT DL (ms)': rng.normal(50, 10, N),
        'Avg RTT UL (ms)': rng.normal(10, 3, N),
        'Avg Bearer TP DL (kbps)': rng.normal(20, 5, N),
        'Avg Bearer TP UL (kbps)': rng.normal(30, 5, N),
        'TCP DL Retrans. Vol (Bytes)': rng.normal(1e6, 2e5, N),
        'TCP UL Retrans. Vol (Bytes)': rng.normal(5e5, 1e5, N),
        'Handset Type': rng.choice(['Apple iPhone 7', 'Samsung Galaxy A5', 'Huawei B528'], N),
        'Handset Manufacturer': rng.choice(['Apple', 'Samsung', 'Huawei'], N),
        'Last Location Name': rng.choice(['LocA', 'LocB'], N),
    }
    for a in APPS:
        data[f'{a} DL (Bytes)'] = rng.normal(1e6, 2e5, N)
        data[f'{a} UL (Bytes)'] = rng.normal(1e5, 2e4, N)

    df = pd.DataFrame(data)

    # inject some missingness and an outlier, like the real file
    df.loc[0, 'Avg RTT DL (ms)'] = np.nan
    df.loc[1, 'TCP DL Retrans. Vol (Bytes)'] = np.nan
    df.loc[2, 'Handset Type'] = np.nan
    df.loc[3, 'Dur. (ms)'] = 1e9  # extreme outlier

    # trailing summary row, like the real export (no Start/MSISDN)
    summary = {c: np.nan for c in df.columns}
    summary['Social Media DL (Bytes)'] = df['Social Media DL (Bytes)'].mean()
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df
