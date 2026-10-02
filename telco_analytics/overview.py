"""Task 1 - User Overview Analysis."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .cleaning import APPS, APP_COLS


def top_handsets(df: pd.DataFrame, n: int = 10) -> pd.Series:
    known = df[df['Handset Type'] != 'undefined']
    return known['Handset Type'].value_counts().head(n)


def top_manufacturers(df: pd.DataFrame, n: int = 3) -> pd.Series:
    known = df[df['Handset Manufacturer'] != 'undefined']
    return known['Handset Manufacturer'].value_counts().head(n)


def top_handsets_per_manufacturer(df: pd.DataFrame, manufacturers=None, n: int = 5) -> dict:
    known = df[(df['Handset Type'] != 'undefined') & (df['Handset Manufacturer'] != 'undefined')]
    if manufacturers is None:
        manufacturers = top_manufacturers(df).index
    return {m: known.loc[known['Handset Manufacturer'] == m, 'Handset Type'].value_counts().head(n)
            for m in manufacturers}


def user_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user session count, duration, DL/UL and per-app totals."""
    user = df.groupby('MSISDN').agg(
        sessions=('Bearer Id', 'count'),
        duration_ms=('Dur. (ms)', 'sum'),
        total_dl=('Total DL (Bytes)', 'sum'),
        total_ul=('Total UL (Bytes)', 'sum'),
        **{f'{a}_bytes': (f'{a} Total', 'sum') for a in APPS},
    )
    user['total_data'] = user['total_dl'] + user['total_ul']
    return user


def decile_summary(user: pd.DataFrame, duration_col: str = 'duration_ms') -> pd.DataFrame:
    """Top-five decile classes by total session duration, with total data per decile."""
    u = user.copy()
    u['decile'] = pd.qcut(u[duration_col].rank(method='first'), 10, labels=False) + 1
    summary = u.groupby('decile').agg(
        users=('total_data', 'size'),
        total_duration_ms=(duration_col, 'sum'),
        total_data=('total_data', 'sum'),
    )
    return summary.loc[6:10]


def app_correlation(user: pd.DataFrame) -> pd.DataFrame:
    app_total_cols = [f'{a}_bytes' for a in APPS]
    return user[app_total_cols].corr()


def app_pca(user: pd.DataFrame, n_components=None):
    """Return (explained_variance_df, loadings_df) for PCA on app totals."""
    app_total_cols = [f'{a}_bytes' for a in APPS]
    X = StandardScaler().fit_transform(user[app_total_cols])
    pca = PCA(n_components=n_components).fit(X)
    explained = pd.DataFrame({
        'PC': [f'PC{i + 1}' for i in range(pca.n_components_)],
        'explained_var_%': pca.explained_variance_ratio_ * 100,
        'cumulative_%': pca.explained_variance_ratio_.cumsum() * 100,
    })
    loadings = pd.DataFrame(pca.components_.T, index=app_total_cols,
                             columns=explained['PC'])
    return explained, loadings
