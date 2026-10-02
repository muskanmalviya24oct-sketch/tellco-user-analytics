"""Task 3 - Experience Analytics."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

EXPERIENCE_FEATURES = ['avg_tcp', 'avg_rtt', 'avg_tput']


def experience_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user average TCP retransmission, RTT, throughput and modal handset."""
    d = df.copy()
    d['tcp'] = d['TCP DL Retrans. Vol (Bytes)'] + d['TCP UL Retrans. Vol (Bytes)']
    d['rtt'] = d['Avg RTT DL (ms)'] + d['Avg RTT UL (ms)']
    d['tput'] = d['Avg Bearer TP DL (kbps)'] + d['Avg Bearer TP UL (kbps)']

    exp = d.groupby('MSISDN').agg(avg_tcp=('tcp', 'mean'), avg_rtt=('rtt', 'mean'),
                                   avg_tput=('tput', 'mean'))

    known = d[d['Handset Type'] != 'undefined']
    mode = (known.groupby(['MSISDN', 'Handset Type']).size().reset_index(name='n')
            .sort_values(['MSISDN', 'n'], ascending=[True, False])
            .drop_duplicates('MSISDN').set_index('MSISDN')['Handset Type'])
    exp['handset'] = mode.reindex(exp.index).fillna('undefined')
    return exp


def top_bottom_frequent(series: pd.Series, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        'top': series.nlargest(n).values,
        'bottom': series.nsmallest(n).values,
        'most_frequent_value': series.value_counts().head(n).index.values,
        'frequency': series.value_counts().head(n).values,
    })


def per_handset_summary(exp: pd.DataFrame, top_n_handsets: int = 10, min_users: int = 100) -> pd.DataFrame:
    known = exp[exp['handset'] != 'undefined']
    top_handsets = known['handset'].value_counts().head(top_n_handsets).index
    sub = known[known['handset'].isin(top_handsets)]
    summary = sub.groupby('handset').agg(users=('avg_tput', 'size'),
                                          mean_tput=('avg_tput', 'mean'),
                                          mean_tcp=('avg_tcp', 'mean'))
    return summary[summary['users'] >= min_users]


def cluster_experience(exp: pd.DataFrame, k: int = 3, random_state: int = 42):
    """Standardize features and run k-means. Returns (exp_with_cluster, scaler, kmeans, X, worst_idx)."""
    exp = exp.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(exp[EXPERIENCE_FEATURES])
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    exp['cluster'] = km.labels_

    centers = pd.DataFrame(km.cluster_centers_, columns=EXPERIENCE_FEATURES)
    # higher throughput is good; higher tcp/rtt are bad
    idx = (centers['avg_tput'] - centers['avg_tcp'] - centers['avg_rtt']).sort_values().index
    worst_idx = int(idx[0])
    return exp, scaler, km, X, worst_idx
