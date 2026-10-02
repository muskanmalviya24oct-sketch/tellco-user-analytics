"""Task 2 - User Engagement Analysis."""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

ENGAGEMENT_METRICS = ['sessions', 'duration_ms', 'total_data']


def engagement_table(user: pd.DataFrame) -> pd.DataFrame:
    return user[ENGAGEMENT_METRICS].copy()


def top_n_per_metric(eng: pd.DataFrame, n: int = 10) -> dict:
    return {m: eng[m].nlargest(n) for m in ENGAGEMENT_METRICS}


def cluster_engagement(eng: pd.DataFrame, k: int = 3, random_state: int = 42):
    """Normalize metrics and run k-means. Returns (eng_with_cluster, scaler, kmeans)."""
    eng = eng.copy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(eng[ENGAGEMENT_METRICS])
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    eng['cluster'] = km.labels_
    return eng, scaler, km, X


def cluster_stats(eng_with_cluster: pd.DataFrame) -> pd.DataFrame:
    return eng_with_cluster.groupby('cluster')[ENGAGEMENT_METRICS].agg(['min', 'max', 'mean', 'sum'])


def elbow_inertias(X, ks=range(1, 11), random_state: int = 42):
    return [KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X).inertia_ for k in ks]


def least_engaged_cluster(km: KMeans) -> int:
    """Index of the cluster whose center has the lowest overall metric values."""
    centers = pd.DataFrame(km.cluster_centers_, columns=ENGAGEMENT_METRICS)
    return int(centers.sum(axis=1).idxmin())
