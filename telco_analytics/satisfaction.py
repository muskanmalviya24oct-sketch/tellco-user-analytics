"""Task 4 - Satisfaction Analysis."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


def engagement_score(X_eng, km_eng, least_engaged_idx: int) -> np.ndarray:
    """Euclidean distance from each user's normalized engagement vector to
    the centre of the least-engaged cluster."""
    return np.linalg.norm(X_eng - km_eng.cluster_centers_[least_engaged_idx], axis=1)


def experience_score(X_exp, km_exp, worst_experience_idx: int) -> np.ndarray:
    """Euclidean distance from each user's standardized experience vector to
    the centre of the worst-experience cluster."""
    return np.linalg.norm(X_exp - km_exp.cluster_centers_[worst_experience_idx], axis=1)


def build_scores(eng_index, exp_index, eng_scores, exp_scores, normalize: bool = False) -> pd.DataFrame:
    """Combine engagement + experience scores (matched on user index) into a
    satisfaction_score = average of the two."""
    eng_s = pd.Series(eng_scores, index=eng_index, name='engagement_score')
    exp_s = pd.Series(exp_scores, index=exp_index, name='experience_score')
    scores = pd.concat([eng_s, exp_s], axis=1, join='inner')
    if normalize:
        scores[['engagement_score', 'experience_score']] = MinMaxScaler().fit_transform(
            scores[['engagement_score', 'experience_score']])
    scores['satisfaction_score'] = scores[['engagement_score', 'experience_score']].mean(axis=1)
    return scores


def top_satisfied(scores: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return scores.nlargest(n, 'satisfaction_score')


def cluster_scores(scores: pd.DataFrame, k: int = 2, random_state: int = 42):
    X = StandardScaler().fit_transform(scores[['engagement_score', 'experience_score']])
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    out = scores.copy()
    out['sat_cluster'] = km.labels_
    return out, km


def cluster_averages(scores_with_cluster: pd.DataFrame) -> pd.DataFrame:
    return scores_with_cluster.groupby('sat_cluster')[
        ['satisfaction_score', 'experience_score', 'engagement_score']].mean()
