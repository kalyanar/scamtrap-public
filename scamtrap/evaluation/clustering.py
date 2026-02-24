"""Clustering quality evaluation."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)


def evaluate_clustering(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int = None,
    n_runs: int = 10,
    seed: int = 42,
) -> dict:
    """K-Means clustering evaluated with NMI, ARI, and Silhouette.

    Runs multiple times with different seeds, reports mean/std.
    """
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    metrics = {"nmi": [], "ari": [], "silhouette": []}

    for i in range(n_runs):
        km = KMeans(n_clusters=n_clusters, random_state=seed + i, n_init=10)
        pred_labels = km.fit_predict(embeddings)

        metrics["nmi"].append(normalized_mutual_info_score(true_labels, pred_labels))
        metrics["ari"].append(adjusted_rand_score(true_labels, pred_labels))
        if len(np.unique(pred_labels)) > 1:
            metrics["silhouette"].append(silhouette_score(embeddings, pred_labels, sample_size=min(5000, len(embeddings))))
        else:
            metrics["silhouette"].append(0.0)

    return {
        metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for metric, vals in metrics.items()
    }
