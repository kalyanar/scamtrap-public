"""Open-set generalization evaluation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def evaluate_openset(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_seen_embeddings: np.ndarray,
    test_seen_labels: np.ndarray,
    test_unseen_embeddings: np.ndarray,
    test_unseen_labels: np.ndarray,
    seed: int = 42,
) -> dict:
    """Open-set evaluation: seen classification + unseen detection/clustering.

    1. Train classifier on seen intents
    2. Evaluate on test_seen: accuracy/F1
    3. Evaluate on test_unseen:
       a. Novelty detection via distance threshold
       b. Clustering quality (do unseen intents cluster coherently?)
    """
    results = {}

    # 1. Seen intent classification
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(train_embeddings, train_labels)

    seen_preds = clf.predict(test_seen_embeddings)
    results["seen_accuracy"] = float(accuracy_score(test_seen_labels, seen_preds))
    results["seen_f1_macro"] = float(f1_score(test_seen_labels, seen_preds, average="macro", zero_division=0))

    # 2. Compute class centroids from training data
    unique_labels = np.unique(train_labels)
    centroids = np.array([
        train_embeddings[train_labels == l].mean(axis=0)
        for l in unique_labels
    ])

    # 3. Novelty detection: unseen samples should be far from all centroids
    # Combine seen test + unseen test, label as "known" vs "novel"
    if len(test_unseen_embeddings) > 0:
        all_test = np.vstack([test_seen_embeddings, test_unseen_embeddings])
        novel_labels = np.array(
            [0] * len(test_seen_embeddings) + [1] * len(test_unseen_embeddings)
        )

        # Distance to nearest centroid
        distances = np.min(
            np.linalg.norm(all_test[:, None] - centroids[None, :], axis=2),
            axis=1,
        )

        # AUROC for novelty detection
        try:
            results["novelty_auroc"] = float(roc_auc_score(novel_labels, distances))
        except ValueError:
            results["novelty_auroc"] = 0.0

        # 4. Clustering quality of unseen intents
        n_unseen_classes = len(np.unique(test_unseen_labels))
        if n_unseen_classes > 1:
            km = KMeans(n_clusters=n_unseen_classes, random_state=seed, n_init=10)
            unseen_preds = km.fit_predict(test_unseen_embeddings)
            results["unseen_nmi"] = float(normalized_mutual_info_score(test_unseen_labels, unseen_preds))
            results["unseen_ari"] = float(adjusted_rand_score(test_unseen_labels, unseen_preds))
        else:
            results["unseen_nmi"] = 0.0
            results["unseen_ari"] = 0.0
    else:
        results["novelty_auroc"] = None
        results["unseen_nmi"] = None
        results["unseen_ari"] = None

    return results
