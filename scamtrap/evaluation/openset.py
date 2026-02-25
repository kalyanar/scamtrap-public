"""Open-set generalization evaluation with multiple OOD scoring methods."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def _cosine_novelty_scores(embeddings, centroids):
    """Max cosine similarity to nearest centroid (lower = more novel)."""
    # Normalize
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    # Cosine similarity to each centroid
    cos_sim = emb_norm @ cent_norm.T  # [N, C]
    # Novelty = negative max similarity (higher = more novel)
    return -np.max(cos_sim, axis=1)


def _euclidean_novelty_scores(embeddings, centroids):
    """Min Euclidean distance to nearest centroid (higher = more novel)."""
    distances = np.min(
        np.linalg.norm(embeddings[:, None] - centroids[None, :], axis=2),
        axis=1,
    )
    return distances


def _mahalanobis_novelty_scores(embeddings, train_embeddings, train_labels):
    """Mahalanobis distance to nearest class-conditional Gaussian.

    Implements Lee et al. (NeurIPS 2018): class-conditional means with
    a shared (tied) covariance matrix across all classes.
    """
    unique_labels = np.unique(train_labels)

    # Compute class-conditional means
    class_means = np.array([
        train_embeddings[train_labels == l].mean(axis=0)
        for l in unique_labels
    ])

    # Compute shared covariance (tied across classes)
    centered = []
    for l in unique_labels:
        class_data = train_embeddings[train_labels == l]
        centered.append(class_data - class_means[unique_labels == l])
    centered = np.vstack(centered)

    # Regularized covariance (add small identity for numerical stability)
    cov = np.cov(centered, rowvar=False)
    cov += 1e-4 * np.eye(cov.shape[0])

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        cov_inv = np.linalg.pinv(cov)

    # Compute Mahalanobis distance to each class mean
    mahal_distances = np.zeros((len(embeddings), len(unique_labels)))
    for c, mean in enumerate(class_means):
        diff = embeddings - mean  # [N, D]
        mahal_distances[:, c] = np.sqrt(
            np.sum(diff @ cov_inv * diff, axis=1).clip(min=0)
        )

    # Novelty score = minimum Mahalanobis distance across classes
    return np.min(mahal_distances, axis=1)


def _energy_novelty_scores(logits):
    """Energy-based OOD score from Liu et al. (NeurIPS 2020).

    Energy(x) = -T * log(sum_c exp(f_c(x)/T))
    Lower energy = more in-distribution, so we negate for novelty scoring.
    """
    T = 1.0  # Temperature
    # LogSumExp for numerical stability
    max_logits = np.max(logits, axis=1, keepdims=True)
    energy = -T * (max_logits.squeeze() + np.log(
        np.sum(np.exp((logits - max_logits) / T), axis=1)
    ))
    # Energy is less negative for OOD samples (flat logits → smaller logsumexp)
    # So energy directly serves as a novelty score: higher (less negative) = more novel
    return energy


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

    Uses three OOD scoring methods:
    1. Cosine distance to nearest centroid (original)
    2. Mahalanobis distance (Lee et al., NeurIPS 2018)
    3. Energy-based score (Liu et al., NeurIPS 2020)

    Returns:
        dict with seen classification metrics and novelty detection AUROC
        for each OOD scoring method.
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

    # 3. Novelty detection with multiple scoring methods
    if len(test_unseen_embeddings) > 0:
        all_test = np.vstack([test_seen_embeddings, test_unseen_embeddings])
        novel_labels = np.array(
            [0] * len(test_seen_embeddings) + [1] * len(test_unseen_embeddings)
        )

        # Method 1: Euclidean distance to nearest centroid (original)
        euclidean_scores = _euclidean_novelty_scores(all_test, centroids)
        try:
            results["novelty_auroc"] = float(roc_auc_score(novel_labels, euclidean_scores))
        except ValueError:
            results["novelty_auroc"] = 0.0

        # Method 2: Cosine distance to nearest centroid
        cosine_scores = _cosine_novelty_scores(all_test, centroids)
        try:
            results["novelty_auroc_cosine"] = float(roc_auc_score(novel_labels, cosine_scores))
        except ValueError:
            results["novelty_auroc_cosine"] = 0.0

        # Method 3: Mahalanobis distance
        mahal_scores = _mahalanobis_novelty_scores(
            all_test, train_embeddings, train_labels
        )
        try:
            results["novelty_auroc_mahalanobis"] = float(roc_auc_score(novel_labels, mahal_scores))
        except ValueError:
            results["novelty_auroc_mahalanobis"] = 0.0

        # Method 4: Energy-based (uses logistic regression logits)
        all_logits = clf.decision_function(all_test)
        if all_logits.ndim == 1:
            # Binary case: expand to 2 columns
            all_logits = np.column_stack([-all_logits, all_logits])
        energy_scores = _energy_novelty_scores(all_logits)
        try:
            results["novelty_auroc_energy"] = float(roc_auc_score(novel_labels, energy_scores))
        except ValueError:
            results["novelty_auroc_energy"] = 0.0

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
        results["novelty_auroc_cosine"] = None
        results["novelty_auroc_mahalanobis"] = None
        results["novelty_auroc_energy"] = None
        results["unseen_nmi"] = None
        results["unseen_ari"] = None

    return results
