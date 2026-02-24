"""Retrieval evaluation using nearest neighbors."""

import numpy as np


def evaluate_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = None,
) -> dict:
    """Compute recall@k: fraction of k nearest neighbors with same label.

    Uses brute-force cosine similarity (fast enough for <50K samples).
    For larger datasets, swap to FAISS.
    """
    k_values = k_values or [1, 5, 10, 20]

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings_norm = embeddings / norms

    # Compute similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T

    # Zero out self-similarity
    np.fill_diagonal(sim_matrix, -1)

    results = {}
    for k in k_values:
        if k >= len(embeddings):
            k = len(embeddings) - 1

        # Get top-k indices for each query
        top_k_indices = np.argpartition(-sim_matrix, k, axis=1)[:, :k]

        # Compute recall: fraction of top-k with same label
        recalls = []
        for i in range(len(embeddings)):
            neighbors = top_k_indices[i]
            same_label = np.sum(labels[neighbors] == labels[i])
            total_same = np.sum(labels == labels[i]) - 1  # exclude self
            if total_same > 0:
                recalls.append(same_label / min(k, total_same))
            else:
                recalls.append(0.0)

        results[f"recall@{k}"] = float(np.mean(recalls))

    return results
