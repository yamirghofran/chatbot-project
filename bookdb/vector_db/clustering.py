from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2


def cluster_seeds_by_embedding(
    seed_vectors: dict[int, list[float]],
    seed_weights: dict[int, float],
    n_clusters: int,
) -> list[tuple[np.ndarray, list[tuple[int, float]]]]:
    """K-means cluster seed books by their embedding vectors.
    Returns list of (weighted_centroid, [(goodreads_id, weight), ...]) per cluster.
    Empty clusters are dropped.
    """
    ids = list(seed_vectors.keys())
    X = np.array([seed_vectors[i] for i in ids], dtype=np.float32)
    weights = np.array([seed_weights.get(i, 1.0) for i in ids], dtype=np.float32)

    _, labels = kmeans2(X, n_clusters, iter=10, minit="++", missing="warn")

    clusters = []
    for k in range(n_clusters):
        mask = labels == k
        if not mask.any():
            continue
        members = [(ids[j], seed_weights.get(ids[j], 1.0)) for j in range(len(ids)) if mask[j]]
        w = weights[mask]
        centroid = (X[mask] * w[:, np.newaxis]).sum(axis=0) / w.sum()
        clusters.append((centroid, members))
    return clusters