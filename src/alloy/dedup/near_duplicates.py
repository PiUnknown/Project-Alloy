from __future__ import annotations
import numpy as np
import faiss


def find_duplicate_clusters(embeddings: np.ndarray, threshold: float, top_k: int) -> list[list[int]]:
    """Returns a list of clusters, each a list of row-indices into
    `embeddings` that are near-duplicates of each other. Uses a FAISS flat
    inner-product index rather than a full n x n matrix -- at ~74k rows, a
    brute-force similarity matrix would need tens of gigabytes of memory.
    Embeddings must already be L2-normalized, so inner product = cosine
    similarity."""
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    similarities, neighbor_ids = index.search(embeddings.astype(np.float32), top_k)

    visited = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        matches = [
            int(neighbor_ids[i][k])
            for k in range(top_k)
            if similarities[i][k] >= threshold and not visited[neighbor_ids[i][k]]
        ]
        if len(matches) > 1:
            for j in matches:
                visited[j] = True
            clusters.append(matches)
        else:
            visited[i] = True
    return clusters


def choose_representative(cluster_row_indices: list[int], scores: list[int]) -> int:
    """Given a cluster of row indices and a parallel list of scores,
    return the row index of the one to keep."""
    return max(cluster_row_indices, key=lambda idx: scores[idx])