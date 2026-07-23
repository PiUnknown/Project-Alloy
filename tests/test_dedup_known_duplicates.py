import numpy as np
from alloy.dedup.near_duplicates import find_duplicate_clusters, choose_representative


def _unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_finds_known_duplicate_pair():
    embeddings = np.stack([
        _unit([1.0, 0.01, 0.0]),
        _unit([1.0, 0.02, 0.0]),
        _unit([0.0, 0.0, 1.0]),
    ])
    clusters = find_duplicate_clusters(embeddings, threshold=0.99, top_k=3)
    assert any(set(c) == {0, 1} for c in clusters)
    assert not any(2 in c for c in clusters)


def test_choose_representative_picks_highest_score():
    assert choose_representative([0, 1, 2], scores=[10, 50, 5]) == 1