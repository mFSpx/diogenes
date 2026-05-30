# DARWIN HAMMER — match 2777, survivor 7
# gen: 6
# parent_a: hybrid_pheromone_hybrid_distributed_l_m41_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_omni_c_hybrid_indy_learning_m2268_s0.py (gen5)
# born: 2026-05-29T23:45:55Z

import sys
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash of a list of float values (64‑bit)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """Decay probability used by the original pheromone broadcast model."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def tokenize(text: str) -> List[Dict[str, Any]]:
    """Simple whitespace tokenization returning token strings and offsets."""
    import re
    WORD_RE = re.compile(r"\S+")
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]


def _deterministic_vector_from_int(seed: int, dim: int = 32) -> np.ndarray:
    """Map an integer seed to a fixed‑dimensional pseudo‑random vector."""
    rng = np.random.default_rng(seed % (2 ** 32))
    return rng.standard_normal(dim)


def token_embedding(token_id: int, dim: int = 32) -> np.ndarray:
    """Embedding vector for a token derived from a hash integer."""
    return _deterministic_vector_from_int(token_id, dim)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def record_pheromone_signal(
    surface_id: str,
    signal_kind: str,
    values: List[float],
    half_life_seconds: int,
    phase: int,
    step: int,
) -> Dict[str, Any]:
    """
    Record a pheromone signal, compute its perceptual hash and associated
    JEPA token embedding.

    Returns a dictionary containing:
        - surface_id
        - signal_kind
        - phash (int)
        - embedding (np.ndarray)
        - decay (float)   # from broadcast_probability
    """
    phash = compute_phash(values)
    decay = broadcast_probability(phase, step)
    embedding = token_embedding(phash, dim=32)
    return {
        "surface_id": surface_id,
        "signal_kind": signal_kind,
        "phash": phash,
        "embedding": embedding,
        "decay": decay,
    }


def build_attention_matrix(signals: List[Dict[str, Any]]) -> np.ndarray:
    """
    Construct a similarity/attention matrix from a list of recorded signals.
    The raw similarity is exp(-hamming_distance). The matrix is then
    row‑normalized to sum to 1 (softmax‑like) and finally multiplied by the
    broadcast decay factor of the source signal.
    """
    n = len(signals)
    if n == 0:
        return np.empty((0, 0))

    hashes = np.array([s["phash"] for s in signals], dtype=np.uint64)
    decays = np.array([s["decay"] for s in signals], dtype=np.float64)

    # Pairwise Hamming distances using vectorized operations
    dist_matrix = np.array([[hamming_distance(int(hashes[i]), int(hashes[j])) for j in range(n)] for i in range(n)], dtype=np.int32)

    # Convert distances to similarities
    sim_matrix = np.exp(-dist_matrix.astype(np.float64))

    # Apply per‑row decay (broadcast_probability) as a scaling factor
    sim_matrix = sim_matrix * decays[:, None]

    # Row‑normalize to obtain attention weights
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    attention = sim_matrix / row_sums
    return attention


def predict_next_state(signals: List[Dict[str, Any]]) -> np.ndarray:
    """
    JEPA‑style prediction: given current embeddings and the attention matrix,
    compute the next‑state embedding for each surface as a weighted sum of
    all embeddings (including self‑attention). The result is a matrix of shape
    (n, d) where d is the embedding dimension.
    """
    if not signals:
        return np.empty((0, 32))

    embeddings = np.stack([s["embedding"] for s in signals])  # shape (n, d)
    attention = build_attention_matrix(signals)               # shape (n, n)

    # Linear JEPA prediction: weighted combination + a simple residual connection
    predicted = attention @ embeddings + 0.1 * embeddings
    return predicted


def cluster_by_hash_similarity(signals: List[Dict[str, Any]], threshold: int = 4) -> List[List[int]]:
    """
    Simple agglomerative clustering based on Hamming distance.
    Two signals belong to the same cluster if their distance <= threshold.
    Returns a list of clusters, each a list of indices into the original signal list.
    """
    n = len(signals)
    if n == 0:
        return []

    hashes = [s["phash"] for s in signals]
    unassigned = set(range(n))
    clusters: List[List[int]] = []

    while unassigned:
        idx = unassigned.pop()
        cluster = [idx]
        to_check = {i for i in unassigned}
        for j in to_check:
            if hamming_distance(int(hashes[idx]), int(hashes[j])) <= threshold:
                cluster.append(j)
                unassigned.remove(j)
        clusters.append(cluster)
    return clusters


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic pheromone traces for three surfaces
    rng = np.random.default_rng(42)
    dummy_values = [rng.random() for _ in range(100)]

    signals = [
        record_pheromone_signal(
            surface_id=f"surface_{i}",
            signal_kind="usage",
            values=[v + random.uniform(-0.1, 0.1) for v in dummy_values],
            half_life_seconds=3600,
            phase=1 + i,
            step=1,
        )
        for i in range(3)
    ]

    # Compute attention and predict next state
    att = build_attention_matrix(signals)
    pred = predict_next_state(signals)

    # Simple sanity checks (no exceptions)
    assert att.shape == (3, 3)
    assert np.allclose(att.sum(axis=1), 1.0, atol=1e-6)
    assert pred.shape == (3, 32)

    # Cluster based on hash similarity
    clusters = cluster_by_hash_similarity(signals, threshold=2)
    print("Attention matrix:\n", att)
    print("Predicted embeddings (first 5 elements):\n", pred[:,:5])