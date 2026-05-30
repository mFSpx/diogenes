# DARWIN HAMMER — match 2777, survivor 6
# gen: 6
# parent_a: hybrid_pheromone_hybrid_distributed_l_m41_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_omni_c_hybrid_indy_learning_m2268_s0.py (gen5)
# born: 2026-05-29T23:45:55Z

"""Hybrid Algorithm integrating Pheromone Surface Signals with Token‑Based Embeddings
Parents:
- hybrid_pheromone_hybrid_distributed_l_m41_s0.py (Pheromone recording, perceptual hashing,
  leader election via Hamming similarity)
- hybrid_hybrid_hybrid_omni_c_hybrid_indy_learning_m2268_s0.py (Tokenization, deterministic
  SHA‑256 hashing, simple vector embeddings)

Mathematical Bridge:
Both parents rely on deterministic hashing to obtain discrete identifiers:
  * Pheromone algorithm → perceptual hash (binary fingerprint of a numeric vector).
  * INDY learning algorithm → SHA‑256‑based integer hash of token strings.

We fuse them by treating the perceptual hash bits as a binary “token” that can be
combined with token‑derived numeric embeddings.  The hybrid state for each node is
represented by a matrix **S** (pheromone signals) and a matrix **E** (token embeddings).
A linear blend
    H = S·W_s + E·W_e
produces a joint representation **H**.  The rows of **H** are re‑hashed with the
perceptual‑hash routine, yielding binary signatures that are clustered using the
Hamming distance.  Within each cluster the node with the highest aggregated signal
becomes the elected leader.  This unifies the surface‑pheromone dynamics with the
text‑driven embedding space in a single mathematically coherent system.
"""

import sys
import math
import random
from pathlib import Path
import json
import hashlib
import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def compute_phash(values: list[float]) -> int:
    """Perceptual hash: 1‑bit per value indicating above/below mean (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer fingerprints."""
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    """Probability used in the original leader‑broadcast schedule."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def sha256_int(value: str) -> int:
    """Deterministic SHA‑256 hash of a string, truncated to 64 bits."""
    h = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(h[:8], byteorder='big', signed=False)

def tokenize(text: str) -> list[dict]:
    """Simple whitespace tokenization with character offsets."""
    import re
    WORD_RE = re.compile(r"\S+")
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]

def token_to_vector(token: str, dim: int = 16) -> np.ndarray:
    """Map a token to a fixed‑dimensional numeric vector via deterministic hashing."""
    h = sha256_int(token)
    rng = np.random.default_rng(h)          # deterministic per‑token RNG
    vec = rng.random(dim)                   # values in [0,1)
    return vec

def text_to_embedding(text: str, dim: int = 16) -> np.ndarray:
    """Aggregate token vectors (mean) to obtain a single embedding for the text."""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(dim)
    vectors = np.stack([token_to_vector(t["token"], dim) for t in tokens])
    return vectors.mean(axis=0)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_pheromone_matrix(signal_values: list[list[float]]) -> np.ndarray:
    """Stack per‑node pheromone signal lists into a 2‑D array (nodes × features)."""
    max_len = max(len(v) for v in signal_values) if signal_values else 0
    padded = [v + [0.0] * (max_len - len(v)) for v in signal_values]
    return np.asarray(padded, dtype=float)

def build_embedding_matrix(texts: list[str], dim: int = 16) -> np.ndarray:
    """Convert a list of node‑associated texts into an embedding matrix."""
    return np.vstack([text_to_embedding(t, dim) for t in texts])

def hybrid_leader_election(
    pheromone_signals: list[list[float]],
    node_texts: list[str],
    phase: int,
    step: int,
    hamming_thresh: int = 2,
) -> dict:
    """
    Perform a single hybrid election round.

    Returns a dict:
        {
            "leaders": [node_index, ...],
            "clusters": {hash_int: [node_indices...]},
            "broadcast_prob": float,
        }
    """
    # 1. Build numeric matrices
    S = build_pheromone_matrix(pheromone_signals)            # (N, F)
    E = build_embedding_matrix(node_texts, dim=16)           # (N, 16)

    N, F = S.shape
    _, D = E.shape

    # 2. Random but deterministic blending matrices (seeded by fixed string)
    rng = np.random.default_rng(sha256_int("hybrid_blend"))
    W_s = rng.standard_normal((F, 32))   # map pheromone -> 32‑dim space
    W_e = rng.standard_normal((D, 32))   # map embedding -> same space

    # 3. Joint representation
    H = S @ W_s + E @ W_e                  # (N, 32)

    # 4. Perceptual hash each joint row
    hashes = [compute_phash(row.tolist()) for row in H]

    # 5. Cluster nodes by Hamming distance ≤ threshold
    clusters: dict[int, list[int]] = {}
    for i, hi in enumerate(hashes):
        placed = False
        for h_rep in list(clusters.keys()):
            if hamming_distance(hi, h_rep) <= hamming_thresh:
                clusters[h_rep].append(i)
                placed = True
                break
        if not placed:
            clusters[hi] = [i]

    # 6. Within each cluster pick the leader (max sum of pheromone signals)
    leaders = []
    for h_rep, members in clusters.items():
        # aggregate original pheromone signal strengths per node
        strengths = [sum(pheromone_signals[m]) for m in members]
        leader_idx = members[int(np.argmax(strengths))]
        leaders.append(leader_idx)

    # 7. Broadcast probability from Parent A
    prob = broadcast_probability(phase, step)

    return {
        "leaders": sorted(leaders),
        "clusters": clusters,
        "broadcast_prob": prob,
    }

def simulate_pheromone_signal(num_nodes: int, max_len: int = 10) -> list[list[float]]:
    """Generate synthetic pheromone signal vectors for testing."""
    rng = np.random.default_rng(42)
    return [rng.random(rng.integers(1, max_len + 1)).tolist() for _ in range(num_nodes)]

def generate_dummy_texts(num_nodes: int) -> list[str]:
    """Create simple deterministic sentences for each node."""
    base = [
        "alpha beta gamma delta",
        "epsilon zeta eta theta iota",
        "kappa lambda mu nu xi omicron",
        "pi rho sigma tau upsilon phi",
        "chi psi omega",
    ]
    return [base[i % len(base)] + f" node{i}" for i in range(num_nodes)]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    NUM_NODES = 12
    pheromone_data = simulate_pheromone_signal(NUM_NODES, max_len=8)
    texts = generate_dummy_texts(NUM_NODES)

    result = hybrid_leader_election(
        pheromone_signals=pheromone_data,
        node_texts=texts,
        phase=3,
        step=1,
        hamming_thresh=1,
    )

    print("Broadcast probability:", result["broadcast_prob"])
    print("Clusters (hash → members):")
    for h, members in result["clusters"].items():
        print(f"  {h:#018x}: {members}")
    print("Elected leaders:", result["leaders"])
    sys.exit(0)