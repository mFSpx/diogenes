# DARWIN HAMMER — match 1295, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_nlms_o_m668_s1.py (gen3)
# parent_b: hybrid_hdc_hybrid_hybrid_ternar_m418_s0.py (gen5)
# born: 2026-05-29T23:35:03Z

"""Hybrid Path‑Signature / NLMS – Bipolar‑Ternary Fusion

Parents:
- hybrid_hybrid_hybrid_path_s_hybrid_hybrid_nlms_o_m668_s1.py (Path‑Signature + NLMS‑Graph‑Tree)
- hybrid_hdc_hybrid_hybrid_ternar_m418_s0.py (Bipolar vector binding + ternary decision‑hygiene)

Mathematical bridge:
The level‑1 and level‑2 path signatures are treated as a low‑dimensional
feature vector **s**.  A deterministic ternary vector **t** (values –1,0,1)
derived from the same command envelope is expanded and bound (element‑wise
multiplication) to a high‑dimensional bipolar vector **b** (values –1,1).
The concatenated hybrid feature  

    x = [ s , b ]

feeds an Normalised‑Least‑Mean‑Squares (NLMS) adaptive filter.  The NLMS
weight vector **w** is used both to predict a scalar target (e.g. a
decision‑hygiene score) and to weight the components of **x** when computing
pairwise edge strengths in a fully‑connected graph:

    edge_ij = (w ⊙ x_i) · (w ⊙ x_j)

where ⊙ denotes element‑wise multiplication and “·” the Euclidean inner
product.  This unifies the geometric algebra of path signatures with the
high‑dimensional binding of bipolar‑ternary representations.

The module implements the full hybrid pipeline and provides a small
smoke‑test.
"""

import math
import random
import sys
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------
# Constants (mirroring parent B)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12
BIPOLAR_DIMS = 10000
NLMS_MU = 0.1          # learning rate
NLMS_EPS = 1e-6        # regularisation to avoid division by zero

# ----------------------------------------------------------------------
# Parent A – Path Signature utilities
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Apply lead‑lag transformation to a multivariate path."""
    n, d = path.shape
    augmented = np.zeros((n, d + 1))
    augmented[:, :d] = path
    # cumulative Euclidean distance as the extra dimension
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    augmented[1:, d] = np.cumsum(diffs)
    return augmented

def compute_signatures(augmented_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return level‑1 (vector) and level‑2 (matrix) signatures."""
    n, d = augmented_path.shape
    # level‑1: time‑integral of each coordinate
    level1 = np.mean(augmented_path, axis=0)
    # level‑2: empirical second‑order iterated integrals
    level2 = np.zeros((d, d))
    for i in range(n - 1):
        delta = augmented_path[i + 1] - augmented_path[i]
        level2 += np.outer(augmented_path[i], delta)
    level2 /= (n - 1)
    return level1, level2

# ----------------------------------------------------------------------
# Parent B – Ternary & Bipolar utilities
# ----------------------------------------------------------------------
def _hash_envelope(raw_command: str, normalized_intent: str, context: dict) -> int:
    """Deterministic SHA‑256 based integer from the command envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return int(hashlib.sha256(encoded).hexdigest(), 16)

def ternary_vector(raw_command: str, normalized_intent: str, context: dict) -> np.ndarray:
    """Generate a ternary vector (values –1,0,1) of length TERNARY_DIMS."""
    h = _hash_envelope(raw_command, normalized_intent, context)
    vec = np.empty(TERNARY_DIMS, dtype=int)
    for i in range(TERNARY_DIMS):
        bits = (h >> (2 * i)) & 0b11
        if bits == 0:
            vec[i] = -1
        elif bits == 1:
            vec[i] = 0
        else:
            vec[i] = 1
    return vec

def bipolar_vector(raw_command: str, normalized_intent: str, context: dict) -> np.ndarray:
    """Generate a bipolar vector (values –1,1) of length BIPOLAR_DIMS."""
    h = _hash_envelope(raw_command, normalized_intent, context)
    vec = np.empty(BIPOLAR_DIMS, dtype=int)
    for i in range(BIPOLAR_DIMS):
        bit = (h >> i) & 1
        vec[i] = 1 if bit else -1
    return vec

def bind_bipolar_ternary(bipolar: np.ndarray, ternary: np.ndarray) -> np.ndarray:
    """
    Bind a bipolar vector with a ternary vector.
    The ternary vector is tiled to match bipolar dimensionality and then
    multiplied element‑wise.  Zero entries in the ternary mask preserve the
    original bipolar sign.
    """
    repeat = BIPOLAR_DIMS // TERNARY_DIMS
    tiled = np.repeat(ternary, repeat)
    # Ensure exact length (in case of non‑divisible dimensions)
    tiled = tiled[:BIPOLAR_DIMS]
    bound = bipolar * tiled
    return bound

# ----------------------------------------------------------------------
# Hybrid core: NLMS adaptation and graph construction
# ----------------------------------------------------------------------
def nlms_update(weights: np.ndarray, x: np.ndarray, d: float,
                mu: float = NLMS_MU, eps: float = NLMS_EPS) -> np.ndarray:
    """
    Perform a single NLMS weight update.
    w_{new} = w + mu * e * x / (eps + ||x||^2)
    where e = d - w·x.
    """
    x_norm_sq = np.dot(x, x) + eps
    y = np.dot(weights, x)
    e = d - y
    delta = (mu * e / x_norm_sq) * x
    return weights + delta

def hybrid_feature_vector(path: np.ndarray,
                          raw_command: str,
                          normalized_intent: str,
                          context: dict) -> np.ndarray:
    """
    Build the hybrid feature vector:
      [ level1_signature , flattened level2_signature , bound bipolar vector ]
    """
    aug = lead_lag_transform(path)
    lvl1, lvl2 = compute_signatures(aug)
    sig_part = np.concatenate([lvl1, lvl2.ravel()])

    tern = ternary_vector(raw_command, normalized_intent, context)
    bipolar = bipolar_vector(raw_command, normalized_intent, context)
    bound = bind_bipolar_ternary(bipolar, tern)

    return np.concatenate([sig_part.astype(float), bound.astype(float)])

def weighted_graph_adjacency(features: list[np.ndarray],
                             weights: np.ndarray) -> np.ndarray:
    """
    Given a list of hybrid feature vectors and an NLMS weight vector,
    compute a symmetric adjacency matrix where

        A_ij = (w ⊙ x_i) · (w ⊙ x_j)

    The diagonal is zeroed (no self‑loops).
    """
    n = len(features)
    dim = weights.shape[0]
    assert all(f.shape[0] == dim for f in features), "Feature dimension mismatch"

    weighted = [weights * f for f in features]  # element‑wise scaling
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = float(np.dot(weighted[i], weighted[j]))
            adj[i, j] = adj[j, i] = val
    np.fill_diagonal(adj, 0.0)
    return adj

# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_hybrid_feature():
    """Create a dummy path and command envelope, then return the hybrid vector."""
    path = np.cumsum(np.random.randn(20, 3), axis=0)  # 20 steps, 3‑D
    raw = "move forward"
    intent = "navigate"
    ctx = {"location": "lab", "timestamp": utc_now()}
    return hybrid_feature_vector(path, raw, intent, ctx)

def demo_nlms_learning(num_iterations: int = 5):
    """
    Run a tiny NLMS learning loop on synthetic data.
    The desired signal is the Euclidean norm of the signature part,
    demonstrating that the filter can learn a simple mapping.
    """
    # Initialise weight vector
    sample_feat = demo_hybrid_feature()
    w = np.zeros_like(sample_feat)

    for _ in range(num_iterations):
        x = demo_hybrid_feature()
        # Desired signal: norm of the low‑dimensional signature portion
        sig_len = len(x) - BIPOLAR_DIMS
        d = np.linalg.norm(x[:sig_len])
        w = nlms_update(w, x, d)
    return w

def demo_graph_construction(node_count: int = 4):
    """Build a weighted graph from several hybrid feature vectors."""
    feats = [demo_hybrid_feature() for _ in range(node_count)]
    w = demo_nlms_learning()
    adj = weighted_graph_adjacency(feats, w)
    return adj

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Hybrid feature vector shape
    hv = demo_hybrid_feature()
    print("Hybrid feature shape:", hv.shape)

    # 2. NLMS learning – final weight norm
    final_w = demo_nlms_learning()
    print("Final NLMS weight norm:", np.linalg.norm(final_w))

    # 3. Graph adjacency matrix
    adjacency = demo_graph_construction()
    print("Adjacency matrix:\n", adjacency)
    # Verify symmetry and zero diagonal
    assert np.allclose(adjacency, adjacency.T), "Adjacency not symmetric"
    assert np.allclose(np.diag(adjacency), 0), "Diagonal not zero"
    print("Smoke test completed successfully.")