# DARWIN HAMMER — match 5074, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1555_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s0.py (gen6)
# born: 2026-05-29T23:59:45Z

"""Hybrid Fusion of Linear Compression/Bandit Learning, Tropical Max‑Plus Evaluation,
and Ollivier‑Ricci Curvature‑Weighted Graph Analysis.

Parents:
- **Parent A** contributes a compressive weight matrix `W`, a contextual bandit
  update mechanism, a SSIM‑derived gain `g`, and a privacy‑risk scalar `r` that
  rescales the flattened coefficients before a tropical max‑plus evaluation
  `max_i (c_i + g)`.
- **Parent B** contributes graph construction from a coefficient matrix, computation
  of Ollivier‑Ricci curvature on that graph, and the use of curvature as an
  additional weight in a linear combination.

**Mathematical bridge** – The flattened weight matrix `c = vec(W)` is the
coefficient vector of the tropical polynomial.  After being rescaled by the
privacy risk `r`, each coefficient is further shifted by the curvature value
associated with its originating row (node) in the graph built from `W`.  The
SSIM score of the compression supplies the tropical gain `g`.  The final decision
value is therefore  


value_i = (1 - r) * c_i + g + κ_i


where `κ_i` is the Ollivier‑Ricci curvature attached to the node that generated
`c_i`.  The index `i* = argmax_i value_i` selects the bandit action with the
largest tropical‑curvature‑augmented gain.  This unifies the two parent
topologies into a single hybrid pipeline.
"""

import sys
import math
import random
import pathlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Bandit structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# Global mutable stores used by the bandit component
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}


def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()
    _STORE.clear()


def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n != 0 else 0.0


def update_policy(update: BanditUpdate) -> None:
    """Incorporate a single bandit observation."""
    total, n = _POLICY.get(update.action_id, [0.0, 0.0])
    total += update.reward
    n += 1
    _POLICY[update.action_id] = [total, n]
    _STORE[update.context_id] = update.reward


# ----------------------------------------------------------------------
# Core Hybrid Functions
# ----------------------------------------------------------------------


def compress_data(X: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Linear compression via a random projection matrix.
    Returns the weight matrix W (target_dim × original_dim) such that
    `X_compressed = W @ X.T`.
    """
    if X.ndim != 2:
        raise ValueError("Input X must be a 2‑D array (samples × features).")
    orig_dim = X.shape[1]
    rng = np.random.default_rng()
    W = rng.standard_normal(size=(target_dim, orig_dim))
    # Normalise rows to unit length (helps later cosine similarity)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    return W


def structural_similarity(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Very lightweight SSIM‑like metric: 1 - MSE / (variance + eps).
    Returns a scalar in [0, 1] where higher is more similar.
    """
    if original.shape != compressed.shape:
        raise ValueError("Shapes of original and compressed must match.")
    mse = np.mean((original - compressed) ** 2)
    var = np.var(original)
    return 1.0 - mse / (var + 1e-12)


def reconstruction_risk(unique_qi: int, total_records: int) -> float:
    """Privacy‑related risk r = unique_qi / total_records ∈ [0,1]."""
    if total_records <= 0:
        return 0.0
    return min(1.0, unique_qi / total_records)


def build_adjacency(W: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Build an undirected adjacency matrix from weight matrix rows.
    Edge weight = cosine similarity; edges kept only if similarity > threshold.
    """
    # Cosine similarity between rows
    normed = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    sim = normed @ normed.T
    A = (sim > threshold).astype(float)
    np.fill_diagonal(A, 0.0)  # no self‑loops
    return A


def ollivier_ricci_curvature(A: np.ndarray) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature for each node.
    For node i, curvature κ_i = 1 - (average shortest‑path distance to neighbours)
    / (average degree of i and its neighbours).  Result is a 1‑D array κ.
    """
    n = A.shape[0]
    # Degree vector
    deg = A.sum(axis=1)
    # Initialise distance matrix with inf, 0 on diagonal
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    # Direct edges have distance 1
    dist[A > 0] = 1.0

    # Floyd‑Warshall (small n assumed)
    for k in range(n):
        dk = dist[:, k][:, None]
        dk2 = dist[k, :][None, :]
        dist = np.minimum(dist, dk + dk2)

    # Compute curvature per node
    curvature = np.zeros(n)
    for i in range(n):
        neighbours = np.where(A[i] > 0)[0]
        if neighbours.size == 0 or deg[i] == 0:
            curvature[i] = 0.0
            continue
        avg_dist = np.mean(dist[i, neighbours])
        avg_deg = (deg[i] + deg[neighbours].mean()) / 2.0
        curvature[i] = 1.0 - avg_dist / (avg_deg + 1e-12)
    return curvature  # shape (n,)


def tropical_max_plus(
    W: np.ndarray,
    gain: float,
    risk: float,
    curvature: np.ndarray,
) -> Tuple[int, float]:
    """
    Perform the tropical max‑plus evaluation on the flattened weight matrix.
    Each coefficient c_i is first rescaled by (1 - risk), then shifted by the
    global gain and the curvature attached to its originating row.
    Returns the index of the maximal value and the value itself.
    """
    # Flatten but keep track of row origin for curvature mapping
    flat_c = W.ravel()
    rows, cols = W.shape
    # Map each flat index to its row index
    row_idx = np.repeat(np.arange(rows), cols)

    # Apply risk scaling and curvature shift
    transformed = (1.0 - risk) * flat_c + gain + curvature[row_idx]
    idx = int(np.argmax(transformed))
    return idx, float(transformed[idx])


def hybrid_decision(
    context_id: str,
    X: np.ndarray,
    target_dim: int,
    unique_qi: int,
    total_records: int,
    policy_actions: List[BanditAction],
) -> BanditAction:
    """
    End‑to‑end hybrid pipeline:
    1. Compress data → weight matrix W.
    2. Compute SSIM gain g.
    3. Compute privacy risk r.
    4. Build adjacency from W and obtain curvature κ.
    5. Tropical max‑plus evaluation selects an index i*.
    6. Map i* to a bandit action (modulo number of actions) and update policy.
    Returns the selected BanditAction.
    """
    # 1. Compression
    W = compress_data(X, target_dim)

    # 2. Reconstruction for SSIM (project back to original space)
    X_rec = (W @ X.T).T  # shape (samples, target_dim) -> back to original dim via pseudo‑inverse
    # Use simple linear decoder (transpose of W) for demonstration
    X_rec = X_rec @ W  # now back to original feature space
    g = structural_similarity(X, X_rec)

    # 3. Privacy risk
    r = reconstruction_risk(unique_qi, total_records)

    # 4. Graph & curvature
    A = build_adjacency(W)
    κ = ollivier_ricci_curvature(A)

    # 5. Tropical evaluation
    idx, _ = tropical_max_plus(W, gain=g, risk=r, curvature=κ)

    # 6. Map to action
    action = policy_actions[idx % len(policy_actions)]

    # Update bandit statistics with a synthetic reward (for demo purposes)
    synthetic_reward = g * (1.0 - r) + κ[idx % κ.size]  # loosely tied to the computed quantities
    update = BanditUpdate(
        context_id=context_id,
        action_id=action.action_id,
        reward=synthetic_reward,
        propensity=action.propensity,
    )
    update_policy(update)

    # Return a fresh BanditAction reflecting the updated expected reward
    updated_reward = _reward(action.action_id)
    return BanditAction(
        action_id=action.action_id,
        propensity=action.propensity,
        expected_reward=updated_reward,
        confidence_bound=math.sqrt(
            2 * math.log(1.0 / max(action.propensity, 1e-12)) / max(_POLICY.get(action.action_id, [0, 1])[1], 1)
        ),
        algorithm=action.algorithm,
    )


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset: 50 samples, 20 features
    rng = np.random.default_rng(42)
    X_demo = rng.normal(size=(50, 20))

    # Define a small set of dummy bandit actions
    dummy_actions = [
        BanditAction(
            action_id=f"act_{i}",
            propensity=0.1 + 0.05 * i,
            expected_reward=0.0,
            confidence_bound=0.0,
            algorithm="hybrid",
        )
        for i in range(5)
    ]

    # Run a single hybrid decision
    selected = hybrid_decision(
        context_id="ctx_001",
        X=X_demo,
        target_dim=8,
        unique_qi=7,
        total_records=100,
        policy_actions=dummy_actions,
    )

    print("Selected action:", selected)
    print("Current policy statistics:", _POLICY)