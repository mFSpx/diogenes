# DARWIN HAMMER — match 5219, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1820_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1204_s3.py (gen6)
# born: 2026-05-30T00:00:54Z

"""Hybrid Algorithm integrating:
- Parent A: ternary‑vector based structural signature and broadcast probability.
- Parent B: sheaf Laplacian, tropical‑maxplus algebra and Fisher‑score weighting.

Mathematical bridge:
The ternary signature **t ∈ {‑1,0,1}ⁿ** is used to scale the edges of the
sheaf Laplacian **L** (produced from node dimensions and edge list).  The
scaled Laplacian **L̃ = (1 + F/κ) · diag(s) · L · diag(s)** where **s** is a
similarity vector between **t** and each node’s dimension vector and **F**
is a Fisher‑score derived from a data sample.  The resulting matrix feeds
the tropical‑maxplus algebra **(A ⊗ B)ᵢⱼ = maxₖ (Aᵢₖ + Bₖⱼ)**, producing a
resource‑aware weight matrix that simultaneously respects the graph‑based
leader‑election signature and the statistical privacy weighting.
"""

import json
import hashlib
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – ternary signature utilities
# ----------------------------------------------------------------------
TERNARY_DIMS = 12  # dimension of ternary signature


def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def payload_hash(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> str:
    """Deterministic SHA‑256 hash of the payload dictionary."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(
    raw_command: str, normalized_intent: str, context: Dict[str, Any]
) -> np.ndarray:
    """Generate a deterministic ternary vector from payload data."""
    h = payload_hash(raw_command, normalized_intent, context)
    hi = int(h, 16)
    vec = np.empty(TERNARY_DIMS, dtype=int)
    for i in range(TERNARY_DIMS):
        vec[i] = (hi % 3) - 1  # map {0,1,2} → {-1,0,1}
        hi //= 3
    return vec


def base_broadcast_probability(phase: int, step: int) -> float:
    """Simple decreasing probability used in the original MIS broadcast."""
    return (0.5 ** phase) * (0.9 ** step)


# ----------------------------------------------------------------------
# Parent B – sheaf Laplacian and tropical‑maxplus utilities
# ----------------------------------------------------------------------
class Sheaf:
    """Light‑weight sheaf storing node dimensions and undirected edges."""

    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)  # node → dimension (scalar for simplicity)
        self.edges = list(edge_list)

    def compute_laplacian(self) -> np.ndarray:
        """Return a signed incidence Laplacian (not normalized)."""
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            L[u, v] = -1.0
            L[v, u] = 1.0
        return L


def tropical_maxplus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical max‑plus product: (A ⊗ B)ᵢⱼ = maxₖ (Aᵢₖ + Bₖⱼ)
    Matrices must be conformable (A.shape[1] == B.shape[0]).
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical max‑plus multiplication.")
    result = np.full((A.shape[0], B.shape[1]), -np.inf, dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = np.max(A[i, :] + B[:, j])
    return result


# ----------------------------------------------------------------------
# Hybrid components – mathematical bridge
# ----------------------------------------------------------------------
def fisher_score(data: np.ndarray) -> float:
    """
    Simple Fisher‑score surrogate: variance‑to‑mean ratio.
    For a 1‑D sample this equals (Var/Mean²) which grows with dispersion.
    """
    if data.size == 0:
        return 0.0
    mean = np.mean(data)
    if mean == 0:
        return 0.0
    var = np.var(data, ddof=1)
    return var / (mean ** 2)


def signature_similarity(
    signature: np.ndarray, node_dims: Dict[int, int]
) -> np.ndarray:
    """
    Compute a similarity weight for each node by dot‑product of the ternary
    signature with a replicated node‑dimension vector (converted to {-1,0,1}
    by sign).  Result is normalized to [0,1].
    """
    n = len(node_dims)
    sim = np.empty(n, dtype=float)
    # Build a simple node feature vector: sign(dim) ∈ {-1,0,1}
    node_feat = np.array([np.sign(dim) for _, dim in sorted(node_dims.items())])
    # Pad or truncate to match signature length
    if node_feat.size < signature.size:
        pad = np.zeros(signature.size - node_feat.size, dtype=int)
        node_feat = np.concatenate([node_feat, pad])
    else:
        node_feat = node_feat[: signature.size]
    # Cosine‑like similarity (dot product) scaled to [0,1]
    dot = signature.astype(float) @ node_feat.astype(float)
    max_abs = signature.size  # max possible absolute dot product
    sim[:] = (dot + max_abs) / (2 * max_abs)
    return sim


def hybrid_weighted_laplacian(
    sheaf: Sheaf,
    signature: np.ndarray,
    fisher_data: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Build the hybrid Laplacian:
        L̃ = (1 + F/κ) · D_s · L · D_s
    where D_s = diag(similarity_vector) and F is the Fisher score.
    """
    L = sheaf.compute_laplacian()
    F = fisher_score(fisher_data)
    scale = 1.0 + F / kappa
    sim_vec = signature_similarity(signature, sheaf.node_dims)
    D = np.diag(sim_vec)
    L_tilde = scale * D @ L @ D
    return L_tilde


def hybrid_broadcast_probability(
    phase: int,
    step: int,
    signature: np.ndarray,
    sheaf: Sheaf,
    fisher_data: np.ndarray,
    kappa: float = 1.0,
) -> float:
    """
    Combine the base broadcast probability with the hybrid Laplacian
    scaling factor.  The scalar factor is the average absolute entry of the
    hybrid Laplacian, acting as a network‑wide weight.
    """
    base = base_broadcast_probability(phase, step)
    L_tilde = hybrid_weighted_laplacian(sheaf, signature, fisher_data, kappa)
    weight = np.mean(np.abs(L_tilde))
    return base * (1.0 + weight)


def hybrid_resource_matrix(
    sheaf: Sheaf,
    signature: np.ndarray,
    fisher_data: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Produce a resource matrix by applying tropical max‑plus to the hybrid
    Laplacian with itself (i.e. L̃ ⊗ L̃).  The resulting matrix can be used
    for routing or leader‑election cost evaluation.
    """
    L_tilde = hybrid_weighted_laplacian(sheaf, signature, fisher_data, kappa)
    # Ensure square for self‑product
    if L_tilde.shape[0] != L_tilde.shape[1]:
        # Pad to square with -inf (neutral element for max‑plus)
        size = max(L_tilde.shape)
        padded = np.full((size, size), -np.inf, dtype=float)
        padded[: L_tilde.shape[0], : L_tilde.shape[1]] = L_tilde
        L_tilde = padded
    return tropical_maxplus(L_tilde, L_tilde)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy payload
    raw_cmd = "GET /resource?id=42"
    intent = "fetch_resource"
    ctx = {"user": "alice", "session": "xyz"}

    # Generate ternary signature (Parent A)
    sig = ternary_vector(raw_cmd, intent, ctx)

    # Define a tiny graph for the sheaf (Parent B)
    node_dimensions = {0: 3, 1: -2, 2: 5}
    edges = [(0, 1), (1, 2), (2, 0)]

    sheaf = Sheaf(node_dimensions, edges)

    # Synthetic data for Fisher score (e.g., latency measurements)
    fisher_sample = np.array([12.3, 11.7, 13.2, 12.9, 11.5])

    # Compute hybrid broadcast probability
    prob = hybrid_broadcast_probability(
        phase=2,
        step=3,
        signature=sig,
        sheaf=sheaf,
        fisher_data=fisher_sample,
        kappa=2.0,
    )
    print(f"Hybrid broadcast probability: {prob:.6f}")

    # Compute hybrid resource matrix
    res_mat = hybrid_resource_matrix(
        sheaf=sheaf,
        signature=sig,
        fisher_data=fisher_sample,
        kappa=2.0,
    )
    print("Hybrid resource matrix (tropical max‑plus of scaled Laplacian):")
    print(res_mat)