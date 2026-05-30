# DARWIN HAMMER — match 5219, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1820_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1204_s3.py (gen6)
# born: 2026-05-30T00:00:54Z

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
# Configuration
# ----------------------------------------------------------------------
TERNARY_DIMS = 12  # dimension of ternary signature
EPS = 1e-12        # numerical tolerance


# ----------------------------------------------------------------------
# Utility functions (Parent A)
# ----------------------------------------------------------------------
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
# Sheaf Laplacian utilities (Parent B)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Minimal sheaf representation.
    node_dims: mapping node → integer dimension (scalar for simplicity)
    edge_list : list of undirected edges (u, v)
    """

    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._validate()

    def _validate(self) -> None:
        n = len(self.node_dims)
        for u, v in self.edges:
            if u not in self.node_dims or v not in self.node_dims:
                raise ValueError("Edge refers to undefined node")
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("Node indices must be contiguous from 0")

    def adjacency_matrix(self) -> np.ndarray:
        """Unsigned adjacency matrix A where A[u, v] = 1 if (u,v) ∈ edges."""
        n = len(self.node_dims)
        A = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            A[u, v] = 1.0
            A[v, u] = 1.0
        return A

    def degree_matrix(self) -> np.ndarray:
        """Diagonal degree matrix D where D[i,i] = sum_j A[i,j]."""
        A = self.adjacency_matrix()
        deg = np.sum(A, axis=1)
        return np.diag(deg)

    def compute_laplacian(self) -> np.ndarray:
        """
        Symmetric combinatorial Laplacian L = D - A.
        This retains the key property L·1 = 0 and is positive semi‑definite.
        """
        return self.degree_matrix() - self.adjacency_matrix()


def tropical_maxplus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical max‑plus product: (A ⊗ B)ᵢⱼ = maxₖ (Aᵢₖ + Bₖⱼ).
    Matrices must be conformable (A.shape[1] == B.shape[0]).
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical max‑plus multiplication.")
    # Use broadcasting for speed: (n, p, 1) + (1, p, m) → (n, p, m) then max over p
    left = A[:, :, np.newaxis]          # (n, p, 1)
    right = B[np.newaxis, :, :]         # (1, p, m)
    summed = left + right                # (n, p, m)
    return np.max(summed, axis=1)        # (n, m)


# ----------------------------------------------------------------------
# Statistical bridge (Parent B)
# ----------------------------------------------------------------------
def fisher_score(data: np.ndarray) -> float:
    """
    Surrogate Fisher‑score: variance‑to‑mean‑squared ratio.
    Returns 0 for empty or zero‑mean data.
    """
    if data.size == 0:
        return 0.0
    mean = np.mean(data)
    if abs(mean) < EPS:
        return 0.0
    var = np.var(data, ddof=1)
    return var / (mean ** 2)


def fisher_diagonal(data: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """
    Build a diagonal matrix I + (F/κ)·I where F is the Fisher score.
    This injects statistical dispersion uniformly across all dimensions.
    """
    F = fisher_score(data)
    scale = 1.0 + F / max(kappa, EPS)
    n = data.shape[0] if data.ndim == 1 else data.shape[1]
    return np.eye(n) * scale


# ----------------------------------------------------------------------
# Signature‑to‑node similarity (deepened integration)
# ----------------------------------------------------------------------
def node_signature_features(node_dims: Dict[int, int]) -> np.ndarray:
    """
    Encode each node dimension as a ternary feature vector of length TERNARY_DIMS.
    The encoding repeats the sign of the dimension to fill the vector.
    """
    signs = np.array([np.sign(dim) for _, dim in sorted(node_dims.items())], dtype=int)
    # Map {-1,0,1} → {0,1,2} for repeat indexing
    signs_mapped = signs + 1
    # Tile to required length and truncate
    tiled = np.tile(signs_mapped, (TERNARY_DIMS // signs_mapped.size + 1))[:TERNARY_DIMS]
    # Convert back to {-1,0,1}
    return tiled - 1


def signature_similarity(
    signature: np.ndarray, node_dims: Dict[int, int]
) -> np.ndarray:
    """
    Compute a per‑node similarity weight.
    For each node we compare the global ternary signature with the node's
    own ternary encoding via cosine similarity, then map to [0,1].
    """
    n = len(node_dims)
    sim = np.empty(n, dtype=float)

    # Global signature normalised
    sig_norm = np.linalg.norm(signature) + EPS

    for idx, (_, dim) in enumerate(sorted(node_dims.items())):
        node_feat = node_signature_features({idx: dim})
        node_norm = np.linalg.norm(node_feat) + EPS
        dot = float(signature @ node_feat)
        cos_sim = dot / (sig_norm * node_norm)  # ∈ [-1,1]
        sim[idx] = (cos_sim + 1.0) / 2.0        # map to [0,1]

    return sim


# ----------------------------------------------------------------------
# Hybrid constructions
# ----------------------------------------------------------------------
def hybrid_weighted_laplacian(
    sheaf: Sheaf,
    signature: np.ndarray,
    fisher_data: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Deeply fused Laplacian:
        L̃ = D_s · F_diag · L · D_s
    where
        L   – symmetric combinatorial Laplacian,
        D_s – diag(similarity_vector) from signature‑node alignment,
        F_diag – Fisher‑information diagonal scaling.
    The construction preserves the zero‑row‑sum property because
    D_s is applied symmetrically on both sides.
    """
    L = sheaf.compute_laplacian()
    sim_vec = signature_similarity(signature, sheaf.node_dims)
    D_s = np.diag(sim_vec)

    # Fisher diagonal must match the size of L (square)
    F_diag = fisher_diagonal(fisher_data, kappa)
    if F_diag.shape[0] != L.shape[0]:
        # Broadcast or truncate to match graph size
        size = L.shape[0]
        if F_diag.shape[0] < size:
            pad = np.eye(size - F_diag.shape[0]) * np.mean(np.diag(F_diag))
            F_diag = np.block([[F_diag, np.zeros((F_diag.shape[0], size - F_diag.shape[0]))],
                               [np.zeros((size - F_diag.shape[0], F_diag.shape[0])), pad]])
        else:
            F_diag = F_diag[:size, :size]

    L_tilde = D_s @ F_diag @ L @ D_s
    # Re‑enforce exact zero row‑sum (numerical drift)
    L_tilde -= np.mean(L_tilde, axis=1, keepdims=True)
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
    Combine the base probability with a spectral‑norm based weight.
    The spectral norm (largest singular value) reflects the strongest
    diffusion mode of the hybrid Laplacian and is less sensitive to outliers
    than the raw mean absolute entry.
    """
    base = base_broadcast_probability(phase, step)
    L_tilde = hybrid_weighted_laplacian(sheaf, signature, fisher_data, kappa)
    # Spectral norm via numpy.linalg.svd (fast for small matrices)
    sigma_max = np.linalg.svd(L_tilde, compute_uv=False, hermitian=True)[0]
    weight = 1.0 + sigma_max / max(L_tilde.shape[0], EPS)
    return base * weight


def hybrid_resource_matrix(
    sheaf: Sheaf,
    signature: np.ndarray,
    fisher_data: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Produce a resource matrix by tropical max‑plus self‑product of the hybrid
    Laplacian.  The result is square and can be interpreted as a tropical
    distance matrix that respects both structural (graph) and statistical
    (Fisher) information.
    """
    L_tilde = hybrid_weighted_laplacian(sheaf, signature, fisher_data, kappa)

    # Ensure finite entries for tropical algebra; replace zeros on diagonal with -inf
    # (the neutral element) only where appropriate.
    L_finite = np.where(np.isfinite(L_tilde), L_tilde, -np.inf)

    return tropical_maxplus(L_finite, L_finite)


# ----------------------------------------------------------------------
# Smoke test (executed when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy payload
    raw_cmd = "GET /resource?id=42"
    intent = "fetch_resource"
    ctx = {"user": "alice", "session": "xyz"}

    # Generate ternary signature (Parent A)
    sig = ternary_vector(raw_cmd, intent, ctx)

    # Tiny graph for the sheaf (Parent B)
    node_dimensions = {0: 3, 1: -2, 2: 5}
    edges = [(0, 1), (1, 2), (2, 0)]

    sheaf = Sheaf(node_dimensions, edges)

    # Synthetic data for Fisher score (e.g., latency measurements)
    fisher_samples = np.array([120.5, 115.2, 130.1, 118.7, 122.3])

    # Compute hybrid objects
    L_hybrid = hybrid_weighted_laplacian(sheaf, sig, fisher_samples, kappa=2.0)
    prob = hybrid_broadcast_probability(phase=2, step=3, signature=sig,
                                        sheaf=sheaf, fisher_data=fisher_samples,
                                        kappa=2.0)
    R = hybrid_resource_matrix(sheaf, sig, fisher_samples, kappa=2.0)

    # Simple sanity prints
    print("Hybrid Laplacian (L̃):")
    print(L_hybrid)
    print("\nHybrid broadcast probability:")
    print(prob)
    print("\nHybrid resource matrix (R = L̃ ⊗ L̃):")
    print(R)