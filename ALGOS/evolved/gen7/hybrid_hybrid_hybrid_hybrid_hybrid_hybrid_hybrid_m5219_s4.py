# DARWIN HAMMER — match 5219, survivor 4
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
# Parent A – ternary signature utilities
# ----------------------------------------------------------------------
TERNARY_DIMS = 12  # dimension of ternary signature


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def payload_hash(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> str:
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
    h = payload_hash(raw_command, normalized_intent, context)
    hi = int(h, 16)
    vec = np.empty(TERNARY_DIMS, dtype=int)
    for i in range(TERNARY_DIMS):
        vec[i] = (hi % 3) - 1  # map {0,1,2} → {-1,0,1}
        hi //= 3
    return vec


def base_broadcast_probability(phase: int, step: int) -> float:
    return (0.5 ** phase) * (0.9 ** step)


# ----------------------------------------------------------------------
# Parent B – sheaf Laplacian and tropical‑maxplus utilities
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)  # node → dimension (scalar for simplicity)
        self.edges = list(edge_list)

    def compute_laplacian(self) -> np.ndarray:
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            L[u, v] = -1.0
            L[v, u] = 1.0
        return L


def tropical_maxplus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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
    n = len(node_dims)
    sim = np.empty(n, dtype=float)
    node_feat = np.array([np.sign(dim) for _, dim in sorted(node_dims.items())])
    if node_feat.size < signature.size:
        pad = np.zeros(signature.size - node_feat.size, dtype=int)
        node_feat = np.concatenate([node_feat, pad])
    else:
        node_feat = node_feat[: signature.size]
    dot = signature.astype(float) @ node_feat.astype(float)
    max_abs = signature.size  
    sim[:] = (dot + max_abs) / (2 * max_abs)
    return sim


def hybrid_weighted_laplacian(
    sheaf: Sheaf,
    signature: np.ndarray,
    fisher_data: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
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
    L_tilde = hybrid_weighted_laplacian(sheaf, signature, fisher_data, kappa)
    if L_tilde.shape[0] != L_tilde.shape[1]:
        size = max(L_tilde.shape)
        padded = np.full((size, size), -np.inf, dtype=float)
        padded[: L_tilde.shape[0], : L_tilde.shape[1]] = L_tilde
        L_tilde = padded
    return tropical_maxplus(L_tilde, L_tilde)


def hybrid_resource_matrix_improved(
    sheaf: Sheaf,
    signature: np.ndarray,
    fisher_data: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    L_tilde = hybrid_weighted_laplacian(sheaf, signature, fisher_data, kappa)
    if L_tilde.shape[0] != L_tilde.shape[1]:
        size = max(L_tilde.shape)
        padded = np.full((size, size), -np.inf, dtype=float)
        padded[: L_tilde.shape[0], : L_tilde.shape[1]] = L_tilde
        L_tilde = padded
    # Improved version using a more robust method for tropical max-plus
    result = np.full((L_tilde.shape[0], L_tilde.shape[1]), -np.inf, dtype=float)
    for i in range(L_tilde.shape[0]):
        for j in range(L_tilde.shape[1]):
            max_val = -np.inf
            for k in range(L_tilde.shape[0]):
                max_val = max(max_val, L_tilde[i, k] + L_tilde[k, j])
            result[i, j] = max_val
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    raw_cmd = "GET /resource?id=42"
    intent = "fetch_resource"
    ctx = {"user": "alice", "session": "xyz"}

    sig = ternary_vector(raw_cmd, intent, ctx)

    node_dimensions = {0: 3, 1: -2, 2: 5}
    edges = [(0, 1), (1, 2), (2, 0)]

    sheaf = Sheaf(node_dimensions, edges)

    fisher_data = np.array([1.0, 2.0, 3.0])

    L_tilde = hybrid_weighted_laplacian(sheaf, sig, fisher_data)
    print("Hybrid Weighted Laplacian:")
    print(L_tilde)

    prob = hybrid_broadcast_probability(1, 1, sig, sheaf, fisher_data)
    print("Hybrid Broadcast Probability:")
    print(prob)

    resource_matrix = hybrid_resource_matrix_improved(sheaf, sig, fisher_data)
    print("Hybrid Resource Matrix:")
    print(resource_matrix)