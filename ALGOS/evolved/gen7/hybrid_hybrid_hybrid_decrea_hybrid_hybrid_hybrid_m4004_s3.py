# DARWIN HAMMER — match 4004, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_decreasing_pr_hybrid_fisher_locali_m215_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_nlms_h_hybrid_hybrid_gliner_m1613_s1.py (gen6)
# born: 2026-05-29T23:53:10Z

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Hashable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Epistemic certainty handling
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Mapping from flag to a confidence weight in [0,1]
_EPISTEMIC_WEIGHT = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "SURE_MAYBE": 0.3,
    "BULLSHIT": 0.0,
}


def epistemic_weight(flag: str) -> float:
    """Return a numeric confidence for a given epistemic flag."""
    if flag not in _EPISTEMIC_WEIGHT:
        raise ValueError(f"Unknown epistemic flag: {flag}")
    return _EPISTEMIC_WEIGHT[flag]


# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Monotonically decreasing pruning probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for a simple Bayesian update."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("All probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def hamming_similarity(a: str, b: str) -> float:
    """Hamming similarity for binary strings of possibly different length."""
    # Pad the shorter string with zeros to align lengths
    max_len = max(len(a), len(b))
    a_padded = a.ljust(max_len, "0")
    b_padded = b.ljust(max_len, "0")
    mismatches = sum(ch1 != ch2 for ch1, ch2 in zip(a_padded, b_padded))
    return 1.0 - mismatches / max_len


def rbf_kernel(a: Tuple[float, float], b: Tuple[float, float], sigma: float = 1.0) -> float:
    """Radial‑basis‑function kernel."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    sq_dist = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return math.exp(-sq_dist / (2 * sigma ** 2))


def fuse_kernels(
    S: np.ndarray,
    K: np.ndarray,
    epistemic: np.ndarray,
    base_alpha: float = 0.5,
) -> np.ndarray:
    """
    Deep fusion of Hamming (S) and RBF (K) kernels.
    The epistemic matrix (values in [0,1]) modulates the local mixing ratio:
        alpha_ij = base_alpha * (1 - epistemic_ij) + (1 - base_alpha) * epistemic_ij
    """
    if not (0.0 <= base_alpha <= 1.0):
        raise ValueError("base_alpha must be in [0,1]")
    # Ensure epistemic is broadcastable to the kernel shape
    epistemic = np.asarray(epistemic, dtype=float)
    if epistemic.shape != S.shape:
        raise ValueError("Epistemic matrix must have the same shape as kernel matrices")
    alpha_local = base_alpha * (1.0 - epistemic) + (1.0 - base_alpha) * epistemic
    return alpha_local * S + (1.0 - alpha_local) * K


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Edge:
    """Edge in the minimum‑cost tree with an epistemic flag."""
    u: Hashable
    v: Hashable
    flag: str  # one of EPISTEMIC_FLAGS


# ----------------------------------------------------------------------
# Pruning that respects epistemic certainty
# ----------------------------------------------------------------------
def prune_edges(
    edges: List[Edge],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: Optional[int] = None,
) -> List[Edge]:
    """Probabilistic pruning where higher epistemic confidence reduces removal chance."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    kept: List[Edge] = []
    for e in edges:
        # Base probability is scaled by (1 - epistemic confidence)
        conf = epistemic_weight(e.flag)
        removal_prob = p * (1.0 - conf)
        if rng.random() >= removal_prob:
            kept.append(e)
    return kept


# ----------------------------------------------------------------------
# Kernel construction with epistemic integration
# ----------------------------------------------------------------------
def build_fused_kernel(
    features: List[Tuple[float, float]],
    epistemic_vec: List[float],
    sigma: float = 1.0,
    base_alpha: float = 0.5,
) -> np.ndarray:
    """
    Build a fused kernel matrix for the feature set.
    Epistemic confidence per node is turned into a matrix via outer product.
    """
    n = len(features)
    # Hamming similarity between binary hashes of coordinates
    # First we convert each coordinate pair to a binary string using a simple quantisation.
    def coord_to_bin(pt: Tuple[float, float]) -> str:
        # 16‑bit quantisation per dimension, concatenated
        scale = 2 ** 16 - 1
        x = int(np.clip(pt[0], 0.0, 1.0) * scale)
        y = int(np.clip(pt[1], 0.0, 1.0) * scale)
        return f"{x:016b}{y:016b}"

    hashes = [coord_to_bin(p) for p in features]

    S = np.empty((n, n), dtype=float)
    K = np.empty((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            s_val = hamming_similarity(hashes[i], hashes[j])
            k_val = rbf_kernel(features[i], features[j], sigma)
            S[i, j] = S[j, i] = s_val
            K[i, j] = K[j, i] = k_val

    # Epistemic matrix via outer product (pairwise confidence)
    epi_mat = np.outer(epistemic_vec, epistemic_vec)
    return fuse_kernels(S, K, epi_mat, base_alpha)


# ----------------------------------------------------------------------
# Label scoring
# ----------------------------------------------------------------------
def label_hash(label: str, bits: int = 32) -> str:
    """Deterministic binary hash of a label (simple built‑in hash, masked to `bits`)."""
    h = hash(label) & ((1 << bits) - 1)
    return f"{h:0{bits}b}"


def score_label(
    label: str,
    features: List[Tuple[float, float]],
    fused_kernel: np.ndarray,
    epistemic_vec: List[float],
) -> float:
    """
    Score a label by aggregating its similarity to every feature.
    Similarity = fused_kernel[i,i] * Hamming(label, feature_hash)
    The contribution of each feature is weighted by its epistemic confidence.
    """
    lbl_bin = label_hash(label)
    n = len(features)

    # Pre‑compute binary hashes for features (same scheme as in `build_fused_kernel`)
    def coord_to_bin(pt: Tuple[float, float]) -> str:
        scale = 2 ** 16 - 1
        x = int(np.clip(pt[0], 0.0, 1.0) * scale)
        y = int(np.clip(pt[1], 0.0, 1.0) * scale)
        return f"{x:016b}{y:016b}"

    feature_hashes = [coord_to_bin(p) for p in features]

    # Vectorised computation
    hamming_vals = np.array(
        [hamming_similarity(lbl_bin, fh) for fh in feature_hashes], dtype=float
    )
    diag_vals = np.diag(fused_kernel)  # self‑similarities capture combined geometry
    epistemic_arr = np.array(epistemic_vec, dtype=float)

    weighted = diag_vals * hamming_vals * epistemic_arr
    return float(weighted.mean()) if n > 0 else 0.0


# ----------------------------------------------------------------------
# Main hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_algorithm(
    edges: List[Edge],
    features: List[Tuple[float, float]],
    labels: List[str],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    sigma: float = 1.0,
    base_alpha: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    End‑to‑end hybrid procedure:
      1. Prune edges using epistemic‑aware probability.
      2. Derive per‑node epistemic confidence from the surviving edges.
      3. Build a deep‑fused kernel that respects both geometry and epistemic certainty.
      4. Score each label against the feature set.
    """
    # ------------------------------------------------------------------
    # 1. Edge pruning
    # ------------------------------------------------------------------
    pruned = prune_edges(edges, t, lam, alpha, seed)

    # ------------------------------------------------------------------
    # 2. Node‑level epistemic confidence
    #    (average confidence of incident edges, fallback to 0.5 if isolated)
    # ------------------------------------------------------------------
    node_conf: Dict[Hashable, List[float]] = {}
    for e in pruned:
        w = epistemic_weight(e.flag)
        node_conf.setdefault(e.u, []).append(w)
        node_conf.setdefault(e.v, []).append(w)

    epistemic_vec = []
    for idx, _ in enumerate(features):
        # Map feature index to a node identifier if available; otherwise use a dummy key.
        node_id = idx  # assumes feature ordering aligns with node identifiers
        confidences = node_conf.get(node_id, [])
        if confidences:
            epistemic_vec.append(sum(confidences) / len(confidences))
        else:
            epistemic_vec.append(0.5)  # neutral prior

    # ------------------------------------------------------------------
    # 3. Kernel construction
    # ------------------------------------------------------------------
    fused = build_fused_kernel(features, epistemic_vec, sigma, base_alpha)

    # ------------------------------------------------------------------
    # 4. Label scoring
    # ------------------------------------------------------------------
    scores = {
        label: score_label(label, features, fused, epistemic_vec) for label in labels
    }
    return scores


# ----------------------------------------------------------------------
# Example usage (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny tree with epistemic flags
    edge_list = [
        Edge(0, 1, "FACT"),
        Edge(1, 2, "PROBABLE"),
        Edge(2, 3, "POSSIBLE"),
        Edge(3, 4, "BULLSHIT"),
    ]

    # Feature coordinates are normalised to [0,1] for the quantisation step
    feature_coords = [
        (0.1, 0.2),
        (0.3, 0.4),
        (0.5, 0.6),
        (0.7, 0.8),
        (0.9, 1.0),
    ]

    label_set = ["cat", "dog", "mouse", "elephant"]

    result = hybrid_algorithm(
        edges=edge_list,
        features=feature_coords,
        labels=label_set,
        t=1.0,
        lam=1.0,
        alpha=0.2,
        sigma=0.5,
        base_alpha=0.6,
        seed=42,
    )
    print(result)