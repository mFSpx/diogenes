# DARWIN HAMMER — match 1510, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1019_s1.py (gen5)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (gen2)
# born: 2026-05-29T23:37:06Z

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CertaintyFlag:
    """Epistemic confidence attached to an edge."""
    label: str
    confidence_bps: int  # basis points, 0‑10000


# ----------------------------------------------------------------------
# Stylometry → weighted graph → curvature → confidence
# ----------------------------------------------------------------------
def stylometry_features(text: str) -> Dict[str, int]:
    """Extract a tiny stylometric fingerprint from *text*."""
    pronouns = {
        "i", "you", "he", "she", "we", "they",
        "me", "him", "her", "us", "them"
    }
    conjunctions = {
        "and", "or", "but", "nor", "so", "yet",
        "for", "although"
    }
    verbs = {
        "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "do", "does"
    }

    tokens = [t.lower().strip(".,!?;:\"'()[]") for t in text.split()]
    counts = Counter(tokens)

    return {
        "pronoun": sum(counts[w] for w in pronouns),
        "conj":    sum(counts[w] for w in conjunctions),
        "verb":    sum(counts[w] for w in verbs),
        "len":     len(tokens),
    }


def build_weighted_graph(
    features_list: List[Dict[str, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a symmetric similarity matrix W (cosine similarity) and node
    strengths (L2 norm of the feature vectors).

    Returns
    -------
    W : (n, n) ndarray
        Cosine similarity in [0, 1].
    strengths : (n,) ndarray
        L2 norm of each feature vector (≥ 1e‑12 to avoid division by zero).
    """
    n = len(features_list)
    # Preserve a deterministic ordering of feature dimensions
    keys = sorted({k for d in features_list for k in d})
    mat = np.zeros((n, len(keys)), dtype=float)

    for i, d in enumerate(features_list):
        for j, k in enumerate(keys):
            mat[i, j] = d.get(k, 0)

    strengths = np.linalg.norm(mat, axis=1) + 1e-12
    normed = mat / strengths[:, None]
    W = np.clip(normed @ normed.T, 0.0, 1.0)
    np.fill_diagonal(W, 1.0)
    return W, strengths


def curvature_to_confidence(
    W: np.ndarray,
    strengths: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], CertaintyFlag]]:
    """
    Compute a surrogate Ollivier‑Ricci curvature on each edge and map it to a
    confidence value in [0, 10000].

    The curvature used here is
        κ_{ij} = 1 - |w_i - w_j| / (w_i + w_j + ε)

    The final confidence combines curvature and the underlying similarity
    matrix W:
        conf_{ij} = κ_{ij} * W_{ij} * 10000

    Returns
    -------
    confidence : (n, n) ndarray of ints in [0, 10000]
    cert_dict : mapping (i, j) → CertaintyFlag (symmetric)
    """
    w_i = strengths[:, None]
    w_j = strengths[None, :]
    curvature = 1.0 - np.abs(w_i - w_j) / (w_i + w_j + eps)
    curvature = np.clip(curvature, 0.0, 1.0)          # ensure valid range

    # Fuse curvature with the original similarity; this deepens the integration
    confidence_float = curvature * W * 10000.0
    confidence = np.rint(confidence_float).astype(int)

    cert_dict: Dict[Tuple[int, int], CertaintyFlag] = {}
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            conf = confidence[i, j]
            label = "high" if conf > 7000 else "low"
            flag = CertaintyFlag(label=label, confidence_bps=int(conf))
            cert_dict[(i, j)] = flag
            cert_dict[(j, i)] = flag  # symmetry

    return confidence, cert_dict


# ----------------------------------------------------------------------
# Tropical broadcast, Hoeffding test, simulated annealing
# ----------------------------------------------------------------------
def max_plus_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical (max‑plus) matrix multiplication:
        (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})
    """
    # Shape tricks: (n, m, 1) + (1, m, n) → (n, m, n) then max over axis=1
    return np.max(A[:, :, None] + B[None, :, :], axis=1)


def tropical_fixed_point(
    confidence: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Compute the tropical power of the confidence matrix until convergence.
    Returns the broadcast vector b where b_i = max_j (C_{ij} + b_j).
    """
    C = confidence.astype(float) / 10000.0          # normalise to [0,1]
    b = np.max(C, axis=1)                           # initialise with self‑influence

    for _ in range(max_iter):
        b_new = np.max(C + b[None, :], axis=1)
        if np.max(np.abs(b_new - b)) < tol:
            break
        b = b_new

    return np.clip(b, 0.0, 1.0)


def hoeffding_candidates(
    broadcast: np.ndarray,
    n_samples: int = 30,
    delta: float = 0.05,
) -> List[int]:
    """
    Apply Hoeffding's inequality to each broadcast value.
    Nodes with a statistically significant excess over 0.5 are returned.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    epsilon = math.sqrt(math.log(2.0 / delta) / (2.0 * n_samples))
    threshold = 0.5 + epsilon
    return [i for i, val in enumerate(broadcast) if val > threshold]


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance probability."""
    if delta_e < 0:
        return 1.0
    try:
        return math.exp(-delta_e / max(temperature, 1e-12))
    except OverflowError:
        return 0.0


def simulated_annealing_update(
    current_leaders: List[int],
    candidates: List[int],
    broadcast: np.ndarray,
    temperature: float,
) -> List[int]:
    """
    Decide whether to replace *current_leaders* with *candidates*.

    Energy is defined as the negative sum of broadcast strengths of the leader
    set (higher broadcast ⇒ lower energy).  ΔE = E_new - E_current.
    """
    energy_current = -float(np.sum(broadcast[current_leaders])) if current_leaders else 0.0
    energy_candidate = -float(np.sum(broadcast[candidates])) if candidates else 0.0
    delta_e = energy_candidate - energy_current
    prob = acceptance_probability(delta_e, temperature)

    return candidates.copy() if random.random() < prob else current_leaders.copy()


# ----------------------------------------------------------------------
# Integrated hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_stylometry_tropical_election(
    texts: List[str],
    temperature: float = 1.0,
    n_samples: int = 30,
    delta: float = 0.05,
    seed: int | None = None,
) -> List[int]:
    """
    End‑to‑end hybrid algorithm.

    Parameters
    ----------
    texts : list of str
        Input documents, one per node.
    temperature : float, optional
        Annealing temperature (higher ⇒ more exploratory).
    n_samples : int, optional
        Number of pseudo‑samples for the Hoeffding bound.
    delta : float, optional
        Confidence level for Hoeffding (smaller ⇒ stricter).
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    leaders : list of int
        Indices of the elected leader nodes.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Stylometry extraction
    features = [stylometry_features(t) for t in texts]

    # 2. Graph construction and curvature‑derived confidence
    W, strengths = build_weighted_graph(features)
    confidence, _ = curvature_to_confidence(W, strengths)

    # 3. Tropical broadcast (fixed‑point)
    broadcast = tropical_fixed_point(confidence)

    # 4. Hoeffding candidate selection
    candidates = hoeffding_candidates(broadcast, n_samples=n_samples, delta=delta)

    # 5. Simulated‑annealing leader update (start with empty set)
    leaders = simulated_annealing_update(
        current_leaders=[],
        candidates=candidates,
        broadcast=broadcast,
        temperature=temperature,
    )
    return leaders

# ----------------------------------------------------------------------
# Small smoke test (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_texts = [
        "I think therefore I am. And you?",
        "She runs and jumps, but he stays.",
        "We have been doing this for a long time.",
        "The quick brown fox jumps over the lazy dog.",
        "Although it was raining, we went out.",
    ]
    elected = hybrid_stylometry_tropical_election(
        demo_texts,
        temperature=0.8,
        n_samples=40,
        delta=0.01,
        seed=42,
    )
    print("Elected leader indices:", elected)