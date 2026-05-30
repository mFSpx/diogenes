# DARWIN HAMMER — match 1061, survivor 4
# gen: 5
# parent_a: hybrid_nlms_hybrid_hybrid_rbf_su_m223_s1.py (gen4)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s4.py (gen2)
# born: 2026-05-29T23:34:13Z

import numpy as np
import math
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# 1. Linguistic function‑category definitions (unchanged)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set("and but or so yet for nor".split()),
}

# ----------------------------------------------------------------------
# 2. Core mathematical primitives
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    return float(np.linalg.norm(a - b))


def compute_phash(values: np.ndarray) -> int:
    """Simple perceptual hash based on mean threshold."""
    if values.size == 0:
        return 0
    avg = values.mean()
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# 3. Kernel and similarity utilities
# ----------------------------------------------------------------------
def rbf_kernel_matrix(features: Dict[int, List[float]], epsilon: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """RBF kernel on raw numeric features."""
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    # vectorised version for clarity and speed
    X = np.stack([np.asarray(features[node], dtype=np.float64) for node in nodes])
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        K[i, :] = np.vectorize(gaussian)(dists, epsilon)
    # Symmetrize (numerical noise may break exact symmetry)
    K = (K + K.T) / 2.0
    return K, nodes


def lsm_vector_from_features(features: List[float]) -> np.ndarray:
    """
    Produce a binary LSM vector for a node.
    The original code used a constant vector; we now derive it from the
    lexical content encoded in the numeric feature vector.
    For demonstration we treat each dimension as a proxy for a word
    belonging to a category (e.g. dimension 0 → pronoun, 1 → article, …).
    """
    # Map each dimension to a category cyclically
    cats = list(FUNCTION_CATS.keys())
    vec = np.zeros(len(cats), dtype=np.float64)
    for idx, val in enumerate(features):
        cat = cats[idx % len(cats)]
        # Threshold the numeric feature to decide presence (simple heuristic)
        if val > 0.5:  # arbitrary but deterministic
            vec[cats.index(cat)] = 1.0
    return vec


def lsm_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine‑like similarity derived from Euclidean distance on binary LSM vectors."""
    # Euclidean distance on binary vectors is sqrt(Hamming distance)
    # Transform to similarity in [0,1]
    dist = euclidean(v1, v2)
    max_dist = math.sqrt(v1.size)  # worst case: all bits differ
    return 1.0 - dist / max_dist


def bayesian_posterior(p_prior: float, likelihood: float, false_pos: float) -> float:
    """
    Standard Bayesian update:
        posterior = (likelihood * p_prior) / (likelihood * p_prior + false_pos * (1-p_prior))
    Guard against division by zero.
    """
    numerator = likelihood * p_prior
    denominator = numerator + false_pos * (1.0 - p_prior)
    return numerator / denominator if denominator != 0.0 else 0.0


def node_belief(p_e: np.ndarray) -> float:
    """Mean posterior over all incident edges of a node."""
    return float(p_e.mean()) if p_e.size > 0 else 0.0


def hybrid_cost(
    posteriors: np.ndarray,
    edge_geom: np.ndarray,
    lambda_: float,
    node_beliefs: np.ndarray,
    edge_weights: np.ndarray,
) -> float:
    """
    Cost = Σ (posterior * edge_geom) + λ * Σ (node_belief * edge_weight)
    Edge‑wise terms are summed over the upper triangle to avoid double counting.
    """
    # Upper‑triangle indices (i<j)
    triu = np.triu_indices_from(posteriors, k=1)
    term1 = float(np.sum(posteriors[triu] * edge_geom[triu]))
    term2 = lambda_ * float(np.sum(node_beliefs[:, None] * edge_weights[triu]))
    return term1 + term2


def nlms_update(
    weights: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalised LMS update:
        w ← w + η * (xᵀ (d - x w)) / (‖x‖² + ε)
    """
    pred = inputs @ weights
    error = targets - pred
    norm = np.linalg.norm(inputs, axis=0) ** 2 + epsilon
    correction = learning_rate * (inputs.T @ error) / norm
    return weights + correction


# ----------------------------------------------------------------------
# 4. Integrated hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_nlms_rbf_lsm_bayesian(
    features: Dict[int, List[float]],
    epsilon: float,
    p_prior: float,
    false_pos: float,
    lambda_: float,
    learning_rate: float,
) -> Tuple[np.ndarray, float]:
    """
    End‑to‑end pipeline:
        1. RBF kernel on raw numeric features.
        2. LSM vectors derived from the same numeric features.
        3. Fuse kernels by element‑wise product (deep integration).
        4. Compute Bayesian posteriors per edge.
        5. Derive node beliefs.
        6. Evaluate hybrid cost.
        7. Perform a single NLMS weight update using the fused kernel.
    Returns the updated weight vector and the scalar hybrid cost.
    """
    # ------------------------------------------------------------------
    # 1️⃣ RBF kernel
    # ------------------------------------------------------------------
    K_rbf, nodes = rbf_kernel_matrix(features, epsilon)

    # ------------------------------------------------------------------
    # 2️⃣ LSM vectors and similarity matrix
    # ------------------------------------------------------------------
    lsm_vectors = np.stack(
        [lsm_vector_from_features(features[n]) for n in nodes], axis=0
    )  # shape (n, c)
    # Pairwise LSM similarity (cosine‑like)
    S_lsm = np.empty_like(K_rbf)
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            sim = lsm_similarity(lsm_vectors[i], lsm_vectors[j])
            S_lsm[i, j] = sim
            S_lsm[j, i] = sim

    # ------------------------------------------------------------------
    # 3️⃣ Kernel fusion (deep integration)
    # ------------------------------------------------------------------
    K_fused = K_rbf * S_lsm  # element‑wise product retains positive‑definiteness

    # ------------------------------------------------------------------
    # 4️⃣ Bayesian posterior per edge (using fused similarity as likelihood)
    # ------------------------------------------------------------------
    posterior = np.empty_like(K_fused)
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            post = bayesian_posterior(p_prior, K_fused[i, j], false_pos)
            posterior[i, j] = post
            posterior[j, i] = post

    # ------------------------------------------------------------------
    # 5️⃣ Node beliefs (average posterior over incident edges)
    # ------------------------------------------------------------------
    node_beliefs = np.empty(len(nodes), dtype=np.float64)
    for i in range(len(nodes)):
        # exclude self‑loop
        peers = np.delete(posterior[i, :], i)
        node_beliefs[i] = node_belief(peers)

    # ------------------------------------------------------------------
    # 6️⃣ Edge geometry (Euclidean distance on raw features)
    # ------------------------------------------------------------------
    raw_matrix = np.stack([np.asarray(features[n], dtype=np.float64) for n in nodes])
    edge_geom = np.empty_like(K_fused)
    for i in range(len(nodes)):
        dists = np.linalg.norm(raw_matrix - raw_matrix[i], axis=1)
        edge_geom[i, :] = dists
    # Symmetrize
    edge_geom = (edge_geom + edge_geom.T) / 2.0

    # ------------------------------------------------------------------
    # 7️⃣ Hybrid cost
    # ------------------------------------------------------------------
    # Edge weights for the λ‑term – we simply use ones (can be replaced by domain‑specific values)
    edge_weights = np.ones_like(posterior)
    cost = hybrid_cost(
        posterior,
        edge_geom,
        lambda_,
        node_beliefs,
        edge_weights,
    )

    # ------------------------------------------------------------------
    # 8️⃣ NLMS weight update
    # ------------------------------------------------------------------
    # Use the fused kernel as the input matrix (each column corresponds to a node)
    inputs = K_fused.T  # shape (n, n)
    targets = node_beliefs  # shape (n,)
    weights = np.random.rand(len(nodes))
    updated_weights = nlms_update(weights, inputs, targets, learning_rate)

    return updated_weights, cost


# ----------------------------------------------------------------------
# 9️⃣ Simple sanity‑check driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example feature set (3 nodes, 3‑dimensional numeric embeddings)
    example_features = {
        0: [1.0, 0.2, 0.8],
        1: [0.4, 1.5, 0.3],
        2: [2.2, 0.1, 1.1],
    }

    eps = 0.8
    prior = 0.6
    fp = 0.05
    lam = 0.2
    lr = 0.01

    w, c = hybrid_nlms_rbf_lsm_bayesian(
        example_features, eps, prior, fp, lam, lr
    )
    print("Updated weights:", w)
    print("Hybrid cost:", c)