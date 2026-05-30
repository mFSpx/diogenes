# DARWIN HAMMER — match 3134, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_model_pool_hy_hybrid_hybrid_minimu_m1971_s4.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s7.py (gen2)
# born: 2026-05-29T23:48:20Z

import math
import random
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Deterministic utilities (from Parent A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Weekday index used by the original doomsday calendar: Mon→1 … Sun→0."""
    return int(
        (np.datetime64(f"{year}-{month:02d}-{day:02d}")
         .astype('datetime64[D]')
         .astype(int) + 3) % 7
    )


def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    return random.Random(seed)


# ----------------------------------------------------------------------
# NLMS core – gradient‑guided version (improved)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_gradient_update(
    weights: np.ndarray,
    grad: np.ndarray,
    error: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Normalised LMS update using a supplied gradient direction.

    new_weights = weights + mu * error * grad / (||grad||² + eps)
    """
    norm_sq = float(np.dot(grad, grad) + eps)
    return weights + (mu * error / norm_sq) * grad


# ----------------------------------------------------------------------
# Token‑signature utilities (adapted from Parent B)
# ----------------------------------------------------------------------
def _char_frequency_vector(text: str) -> np.ndarray:
    """Return a 26‑dim L2‑normalised vector of lowercase alphabet frequencies."""
    vec = np.zeros(26, dtype=float)
    for ch in text.lower():
        if "a" <= ch <= "z":
            vec[ord(ch) - ord("a")] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def signature_from_text(text: str) -> np.ndarray:
    """
    Produce a deterministic “signature” for a piece of text.
    Here we simply reuse the character‑frequency vector; in a full system
    this could be replaced by a MinHash sketch.
    """
    return _char_frequency_vector(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1‑D arrays (guarded against zero norm)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------
# Curvature‑Weighted Bayesian selector (Parent A)
# ----------------------------------------------------------------------
def curvature_weights(v: np.ndarray) -> np.ndarray:
    """
    From a raw curvature vector `v` compute per‑model weights w_i = v_i²
    and normalise to a probability simplex.
    """
    w = np.square(v)
    total = np.sum(w)
    return w / total if total > 0 else w


def edge_matrix(curv_w: np.ndarray, sigs: List[np.ndarray]) -> np.ndarray:
    """
    Build a symmetric matrix `E` where
        E_ij = avg(curv_w_i, curv_w_j) * similarity(sig_i, sig_j)

    The diagonal is set to the curvature weight itself (self‑edge).
    The matrix is normalised to a probability distribution over edges.
    """
    n = len(sigs)
    E = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            avg_w = (curv_w[i] + curv_w[j]) / 2.0
            sim = cosine_similarity(sigs[i], sigs[j]) if i != j else 1.0
            val = avg_w * sim
            E[i, j] = E[j, i] = val
    total = np.sum(E)
    return E / total if total > 0 else E


def posterior_distribution(edge_mat: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """
    Bayesian update: posterior ∝ prior (as node prior) * sum_j edge_ij
    then normalised to sum to 1.
    """
    likelihood = np.sum(edge_mat, axis=1)
    unnorm = prior * likelihood
    total = np.sum(unnorm)
    return unnorm / total if total > 0 else unnorm


def shannon_entropy(p: np.ndarray) -> float:
    """Standard Shannon entropy (base e) for a probability vector."""
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    return -float(np.sum(p_safe * np.log(p_safe)))


def expected_posterior_entropy(
    edge_mat: np.ndarray,
    prior: np.ndarray,
    sigs: List[np.ndarray],
) -> np.ndarray:
    """
    For each candidate model i, simulate the effect of observing i
    (by boosting its self‑edge to the maximum similarity) and compute
    the resulting posterior entropy.
    Returns an array `E_i`.
    """
    n = len(prior)
    entropies = np.zeros(n, dtype=float)

    for i in range(n):
        # Clone the edge matrix and boost the i‑th self‑edge.
        simulated = edge_mat.copy()
        # Replace row/col i with a deterministic “perfect” observation:
        # self‑edge weight = prior[i] (max similarity) and zero elsewhere.
        simulated[i, :] = 0.0
        simulated[:, i] = 0.0
        simulated[i, i] = prior[i]

        # Renormalise simulated matrix to keep it a probability distribution.
        total = np.sum(simulated)
        if total > 0:
            simulated /= total

        post = posterior_distribution(simulated, prior)
        entropies[i] = shannon_entropy(post)

    return entropies


def select_min_entropy_model(exp_entropies: np.ndarray) -> int:
    """Return the index of the model with minimal expected posterior entropy."""
    return int(np.argmin(exp_entropies))


# ----------------------------------------------------------------------
# Gradient estimation utilities (finite‑difference)
# ----------------------------------------------------------------------
def _entropy_given_v(
    v: np.ndarray,
    texts: List[str],
    target_entropy: float,
) -> float:
    """
    Helper that runs the Bayesian part of the pipeline and returns the
    entropy of the posterior that would be obtained after the selection
    step (i.e. the minimal expected entropy).
    """
    sigs = [signature_from_text(t) for t in texts]
    curv_w = curvature_weights(v)
    E = edge_matrix(curv_w, sigs)
    post = posterior_distribution(E, curv_w)

    # Expected entropies for each candidate
    exp_ent = expected_posterior_entropy(E, curv_w, sigs)
    chosen = select_min_entropy_model(exp_ent)
    return exp_ent[chosen]


def entropy_gradient(
    v: np.ndarray,
    texts: List[str],
    target_entropy: float,
    delta: float = 1e-6,
) -> np.ndarray:
    """
    Approximate ∂entropy/∂v using central finite differences.
    """
    grad = np.zeros_like(v)
    base_entropy = _entropy_given_v(v, texts, target_entropy)

    for i in range(len(v)):
        v_pert = v.copy()
        v_pert[i] += delta
        pert_entropy = _entropy_given_v(v_pert, texts, target_entropy)
        grad[i] = (pert_entropy - base_entropy) / delta

    return grad


# ----------------------------------------------------------------------
# Hybrid pipeline – deepened integration (improved)
# ----------------------------------------------------------------------
def hybrid_step_improved(
    v: np.ndarray,
    texts: List[str],
    target_entropy: float,
    mu: float = 0.5,
    grad_delta: float = 1e-6,
) -> Tuple[np.ndarray, int, float]:
    """
    Perform one hybrid iteration with a gradient‑guided NLMS update.

    1. Build signatures from `texts`.
    2. Compute curvature weights from `v`.
    3. Assemble the edge matrix and posterior.
    4. Evaluate expected posterior entropies and pick the best model.
    5. Estimate the gradient of the achieved entropy w.r.t. `v`.
    6. Apply a normalised LMS step that moves `v` in the direction that
       reduces the error (target_entropy – achieved_entropy).

    Returns (new_v, selected_index, achieved_entropy).
    """
    # ------------------------------------------------------------------
    # 1 – signatures
    # ------------------------------------------------------------------
    sigs = [signature_from_text(t) for t in texts]

    # ------------------------------------------------------------------
    # 2 – curvature weights
    # ------------------------------------------------------------------
    curv_w = curvature_weights(v)

    # ------------------------------------------------------------------
    # 3 – edge matrix and posterior
    # ------------------------------------------------------------------
    E = edge_matrix(curv_w, sigs)
    post = posterior_distribution(E, curv_w)

    # ------------------------------------------------------------------
    # 4 – expected entropies & selection
    # ------------------------------------------------------------------
    exp_ent = expected_posterior_entropy(E, curv_w, sigs)
    selected = select_min_entropy_model(exp_ent)
    achieved_entropy = exp_ent[selected]

    # ------------------------------------------------------------------
    # 5 – gradient estimation (finite differences)
    # ------------------------------------------------------------------
    grad = entropy_gradient(v, texts, target_entropy, delta=grad_delta)

    # ------------------------------------------------------------------
    # 6 – NLMS‑style update using the entropy error and the estimated gradient
    # ------------------------------------------------------------------
    error = target_entropy - achieved_entropy
    new_v = nlms_gradient_update(v, grad, error, mu=mu)

    # Enforce non‑negativity (curvature vector represents a magnitude)
    new_v = np.maximum(new_v, 0.0)

    return new_v, selected, achieved_entropy


# ----------------------------------------------------------------------
# Example entry‑point (kept for compatibility – can be removed in production)
# ----------------------------------------------------------------------
def run_example():
    """Simple sanity‑check driver."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "To be, or not to be, that is the question."
    ]
    v0 = np.ones(len(texts), dtype=float)  # initialise curvature vector
    target = 0.8  # desired entropy (arbitrary)
    v, idx, ent = hybrid_step_improved(v0, texts, target_entropy=target, mu=0.3)
    print(f"Selected model: {idx}, achieved entropy: {ent:.4f}, new v: {v}")

if __name__ == "__main__":
    run_example()