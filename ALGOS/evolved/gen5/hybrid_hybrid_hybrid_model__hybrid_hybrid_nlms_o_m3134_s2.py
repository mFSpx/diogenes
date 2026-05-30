# DARWIN HAMMER — match 3134, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_model_pool_hy_hybrid_hybrid_minimu_m1971_s4.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s7.py (gen2)
# born: 2026-05-29T23:48:20Z

"""Hybrid Curvature‑NLMS Bayesian Selector

This module fuses the two parent algorithms:

* **Parent A** – Curvature‑Weighted MinHash Bayesian Selector.  
  It builds a curvature weight vector `w = v²` from a feature vector `v`,
  combines it with pairwise token‑signature similarities to form edge
  probabilities, and uses a Bayesian update + expected‑entropy criterion to
  select a model.

* **Parent B** – Normalised Least‑Mean‑Squares (NLMS) adaptation and simple
  character‑frequency based span extraction.  
  The NLMS step provides a deterministic way to adapt the curvature vector
  `v` based on a scalar error signal.

**Mathematical Bridge**  
The bridge is the curvature vector `v`.  It is updated by the NLMS rule
(`v ← v + μ·error·x / (‖x‖²+ε)`) where the input `x` is the concatenated
character‑frequency vectors of the candidate texts.  After each update the
new `v` yields curvature weights `w_i = v_i²`.  These weights are used as
priors in the Bayesian selector of Parent A.  Thus the adaptive filter of
Parent B drives the probabilistic model‑selection machinery of Parent A,
creating a single unified optimisation loop.

The core functions below implement this hybrid pipeline.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Deterministic utilities (from Parent A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Weekday index used by the original doomsday calendar: Mon→1 … Sun→0."""
    return (int((np.datetime64(f"{year}-{month:02d}-{day:02d}").astype('datetime64[D]').astype(int) + 3) % 7))


def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    return random.Random(seed)


# ----------------------------------------------------------------------
# NLMS core (Parent B)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Returns (new_weights, error) where error = target - y.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / power) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Token‑signature utilities (adapted from Parent B)
# ----------------------------------------------------------------------
def _char_frequency_vector(text: str) -> np.ndarray:
    """Return a 26‑dim L2‑normalised vector of lowercase alphabet frequencies."""
    vec = np.zeros(26, dtype=float)
    for ch in text.lower():
        if 'a' <= ch <= 'z':
            vec[ord(ch) - ord('a')] += 1.0
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
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------
# Curvature‑Weighted Bayesian selector (Parent A)
# ----------------------------------------------------------------------
def curvature_weights(v: np.ndarray) -> np.ndarray:
    """From a raw curvature vector `v` compute per‑model weights w_i = v_i² and normalise."""
    w = np.square(v)
    total = np.sum(w)
    return w / total if total > 0 else w


def edge_matrix(curv_w: np.ndarray, sigs: List[np.ndarray]) -> np.ndarray:
    """
    Build a symmetric matrix `E` where
        E_ij = avg(curv_w_i, curv_w_j) * similarity(sig_i, sig_j)

    The diagonal is set to the curvature weight itself (self‑edge).
    """
    n = len(sigs)
    E = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            avg_w = (curv_w[i] + curv_w[j]) / 2.0
            sim = cosine_similarity(sigs[i], sigs[j]) if i != j else 1.0
            val = avg_w * sim
            E[i, j] = E[j, i] = val
    # Normalise to a probability distribution over edges
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


def expected_posterior_entropy(edge_mat: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """
    For each candidate model i, compute the posterior that would result
    if i were chosen as the “next observation”, then return the entropy
    of each such posterior.  The result is an array `E_i`.
    """
    n = len(prior)
    entropies = np.zeros(n, dtype=float)
    for i in range(n):
        # Simulate observing model i: boost its self‑edge to 1 (max similarity)
        simulated = edge_mat.copy()
        simulated[i, :] = prior  # treat prior as uniform likelihood for simplicity
        simulated[:, i] = prior
        post = posterior_distribution(simulated, prior)
        entropies[i] = shannon_entropy(post)
    return entropies


def select_min_entropy_model(exp_entropies: np.ndarray) -> int:
    """Return the index of the model with minimal expected posterior entropy."""
    return int(np.argmin(exp_entropies))


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_step(
    v: np.ndarray,
    texts: List[str],
    target_entropy: float,
    mu: float = 0.5,
) -> Tuple[np.ndarray, int, float]:
    """
    Perform one hybrid iteration:

    1. Build signatures from `texts`.
    2. Compute curvature weights from `v`.
    3. Assemble the edge matrix and posterior.
    4. Evaluate expected posterior entropies and pick the best model.
    5. Use NLMS to adapt `v` so that the achieved entropy moves toward
       `target_entropy`.

    Returns (new_v, selected_index, achieved_entropy).
    """
    # 1 – signatures
    sigs = [signature_from_text(t) for t in texts]

    # 2 – curvature weights
    curv_w = curvature_weights(v)

    # 3 – edge matrix and posterior
    E = edge_matrix(curv_w, sigs)
    post = posterior_distribution(E, curv_w)

    # 4 – expected entropies & selection
    exp_ent = expected_posterior_entropy(E, curv_w)
    selected = select_min_entropy_model(exp_ent)
    achieved_entropy = exp_ent[selected]

    # 5 – NLMS adaptation
    # Input vector `x` for NLMS is the concatenated signatures (flattened)
    x = np.concatenate(sigs)
    new_v, _ = nlms_update(v, x, target=target_entropy, mu=mu)

    return new_v, selected, achieved_entropy


def initialize_curvature_vector(dim: int = 24, seed_text: str = "initial seed") -> np.ndarray:
    """Deterministic initial curvature vector using a text‑derived RNG."""
    rng = _rng_from_text(seed_text)
    return np.array([rng.random() for _ in range(dim)], dtype=float)


def demo_hybrid_process() -> None:
    """Run a short demonstration of the hybrid algorithm."""
    # Example model descriptions (could be any textual representation)
    model_texts = [
        "Alpha model: predicts linear trends with low variance.",
        "Beta model: uses quadratic features for medium complexity.",
        "Gamma model: deep network with many hidden layers.",
        "Delta model: ensemble of decision trees."
    ]

    # Initialise curvature vector
    v = initialize_curvature_vector(dim=24, seed_text="demo seed")

    # Target entropy we would like to achieve (arbitrary)
    target = 0.5

    print("Starting hybrid optimisation...")
    for iteration in range(5):
        v, sel, ent = hybrid_step(v, model_texts, target_entropy=target, mu=0.3)
        print(
            f"Iter {iteration+1:02d}: selected model {sel} "
            f"('{model_texts[sel][:30]}...'), entropy={_pct(ent)}"
        )
    print("Demo finished.")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_process()