# DARWIN HAMMER — match 2494, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_hybrid_m805_s0.py (gen4)
# born: 2026-05-29T23:42:34Z

"""Hybrid Fusion of:
- Parent A: Geometric Algebra with Koopman operator dynamics & Count‑Min sketch.
- Parent B: High‑dimensional bipolar hypervectors & variational free‑energy similarity.

Mathematical Bridge
------------------
The Count‑Min sketch produces a non‑negative frequency vector `s ∈ ℝ^d`.  
We interpret `s` as the coefficient vector of a multivector in the Clifford
algebra `Cl(d,0)`, i.e. each sketch bucket corresponds to a 1‑vector basis blade.
The Koopman operator `K ∈ ℝ^{d×d}` (learned from paired state matrices) evolves
these coefficients linearly: `c' = K c`.

A morphology is encoded as a bipolar hypervector `h ∈ {‑1,+1}^d`.  Binding the
evolved multivector coefficients with the hypervector is performed by the
geometric‑algebra product, which for 1‑vectors reduces to element‑wise
multiplication (XOR‑like binding).  The bound vector `b = c' ⊙ h` serves as the
observation for a variational free‑energy model.  Assuming a Gaussian belief
with mean `μ` and covariance `Σ`, the free energy

    F(b) = ½ (b‑μ)ᵀ Σ⁻¹ (b‑μ) + ½ log det Σ + (d/2) log(2π)

provides a scalar similarity score that drives downstream decisions.

The pipeline therefore fuses:
1. Sketch → multivector coefficients.
2. Koopman linear dynamics on the algebraic space.
3. Probabilistic normalization (softmax) → probability distribution.
4. Binding with a morphology hypervector.
5. Variational free‑energy evaluation.

The code below implements this fused system with three core functions and a
smoke test.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# 1. Count‑Min Sketch utilities (Parent A)
# ----------------------------------------------------------------------
def build_count_min_sketch(
    stream: List[int],
    depth: int,
    width: int,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a Count‑Min sketch from an integer stream.

    Returns
    -------
    sketch : np.ndarray
        A `depth × width` matrix of non‑negative counts.
    hash_seeds : List[int]
        Random seeds used for each hash function (needed for reproducibility).
    """
    rng = random.Random(seed)
    hash_seeds = [rng.randint(0, 2**31 - 1) for _ in range(depth)]
    sketch = np.zeros((depth, width), dtype=np.int64)

    for item in stream:
        for i, h in enumerate(hash_seeds):
            idx = (hash(item) ^ h) % width
            sketch[i, idx] += 1
    return sketch, hash_seeds


def sketch_to_multivector(sketch: np.ndarray) -> np.ndarray:
    """
    Collapse the sketch matrix into a single coefficient vector of length `width`
    by summing over the depth dimension.  The resulting vector is interpreted as
    the coefficients of the basis 1‑vectors of `Cl(width,0)`.
    """
    # Simple aggregation; alternative reductions are possible.
    coeffs = sketch.sum(axis=0).astype(np.float64)
    return coeffs


# ----------------------------------------------------------------------
# 2. Koopman operator learning & evolution (Parent A)
# ----------------------------------------------------------------------
def learn_koopman_operator(X: np.ndarray, X_prime: np.ndarray) -> np.ndarray:
    """
    Given paired state matrices X (n×d) and X' (n×d), compute the least‑squares
    Koopman operator K that satisfies X' ≈ X Kᵀ.
    """
    # Solve Kᵀ = pinv(X) X'
    pinv = np.linalg.pinv(X)
    K_T = pinv @ X_prime
    K = K_T.T
    return K


def evolve_coefficients(coeffs: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Apply the Koopman operator to the multivector coefficients.
    """
    return K @ coeffs


def normalize_to_distribution(vec: np.ndarray) -> np.ndarray:
    """
    Convert a real‑valued vector to a probability distribution via softmax.
    """
    max_val = np.max(vec)
    exp_vec = np.exp(vec - max_val)  # stability
    return exp_vec / exp_vec.sum()


# ----------------------------------------------------------------------
# 3. Morphology → bipolar hypervector (Parent B)
# ----------------------------------------------------------------------
def morphology_to_hypervector(morph: Dict[str, float], dim: int, seed: int = 0) -> np.ndarray:
    """
    Encode a morphology (dictionary of scalar features) into a bipolar hypervector.
    The sign of each component is determined by a deterministic hash of the
    feature values to ensure reproducibility.
    """
    rng = random.Random(seed)
    # Simple deterministic mapping: for each dimension, pick a feature at random
    # and set sign according to whether its normalized value exceeds 0.5.
    features = list(morph.values())
    if not features:
        raise ValueError("Morphology must contain at least one feature")
    hv = np.empty(dim, dtype=np.int8)
    for i in range(dim):
        f = rng.choice(features)
        # Normalize feature to [0,1] using a crude min‑max based on the current set
        f_min, f_max = min(features), max(features)
        norm = (f - f_min) / (f_max - f_min + 1e-12)
        hv[i] = 1 if norm > 0.5 else -1
    return hv


def bind_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Geometric‑algebra binding for 1‑vectors reduces to element‑wise multiplication.
    Both inputs must have the same shape.
    """
    if a.shape != b.shape:
        raise ValueError("Shapes must match for binding")
    return a * b


# ----------------------------------------------------------------------
# 4. Variational free‑energy evaluation (Parent B)
# ----------------------------------------------------------------------
def variational_free_energy(
    observation: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Compute the Gaussian variational free energy:
        F = 0.5 * (obs-μ)^T Σ⁻¹ (obs-μ) + 0.5 * log det Σ + (d/2) * log(2π)

    Parameters
    ----------
    observation : np.ndarray
        Vector of shape (d,).
    mean : np.ndarray
        Belief mean vector of shape (d,).
    cov : np.ndarray
        Positive‑definite covariance matrix of shape (d,d).

    Returns
    -------
    float
        Scalar free‑energy value (lower is better similarity).
    """
    d = observation.shape[0]
    diff = observation - mean
    # Solve Σ⁻¹ diff efficiently via Cholesky
    try:
        L = np.linalg.cholesky(cov)
        # Solve L y = diff
        y = np.linalg.solve(L, diff)
        # Solve Lᵀ x = y  => x = Σ⁻¹ diff
        x = np.linalg.solve(L.T, y)
        quad = diff @ x
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # Fallback to generic inversion (less stable)
        inv_cov = np.linalg.inv(cov)
        quad = diff @ inv_cov @ diff
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite")
    return 0.5 * quad + 0.5 * logdet + (d / 2.0) * math.log(2 * math.pi)


# ----------------------------------------------------------------------
# 5. High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_score(
    stream: List[int],
    morphology: Dict[str, float],
    depth: int = 4,
    width: int = 128,
    koopman_K: np.ndarray = None,
    seed: int = 0,
) -> float:
    """
    End‑to‑end hybrid computation:

    1. Build Count‑Min sketch → multivector coefficients.
    2. Apply Koopman dynamics (learned if not supplied).
    3. Normalise to a probability distribution.
    4. Encode morphology as a bipolar hypervector.
    5. Bind distribution with hypervector.
    6. Evaluate variational free energy using a simple Gaussian belief
       (mean = uniform, covariance = diagonal ε·I).

    Returns the free‑energy score (lower ⇒ higher similarity).
    """
    # 1. Sketch & multivector
    sketch, _ = build_count_min_sketch(stream, depth, width, seed)
    coeffs = sketch_to_multivector(sketch)

    # 2. Koopman operator (learn on‑the‑fly if not provided)
    if koopman_K is None:
        # Create synthetic paired data: X = coeffs (as a column), X' = coeffs shifted by 1
        X = coeffs.reshape(1, -1)  # shape (1, d)
        X_prime = np.roll(coeffs, -1).reshape(1, -1)
        koopman_K = learn_koopman_operator(X, X_prime)

    evolved = evolve_coefficients(coeffs, koopman_K)

    # 3. Normalise
    prob_dist = normalize_to_distribution(evolved)

    # 4. Hypervector from morphology
    hv = morphology_to_hypervector(morphology, dim=width, seed=seed)

    # 5. Bind (distribution is real‑valued, hypervector is ±1)
    bound = bind_vectors(prob_dist, hv.astype(np.float64))

    # 6. Free‑energy with simple belief
    belief_mean = np.full(width, 0.5)  # uniform prior mean
    eps = 1e-3
    belief_cov = np.eye(width) * eps  # small isotropic covariance

    energy = variational_free_energy(bound, belief_mean, belief_cov)
    return energy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a random integer stream
    random_stream = [random.randint(0, 1000) for _ in range(500)]

    # Define a simple morphology
    example_morph = {
        "length": 12.3,
        "width": 7.8,
        "height": 4.5,
        "mass": 2.1,
    }

    # Run hybrid score
    try:
        score = hybrid_score(
            stream=random_stream,
            morphology=example_morph,
            depth=5,
            width=256,
            seed=42,
        )
        print(f"Hybrid free‑energy score: {score:.6f}")
    except Exception as e:
        print(f"Error during hybrid execution: {e}", file=sys.stderr)
        sys.exit(1)