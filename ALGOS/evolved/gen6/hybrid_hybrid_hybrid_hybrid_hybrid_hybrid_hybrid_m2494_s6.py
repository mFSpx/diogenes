# DARWIN HAMMER — match 2494, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_hybrid_m805_s0.py (gen4)
# born: 2026-05-29T23:42:34Z

"""Hybrid algorithm merging:
- Parent A: Geometric Algebra multivector dynamics via a learned Koopman operator,
  Count‑Min sketch frequency tables, and Bayesian Beta updates.
- Parent B: High‑dimensional bipolar hypervectors representing morphologies and a
  variational free‑energy (VFE) similarity measure.

Mathematical bridge:
A Count‑Min sketch of streamed morphology observations is interpreted as a
multivector in a Clifford algebra: each hash bucket corresponds to a basis
blade and its count becomes the blade coefficient.  The coefficient vector is
therefore a high‑dimensional hypervector.  A linear Koopman operator K, learned
from paired multivector states (X, X′), evolves this hypervector in time.
The evolved hypervector is projected back to physical morphology space via a
learned linear decoder D.  The reconstruction error together with a KL‑term
between the Bayesian posterior (Beta per bucket) and a uniform prior yields a
variational free‑energy score that drives downstream decisions.

The implementation below provides the core pipeline:
1. streaming sketch → multivector coefficients,
2. Koopman learning & evolution,
3. Bayesian update of sketch counts,
4. VFE evaluation of morphology similarity.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Any

# ----------------------------------------------------------------------
# Count‑Min Sketch with simple hash functions
# ----------------------------------------------------------------------
class CountMinSketch:
    def __init__(self, depth: int = 4, width: int = 256, seed: int = 0):
        self.depth = depth
        self.width = width
        self.table = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        # generate a list of pairwise‑independent hash salts
        self.salts = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: Any, i: int) -> int:
        h = hash((item, self.salts[i]))
        return h % self.width

    def update(self, item: Any, inc: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += inc

    def query(self, item: Any) -> int:
        """Return the minimum count over all hash rows (standard CM estimate)."""
        mins = [self.table[i, self._hash(item, i)] for i in range(self.depth)]
        return min(mins)

    def flatten(self) -> np.ndarray:
        """Flatten the sketch matrix to a 1‑D coefficient vector."""
        return self.table.ravel().astype(np.float64)

# ----------------------------------------------------------------------
# Simple morphology representation (Parent B)
# ----------------------------------------------------------------------
class Morphology:
    __slots__ = ("length", "width", "height", "mass")
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def as_vector(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height, self.mass], dtype=np.float64)

# ----------------------------------------------------------------------
# Mapping between sketch coefficients (multivector) and morphology space
# ----------------------------------------------------------------------
def random_decoder(input_dim: int, output_dim: int, seed: int = 42) -> np.ndarray:
    """Generate a fixed random linear decoder D (output_dim × input_dim)."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.01, size=(output_dim, input_dim))

# ----------------------------------------------------------------------
# Koopman operator learning (Parent A)
# ----------------------------------------------------------------------
def learn_koopman(X: np.ndarray, X_prime: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Least‑squares solution K ≈ X′ X⁺ where X⁺ is the Moore‑Penrose pseudoinverse.
    Regularisation term added to avoid singularities.
    """
    assert X.shape[0] == X_prime.shape[0], "state dimension mismatch"
    # X, X_prime are (state_dim, n_samples)
    Xt = X.T
    # Regularised inverse via (X Xᵀ + reg·I)⁻¹ X
    gram = X @ Xt + reg * np.eye(X.shape[1])
    inv_gram = np.linalg.inv(gram)
    K = X_prime @ Xt @ inv_gram
    return K

def evolve_multivector(K: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply the Koopman operator to a coefficient vector."""
    return K @ coeffs

# ----------------------------------------------------------------------
# Bayesian Beta update per sketch bucket (Parent A)
# ----------------------------------------------------------------------
def beta_posterior(mean_counts: np.ndarray, alpha0: float = 1.0, beta0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each bucket treat the count as successes in a Bernoulli process.
    Returns posterior (alpha, beta) parameters.
    """
    alpha = alpha0 + mean_counts
    beta = beta0 + (np.max(mean_counts) - mean_counts)  # simple complement
    return alpha, beta

def beta_mean(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Posterior mean of the Beta distribution."""
    return alpha / (alpha + beta + 1e-12)

# ----------------------------------------------------------------------
# Variational Free Energy (Parent B)
# ----------------------------------------------------------------------
def variational_free_energy(
    observed: Morphology,
    predicted_vec: np.ndarray,
    posterior_mean: np.ndarray,
    prior_mean: np.ndarray = None,
    prior_var: float = 1.0,
) -> float:
    """
    VFE = reconstruction_error + KL divergence (Gaussian approx).
    - reconstruction_error = 0.5 * ||predicted - observed||²
    - KL = 0.5 * Σ [ (σ_post² + (μ_post-μ_prior)²) / σ_prior² - 1 + log(σ_prior²/σ_post²) ]
    For simplicity we treat posterior variance as the variance of the Beta means.
    """
    obs_vec = observed.as_vector()
    recon_err = 0.5 * np.sum((predicted_vec - obs_vec) ** 2)

    if prior_mean is None:
        prior_mean = np.full_like(posterior_mean, 0.5)

    post_var = np.var(posterior_mean) + 1e-12
    prior_var = prior_var + 1e-12

    kl = 0.5 * np.sum(
        (post_var + (posterior_mean - prior_mean) ** 2) / prior_var
        - 1
        + np.log(prior_var / post_var)
    )
    return recon_err + kl

# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_step(
    sketch: CountMinSketch,
    K: np.ndarray,
    decoder: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform one hybrid iteration:
    1. Convert sketch to multivector coefficients.
    2. Evolve coefficients with Koopman operator K.
    3. Decode to morphology space.
    4. Update Bayesian posterior on sketch counts.
    5. Compute VFE against a synthetic observation (for demo).
    Returns (evolved_coeffs, posterior_means, vfe_score).
    """
    coeffs = sketch.flatten()
    evolved = evolve_multivector(K, coeffs)

    # Decode to morphology prediction
    pred_morph_vec = decoder @ evolved  # shape (4,)

    # Bayesian update on the raw counts
    mean_counts = coeffs / (np.max(coeffs) + 1e-12)
    alpha, beta = beta_posterior(mean_counts, alpha0, beta0)
    post_mean = beta_mean(alpha, beta)

    # Synthetic observation for demonstration (could be replaced by real data)
    obs = Morphology(
        length=random.uniform(0.5, 2.0),
        width=random.uniform(0.5, 2.0),
        height=random.uniform(0.5, 2.0),
        mass=random.uniform(0.1, 5.0),
    )

    vfe = variational_free_energy(obs, pred_morph_vec, post_mean)

    return evolved, post_mean, vfe

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create a sketch and stream synthetic morphology observations
    sketch = CountMinSketch(depth=5, width=128, seed=123)
    rng = np.random.default_rng(2026)

    for _ in range(500):
        m = Morphology(
            length=rng.uniform(0.5, 3.0),
            width=rng.uniform(0.5, 3.0),
            height=rng.uniform(0.5, 3.0),
            mass=rng.uniform(0.2, 10.0),
        )
        # Hash each scalar feature as a separate item
        for feat_name, value in zip(("len", "wid", "hei", "mas"), m.as_vector()):
            # discretise value to a bucket string to keep sketch size modest
            bucket = f"{feat_name}:{int(value * 10)}"
            sketch.update(bucket, inc=1)

    # 2. Build training data for Koopman: two consecutive snapshots
    coeffs_t0 = sketch.flatten()
    # Simulate a small evolution by updating the sketch with a few more items
    for _ in range(50):
        m = Morphology(
            length=rng.uniform(0.5, 3.0),
            width=rng.uniform(0.5, 3.0),
            height=rng.uniform(0.5, 3.0),
            mass=rng.uniform(0.2, 10.0),
        )
        for feat_name, value in zip(("len", "wid", "hei", "mas"), m.as_vector()):
            bucket = f"{feat_name}:{int(value * 10)}"
            sketch.update(bucket, inc=1)
    coeffs_t1 = sketch.flatten()

    # Stack columns to obtain (state_dim, n_samples) matrices with a single sample each
    X = coeffs_t0[:, np.newaxis]
    X_prime = coeffs_t1[:, np.newaxis]

    # 3. Learn Koopman operator
    K = learn_koopman(X, X_prime, reg=1e-4)

    # 4. Random decoder from multivector space to 4‑D morphology space
    decoder = random_decoder(input_dim=coeffs_t0.size, output_dim=4, seed=777)

    # 5. Run hybrid step and print results
    evolved, posterior_means, vfe_score = hybrid_step(sketch, K, decoder)

    print(f"Evolved coefficient norm: {np.linalg.norm(evolved):.4f}")
    print(f"Posterior mean (first 5 buckets): {posterior_means[:5]}")
    print(f"Variational Free Energy score: {vfe_score:.6f}")

    sys.exit(0)