# DARWIN HAMMER — match 3355, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_fractional_hd_hybrid_hybrid_nlms_o_m1127_s1.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_krampus_brain_m74_s2.py (gen2)
# born: 2026-05-29T23:49:40Z

"""
Hybrid Causal‑Geometry NLMS (HCG‑NLMS)

This module fuses the two parent algorithms:

* **Parent A** – hyperdimensional representation of causal effects together with a
  fractional binding operator and an adaptive Normalised Least‑Mean‑Squares (NLMS)
  weight update.
* **Parent B** – deterministic + stochastic feature extraction, weighted fusion of the
  two streams, Euclidean embedding of the resulting vectors and an
  Ollivier‑Ricci curvature approximation on the induced “brain‑map” graph.

**Mathematical bridge**

1. A causal effect is encoded as a *hypervector* **h** ∈ ℂᴰ (Parent A).  
2. The same effect is also represented by a *feature vector* **f** ∈ ℝᴺ obtained by
   fusing deterministic and stochastic extractions (Parent B).  
3. The inner product 𝜙 = Re⟨h, f⟩ yields a scalar “causal‑feature projection”.  
4. A set of such projections for many effects forms a point cloud; pairwise Euclidean
   distances d(i,j) define a metric tensor on the cloud.  
5. Using the lazy‑random‑walk neighbourhoods of this metric we compute an
   Ollivier‑Ricci curvature κ(i,j) = 1 − W₁(μᵢ, μⱼ)/d(i,j), where W₁ is approximated
   by the mean absolute difference of neighbour vectors (Parent B).  
6. The curvature matrix κ is fed back as a *geometric regulariser* for the NLMS
   adaptation: the NLMS error term is multiplied by (1 + κ) so that updates respect
   the underlying causal geometry.

The three core public functions illustrate the hybrid pipeline:
* ``generate_causal_hypervector`` – builds a hypervector for a ``CausalEffect``.
* ``fuse_text_features`` – deterministic + stochastic feature fusion for a text.
* ``nlms_curvature_update`` – NLMS weight update that incorporates Ollivier‑Ricci
  curvature derived from the current set of representations.

The ``if __name__ == "__main__"`` block runs a smoke test on synthetic data."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CausalEffect:
    """Immutable container for a causal effect estimate."""
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]


# ----------------------------------------------------------------------
# 1. Hyperdimensional utilities (Parent A)
# ----------------------------------------------------------------------


def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """
    Generate a random hypervector of dimension *d*.

    - ``kind='complex'``  → unit‑modulus complex numbers e^{iθ}
    - ``kind='bipolar'``  → values in {‑1, +1}
    - ``kind='real'``     → L2‑normalised real Gaussian vector
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        norm = np.linalg.norm(v) + 1e-30
        return v / norm
    raise ValueError(f"kind must be 'complex', 'bipolar', or 'real'; got {kind!r}")


def fractional_bind(hv_a: np.ndarray, hv_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Fractional binding between two hypervectors.

    The operation is defined as:
        bind = hv_a * (hv_b ** alpha)

    For complex hypervectors this corresponds to a rotation by a fraction
    ``alpha`` of the phase of ``hv_b``.
    """
    if hv_a.shape != hv_b.shape:
        raise ValueError("Hypervectors must share the same shape for binding.")
    return hv_a * (hv_b ** alpha)


def generate_causal_hypervector(effect: CausalEffect,
                                dim: int = 10000,
                                alpha: float = 0.5,
                                seed: int | None = None) -> np.ndarray:
    """
    Encode a ``CausalEffect`` as a hypervector.

    Steps:
    1. Create base hypervectors for treatment, outcome and each confounder.
    2. Bind them together using fractional binding.  The binding exponent ``alpha``
       controls the “strength” of each component.
    3. If an ATE estimate is available, rotate the resulting vector by an angle
       proportional to the estimate (mod 2π) to embed magnitude information.
    """
    rng = np.random.default_rng(seed)
    # Base vectors
    hv_treat = random_hv(dim, kind="complex", seed=rng.integers(0, 2**63))
    hv_out = random_hv(dim, kind="complex", seed=rng.integers(0, 2**63))
    hv = fractional_bind(hv_treat, hv_out, alpha)

    # Incorporate confounders
    for conf in effect.confounders:
        hv_conf = random_hv(dim, kind="complex", seed=rng.integers(0, 2**63))
        hv = fractional_bind(hv, hv_conf, alpha)

    # Encode the ATE magnitude as a global phase shift
    if effect.ate_estimate is not None:
        angle = (effect.ate_estimate % (2 * math.pi))
        hv = hv * np.exp(1j * angle)

    # Normalise to unit magnitude to keep the representation on the hypersphere
    hv = hv / (np.abs(hv).mean() + 1e-30)
    return hv


# ----------------------------------------------------------------------
# 2. Deterministic / Stochastic feature extraction (Parent B)
# ----------------------------------------------------------------------


def _deterministic_features(text: str, size: int = 128) -> np.ndarray:
    """
    Produce a reproducible feature vector by seeding ``random.Random`` with the
    hash of the input text.
    """
    rnd = random.Random(hash(text))
    return np.fromiter((rnd.random() for _ in range(size)), dtype=float)


def _stochastic_features(text: str, size: int = 128) -> np.ndarray:
    """
    Produce a stochastic feature vector that depends on the global RNG state.
    """
    # Use the text only to diversify the pattern a bit; the global RNG still
    # introduces stochasticity.
    base = sum(ord(c) for c in text) % 1000
    random.seed(base)  # affect the global RNG in a lightweight way
    return np.random.rand(size)


def fuse_text_features(text: str,
                       alpha: float = 0.5,
                       size: int = 128) -> np.ndarray:
    """
    Fuse deterministic and stochastic feature streams.

    The fused vector is ``alpha * det + (1‑alpha) * stoch``.
    """
    det = _deterministic_features(text, size)
    stoch = _stochastic_features(text, size)
    return alpha * det + (1.0 - alpha) * stoch


# ----------------------------------------------------------------------
# 3. Geometry: pairwise distances and Ollivier‑Ricci curvature
# ----------------------------------------------------------------------


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix for rows of ``X``.
    """
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _neighbour_mask(distances: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Boolean mask where ``True`` indicates that the distance is ≤ ``cutoff`` and
    not the diagonal element.
    """
    mask = (distances <= cutoff) & (distances > 0.0)
    return mask


def approximate_ollivier_ricci(distances: np.ndarray,
                               cutoff: float = 1.0) -> np.ndarray:
    """
    Approximate the Ollivier‑Ricci curvature matrix κ using the scheme described
    in Parent B.

    For each node i we define a lazy random‑walk measure μᵢ that is uniform over
    the neighbours within ``cutoff``.  The 1‑Wasserstein distance W₁(μᵢ, μⱼ) is
    approximated by the average absolute difference of the neighbour vectors.
    """
    n = distances.shape[0]
    κ = np.zeros((n, n), dtype=float)

    # Pre‑compute neighbour masks
    neighbour_masks = [_neighbour_mask(distances, cutoff) for _ in range(n)]

    for i in range(n):
        mask_i = neighbour_masks[i]
        if not mask_i.any():
            continue
        for j in range(i + 1, n):
            mask_j = neighbour_masks[j]
            if not mask_j.any():
                continue

            # Approximate W₁ by mean absolute difference of distances to common neighbours
            # (a cheap proxy that satisfies the required symmetry).
            common = mask_i & mask_j
            if common.any():
                w1 = np.mean(np.abs(distances[i, common] - distances[j, common]))
            else:
                # If no common neighbours, fallback to average of individual neighbourhood spreads
                w1_i = np.mean(distances[i, mask_i])
                w1_j = np.mean(distances[j, mask_j])
                w1 = 0.5 * (w1_i + w1_j)

            d_ij = distances[i, j] + 1e-30  # avoid division by zero
            κ[i, j] = κ[j, i] = 1.0 - w1 / d_ij

    return κ


# ----------------------------------------------------------------------
# 4. NLMS adaptation with curvature regularisation
# ----------------------------------------------------------------------


def nlms_curvature_update(w: np.ndarray,
                          x: np.ndarray,
                          d: float,
                          curvature: float,
                          mu: float = 0.1,
                          eps: float = 1e-6) -> np.ndarray:
    """
    Perform a single Normalised Least‑Mean‑Squares (NLMS) weight update,
    modulated by a scalar curvature term.

    The classic NLMS update:
        e   = d - w·x
        w'  = w + (mu / (||x||² + eps)) * e * x

    Here the error ``e`` is multiplied by ``(1 + curvature)`` so that updates
    are amplified in regions of positive curvature (tightly clustered points)
    and dampened in negatively curved regions.
    """
    x_norm_sq = np.dot(x, x) + eps
    y = np.dot(w, x)
    e = (d - y) * (1.0 + curvature)
    step = (mu / x_norm_sq) * e
    return w + step * x


# ----------------------------------------------------------------------
# 5. High‑level hybrid pipeline
# ----------------------------------------------------------------------


def hybrid_causal_learning(effects: List[CausalEffect],
                           texts: List[str],
                           hv_dim: int = 4096,
                           feat_dim: int = 128,
                           alpha_fuse: float = 0.5,
                           mu: float = 0.05,
                           curvature_cutoff: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    End‑to‑end hybrid learning routine.

    Returns
    -------
    w : np.ndarray
        Final NLMS weight vector (size = feat_dim).
    κ : np.ndarray
        Ollivier‑Ricci curvature matrix for the combined representations.
    """
    if len(effects) != len(texts):
        raise ValueError("The number of effects must match the number of texts.")

    # 1. Encode causal effects as hypervectors
    hypervectors = np.vstack([
        generate_causal_hypervector(eff, dim=hv_dim).view(np.float64)  # split real/imag
        for eff in effects
    ])  # shape (N, 2*hv_dim)

    # 2. Fuse text features
    feature_matrix = np.vstack([
        fuse_text_features(txt, alpha=alpha_fuse, size=feat_dim)
        for txt in texts
    ])  # shape (N, feat_dim)

    # 3. Project hypervectors onto feature space via a simple real inner product
    #    (take the real part of the complex hypervector).
    proj = np.real(hypervectors[:, :hv_dim]) @ np.random.randn(hv_dim, feat_dim)

    # 4. Combine projection with the raw features to obtain the final representation
    representation = proj + feature_matrix  # shape (N, feat_dim)

    # 5. Compute geometric quantities
    distances = pairwise_euclidean(representation)
    κ = approximate_ollivier_ricci(distances, cutoff=curvature_cutoff)

    # 6. Initialise NLMS weights (one weight per feature dimension)
    w = np.zeros(feat_dim, dtype=float)

    # 7. Perform one NLMS pass per sample, using the *average* curvature of the
    #    neighbourhood of the sample as the regulariser.
    for idx, eff in enumerate(effects):
        x = representation[idx]
        # Desired scalar: use the ATE estimate if present, else 0.0
        d_target = float(eff.ate_estimate) if eff.ate_estimate is not None else 0.0

        # Curvature regulariser: average κ over neighbours within the cutoff
        neigh_mask = distances[idx] <= curvature_cutoff
        curv = float(κ[idx, neigh_mask].mean()) if neigh_mask.any() else 0.0

        w = nlms_curvature_update(w, x, d_target, curvature=curv, mu=mu)

    return w, κ


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a handful of synthetic causal effects
    synth_effects = [
        CausalEffect(
            effect_id=f"e{i}",
            treatment=f"T{i}",
            outcome=f"O{i}",
            confounders=(f"C{i}_1", f"C{i}_2"),
            ate_estimate=random.uniform(-1.0, 1.0),
            ate_confidence_interval=(random.uniform(-1.5, -0.5), random.uniform(0.5, 1.5)),
            refutation_passed=bool(random.getrandbits(1)),
            refutation_methods=("placebo_test", "subset_analysis"),
            heterogeneous_effects={}
        )
        for i in range(8)
    ]

    # Corresponding dummy texts
    texts = [f"This is a synthetic document number {i}." for i in range(8)]

    # Run the hybrid pipeline
    final_weights, curvature_matrix = hybrid_causal_learning(
        effects=synth_effects,
        texts=texts,
        hv_dim=2048,
        feat_dim=64,
        alpha_fuse=0.6,
        mu=0.07,
        curvature_cutoff=1.2,
    )

    print("Final NLMS weight vector (first 10 entries):")
    print(final_weights[:10])
    print("\nCurvature matrix (rounded, first 4×4 block):")
    print(np.round(curvature_matrix[:4, :4], 3))

    # Basic sanity checks
    assert final_weights.shape == (64,)
    assert curvature_matrix.shape == (8, 8)
    print("\nSmoke test completed successfully.")