# DARWIN HAMMER — match 3770, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_krampu_m11_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m413_s5.py (gen4)
# born: 2026-05-29T23:51:32Z

"""Hybrid Fisher‑SSIM / MinHash‑Infotaxis Algorithm

Parents:
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_krampu_m11_s0 (Fisher‑SSIM routing,
  decision‑hygiene pruning, adjacency matrix for Ollivier‑Ricci curvature)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m413_s5 (Semantic neighbours,
  MinHash priors, Bayesian update, entropy‑driven infotaxis, minimum‑cost tree)

Mathematical bridge:
Both parents operate on a graph G=(V,E).  In the Fisher‑SSIM side the adjacency
matrix A encodes edge weights that are later used to compute Ollivier‑Ricci
curvature.  In the MinHash side the same edge‑weight matrix is interpreted as
the prior probability distribution p(e) for each edge.  The hybrid therefore
maps

    prior(e)   ← normalized MinHash signature of the token sets of the incident
                  vertices,
    likelihood(e) ← exp(‑α·d_fisher(e)) where d_fisher(e)=1‑SSIM(x_u,x_v) is a
                  distance derived from the Fisher‑SSIM similarity of the node
                  feature vectors,
    posterior(e) ← Bayesian update of prior with likelihood,
    curvature(e) ← Ollivier‑Ricci curvature computed on the posterior‑weighted
                  adjacency matrix.

The decision metric M from the Fisher‑SSIM parent is also reused as a
pruning factor p(t)·[w_f·SSIM+ w_h·H·Σ_i w_i·f_i] that modulates the edge‑wise
likelihood before the Bayesian step.

The code below implements this fused pipeline."""
import math
import random
import sys
from pathlib import Path
import numpy as np
from collections import Counter

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    numerator = (2 * mx * my + C1) * (2 * cov + C2)
    denominator = (mx * mx + my * my + C1) * (vx + vy + C2)
    return numerator / denominator


def decision_metric(p_t: float,
                    w_f: float,
                    ssim_val: float,
                    w_h: float,
                    H: float,
                    feature_counts: np.ndarray,
                    feature_flags: np.ndarray) -> float:
    """Unified decision metric M = p(t)·[w_f·SSIM + w_h·H·Σ w_i·f_i]."""
    hygiene_term = H * np.sum(feature_counts * feature_flags)
    return p_t * (w_f * ssim_val + w_h * hygiene_term)


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def minhash_signature(tokens: set[str], num_buckets: int = 128) -> np.ndarray:
    """Simple MinHash signature: for each bucket keep the minimum 64‑bit hash."""
    max_hash = (1 << 64) - 1
    signature = np.full(num_buckets, max_hash, dtype=np.uint64)
    for token in tokens:
        h = hash(token) & max_hash
        bucket = h % num_buckets
        if h < signature[bucket]:
            signature[bucket] = h
    return signature


def prior_from_signature(sig: np.ndarray) -> np.ndarray:
    """Convert a MinHash signature into a probability distribution."""
    # Presence of a finite hash indicates the bucket is “covered”.
    present = sig != ((1 << 64) - 1)
    if not np.any(present):
        # Avoid division by zero – fall back to uniform prior.
        return np.full_like(sig, 1.0 / sig.size, dtype=float)
    prior = present.astype(float)
    return prior / prior.sum()


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = L·P + FP·(1‑P)."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior = L·P / P(E)."""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def compute_fisher_weights(theta_vals: np.ndarray,
                           center: float,
                           width: float,
                           eps: float = 1e-12) -> np.ndarray:
    """Vectorised Fisher information for an array of theta values."""
    intensity = np.maximum(
        np.exp(-0.5 * ((theta_vals - center) / width) ** 2), eps
    )
    derivative = intensity * (-(theta_vals - center) / (width * width))
    return (derivative * derivative) / intensity


def hybrid_edge_update(adj_matrix: np.ndarray,
                       node_features: list[np.ndarray],
                       token_sets: list[set[str]],
                       alpha: float = 1.0,
                       false_positive: float = 1e-3,
                       p_t: float = 1.0,
                       eps: float = 1e-12) -> np.ndarray:
    """
    Perform a single hybrid update of the adjacency matrix.

    Steps for each unordered pair (i, j):
      1. Compute SSIM between feature vectors -> s.
      2. Compute Fisher weight w_f = I/(I+ε) where I = fisher_score(θ,…) with
         θ taken as the SSIM value (treated as “theta”).
      3. Build MinHash signatures for both vertices, derive priors p_i, p_j,
         and combine them by averaging.
      4. Compute Shannon entropy H of the combined prior.
      5. Compute hygiene term using dummy binary flags derived from token
         presence (f_i = 1 if token appears in either set).
      6. Assemble decision metric M and use it to scale the likelihood:
         likelihood = exp(-α·(1‑s))·M.
      7. Bayesian update of the prior with the scaled likelihood.
      8. Replace the original adjacency weight with the posterior.

    The function returns a new adjacency matrix with the same shape.
    """
    n = adj_matrix.shape[0]
    if adj_matrix.shape != (n, n):
        raise ValueError("Adjacency matrix must be square")
    if len(node_features) != n or len(token_sets) != n:
        raise ValueError("Feature and token lists must match number of nodes")

    # Pre‑compute MinHash priors for each node.
    priors = [prior_from_signature(minhash_signature(ts)) for ts in token_sets]

    # Prepare output matrix.
    new_adj = adj_matrix.copy().astype(float)

    for i in range(n):
        for j in range(i + 1, n):
            # 1. SSIM similarity.
            s = ssim(node_features[i], node_features[j])

            # 2. Fisher weight.
            I = fisher_score(s, center=0.5, width=0.2, eps=eps)
            w_f = I / (I + eps)

            # 3. Prior from combined MinHash signatures.
            combined_prior = (priors[i] + priors[j]) / 2.0

            # 4. Entropy of the combined prior.
            H = -np.sum(combined_prior * np.log(combined_prior + eps))

            # 5. Hygiene flags – binary indicator for each token bucket.
            flags = (priors[i] > 0).astype(float) | (priors[j] > 0).astype(float)
            counts = np.ones_like(flags)  # uniform raw counts for illustration.

            # 6. Decision metric M and scaled likelihood.
            M = decision_metric(p_t, w_f, s, w_h=H / (H + eps), H=H,
                                feature_counts=counts,
                                feature_flags=flags)
            likelihood = math.exp(-alpha * (1.0 - s)) * M

            # 7. Bayesian posterior for the edge.
            prior_edge = combined_prior.mean()  # collapse distribution to a single scalar.
            posterior = bayes_update(prior_edge, likelihood, false_positive)

            # 8. Update adjacency weight.
            new_adj[i, j] = new_adj[j, i] = posterior

    return new_adj


def ollivier_ricci_curvature(adj_matrix: np.ndarray,
                             epsilon: float = 1e-12) -> np.ndarray:
    """
    Very light‑weight Ollivier‑Ricci curvature estimator.

    For each edge (i, j) we treat the normalized rows of the adjacency matrix
    as probability measures μ_i, μ_j and compute the 1‑Wasserstein distance
    using the graph distance (here approximated by the inverse of the weight).
    The curvature is κ_ij = 1 - W(μ_i, μ_j) / d_ij.

    This implementation is not numerically optimal but suffices for the
    hybrid demonstration.
    """
    n = adj_matrix.shape[0]
    curvature = np.zeros((n, n), dtype=float)

    # Normalise rows to obtain probability measures.
    row_sums = adj_matrix.sum(axis=1, keepdims=True) + epsilon
    probs = adj_matrix / row_sums

    for i in range(n):
        for j in range(i + 1, n):
            # Graph distance approximated as inverse of weight; fall back to large value.
            weight = adj_matrix[i, j] if adj_matrix[i, j] > 0 else epsilon
            d_ij = 1.0 / weight

            # Wasserstein distance under L1 metric on the probability vectors.
            w_dist = np.sum(np.abs(probs[i] - probs[j]))

            kappa = 1.0 - w_dist / d_ij
            curvature[i, j] = curvature[j, i] = kappa

    return curvature


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph with 3 nodes.
    n_nodes = 3
    # Random feature vectors (1‑D signals of length 8).
    rng = np.random.default_rng(42)
    features = [rng.integers(0, 256, size=8).astype(float) for _ in range(n_nodes)]

    # Token sets for MinHash priors.
    token_sets = [
        {"alpha", "beta", "gamma"},
        {"beta", "delta", "epsilon"},
        {"gamma", "zeta", "eta"}
    ]

    # Initial adjacency matrix (fully connected, weight=1.0).
    A0 = np.full((n_nodes, n_nodes), 1.0, dtype=float)
    np.fill_diagonal(A0, 0.0)

    # Perform hybrid update.
    A1 = hybrid_edge_update(
        adj_matrix=A0,
        node_features=features,
        token_sets=token_sets,
        alpha=2.0,
        false_positive=1e-4,
        p_t=0.9
    )

    # Compute curvature on the updated matrix.
    curvature = ollivier_ricci_curvature(A1)

    # Simple sanity prints – should not raise.
    print("Updated adjacency matrix:\n", A1)
    print("Estimated Ollivier‑Ricci curvature:\n", curvature)