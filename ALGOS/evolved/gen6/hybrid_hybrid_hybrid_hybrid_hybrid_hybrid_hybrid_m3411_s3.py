# DARWIN HAMMER — match 3411, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1038_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_percyphon_hyb_m992_s3.py (gen3)
# born: 2026-05-29T23:49:56Z

"""Hybrid Algorithm: Geometric‑Koopman ↔ Bayesian‑Fisher Fusion

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1038_s0.py (Geometric algebra + Koopman operator)
- hybrid_hybrid_hybrid_minimu_hybrid_percyphon_hyb_m992_s3.py (Bayesian edge posteriors, procedural slots)

Mathematical Bridge
------------------
1. **Geometric ↔ Probabilistic Mapping**  
   Grade‑1 blades (vectors) of a multivector are interpreted as *node priors*:
   `p_i = |c_i| / Σ|c|`, where `c_i` is the coefficient of basis blade `e_i`.  
   Grade‑2 blades (bivectors) are interpreted as *edge likelihoods*:
   `ℓ_{ij} = |c_{ij}| / Σ|c_{grade2}|`.

2. **Koopman Linear Evolution**  
   The coefficient vector of the multivector, `v ∈ ℝ^m`, is advanced by a
   Koopman matrix `K ∈ ℝ^{m×m}`: `v' = K·v`. This provides a linearised
   propagation of the underlying nonlinear geometric dynamics.

3. **Bayesian‑Fisher Fusion**  
   After Koopman evolution the node priors are updated via Bayes rule using
   the edge likelihoods as evidence.  The Fisher information of the
   multivector coefficients (`I = Σ (∂log p/∂θ)^2`) is employed as a
   weighting factor `w` that scales the posterior contribution of each edge.

The three functions below demonstrate the complete hybrid pipeline:
`multivector_to_priors`, `koopman_evolve`, and `bayesian_fisher_fusion`.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, FrozenSet, Any

import numpy as np

# ----------------------------------------------------------------------
# Geometric Algebra core (excerpt from Parent A)
# ----------------------------------------------------------------------


def _blade_sign(indices: list) -> Tuple[list, int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # prune near‑zero components
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def __add__(self, other: "Multivector") -> "Multivector":
        assert self.n == other.n
        new_comp = dict(self.components)
        for b, c in other.components.items():
            new_comp[b] = new_comp.get(b, 0.0) + c
        return Multivector(new_comp, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    def grade(self, k: int) -> "Multivector":
        return Multivector({b: c for b, c in self.components.items() if len(b) == k}, self.n)

    def coeff_vector(self, ordered_blades: Tuple[FrozenSet[int], ...]) -> np.ndarray:
        """Return a dense vector of coefficients ordered by `ordered_blades`."""
        return np.array([self.components.get(b, 0.0) for b in ordered_blades], dtype=float)

    @staticmethod
    def from_coeff_vector(vec: np.ndarray, ordered_blades: Tuple[FrozenSet[int], ...], n: int) -> "Multivector":
        comp = {b: float(v) for b, v in zip(ordered_blades, vec) if abs(v) > 1e-15}
        return Multivector(comp, n)


# ----------------------------------------------------------------------
# Bayesian helpers (excerpt from Parent B)
# ----------------------------------------------------------------------


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability P(evidence)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Return the posterior P(H|E) = P(E|H)P(H)/P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def bayes_edge_posterior(
    node_prior: float,
    edge_likelihood: float,
    false_positive: float = 0.1,
) -> float:
    """Posterior that an edge is useful given the node prior and edge likelihood."""
    marginal = bayes_marginal(node_prior, edge_likelihood, false_positive)
    return bayes_update(node_prior, edge_likelihood, marginal)


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------


def multivector_to_priors(mv: Multivector) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """
    Convert a multivector into:
      - node priors (grade‑1 blades)
      - edge likelihoods (grade‑2 blades)

    The absolute values of the coefficients are normalised to form proper
    probability distributions.
    """
    grade1 = mv.grade(1).components
    grade2 = mv.grade(2).components

    # Node priors
    total1 = sum(abs(c) for c in grade1.values()) + 1e-12
    node_priors = {list(b)[0]: abs(c) / total1 for b, c in grade1.items()}

    # Edge likelihoods (unordered pairs)
    total2 = sum(abs(c) for c in grade2.values()) + 1e-12
    edge_likelihoods = {}
    for blade, coeff in grade2.items():
        idx = tuple(sorted(blade))
        if len(idx) == 2:
            edge_likelihoods[idx] = abs(coeff) / total2
    return node_priors, edge_likelihoods


def koopman_evolve(mv: Multivector, K: np.ndarray, ordered_blades: Tuple[FrozenSet[int], ...]) -> Multivector:
    """
    Apply a Koopman operator (linear matrix) to the coefficient vector of `mv`.
    Returns a new Multivector with evolved coefficients.
    """
    vec = mv.coeff_vector(ordered_blades)
    if K.shape != (len(vec), len(vec)):
        raise ValueError("Koopman matrix shape mismatch with coefficient vector length")
    new_vec = K @ vec
    return Multivector.from_coeff_vector(new_vec, ordered_blades, mv.n)


def fisher_information(weights: np.ndarray) -> float:
    """
    Simple scalar Fisher information for a discrete distribution `weights`
    (assumed to sum to 1).  I = Σ (∂log w_i / ∂θ)^2, here we approximate
    ∂log w_i / ∂θ ≈ 1/w_i, giving I = Σ 1/w_i.
    """
    eps = 1e-12
    w = np.clip(weights, eps, None)
    return float(np.sum(1.0 / w))


def bayesian_fisher_fusion(
    node_priors: Dict[int, float],
    edge_likelihoods: Dict[Tuple[int, int], float],
    false_positive: float = 0.1,
) -> Dict[Tuple[int, int], float]:
    """
    For every edge (i,j) compute a posterior using Bayes rule where the node
    prior is the average of the two incident node priors.  The result is
    weighted by a Fisher‑information factor derived from the edge likelihoods.
    Returns a dictionary mapping edges to weighted posterior probabilities.
    """
    # Build arrays for Fisher weighting
    edge_vals = np.array(list(edge_likelihoods.values()))
    fisher_weight = fisher_information(edge_vals)

    posteriors = {}
    for (i, j), ℓ in edge_likelihoods.items():
        prior_i = node_priors.get(i, 0.5)
        prior_j = node_priors.get(j, 0.5)
        avg_prior = 0.5 * (prior_i + prior_j)
        post = bayes_edge_posterior(avg_prior, ℓ, false_positive)
        # Apply Fisher scaling (normalised)
        scaled = post * fisher_weight / (fisher_weight + 1e-12)
        posteriors[(i, j)] = scaled
    return posteriors


def hybrid_step(
    mv: Multivector,
    K: np.ndarray,
    ordered_blades: Tuple[FrozenSet[int], ...],
    false_positive: float = 0.1,
) -> Tuple[Multivector, Dict[Tuple[int, int], float]]:
    """
    Execute one full hybrid iteration:
      1. Evolve the multivector via the Koopman operator.
      2. Extract node priors and edge likelihoods.
      3. Fuse them with Bayesian‑Fisher weighting.
    Returns the evolved multivector and the edge‑posterior map.
    """
    evolved_mv = koopman_evolve(mv, K, ordered_blades)
    node_priors, edge_likelihoods = multivector_to_priors(evolved_mv)
    edge_post = bayesian_fisher_fusion(node_priors, edge_likelihoods, false_positive)
    return evolved_mv, edge_post


# ----------------------------------------------------------------------
# Utility for generating the ordered blade list (required for Koopman)
# ----------------------------------------------------------------------


def generate_ordered_blades(n: int) -> Tuple[FrozenSet[int], ...]:
    """
    Produce a deterministic ordering of all blades up to grade n.
    For simplicity we generate blades of grades 0,1,2 only (scalar, vector,
    bivector) which suffices for the demo.
    """
    blades = [frozenset()]  # scalar
    # grade‑1
    for i in range(1, n + 1):
        blades.append(frozenset([i]))
    # grade‑2
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            blades.append(frozenset([i, j]))
    return tuple(blades)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a 3‑dimensional geometric algebra (e1, e2, e3)
    n_dim = 3
    blades = generate_ordered_blades(n_dim)

    # Construct an initial multivector:
    #   0.6*e1 + 0.4*e2 + 0.2*e1^e2
    init_components = {
        frozenset([1]): 0.6,
        frozenset([2]): 0.4,
        frozenset([1, 2]): 0.2,
    }
    mv = Multivector(init_components, n_dim)

    # Simple Koopman matrix: slight rotation in the subspace of (e1, e2, e1^e2)
    dim = len(blades)
    K = np.eye(dim) * 0.9
    # add a coupling between e1 and e2
    idx_e1 = blades.index(frozenset([1]))
    idx_e2 = blades.index(frozenset([2]))
    K[idx_e1, idx_e2] = 0.1
    K[idx_e2, idx_e1] = 0.1
    # couple bivector with e1
    idx_b12 = blades.index(frozenset([1, 2]))
    K[idx_b12, idx_e1] = 0.05
    K[idx_e1, idx_b12] = 0.05

    # Run a single hybrid iteration
    evolved_mv, edge_posteriors = hybrid_step(mv, K, blades)

    print("Evolved multivector components:")
    for blade, coeff in evolved_mv.components.items():
        print(f"  Blade {sorted(blade)} : {coeff:.4f}")

    print("\nEdge posterior probabilities (weighted by Fisher information):")
    for (i, j), prob in edge_posteriors.items():
        print(f"  Edge ({i},{j}) -> {prob:.4f}")

    # Ensure that the code runs without raising exceptions
    sys.exit(0)