# DARWIN HAMMER — match 4141, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1968_s4.py (gen6)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py (gen4)
# born: 2026-05-29T23:53:52Z

"""Hybrid Algorithm combining Bayesian edge modeling (Parent A) with Geometric Algebra
operations (Parent B).

Mathematical bridge
-------------------
* In Parent A each graph edge is described by a **Beta prior** whose posterior mean
  (a scalar in \[0,1\]) quantifies the reliability of that edge.
* In Parent B blades are identified by frozensets of integer basis indices and
  multivectors are linear combinations of those blades.

The fusion treats every edge *e* as a **grade‑1 blade** whose index is the edge
identifier. The posterior mean of the edge becomes the scalar coefficient of
that blade.  Geometric products of such blades therefore combine edge reliabilities
according to Clifford algebra rules, while the scalar (grade‑0) part of the
resulting multivector can be interpreted as an **effective energy** of a
sub‑graph.  This scalar energy is finally fed to the Metropolis‑style
`acceptance_probability` from Parent A, giving a unified acceptance criterion
that depends on both Bayesian statistics and geometric structure.

The module provides three illustrative hybrid operations:
1. `bayesian_edges_to_mv` – convert a mapping of edge priors into a multivector.
2. `geometric_product_mv` – perform a Clifford product of two multivectors.
3. `metropolis_accept_mv` – evaluate Metropolis acceptance using the scalar part
   of a multivector (e.g. the effective energy after geometric combination).

All functions are pure Python and rely only on the standard library and NumPy.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Dict, Tuple, List, FrozenSet, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (statistical / Monte‑Carlo)
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability, never exactly zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    try:
        prob = math.exp(-delta_energy / temperature)
    except OverflowError:
        prob = float('inf')
    return max(min(prob, 1.0), 1e-12)


def hoeffding_decision(num_samples: int, epsilon: float, delta: float = 0.05) -> bool:
    """Return True if Hoeffding bound is tighter than *epsilon*."""
    if num_samples <= 0:
        return False
    bound = math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))
    return bound < epsilon


@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior for Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0


def bayesian_edge_update(
    prior: EdgeBetaPrior,
    successes: int,
    failures: int,
) -> Tuple[float, float, EdgeBetaPrior]:
    """Posterior mean, variance and updated prior."""
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    total = new_alpha + new_beta
    posterior_mean = new_alpha / total
    posterior_var = (new_alpha * new_beta) / (total ** 2 * (total + 1))
    return posterior_mean, posterior_var, EdgeBetaPrior(new_alpha, new_beta)


# ----------------------------------------------------------------------
# Parent B components (geometric algebra)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort indices and cancel pairs of equal indices (which square to +1).
    Returns the sorted tuple of remaining indices and the sign (+1 / -1)
    induced by the swaps needed to bring the original list into the sorted order.
    """
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # Keep only those with odd multiplicity
    odd_indices = [i for i, c in counts.items() if c % 2 == 1]

    # Build a list preserving original order but dropping even copies
    cleaned: List[int] = []
    seen: Dict[int, int] = {}
    for i in indices:
        if counts[i] % 2 == 1:
            # keep the first occurrence of each odd‑count index
            if seen.get(i, 0) == 0:
                cleaned.append(i)
                seen[i] = 1

    # Bubble‑sort while tracking sign flips
    sign = 1
    n = len(cleaned)
    for i in range(n):
        for j in range(n - 1 - i):
            if cleaned[j] > cleaned[j + 1]:
                cleaned[j], cleaned[j + 1] = cleaned[j + 1], cleaned[j]
                sign = -sign
    return tuple(cleaned), sign


def _multiply_blades(
    blade_a: FrozenSet[int],
    blade_b: FrozenSet[int],
) -> Tuple[FrozenSet[int], int]:
    """
    Clifford (geometric) product of two basis blades.
    Returns (resulting_blade, sign).
    """
    combined = tuple(list(blade_a) + list(blade_b))
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign


def vector_to_mv(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """Convert a 1‑D NumPy array into a multivector containing only grade‑1 blades."""
    mv: Dict[FrozenSet[int], float] = {}
    for i, coeff in enumerate(v):
        if coeff != 0.0:
            mv[frozenset({i})] = float(coeff)
    return mv


def mv_to_scalar(mv: Dict[FrozenSet[int], float]) -> float:
    """Extract the grade‑0 (scalar) component of a multivector; returns 0.0 if absent."""
    return mv.get(frozenset(), 0.0)


def geometric_product_mv(
    mv_a: Dict[FrozenSet[int], float],
    mv_b: Dict[FrozenSet[int], float],
) -> Dict[FrozenSet[int], float]:
    """
    Compute the geometric (Clifford) product of two multivectors.
    The result is a new multivector dictionary.
    """
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            new_blade, sign = _multiply_blades(blade_a, blade_b)
            contribution = coeff_a * coeff_b * sign
            result[new_blade] = result.get(new_blade, 0.0) + contribution
    # Remove near‑zero entries for numerical cleanliness
    eps = 1e-14
    result = {b: c for b, c in result.items() if abs(c) > eps}
    return result


# ----------------------------------------------------------------------
# Hybrid layer – marrying Bayesian edges with geometric algebra
# ----------------------------------------------------------------------
def bayesian_edges_to_mv(
    edge_priors: Dict[int, EdgeBetaPrior],
    successes: Dict[int, int],
    failures: Dict[int, int],
) -> Dict[FrozenSet[int], float]:
    """
    Convert a collection of edges (identified by integer IDs) into a multivector.
    Each edge becomes a grade‑1 blade whose coefficient is the posterior mean
    reliability obtained from the Beta update.
    """
    mv: Dict[FrozenSet[int], float] = {}
    for edge_id, prior in edge_priors.items():
        s = successes.get(edge_id, 0)
        f = failures.get(edge_id, 0)
        mean, _, _ = bayesian_edge_update(prior, s, f)
        if mean != 0.0:
            mv[frozenset({edge_id})] = mean
    return mv


def metropolis_accept_mv(
    delta_mv: Dict[FrozenSet[int], float],
    temperature: float,
) -> bool:
    """
    Apply Metropolis acceptance to the scalar part of a multivector.
    The scalar (grade‑0) component is interpreted as an energy difference.
    """
    delta_energy = mv_to_scalar(delta_mv)
    prob = acceptance_probability(delta_energy, temperature)
    return random.random() < prob


def hybrid_step(
    edge_priors: Dict[int, EdgeBetaPrior],
    successes: Dict[int, int],
    failures: Dict[int, int],
    geometry_vectors: List[np.ndarray],
    temperature: float,
) -> Tuple[bool, Dict[FrozenSet[int], float]]:
    """
    Perform a full hybrid iteration:

    1. Build a multivector from Bayesian edge updates.
    2. Convert each geometry vector to a multivector and multiply them all together.
    3. Geometrically combine the edge MV with the geometry MV via Clifford product.
    4. Decide acceptance using the scalar part of the combined MV.

    Returns (accepted, combined_multivector).
    """
    # 1. Edge‑based MV
    edge_mv = bayesian_edges_to_mv(edge_priors, successes, failures)

    # 2. Geometry MV (product of all supplied vectors)
    if not geometry_vectors:
        raise ValueError("at least one geometry vector required")
    geom_mv = vector_to_mv(geometry_vectors[0])
    for vec in geometry_vectors[1:]:
        geom_mv = geometric_product_mv(geom_mv, vector_to_mv(vec))

    # 3. Combine edge and geometry information
    combined_mv = geometric_product_mv(edge_mv, geom_mv)

    # 4. Acceptance decision
    accepted = metropolis_accept_mv(combined_mv, temperature)
    return accepted, combined_mv


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph with three edges
    edge_priors = {
        0: EdgeBetaPrior(alpha=2.0, beta=3.0),
        1: EdgeBetaPrior(alpha=5.0, beta=2.0),
        2: EdgeBetaPrior(alpha=1.0, beta=1.0),
    }
    successes = {0: 1, 1: 4, 2: 0}
    failures = {0: 2, 1: 1, 2: 1}

    # Geometry: two 3‑D vectors
    v1 = np.array([1.0, 0.0, 2.0])
    v2 = np.array([0.5, -1.0, 0.0])

    # Run one hybrid step
    temperature = 1.5
    accepted, combined = hybrid_step(
        edge_priors,
        successes,
        failures,
        geometry_vectors=[v1, v2],
        temperature=temperature,
    )

    print(f"Accepted: {accepted}")
    print("Combined multivector (blade -> coefficient):")
    for blade, coeff in combined.items():
        # pretty‑print blade as sorted tuple
        print(f"  {tuple(sorted(blade))}: {coeff:.6f}")

    # Demonstrate Hoeffding decision on sample count from combined MV
    sample_count = len(combined)
    eps = 0.1
    decision = hoeffding_decision(sample_count, eps)
    print(f"Hoeffding decision (samples={sample_count}, eps={eps}): {decision}")