# DARWIN HAMMER — match 1402, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s2.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m601_s0.py (gen4)
# born: 2026-05-29T23:36:00Z

"""
Hybrid Algorithm: Pheromone-Weighted Minimum-Cost Tree with Hoeffding‑Guided Bayesian Updates

Parents
-------
* **HybridPheromoneSystem** – from `hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s2.py`
  Provides time‑decaying pheromone signals and entropy of probability
  distributions.

* **Endpoint‑Hoeffding Minimum‑Cost Tree** – from
  `hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m601_s0.py`
  Supplies health‑score endpoints, a Hoeffding bound to trigger Bayesian
  posterior updates, and a tropical (max‑plus) ReLU mapping.

Mathematical Bridge
-------------------
The bridge is the *edge weight* of the minimum‑cost tree.  In the parent
algorithm the weight is a function of the endpoint health score.  In the
pheromone system the pheromone signal encodes the confidence (entropy) of a
distribution associated with a network surface.  By **multiplying** the raw
health‑score weight with a *pheromone‑modulation factor* derived from the
signal’s entropy we obtain a unified edge weight:


w_ij = health_i * health_j * exp( -α * Entropy(pheromone_ij) )


where `α` controls the influence of uncertainty.  This unified weight is then
used in the Bayesian edge‑posterior update; the Hoeffding bound decides when
enough observations have been collected to perform the update.

The three public functions below illustrate the full hybrid workflow:
1. `compute_pheromone_modulated_weights`
2. `hoeffding_guided_bayes_update`
3. `evaluate_hybrid_cost`
"""

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Pheromone subsystem (trimmed to essentials)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Manages time‑decaying pheromone signals and computes their entropy."""
    def __init__(self):
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """Create or update a pheromone entry and return the fresh signal value."""
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        else:
            prev = self.pheromones[surface_key]
            elapsed = (now - prev["created_time"]).total_seconds()
            decayed = prev["signal_value"] * math.pow(
                0.5, elapsed / prev["half_life_seconds"]
            )
            # Simple exponential decay + new injection
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": decayed + signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        return self.pheromones[surface_key]["signal_value"]

    def calculate_entropy(self, probabilities: np.ndarray, eps: float = 1e-12) -> float:
        """Shannon entropy of a probability vector."""
        probs = np.clip(probabilities, eps, 1.0)
        probs /= probs.sum()
        return -float(np.sum(probs * np.log(probs)))

    def get_normalized_signal(self, surface_key: str) -> float:
        """Return a normalized (0‑1) signal for a given surface."""
        if surface_key not in self.pheromones:
            return 0.0
        raw = self.pheromones[surface_key]["signal_value"]
        # Simple sigmoid normalisation
        return 1.0 / (1.0 + math.exp(-raw))


# ----------------------------------------------------------------------
# Parent B – Endpoint + Hoeffding + Minimum‑Cost Tree (core pieces)
# ----------------------------------------------------------------------
@dataclass
class Endpoint:
    """Endpoint with health‑related attributes."""
    id: int
    health_score: float
    recovery_priority: float = 0.0


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a variable bounded in [0, r]."""
    if n <= 0:
        raise ValueError("n must be positive")
    return r * math.sqrt((math.log(2.0 / delta)) / (2.0 * n))


def tropical_relu(x: np.ndarray, a: float = 1.0, b: float = 0.0) -> np.ndarray:
    """
    Tropical (max‑plus) ReLU: max(0, a·x + b).
    Implements a piecewise‑linear mapping used as a cheap surrogate for a
    tropical neural network.
    """
    return np.maximum(0.0, a * x + b)


# ----------------------------------------------------------------------
# Hybrid Core – mathematical fusion of the two parents
# ----------------------------------------------------------------------
def compute_pheromone_modulated_weights(
    endpoints: List[Endpoint],
    pheromone_system: HybridPheromoneSystem,
    alpha: float = 0.5,
) -> Dict[Tuple[int, int], float]:
    """
    Build a complete graph over the endpoints.  For each unordered pair (i, j)
    compute a weight that merges the health scores with pheromone‑derived
    uncertainty.

    w_ij = health_i * health_j * exp( -α * Entropy(pheromone_ij) )

    The pheromone key is ``f"{i}-{j}"``.  If no pheromone exists we treat the
    entropy as maximal (i.e. entropy of a uniform distribution).
    """
    weights: Dict[Tuple[int, int], float] = {}
    # Prepare a dummy uniform distribution for missing pheromones
    uniform_entropy = -float(np.sum(np.full(2, 0.5) * np.log(0.5)))
    for i, ep_i in enumerate(endpoints):
        for j, ep_j in enumerate(endpoints):
            if j <= i:
                continue
            key = f"{ep_i.id}-{ep_j.id}"
            if key in pheromone_system.pheromones:
                # Build a 2‑element probability vector from the normalized signal
                prob = np.array(
                    [
                        pheromone_system.get_normalized_signal(key),
                        1.0
                        - pheromone_system.get_normalized_signal(key),
                    ]
                )
                entropy = pheromone_system.calculate_entropy(prob)
            else:
                entropy = uniform_entropy
            mod_factor = math.exp(-alpha * entropy)
            weight = ep_i.health_score * ep_j.health_score * mod_factor
            weights[(ep_i.id, ep_j.id)] = weight
    return weights


def hoeffding_guided_bayes_update(
    edge_weights: Dict[Tuple[int, int], float],
    observations: Dict[Tuple[int, int], List[float]],
    delta: float = 0.05,
) -> Dict[Tuple[int, int], float]:
    """
    Perform a Bayesian‑like update of edge posteriors only when the Hoeffding
    bound indicates sufficient confidence.

    For each edge we keep a running posterior mean μ and count n.
    When the bound ≤ tolerance (here we use 0.1 * μ) we accept the new mean.
    """
    updated_posteriors: Dict[Tuple[int, int], float] = {}
    for edge, prior in edge_weights.items():
        obs = observations.get(edge, [])
        n = len(obs)
        if n == 0:
            updated_posteriors[edge] = prior
            continue
        sample_mean = float(np.mean(obs))
        r = max(obs) - min(obs) if n > 1 else 1.0  # range for Hoeffding
        bound = hoeffding_bound(r, delta, n)
        tolerance = 0.1 * prior
        if bound <= tolerance:
            # Accept Bayesian update: posterior = weighted average
            posterior = (prior + sample_mean) / 2.0
        else:
            posterior = prior  # keep old value
        updated_posteriors[edge] = posterior
    return updated_posteriors


def evaluate_hybrid_cost(
    endpoints: List[Endpoint],
    posteriors: Dict[Tuple[int, int], float],
    a: float = 1.0,
    b: float = 0.0,
) -> float:
    """
    Compute a global cost by feeding the vector of posterior edge weights
    through a tropical ReLU network and aggregating.

    cost = Σ ReLU_a,b( w_ij )
    """
    if not posteriors:
        return 0.0
    weight_vec = np.array(list(posteriors.values()))
    relu_out = tropical_relu(weight_vec, a, b)
    return float(np.sum(relu_out))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise pheromone system and inject a few signals
    pher = HybridPheromoneSystem()
    pher.calculate_pheromone_signal("1-2", "typeA", 0.8, half_life_seconds=30)
    pher.calculate_pheromone_signal("2-3", "typeA", 0.3, half_life_seconds=30)
    pher.calculate_pheromone_signal("1-3", "typeA", 0.5, half_life_seconds=30)

    # Create a small endpoint pool
    eps = [
        Endpoint(id=1, health_score=0.9, recovery_priority=0.2),
        Endpoint(id=2, health_score=0.6, recovery_priority=0.5),
        Endpoint(id=3, health_score=0.4, recovery_priority=0.8),
    ]

    # Step 1: fuse pheromones with health scores
    edge_w = compute_pheromone_modulated_weights(eps, pheromone_system=pher, alpha=0.7)
    print("Initial edge weights (pheromone‑modulated):")
    for e, w in edge_w.items():
        print(f"  {e}: {w:.4f}")

    # Simulate some observations per edge (e.g., measured latencies)
    obs = {
        (1, 2): [0.12, 0.15, 0.11],
        (2, 3): [0.30, 0.28],
        (1, 3): [0.22],
    }

    # Step 2: Hoeffding‑guided Bayesian update
    post = hoeffding_guided_bayes_update(edge_w, observations=obs, delta=0.05)
    print("\nPost‑update edge posteriors:")
    for e, p in post.items():
        print(f"  {e}: {p:.4f}")

    # Step 3: Global hybrid cost evaluation
    total_cost = evaluate_hybrid_cost(eps, post, a=1.2, b=-0.1)
    print(f"\nHybrid total cost (tropical ReLU aggregation): {total_cost:.4f}")

    sys.exit(0)