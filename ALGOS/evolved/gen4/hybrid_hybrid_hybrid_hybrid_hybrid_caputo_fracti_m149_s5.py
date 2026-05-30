# DARWIN HAMMER — match 149, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_bandit_m40_s0.py (gen3)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s4.py (gen1)
# born: 2026-05-29T23:27:09Z

"""
Hybrid Algorithm: hybrid_pheromone_fractional_tree.py

Parents:
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_bandit_m40_s0.py (Pheromone‑Store‑Bandit system)
- hybrid_caputo_fractional_minimum_cost_tree_m35_s4.py (Caputo fractional calculus + minimum‑cost tree)

Mathematical Bridge:
The bridge is the *fractional decay kernel* from the Caputo‑fractional side, which we use as a
time‑varying decay factor for the pheromone signal that drives the StoreState dynamics.
Conversely, the current StoreState (its level) is fed back as the “time” argument of the
fractional decay, thereby modulating the edge weights of a minimum‑cost spanning tree.
Thus the pheromone decay and the tree‑edge decay are coupled through the store level,
creating a closed hybrid loop:

    pheromone_signal(t) = signal₀·½^{Δt/half_life}·fractional_decay(α, store_level)
    store.update(inflow=∑pheromone_signal, outflow=...)
    edge_weight(t) = base_weight·fractional_decay(α, store_level)

The three core functions below demonstrate this fused dynamics.
"""

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Parent A – StoreState and basic pheromone handling
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0      # inflow scaling
    beta: float = 1.0       # outflow scaling
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Euler update of the store level."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded signal derived from the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Parent B – Caputo fractional tools
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def fractional_decay(alpha: float, t: float) -> float:
    """Fractional decay kernel  t^{α-1} / Γ(α) ."""
    if t <= 0:
        return 0.0
    return t ** (alpha - 1) / gamma_lanczos(alpha)

# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Manages pheromone signals with fractional decay influenced by the store."""
    def __init__(self):
        self.pheromones: Dict[str, Tuple[float, datetime]] = {}  # key -> (base_signal, last_update)

    def calculate_pheromone_signal(
        self,
        key: str,
        base_signal: float,
        half_life_seconds: float,
        store_level: float,
        alpha_frac: float = 0.8,
    ) -> float:
        """
        Compute a pheromone signal that decays exponentially (half‑life) and is
        additionally weighted by a fractional decay kernel driven by the store level.
        """
        now = datetime.now(timezone.utc)
        if key not in self.pheromones:
            self.pheromones[key] = (base_signal, now)
            elapsed = 0.0
        else:
            _, last = self.pheromones[key]
            elapsed = (now - last).total_seconds()
            self.pheromones[key] = (base_signal, now)

        # Classic exponential half‑life decay
        exp_decay = base_signal * (0.5 ** (elapsed / half_life_seconds))

        # Fractional decay factor using the current store level as “time”
        frac_factor = fractional_decay(alpha_frac, store_level + 1e-9)  # avoid zero

        return exp_decay * frac_factor


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def update_store_with_pheromones(
    store: StoreState,
    pheromone_system: HybridPheromoneSystem,
    sources: List[Tuple[str, float, float]],
    sinks: List[float],
    half_life: float,
    alpha_frac: float = 0.8,
) -> Tuple[float, float]:
    """
    Update the StoreState using pheromone inflows.
    `sources` is a list of (key, base_signal, weight) tuples.
    `sinks` is a list of outflow magnitudes.
    """
    inflow = [
        pheromone_system.calculate_pheromone_signal(
            key, base_signal * weight, half_life, store.level, alpha_frac
        )
        for key, base_signal, weight in sources
    ]
    outflow = sinks
    level, delta = store.update(inflow, outflow)
    return level, delta


def fractional_tree_cost(
    nodes: List[int],
    edges: List[Tuple[int, int, float]],
    store_level: float,
    alpha_frac: float = 0.8,
) -> float:
    """
    Compute a minimum‑cost tree cost where each edge weight is modulated by the
    fractional decay kernel evaluated at the current store level.
    """
    # Apply fractional decay to each edge weight
    decayed_weights = [
        (u, v, w * fractional_decay(alpha_frac, store_level + 1e-9))
        for (u, v, w) in edges
    ]

    # Simple Prim‑like aggregation (since we lack a full graph library)
    visited = {nodes[0]}
    total = 0.0
    while len(visited) < len(nodes):
        # Find cheapest edge crossing the cut
        candidate = None
        min_w = float('inf')
        for u, v, w in decayed_weights:
            if (u in visited) ^ (v in visited):
                if w < min_w:
                    min_w = w
                    candidate = (u, v, w)
        if candidate is None:
            break  # disconnected graph
        total += candidate[2]
        visited.update({candidate[0], candidate[1]})
    return total


def hybrid_step(
    store: StoreState,
    pheromone_system: HybridPheromoneSystem,
    nodes: List[int],
    edges: List[Tuple[int, int, float]],
    source_specs: List[Tuple[str, float, float]],
    sink_vals: List[float],
    half_life: float,
    alpha_frac: float = 0.8,
) -> Dict[str, float]:
    """
    Perform one hybrid iteration:
      1. Update store using pheromone‑driven inflows.
      2. Compute fractional tree cost based on the new store level.
      3. Return a dictionary of diagnostics.
    """
    level, delta = update_store_with_pheromones(
        store, pheromone_system, source_specs, sink_vals, half_life, alpha_frac
    )
    tree_cost = fractional_tree_cost(nodes, edges, store.level, alpha_frac)
    diagnostics = {
        "store_level": level,
        "store_delta": delta,
        "store_dance": store.dance,
        "tree_cost": tree_cost,
        "combined_metric": level + tree_cost,
    }
    return diagnostics


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    store = StoreState(level=5.0, alpha=1.2, beta=0.8, dt=0.5, base=0.5, gain=2.0, limit=15.0)
    pheromone_system = HybridPheromoneSystem()

    # Simple graph (5 nodes, a few edges with base weights)
    nodes = [0, 1, 2, 3, 4]
    edges = [
        (0, 1, 3.0),
        (0, 2, 2.5),
        (1, 3, 4.0),
        (2, 3, 1.5),
        (3, 4, 2.0),
        (1, 4, 5.0),
    ]

    # Define pheromone sources: (key, base_signal, weight)
    source_specs = [
        ("hive_a", 1.0, 1.0),
        ("hive_b", 0.8, 0.5),
        ("hive_c", 0.5, 0.2),
    ]

    # Outflows (e.g., consumption rates)
    sink_vals = [0.3, 0.1]

    half_life_seconds = 30.0
    alpha_frac = 0.75

    # Run a few hybrid steps
    for step in range(5):
        diag = hybrid_step(
            store,
            pheromone_system,
            nodes,
            edges,
            source_specs,
            sink_vals,
            half_life_seconds,
            alpha_frac,
        )
        print(f"Step {step+1}: {diag}")
    sys.exit(0)