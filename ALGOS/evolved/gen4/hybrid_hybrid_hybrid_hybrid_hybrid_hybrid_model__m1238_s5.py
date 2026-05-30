# DARWIN HAMMER — match 1238, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s2.py (gen2)
# born: 2026-05-29T23:34:44Z

"""Hybrid Privacy‑VRAM Curvature Scheduler

Parents:
- hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py (privacy risk,
  differential‑privacy aggregation, endpoint circuit‑breaker)
- hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s2.py (VRAM‑aware
  artefact planner + Ollivier‑Ricci curvature on a weighted graph)

Mathematical bridge:
Each artefact registered in :class:`VramPlanner` becomes a node *i* in an
undirected graph *G*.  The node attribute ``estimated_mb`` is turned into a
mass weight  

    w_i = estimated_mb / Σ_j estimated_mb .

When computing the lazy random‑walk measure μ_i(v) for Ollivier‑Ricci curvature
the base laziness α is multiplied by the mass weight, i.e.

    μ_i(v) = α·w_i·δ_{i=v} + (1‑α)·w_i·(1/deg(i))·∑_{u∈N(i)} δ_{u=v} .

Thus curvature values reflect the VRAM allocation landscape.  The curvature
is then fed back into the privacy side: higher curvature (more “flat” VRAM
distribution) allows a larger ε for differential‑privacy aggregation, while
low curvature tightens the privacy budget.  The endpoint circuit‑breaker
monitors successive failures of privacy‑budget checks and opens if a
threshold is exceeded.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


# Example tiers (mirroring parent A)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")


def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Probability that a record can be re‑identified (Parent A)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """
    Simple Laplace mechanism: sum(values) + Laplace(0, sensitivity/epsilon).
    (Parent A)
    """
    total = float(np.sum(list(values)))
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise


# ----------------------------------------------------------------------
# Endpoint circuit‑breaker (Parent A, truncated)
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.failure_threshold:
            self.open = True


# ----------------------------------------------------------------------
# VRAM planner & curvature (Parent B)
# ----------------------------------------------------------------------


class VramPlanner:
    """Tracks artefacts, their VRAM consumption and builds a weighted graph."""

    def __init__(self,
                 budget_mb: int = 4096,
                 reserve_mb: int = 768):
        self.budget_mb = budget_mb
        self.reserve_mb = reserve_mb
        self._artefacts: Dict[str, ModelTier] = {}
        self._graph: Dict[str, List[str]] = {}          # adjacency list
        self._mass: Dict[str, float] = {}               # w_i values

    # ------------------------------------------------------------------
    # Artefact registration
    # ------------------------------------------------------------------
    def register_artifact(self, name: str, tier: ModelTier) -> bool:
        """Attempt to register an artefact. Returns True on success."""
        projected = self.current_usage_mb() + tier.ram_mb
        if projected > self.budget_mb - self.reserve_mb:
            return False
        self._artefacts[name] = tier
        self._graph.setdefault(name, [])
        # connect to all existing nodes (fully‑connected for simplicity)
        for other in self._artefacts:
            if other != name:
                self._graph[name].append(other)
                self._graph[other].append(name)
        self._recompute_mass()
        return True

    def current_usage_mb(self) -> int:
        return sum(t.ram_mb for t in self._artefacts.values())

    # ------------------------------------------------------------------
    # Mass (VRAM weight) handling
    # ------------------------------------------------------------------
    def _recompute_mass(self) -> None:
        total = self.current_usage_mb()
        if total == 0:
            self._mass = {}
            return
        self._mass = {
            name: tier.ram_mb / total
            for name, tier in self._artefacts.items()
        }

    def mass_of(self, name: str) -> float:
        return self._mass.get(name, 0.0)

    # ------------------------------------------------------------------
    # Graph accessors
    # ------------------------------------------------------------------
    def neighbors(self, name: str) -> List[str]:
        return self._graph.get(name, [])

    def all_nodes(self) -> List[str]:
        return list(self._artefacts.keys())

    # ------------------------------------------------------------------
    # Curvature integration
    # ------------------------------------------------------------------
    def compute_edge_curvatures(self, alpha: float = 0.5) -> Dict[Tuple[str, str], float]:
        """Compute Ollivier‑Ricci curvature for each edge using weighted lazy walks."""
        curvatures: Dict[Tuple[str, str], float] = {}
        for i in self.all_nodes():
            for j in self.neighbors(i):
                if (j, i) in curvatures:   # avoid duplicate computation
                    continue
                curv = _ollivier_ricci_edge(self, i, j, alpha)
                curvatures[(i, j)] = curv
        return curvatures


# ----------------------------------------------------------------------
# Weighted lazy random‑walk distribution (bridge equation)
# ----------------------------------------------------------------------
def _lazy_walk_distribution(planner: VramPlanner,
                            node: str,
                            alpha: float = 0.5) -> Dict[str, float]:
    """
    μ_i(v) = α·w_i·δ_{i=v} + (1‑α)·w_i·(1/deg(i))·∑_{u∈N(i)} δ_{u=v}
    Returns a probability distribution over the node and its neighbours.
    """
    w_i = planner.mass_of(node)
    deg_i = max(1, len(planner.neighbors(node)))   # avoid division by zero
    dist: Dict[str, float] = {node: alpha * w_i}
    for nb in planner.neighbors(node):
        dist[nb] = dist.get(nb, 0.0) + (1 - alpha) * w_i / deg_i
    # renormalise (should already sum to w_i, but we need a proper prob.)
    total = sum(dist.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in dist.items()}


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature for a single edge
# ----------------------------------------------------------------------
def _wasserstein_1(dist_p: Dict[str, float],
                   dist_q: Dict[str, float],
                   planner: VramPlanner) -> float:
    """
    Simplified 1‑Wasserstein distance on the graph where each edge length = 1.
    Since the graph is unweighted and distances are 0 (same node) or 1 (different),
    the distance reduces to 1‑∑_v min(p(v), q(v)).
    """
    overlap = sum(min(dist_p.get(v, 0.0), dist_q.get(v, 0.0)) for v in set(dist_p) | set(dist_q))
    return 1.0 - overlap


def _ollivier_ricci_edge(planner: VramPlanner,
                         i: str,
                         j: str,
                         alpha: float = 0.5) -> float:
    """
    Curvature κ(i,j) = 1 - W1(μ_i, μ_j) / d(i,j)
    where d(i,j) = 1 for adjacent nodes.
    """
    mu_i = _lazy_walk_distribution(planner, i, alpha)
    mu_j = _lazy_walk_distribution(planner, j, alpha)
    w1 = _wasserstein_1(mu_i, mu_j, planner)
    return 1.0 - w1   # denominator d(i,j)=1


# ----------------------------------------------------------------------
# Hybrid operations (demonstrating the fused system)
# ----------------------------------------------------------------------


def hybrid_register(planner: VramPlanner,
                    name: str,
                    tier: ModelTier,
                    unique_qi: int,
                    total_records: int,
                    circuit: EndpointCircuitBreaker,
                    alpha: float = 0.5,
                    curvature_threshold: float = 0.2) -> bool:
    """
    Registers an artefact only if:
    1. VRAM budget permits (handled by VramPlanner).
    2. Reconstruction risk is below a dynamic ceiling derived from curvature.
       ε_max = 1 + curvature (higher curvature → larger privacy budget).
    3. The endpoint circuit‑breaker is not open.
    Returns True on successful registration.
    """
    if circuit.open:
        return False

    # Step 1 – VRAM check
    if not planner.register_artifact(name, tier):
        circuit.record_failure()
        return False

    # Step 2 – compute local curvature around the new node
    curvatures = planner.compute_edge_curvatures(alpha)
    # take the minimum curvature of incident edges as a conservative estimate
    incident = [(name, nb) for nb in planner.neighbors(name)]
    if incident:
        min_curv = min(curvatures.get(e, 0.0) for e in incident)
    else:
        min_curv = 0.0

    # Step 3 – privacy risk check
    risk = reconstruction_risk_score(unique_qi, total_records)
    epsilon_allowed = 1.0 + max(0.0, min_curv)   # ε grows with curvature
    if risk > epsilon_allowed:
        # privacy budget exceeded – rollback registration
        # (simple rollback: remove node and its edges)
        for nb in planner.neighbors(name):
            planner._graph[nb].remove(name)
        del planner._graph[name]
        del planner._artefacts[name]
        planner._recompute_mass()
        circuit.record_failure()
        return False

    # All checks passed
    circuit.record_success()
    return True


def dp_aggregate_with_curvature(values: Iterable[float],
                                planner: VramPlanner,
                                alpha: float = 0.5,
                                base_epsilon: float = 1.0) -> float:
    """
    Adjusts the Laplace scale by a factor derived from the average curvature
    of the current graph.  Higher curvature ⇒ lower noise (larger ε).

    ε_eff = base_epsilon * (1 + avg_curvature)
    """
    curv_dict = planner.compute_edge_curvatures(alpha)
    if curv_dict:
        avg_curv = sum(curv_dict.values()) / len(curv_dict)
    else:
        avg_curv = 0.0
    epsilon_eff = base_epsilon * (1.0 + avg_curv)
    return dp_aggregate(values, epsilon=epsilon_eff)


def monitor_endpoint(circuit: EndpointCircuitBreaker,
                     success: bool,
                     log_path: Path | str = "endpoint_log.json") -> None:
    """
    Records the circuit‑breaker state after each request and writes a tiny
    JSON log.  This function showcases the endpoint‑side of the hybrid system.
    """
    if success:
        circuit.record_success()
    else:
        circuit.record_failure()
    entry = {
        "timestamp": now_z(),
        "open": circuit.open,
        "failures": circuit.failures
    }
    path = Path(log_path)
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []
    data.append(entry)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    planner = VramPlanner(budget_mb=8192, reserve_mb=1024)
    circuit = EndpointCircuitBreaker(failure_threshold=2)

    # Register a few artefacts with varying privacy risk
    artefacts = [
        ("model_A", TIER_T1_QWEN_0_5B, 5, 1000),
        ("model_B", TIER_T2_REASONING, 20, 1000),
        ("model_C", TIER_T3_QWEN_7B, 200, 1000),   # high risk
    ]

    for name, tier, uq, total in artefacts:
        ok = hybrid_register(planner, name, tier, uq, total, circuit)
        monitor_endpoint(circuit, ok)
        print(f"Register {name}: {'OK' if ok else 'FAILED'}  | Circuit open: {circuit.open}")

    # Perform a DP aggregation on some dummy metrics
    metrics = [random.random() for _ in range(10)]
    agg = dp_aggregate_with_curvature(metrics, planner)
    print(f"DP‑aggregated value (curvature‑aware): {agg:.4f}")

    # Show a few curvature values
    curvs = planner.compute_edge_curvatures()
    print("Sample edge curvatures:")
    for (i, j), c in list(curvs.items())[:5]:
        print(f"  ({i}, {j}) -> {c:.3f}")

    # Final circuit state
    print(f"Final circuit state: open={circuit.open}, failures={circuit.failures}")