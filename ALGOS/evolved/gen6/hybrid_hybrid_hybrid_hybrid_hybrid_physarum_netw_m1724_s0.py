# DARWIN HAMMER — match 1724, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s1.py (gen3)
# parent_b: hybrid_physarum_network_hybrid_hybrid_hdc_hy_m683_s4.py (gen5)
# born: 2026-05-29T23:38:31Z

"""Hybrid Algorithm combining Morphology‑Shapley analysis (Parent A) with
Physarum‑inspired conductance dynamics and hyperdimensional vector operations
(Parent B).

Mathematical bridge:
- Each physical entity is described by a ``Morphology`` (Parent A).  Its
  sphericity index is interpreted as a scalar “pressure” node in a
  Physarum‑type flow network (Parent B).
- Feature importance for a morphology is obtained via exact Shapley values.
  These values weight the conductance of edges incident to the node, thus
  coupling game‑theoretic attribution to the transport dynamics.
- Nodes are also encoded as binary hypervectors (binding/bundling from
  Parent B).  Edge vectors are formed by binding the endpoint vectors; a
  circuit‑breaker (Parent A) disables edges whose conductance repeatedly
  falls below a tolerance, modelling failure‑aware flow.

The module provides three core hybrid functions:
1. ``morphology_to_hypervector`` – encodes a Morphology into a hypervector.
2. ``shapley_weighted_conductance`` – computes conductances scaled by Shapley
   attributions.
3. ``physarum_shapley_step`` – performs one Physarum update step using the
   pressures derived from sphericity and the Shapley‑scaled conductances,
   while respecting circuit‑breaker states.

All components are pure Python with only ``numpy``, ``math``, ``random``,
``sys``, ``pathlib`` and standard library imports.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations, chain
from typing import Any, Callable, Dict, List, Sequence, Tuple, Set, FrozenSet

import numpy as np

# ---------- Parent A core ----------

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def exact_shapley_value(
    value_fn: Callable[[FrozenSet[int]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    """Exact generic Shapley value by enumerating every coalition.

    Parameters
    ----------
    value_fn : Callable[[frozenset[int]], float]
        Function that maps a set of feature indices to a scalar worth.
    feature_index : int
        Index of the feature whose Shapley value is requested.
    feature_count : int
        Total number of features.

    Returns
    -------
    float
        Shapley value for the specified feature.
    """
    if not (0 <= feature_index < feature_count):
        raise ValueError("feature_index out of range")

    total = 0.0
    all_features = set(range(feature_count))
    for r in range(feature_count):
        for subset in combinations(all_features - {feature_index}, r):
            subset_frozen = frozenset(subset)
            weight = shapley_kernel_weight(r, feature_count)
            marginal = value_fn(subset_frozen | {feature_index}) - value_fn(
                subset_frozen
            )
            total += weight * marginal
    return total


# ---------- Parent B core ----------

Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    import hashlib

    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if x >= 0 else -1 for x in sums]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> float:
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(
    conductance: float,
    q: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    if dt < 0 or decay < 0:
        raise ValueError("dt and decay must be non‑negative")
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ---------- Hybrid Layer ----------

EdgeKey = Tuple[int, int]  # (node_a, node_b)


@dataclass
class HybridNode:
    """Container that couples morphology, hypervector and circuit‑breaker."""
    morph: Morphology
    vector: Vector
    breaker: EndpointCircuitBreaker

    @property
    def pressure(self) -> float:
        """Pressure derived from sphericity; higher sphericity → higher pressure."""
        return sphericity_index(
            self.morph.length, self.morph.width, self.morph.height
        )


def morphology_to_hypervector(morph: Morphology, dim: int = 4096) -> Vector:
    """
    Encode a Morphology into a binary hypervector by hashing each attribute
    and binding the resulting symbol vectors.
    """
    # Create a deterministic symbol for each attribute using its numeric value.
    length_sym = f"len:{morph.length:.4f}"
    width_sym = f"wid:{morph.width:.4f}"
    height_sym = f"hei:{morph.height:.4f}"
    mass_sym = f"mas:{morph.mass:.4f}"

    vec_len = symbol_vector(length_sym, dim)
    vec_wid = symbol_vector(width_sym, dim)
    vec_hei = symbol_vector(height_sym, dim)
    vec_mas = symbol_vector(mass_sym, dim)

    # Bind all attribute vectors together to obtain a single representation.
    bound = bind(vec_len, vec_wid)
    bound = bind(bound, vec_hei)
    bound = bind(bound, vec_mas)
    return bound


def shapley_weighted_conductance(
    node_a: HybridNode,
    node_b: HybridNode,
    base_conductance: float = 1.0,
) -> float:
    """
    Compute a conductance value for an edge (a, b) where the base conductance
    is scaled by the product of Shapley importance of the *mass* feature for
    each endpoint.  Mass is chosen because it is a scalar that directly
    influences physical flow.
    """
    # Define a simple value function: pressure * mass.
    def value_fn(features: FrozenSet[int]) -> float:
        # Feature ordering: 0‑length, 1‑width, 2‑height, 3‑mass
        selected = [0, 0, 0, 0]
        for idx in features:
            if idx == 0:
                selected[0] = node_a.morph.length
            elif idx == 1:
                selected[1] = node_a.morph.width
            elif idx == 2:
                selected[2] = node_a.morph.height
            elif idx == 3:
                selected[3] = node_a.morph.mass
        # Simple linear worth: sum of selected dimensions.
        return sum(selected)

    # Shapley value for the mass feature (index 3) of node_a.
    shap_a = exact_shapley_value(value_fn, feature_index=3, feature_count=4)

    # Repeat for node_b (swap morphologies inside value_fn).
    def value_fn_b(features: FrozenSet[int]) -> float:
        selected = [0, 0, 0, 0]
        for idx in features:
            if idx == 0:
                selected[0] = node_b.morph.length
            elif idx == 1:
                selected[1] = node_b.morph.width
            elif idx == 2:
                selected[2] = node_b.morph.height
            elif idx == 3:
                selected[3] = node_b.morph.mass
        return sum(selected)

    shap_b = exact_shapley_value(value_fn_b, feature_index=3, feature_count=4)

    # Scale conductance; ensure positivity.
    scale = max(0.0, shap_a) * max(0.0, shap_b)
    return base_conductance * (1.0 + scale)


def physarum_shapley_step(
    nodes: Dict[int, HybridNode],
    edges: Dict[EdgeKey, float],
    edge_lengths: Dict[EdgeKey, float],
    dt: float = 1.0,
) -> Tuple[Dict[EdgeKey, float], Dict[EdgeKey, bool]]:
    """
    Perform a single Physarum‑style update on all edges.

    Returns
    -------
    new_conductances : dict
        Updated conductance per edge.
    edge_active : dict
        Boolean flag indicating whether the edge is still usable (circuit
        breaker not opened).
    """
    new_conductances: Dict[EdgeKey, float] = {}
    edge_active: Dict[EdgeKey, bool] = {}

    # Pre‑compute pressures for all nodes.
    pressures = {nid: node.pressure for nid, node in nodes.items()}

    for (a, b), conductance in edges.items():
        # Skip if either endpoint has its breaker opened.
        if not (nodes[a].breaker.allow() and nodes[b].breaker.allow()):
            edge_active[(a, b)] = False
            new_conductances[(a, b)] = 0.0
            continue

        # Compute Shapley‑scaled conductance.
        base_cond = shapley_weighted_conductance(nodes[a], nodes[b], conductance)

        # Flux based on pressures.
        q = flux(
            conductance=base_cond,
            edge_length=edge_lengths[(a, b)],
            pressure_a=pressures[a],
            pressure_b=pressures[b],
        )

        # Update conductance.
        updated = update_conductance(base_cond, q, dt=dt)

        # Register failure if conductance falls below a tiny threshold.
        if updated < 1e-6:
            nodes[a].breaker.record_failure()
            nodes[b].breaker.record_failure()
        else:
            nodes[a].breaker.record_success()
            nodes[b].breaker.record_success()

        edge_active[(a, b)] = nodes[a].breaker.allow() and nodes[b].breaker.allow()
        new_conductances[(a, b)] = updated if edge_active[(a, b)] else 0.0

    return new_conductances, edge_active


# ---------- Demonstration Functions ----------

def build_demo_network(num_nodes: int = 5, dim: int = 4096) -> Tuple[
    Dict[int, HybridNode],
    Dict[EdgeKey, float],
    Dict[EdgeKey, float],
]:
    """
    Construct a small fully‑connected network of HybridNode objects with
    random morphologies and initial conductances.
    """
    rng = random.Random(42)

    nodes: Dict[int, HybridNode] = {}
    for i in range(num_nodes):
        morph = Morphology(
            length=rng.uniform(0.5, 2.0),
            width=rng.uniform(0.5, 2.0),
            height=rng.uniform(0.5, 2.0),
            mass=rng.uniform(1.0, 10.0),
        )
        vec = morphology_to_hypervector(morph, dim=dim)
        breaker = EndpointCircuitBreaker(failure_threshold=2)
        nodes[i] = HybridNode(morph=morph, vector=vec, breaker=breaker)

    edges: Dict[EdgeKey, float] = {}
    edge_lengths: Dict[EdgeKey, float] = {}
    for a, b in combinations(range(num_nodes), 2):
        # Euclidean distance between the two hypervectors as a proxy for edge length.
        length = euclidean(nodes[a].vector, nodes[b].vector)
        edge_lengths[(a, b)] = max(0.1, length)  # avoid zero length
        edges[(a, b)] = rng.uniform(0.5, 1.5)   # initial conductance

    return nodes, edges, edge_lengths


def run_hybrid_simulation(steps: int = 10) -> None:
    """
    Execute a short simulation, printing conductance statistics at each step.
    """
    nodes, edges, edge_lengths = build_demo_network()
    for step in range(steps):
        edges, active = physarum_shapley_step(nodes, edges, edge_lengths, dt=0.5)
        avg_cond = np.mean(list(edges.values()))
        alive = sum(active.values())
        print(
            f"Step {step+1:02d}: avg conductance = {avg_cond:.4f}, active edges = {alive}"
        )


def bundled_network_vector(nodes: Dict[int, HybridNode]) -> Vector:
    """
    Produce a single hypervector representing the whole network by bundling
    all node vectors.
    """
    return bundle([node.vector for node in nodes.values()])


# ---------- Smoke Test ----------

if __name__ == "__main__":
    print("=== Hybrid Physarum‑Shapley Demo ===")
    run_hybrid_simulation(steps=5)
    # Demonstrate bundling of node vectors.
    demo_nodes, _, _ = build_demo_network(num_nodes=3)
    net_vec = bundled_network_vector(demo_nodes)
    print(f"Bundled network hypervector (first 20 bits): {net_vec[:20]}")