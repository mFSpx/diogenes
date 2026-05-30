# DARWIN HAMMER — match 910, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s2.py (gen3)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s2.py (gen2)
# born: 2026-05-29T23:31:44Z

"""Hybrid Algorithm: Morphological Labeling + Bayesian Minimum‑Cost Routing

Parent A: ``hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s2.py`` – provides
weak‑supervision labeling functions, morphological feature extraction and a
KAN‑style universal approximator that maps a morphology vector to a confidence
score.

Parent B: ``hybrid_ternary_router_hybrid_minimum_cost__m36_s2.py`` – defines a
ternary router whose engine‑channel selection is driven by a Bayesian‑updated
minimum‑cost tree.  Edge priors are updated with evidence and the expected
cost of each channel is computed as  

    C(ch) = Σ_{e∈path(ch)} base_cost(e)·(1‑p(e|evidence))

Mathematical Bridge
-------------------
Both parents manipulate probability‑like quantities:

* The KAN approximator of Parent A yields a confidence *c∈[0,1]* for a given
  document.  This confidence can be interpreted as a prior probability that
  the document belongs to the correct label class.
* The routing tree of Parent B requires priors *p(e)* on edges.  We lift the
  confidence *c* to an edge prior by assigning it to all edges that belong to
  the routing path suggested by the labeling function.

Thus a unified system can be built:  
1. Extract a morphology vector from the input document.  
2. Pass it through the KAN approximator → confidence *c*.  
3. Use *c* to initialise / update edge priors in the routing tree.  
4. Compute the Bayesian‑updated expected cost for each engine channel.  
5. Select the channel with minimal expected cost and emit the final label.

The code below implements this fusion, providing three core functions that
demonstrate the hybrid operation and a smoke‑test that runs end‑to‑end.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared between both parent logics)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an endpoint/document."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # ∈ [0,1]


@dataclass(frozen=True)
class EdgePrior:
    """Prior probability associated with an edge in the routing tree."""
    edge: Tuple[str, str]  # (parent, child)
    prior: float           # ∈ (0,1)


# ----------------------------------------------------------------------
# Parent‑A core: morphological feature → KAN‑style confidence
# ----------------------------------------------------------------------


def morphology_vector(morph: Morphology) -> np.ndarray:
    """
    Convert a Morphology instance into a normalized feature vector.
    The vector is L2‑normalised to keep the KAN mapping scale‑invariant.
    """
    vec = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Morphology vector cannot be all zeros")
    return vec / norm


def kan_approximation(vec: np.ndarray, weight: np.ndarray | None = None) -> float:
    """
    Very light‑weight KAN surrogate.
    The original KAN uses spline‑based activation; we approximate with
    a single hidden unit and a smooth exponential activation:

        y = σ(w·x + b)   where σ(z)=exp(z)/(1+exp(z))

    The function returns a confidence in [0,1].
    """
    if weight is None:
        # Initialise a deterministic weight vector for reproducibility.
        rng = np.random.default_rng(42)
        weight = rng.normal(loc=0.0, scale=1.0, size=vec.shape)
    bias = 0.1
    z = float(np.dot(weight, vec) + bias)
    # Logistic‑style activation bounded in (0,1)
    confidence = math.exp(z) / (1.0 + math.exp(z))
    # Clamp for numerical safety
    return max(0.0, min(1.0, confidence))


def labeling_function(name: str | None = None) -> Callable[[Callable[[dict], int]], Callable[[dict], int]]:
    """
    Decorator used by the original Parent A to tag labeling functions.
    Here it simply records the name.
    """
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


@labeling_function("simple_length_based")
def lf_length_based(doc: dict) -> int:
    """Trivial labeling: 1 if length > width else 0."""
    return int(doc.get("length", 0) > doc.get("width", 0))


# ----------------------------------------------------------------------
# Parent‑B core: Bayesian minimum‑cost routing
# ----------------------------------------------------------------------


class BayesianRouter:
    """
    Maintains a directed graph of engine channels, edge priors and base costs.
    Provides methods to compute expected costs and to update priors with evidence.
    """

    def __init__(self, channels: List[str]):
        self.channels = channels
        self.graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.edge_priors: Dict[Tuple[str, str], float] = {}
        self._build_fully_connected_graph()

    def _build_fully_connected_graph(self):
        """Create a complete directed graph among channels with unit base cost."""
        for src in self.channels:
            for dst in self.channels:
                if src == dst:
                    continue
                edge = (src, dst)
                self.graph[src].append((dst, 1.0))          # base cost = 1.0
                self.edge_priors[edge] = 0.5                # uninformed prior

    def expected_cost(self, target: str) -> float:
        """
        Compute the expected cost of reaching *target* from a virtual root.
        For simplicity we treat the root as connecting to every channel with
        base cost 1.0 and use the current priors to discount the cost:

            C = Σ_{e∈path} base_cost(e)·(1‑p(e))

        Because the graph is fully connected we take the cheapest single‑edge
        path (root → target).
        """
        # root→target edge is not stored; we treat its prior as the average of
        # outgoing priors from the target node.
        outgoing = [self.edge_priors[(target, nxt)] for nxt, _ in self.graph[target]]
        avg_prior = sum(outgoing) / len(outgoing) if outgoing else 0.5
        cost = 1.0 * (1.0 - avg_prior)  # base cost of root→target is 1.0
        return cost

    def update_priors(self, evidence: Dict[Tuple[str, str], bool]) -> None:
        """
        Perform a Bayesian update for each edge given binary evidence.
        evidence[edge] = True  → observation supports the edge,
        evidence[edge] = False → observation contradicts the edge.

        Prior p, likelihood L(True|edge)=0.9, L(False|edge)=0.1 (tunable).
        Posterior ∝ prior·likelihood.
        """
        lik_pos = 0.9
        lik_neg = 0.1
        for edge, obs in evidence.items():
            prior = self.edge_priors.get(edge, 0.5)
            likelihood = lik_pos if obs else lik_neg
            numer = prior * likelihood
            denom = numer + (1 - prior) * (1 - likelihood)
            if denom == 0:
                posterior = prior  # avoid division by zero
            else:
                posterior = numer / denom
            self.edge_priors[edge] = posterior


def hybrid_route_packet(
    packet: dict,
    router: BayesianRouter,
    confidence: float,
) -> Tuple[str, float]:
    """
    Fuse labeling confidence with routing cost.
    The confidence is injected as evidence for edges outgoing from the
    candidate channel: higher confidence → higher prior that the edge is
    “good”.  After updating priors, the channel with minimal expected cost
    is selected.

    Returns:
        (selected_channel, expected_cost)
    """
    # Build evidence: for each edge (c, nxt) we assert that the edge is good
    # with probability equal to confidence.
    evidence: Dict[Tuple[str, str], bool] = {}
    for src in router.channels:
        for dst, _ in router.graph[src]:
            # Stochastic evidence: treat confidence > 0.5 as True, else False.
            # In a real system we would weight by confidence; here we keep it binary.
            evidence[(src, dst)] = confidence > 0.5

    router.update_priors(evidence)

    # Score each channel
    scores = {ch: router.expected_cost(ch) for ch in router.channels}
    best_channel = min(scores, key=scores.get)
    return best_channel, scores[best_channel]


# ----------------------------------------------------------------------
# Hybrid operation combining both parents
# ----------------------------------------------------------------------


def hybrid_label_and_route(
    doc: dict,
    packet: dict,
    router: BayesianRouter,
    weight: np.ndarray | None = None,
) -> Tuple[ProbabilisticLabel, str, float]:
    """
    End‑to‑end hybrid pipeline:

    1. Extract morphology from *doc* and compute a normalized vector.
    2. Pass the vector through the KAN surrogate → confidence *c*.
    3. Produce a probabilistic label using *c*.
    4. Feed *c* into the Bayesian router to obtain the best engine channel
       and its expected cost.

    Returns:
        (ProbabilisticLabel, selected_channel, expected_cost)
    """
    # 1. Morphology extraction (fallback defaults if fields missing)
    morph = Morphology(
        length=float(doc.get("length", 1.0)),
        width=float(doc.get("width", 1.0)),
        height=float(doc.get("height", 1.0)),
        mass=float(doc.get("mass", 1.0)),
    )
    vec = morphology_vector(morph)

    # 2. KAN confidence
    confidence = kan_approximation(vec, weight)

    # 3. Deterministic label via a simple labeling function
    label = lf_length_based(doc)
    prob_label = ProbabilisticLabel(
        doc_id=str(doc.get("doc_id", "unknown")),
        label=label,
        confidence=confidence,
    )

    # 4. Routing
    selected_channel, expected_cost = hybrid_route_packet(packet, router, confidence)

    return prob_label, selected_channel, expected_cost


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny set of engine channels
    channels = ["alpha", "beta", "gamma"]
    router = BayesianRouter(channels)

    # Dummy document and packet
    document = {
        "doc_id": "doc_001",
        "length": 12.3,
        "width": 7.4,
        "height": 2.1,
        "mass": 0.8,
    }

    packet = {
        "payload": "example request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Run the hybrid pipeline
    prob_label, best_chan, cost = hybrid_label_and_route(document, packet, router)

    print(f"Probabilistic label: {prob_label}")
    print(f"Selected engine channel: {best_chan}")
    print(f"Expected routing cost: {cost:.4f}")

    # Verify that the router's priors have been updated (sanity check)
    print("\nUpdated edge priors (sample):")
    sample_edges = list(router.edge_priors.items())[:5]
    for edge, prior in sample_edges:
        print(f"  Edge {edge}: prior={prior:.3f}")

    sys.exit(0)