# DARWIN HAMMER — match 910, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s2.py (gen3)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s2.py (gen2)
# born: 2026-05-29T23:31:44Z

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

    Args:
    packet (dict): Input document
    router (BayesianRouter): Bayesian router instance
    confidence (float): Labeling confidence in [0,1]

    Returns:
    Tuple[str, float]: Selected channel and its expected cost
    """
    # Map confidence to edge prior boost
    prior_boost = 0.9 * confidence + 0.1  # ∈ [0.1, 1.0]

    # Evidence for edges outgoing from candidate channel
    evidence = {}
    for ch in router.channels:
        for nxt, _ in router.graph[ch]:
            edge = (ch, nxt)
            evidence[edge] = prior_boost if ch == lf_length_based(packet) else False

    # Update edge priors with evidence
    router.update_priors(evidence)

    # Select channel with minimal expected cost
    costs = {ch: router.expected_cost(ch) for ch in router.channels}
    best_channel = min(costs, key=costs.get)

    return best_channel, costs[best_channel]


def smoke_test_hybrid():
    # Create a Bayesian router
    channels = ["chan1", "chan2", "chan3"]
    router = BayesianRouter(channels)

    # Create a packet
    packet = {"length": 10, "width": 5}

    # Compute labeling confidence
    morph = Morphology(length=10, width=5, height=2, mass=1)
    vec = morphology_vector(morph)
    confidence = kan_approximation(vec)

    # Route packet
    best_channel, cost = hybrid_route_packet(packet, router, confidence)
    print(f"Best channel: {best_channel}, Cost: {cost}")


if __name__ == "__main__":
    smoke_test_hybrid()