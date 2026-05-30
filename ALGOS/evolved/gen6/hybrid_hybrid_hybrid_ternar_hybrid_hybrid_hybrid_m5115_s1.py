# DARWIN HAMMER — match 5115, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_hybrid_m2532_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s3.py (gen4)
# born: 2026-05-29T23:59:50Z

"""
Hybrid algorithm combining:
- Parent A: FairyFuse ternary router with Bayesian edge uncertainty (hybrid_tree_cost, bayes_update).
- Parent B: Epistemic certainty based labeling with certainty flags.

Mathematical bridge:
The posterior probability of each edge (from the Bayesian update) is used to modulate 
the confidence of the certainty flags. This yields a unified system that respects 
both probabilistic evidence and epistemic certainty.

"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass

Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Geometry utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian primitives
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)"""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)"""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def edge_posterior(prior: float, likelihood: float, false_positive: float) -> float:
    """Convenience wrapper that returns the posterior probability for an edge."""
    m = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, m)

# ----------------------------------------------------------------------
# Epistemic certainty helpers
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")


    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: List[str] = [],
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def modulate_certainty(
    posterior: float, 
    certainty_flag: CertaintyFlag
) -> CertaintyFlag:
    """
    Modulate the certainty of a certainty flag 
    based on the posterior probability of an edge.

    Args:
    posterior (float): The posterior probability of an edge.
    certainty_flag (CertaintyFlag): The certainty flag to be modulated.

    Returns:
    CertaintyFlag: The modulated certainty flag.
    """
    modulated_confidence_bps = int(certainty_flag.confidence_bps * posterior)
    return CertaintyFlag(
        label=certainty_flag.label,
        confidence_bps=modulated_confidence_bps,
        authority_class=certainty_flag.authority_class,
        rationale=certainty_flag.rationale,
        evidence_refs=certainty_flag.evidence_refs,
    )


def hybrid_edge_cost(
    prior: float, 
    likelihood: float, 
    false_positive: float, 
    certainty_flag: CertaintyFlag
) -> Tuple[float, CertaintyFlag]:
    """
    Calculate the cost of an edge based on its posterior probability 
    and the certainty of its associated certainty flag.

    Args:
    prior (float): The prior probability of an edge.
    likelihood (float): The likelihood of an edge.
    false_positive (float): The false positive rate of an edge.
    certainty_flag (CertaintyFlag): The certainty flag associated with the edge.

    Returns:
    Tuple[float, CertaintyFlag]: A tuple containing the cost of the edge 
    and its modulated certainty flag.
    """
    posterior = edge_posterior(prior, likelihood, false_positive)
    modulated_certainty_flag = modulate_certainty(posterior, certainty_flag)
    edge_cost = 1 / (1 + posterior)
    return edge_cost, modulated_certainty_flag


def hybrid_route_cost(
    edges: List[Edge], 
    points: List[Point], 
    prior: float, 
    likelihood: float, 
    false_positive: float, 
    certainty_flags: Dict[Edge, CertaintyFlag]
) -> Tuple[float, Dict[Edge, CertaintyFlag]]:
    """
    Calculate the cost of a route based on the costs of its edges 
    and their associated certainty flags.

    Args:
    edges (List[Edge]): The edges in the route.
    points (List[Point]): The points in the route.
    prior (float): The prior probability of an edge.
    likelihood (float): The likelihood of an edge.
    false_positive (float): The false positive rate of an edge.
    certainty_flags (Dict[Edge, CertaintyFlag]): The certainty flags 
    associated with the edges.

    Returns:
    Tuple[float, Dict[Edge, CertaintyFlag]]: A tuple containing the 
    cost of the route and its modulated certainty flags.
    """
    route_cost = 0.0
    modulated_certainty_flags = {}
    for edge in edges:
        point_a, point_b = points[edge[0]], points[edge[1]]
        edge_length = length(point_a, point_b)
        edge_cost, modulated_certainty_flag = hybrid_edge_cost(
            prior, 
            likelihood, 
            false_positive, 
            certainty_flags[edge]
        )
        route_cost += edge_length * edge_cost
        modulated_certainty_flags[edge] = modulated_certainty_flag
    return route_cost, modulated_certainty_flags


if __name__ == "__main__":
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    edges = [("0", "1"), ("1", "2")]
    prior = 0.5
    likelihood = 0.8
    false_positive = 0.2

    certainty_flags = {
        ("0", "1"): certainty("PROBABLE", confidence_bps=5000, authority_class="expert", rationale="expert opinion"),
        ("1", "2"): certainty("POSSIBLE", confidence_bps=2000, authority_class="layman", rationale="personal experience"),
    }

    route_cost, modulated_certainty_flags = hybrid_route_cost(
        edges, 
        points, 
        prior, 
        likelihood, 
        false_positive, 
        certainty_flags
    )
    print(route_cost)
    for edge, certainty_flag in modulated_certainty_flags.items():
        print(f"Edge {edge}: {certainty_flag.as_dict()}")