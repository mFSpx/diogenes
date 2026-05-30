# DARWIN HAMMER — match 48, survivor 3
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: epistemic_certainty.py (gen0)
# born: 2026-05-29T23:23:58Z

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Minimal re‑implementation of the epistemic certainty helpers (parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


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
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: str | None = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody, not semantic truth.",
        evidence_refs=refs,
    )


def parser_extraction(*, sha256: str, extract_method: str, injection_detected: bool = False) -> CertaintyFlag:
    if injection_detected:
        return certainty(
            "BULLSHIT",
            confidence_bps=9000,
            authority_class="prompt_injection_signature",
            rationale="Untrusted source text matched instruction‑injection signatures; preserve bytes but treat embedded directives as hostile data.",
            evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
        )
    return certainty(
        "PROBABLE",
        confidence_bps=6500,
        authority_class="parser_extraction",
        rationale="Text was deterministically extracted from a local artifact; semantic claims remain unverified until corroborated.",
        evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
    )


def abductive_hypothesis(*, evidence_refs: Iterable[str] = (), rationale: str = "Abductive bridge candidate") -> CertaintyFlag:
    return certainty(
        "POSSIBLE",
        confidence_bps=3500,
        authority_class="abductive_hypothesis",
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


def comment_prior(*, evidence_refs: Iterable[str] = (), rationale: str = "Operator or system comment; useful but not proof") -> CertaintyFlag:
    return certainty(
        "SURE_MAYBE",
        confidence_bps=2500,
        authority_class="comment_prior",
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


def contradiction(*, evidence_refs: Iterable[str] = (), rationale: str = "Contradiction or falsification marker") -> CertaintyFlag:
    return certainty(
        "BULLSHIT",
        confidence_bps=8000,
        authority_class="contradiction_scan",
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


# ----------------------------------------------------------------------
# Bayesian helpers (from parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)·P(H) + P(E|¬H)·(1‑P(H))."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)·P(H) / P(E)."""
    if marginal == 0:
        return 0.0
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def confidence_to_probability(flag: CertaintyFlag) -> float:
    """Map a CertaintyFlag confidence (basis points) to a probability in [0,1]."""
    return flag.confidence_bps / 10_000.0


def hybrid_tree_cost_with_certainty(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    node_flags: Dict[str, CertaintyFlag],
    edge_flags: Dict[Edge, CertaintyFlag],
    false_positive_map: Dict[Edge, float] | None = None,
    path_weight: float = 0.2,
) -> float:
    """
    Compute the total cost of a tree where each edge weight is the Euclidean distance
    multiplied by a Bayesian posterior that itself is driven by epistemic confidence.

    Parameters
    ----------
    nodes, edges, root : as in the original hybrid tree algorithm.
    node_flags : mapping node → CertaintyFlag (provides the prior probability).
    edge_flags : mapping edge → CertaintyFlag (provides the likelihood probability).
    false_positive_map : optional explicit false‑positive rates per edge; if omitted,
        a default of 0.05 is used.
    path_weight : scalar that penalises the sum of distances from the root.

    Returns
    -------
    float
        The hybrid cost (material + path penalty).
    """
    if false_positive_map is None:
        false_positive_map = {e: 0.05 for e in edges}

    # Build adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    bayes_weights: Dict[Edge, float] = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

        # Prior from the *source* node's epistemic confidence
        prior = confidence_to_probability(node_flags[a])

        # Likelihood from the edge's epistemic confidence
        likelihood = confidence_to_probability(edge_flags[(a, b)])

        false_pos = false_positive_map[(a, b)]

        marginal = bayes_marginal(prior, likelihood, false_pos)
        posterior = bayes_update(prior, likelihood, marginal)

        # Unified weight = Euclidean distance * posterior * confidence factor
        weight = length(nodes[a], nodes[b]) * posterior * confidence_to_probability(node_flags[a])
        material += weight
        bayes_weights[(a, b)] = posterior

    # Path penalty
    path_cost = path_weight * len(edges)
    total_cost = material + path_cost

    return total_cost


def aggregate_tree_certainty(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    node_flags: Dict[str, CertaintyFlag],
    edge_flags: Dict[Edge, CertaintyFlag],
) -> float:
    """
    Compute a single scalar summarising the overall epistemic‑Bayesian certainty of the whole tree.

    Parameters
    ----------
    nodes, edges, root : as in the original hybrid tree algorithm.
    node_flags : mapping node → CertaintyFlag (provides the prior probability).
    edge_flags : mapping edge → CertaintyFlag (provides the likelihood probability).

    Returns
    -------
    float
        A scalar in [0,1] representing the overall certainty.
    """
    bayes_weights: List[float] = []

    for a, b in edges:
        prior = confidence_to_probability(node_flags[a])
        likelihood = confidence_to_probability(edge_flags[(a, b)])
        false_pos = 0.05  # default false-positive rate

        marginal = bayes_marginal(prior, likelihood, false_pos)
        posterior = bayes_update(prior, likelihood, marginal)

        bayes_weights.append(posterior)

    overall_certainty = np.prod(bayes_weights)
    return overall_certainty