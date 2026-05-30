# DARWIN HAMMER — match 48, survivor 2
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: epistemic_certainty.py (gen0)
# born: 2026-05-29T23:23:58Z

"""Hybrid Epistemic‑Bayesian Minimum‑Cost Tree.

Parents:
- `hybrid_minimum_cost_tree_bayes_update_m6_s0.py` – builds a tree cost where each edge
  weight is updated by a Bayesian rule using a prior, a likelihood and a false‑positive term.
- `epistemic_certainty.py` – defines `CertaintyFlag` objects that carry a confidence
  (basis points) together with epistemic metadata.

Mathematical bridge:
Each `CertaintyFlag` expresses a confidence in the interval [0, 1] via
`p = confidence_bps / 10_000`.  We treat this probability as the *prior* for a node
and as the *likelihood* for an edge.  The Bayesian marginal and update equations
from the tree module are then applied, and the resulting posterior is finally
scaled by the same confidence factor, yielding a unified weight that reflects both
physical distance and epistemic certainty.

The three core functions below demonstrate the hybrid operation:
1. `confidence_to_probability` – maps a `CertaintyFlag` to a probability.
2. `hybrid_tree_cost_with_certainty` – computes the total cost of a tree where
   every edge weight incorporates Bayesian updating *and* epistemic confidence.
3. `aggregate_tree_certainty` – produces a single scalar summarising the overall
   epistemic‑Bayesian certainty of the whole tree (product of posteriors)."""

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

        # Scale the posterior again by the edge confidence (acts like a second‑order weight)
        posterior_scaled = posterior * confidence_to_probability(edge_flags[(a, b)])

        bayes_weights[(a, b)] = posterior_scaled
        material += length(nodes[a], nodes[b]) * posterior_scaled

    # Depth‑first accumulation of root‑to‑node distances using the hybrid weights
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # Edge direction matters for the prior; we use the stored weight irrespective of direction
                w = bayes_weights.get((cur, nxt)) or bayes_weights.get((nxt, cur), 1.0)
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt]) * w
                stack.append(nxt)

    return material + path_weight * sum(dist.values())


def aggregate_tree_certainty(
    node_flags: Dict[str, CertaintyFlag],
    edge_flags: Dict[Edge, CertaintyFlag],
) -> float:
    """
    Produce a single scalar representing the overall epistemic‑Bayesian certainty of the tree.
    The metric is the product of all node priors and edge posteriors (after a single Bayesian update
    assuming a uniform false‑positive rate of 0.05).  The product is kept in log‑space to avoid underflow.
    """
    log_product = 0.0
    false_pos = 0.05

    for node, flag in node_flags.items():
        p = confidence_to_probability(flag)
        # Clamp to avoid log(0)
        p = max(p, 1e-12)
        log_product += math.log(p)

    for edge, flag in edge_flags.items():
        prior = confidence_to_probability(flag)  # use edge flag as a proxy prior
        likelihood = confidence_to_probability(flag)
        marginal = bayes_marginal(prior, likelihood, false_pos)
        posterior = bayes_update(prior, likelihood, marginal)
        posterior = max(posterior, 1e-12)
        log_product += math.log(posterior)

    return math.exp(log_product)


def generate_certainty_tree(num_nodes: int, extra_edge_factor: float = 1.5) -> Tuple[
    Dict[str, Point],
    List[Edge],
    str,
    Dict[str, CertaintyFlag],
    Dict[Edge, CertaintyFlag],
]:
    """
    Randomly generate a connected tree (plus a few extra edges) together with
    epistemic CertaintyFlag objects for every node and edge.

    Returns
    -------
    nodes, edges, root, node_flags, edge_flags
    """
    # Nodes with random coordinates
    nodes = {f"node_{i}": (random.uniform(0, 10), random.uniform(0, 10)) for i in range(num_nodes)}
    node_names = list(nodes.keys())

    # Ensure connectivity by building a random spanning tree first
    edges: List[Edge] = []
    shuffled = node_names[:]
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        a = shuffled[i]
        b = random.choice(shuffled[:i])
        edges.append((a, b))

    # Add extra random edges
    extra_edges = int(num_nodes * extra_edge_factor) - len(edges)
    possible = [(a, b) for a in node_names for b in node_names if a != b]
    while extra_edges > 0:
        a, b = random.choice(possible)
        if (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
            extra_edges -= 1

    root = random.choice(node_names)

    # Create CertaintyFlag for each node (filesystem observation)
    node_flags = {
        name: filesystem_observation(sha256=f"nodehash_{i}", path=f"/tmp/{name}.bin")
        for i, name in enumerate(node_names)
    }

    # Create CertaintyFlag for each edge (parser extraction, with occasional injection)
    edge_flags = {}
    for a, b in edges:
        sha = f"edgehash_{a}_{b}"
        inject = random.random() < 0.1  # 10 % chance of injection flag
        edge_flags[(a, b)] = parser_extraction(sha256=sha, extract_method="auto", injection_detected=inject)

    return nodes, edges, root, node_flags, edge_flags


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    nodes, edges, root, node_flags, edge_flags = generate_certainty_tree(num_nodes=12)

    cost = hybrid_tree_cost_with_certainty(
        nodes,
        edges,
        root,
        node_flags,
        edge_flags,
        false_positive_map=None,
        path_weight=0.2,
    )
    overall_certainty = aggregate_tree_certainty(node_flags, edge_flags)

    print(f"Hybrid tree cost: {cost:.4f}")
    print(f"Overall epistemic‑Bayesian certainty (product): {overall_certainty:.6g}")