# DARWIN HAMMER — match 84, survivor 1
# gen: 3
# parent_a: bayes_claim_kernel.py (gen0)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s1.py (gen2)
# born: 2026-05-29T23:25:44Z

"""Hybrid algorithm combining Bayesian hypothesis updating (bayes_claim_kernel) with
FairyFuse ternary router minimum‑cost tree evaluation (hybrid_ternary_router_hybrid_minimum_cost).

Mathematical bridge:
Each edge in a tree is associated with a *reliability hypothesis* H.  The posterior
probability P(H|E) obtained from the Bayesian update (likelihood ratio → odds) is used
as a multiplicative confidence factor on the physical length of the edge.  The expected
material cost becomes Σ length(e)·P(H_e|E_e) and the path‑weight term is scaled by the
same confidences, yielding a single cost functional that fuses the two parent
topologies."""
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Minimal type definitions (stand‑ins for the original .types module)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathEvidence:
    """An observation that can be used to update an edge hypothesis."""
    id: str
    measurement: float  # e.g., observed length or signal strength
    noise_std: float    # standard deviation of measurement noise


@dataclass(frozen=True)
class MathHypothesis:
    """Bayesian hypothesis attached to a tree edge."""
    id: str
    prior: float                # prior probability that the edge is reliable
    posterior: float            # current posterior after evidence
    evidence_ids: Tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Parent A – Bayesian update (bayes_claim_kernel)
# ----------------------------------------------------------------------
def update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    likelihood_ratio: float,
) -> MathHypothesis:
    """Update posterior of a hypothesis using a likelihood ratio.

    The odds form is used to keep the operation numerically stable.
    """
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")

    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or likelihood_ratio == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)

    posterior = max(0.0, min(1.0, posterior))
    ids = tuple(list(hypothesis.evidence_ids) + [evidence.id])
    return MathHypothesis(
        id=hypothesis.id,
        prior=hypothesis.posterior,
        posterior=posterior,
        evidence_ids=ids,
    )


def gaussian_likelihood_ratio(
    evidence: MathEvidence,
    expected: float,
) -> float:
    """Compute a likelihood ratio assuming Gaussian noise.

    The ratio is  p(e|H) / p(e|¬H) where the alternative hypothesis (¬H)
    is modelled as a very broad uniform distribution over a wide interval.
    """
    # likelihood under H (Gaussian)
    var = evidence.noise_std ** 2
    gauss = math.exp(-0.5 * ((evidence.measurement - expected) ** 2) / var) / math.sqrt(2 * math.pi * var)

    # likelihood under ¬H (uniform over [0, 2*expected] – a very vague model)
    uniform_width = max(1e-6, 2 * expected)
    uniform = 1.0 / uniform_width

    return gauss / uniform


# ----------------------------------------------------------------------
# Parent B – Tree cost (ternary router) and corrected marginal
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_posteriors: Dict[Edge, float],
    path_weight: float = 0.2,
) -> float:
    """Hybrid cost: material cost weighted by edge posteriors plus path weight.

    material = Σ length(e)·P(H_e|E_e)
    path term = path_weight · Σ dist(v, root)·P(H_path(v)|E_path(v))
    where dist is the accumulated Euclidean distance from the root.
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Compute weighted material cost
    material = 0.0
    for a, b in edges:
        w = edge_posteriors.get((a, b), edge_posteriors.get((b, a), 1.0))
        material += w * length(nodes[a], nodes[b])

    # BFS/DFS to get distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge = (cur, nxt) if (cur, nxt) in edge_posteriors else (nxt, cur)
                w = edge_posteriors.get(edge, 1.0)
                # distance accumulates the *expected* length (posterior‑scaled)
                step = w * length(nodes[cur], nodes[nxt])
                dist[nxt] = dist[cur] + step
                stack.append(nxt)

    path_term = path_weight * sum(dist.values())
    return material + path_term


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Corrected marginal probability: P(E) = likelihood·prior + false_positive·(1‑prior)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_edge_hypotheses(
    edges: List[Edge],
    nodes: Dict[str, Point],
    evidences: List[MathEvidence],
    prior: float = 0.5,
) -> Dict[Edge, MathHypothesis]:
    """Create a hypothesis for every edge and update it with the matching evidence.

    The matching rule is simple: an evidence id must contain the concatenated
    node identifiers (e.g., \"A_B\") to be associated with edge (A, B).
    """
    hyps: Dict[Edge, MathHypothesis] = {}
    # initialise hypotheses with the given prior
    for a, b in edges:
        hyp_id = f"{a}_{b}"
        hyps[(a, b)] = MathHypothesis(id=hyp_id, prior=prior, posterior=prior, evidence_ids=())

    # associate evidences and perform Bayesian updates
    for ev in evidences:
        # find the edge(s) whose identifier is a substring of ev.id
        for edge, hyp in hyps.items():
            if edge[0] in ev.id and edge[1] in ev.id:
                expected_len = length(nodes[edge[0]], nodes[edge[1]])
                lr = gaussian_likelihood_ratio(ev, expected_len)
                hyps[edge] = update_hypothesis(hyp, ev, lr)

    return hyps


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    evidences: List[MathEvidence],
    prior: float = 0.5,
    path_weight: float = 0.2,
) -> float:
    """One‑stop function that builds edge posteriors from evidence and evaluates the
    hybrid cost."""
    hyps = compute_edge_hypotheses(edges, nodes, evidences, prior)
    edge_post = {edge: hyp.posterior for edge, hyp in hyps.items()}
    return tree_cost(nodes, edges, root, edge_post, path_weight)


def sample_random_tree(num_nodes: int = 5) -> Tuple[Dict[str, Point], List[Edge], str]:
    """Generate a small random tree (connected, acyclic) for testing."""
    points: Dict[str, Point] = {}
    for i in range(num_nodes):
        points[f"N{i}"] = (random.uniform(0, 10), random.uniform(0, 10))

    # create a random spanning tree using a simple Prim‑like process
    nodes = list(points.keys())
    root = nodes[0]
    visited = {root}
    edges: List[Edge] = []
    while len(visited) < len(nodes):
        a = random.choice(list(visited))
        b = random.choice([n for n in nodes if n not in visited])
        edges.append((a, b))
        visited.add(b)
    return points, edges, root


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    # Build a random tree
    pts, edgs, rt = sample_random_tree(6)

    # Create synthetic evidences: each edge gets a noisy measurement of its true length
    evs: List[MathEvidence] = []
    for a, b in edgs:
        true_len = length(pts[a], pts[b])
        noise = random.uniform(0.1, 0.5)
        meas = true_len + random.gauss(0, noise)
        evs.append(MathEvidence(id=f"{a}_{b}", measurement=meas, noise_std=noise))

    # Compute hybrid cost
    cost = hybrid_tree_cost(pts, edgs, rt, evs, prior=0.6, path_weight=0.3)
    print(f"Hybrid cost on random tree: {cost:.4f}")

    # Verify that posterior values are within [0,1] and that the cost is finite
    hyps = compute_edge_hypotheses(edgs, pts, evs, prior=0.6)
    for edge, hyp in hyps.items():
        assert 0.0 <= hyp.posterior <= 1.0, f"Posterior out of bounds for {edge}"
    assert math.isfinite(cost), "Cost should be a finite number"
    print("All checks passed.")