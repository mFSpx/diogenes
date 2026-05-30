# DARWIN HAMMER — match 84, survivor 2
# gen: 3
# parent_a: bayes_claim_kernel.py (gen0)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s1.py (gen2)
# born: 2026-05-29T23:25:44Z

import math
import random
import numpy as np
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple

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
    return replace(hypothesis, posterior=posterior, evidence_ids=ids)

def gaussian_likelihood_ratio(
    evidence: MathEvidence,
    expected: float,
) -> float:
    """Compute a likelihood ratio assuming Gaussian noise.

    The ratio is  p(e|H) / p(e|¬H) where the alternative hypothesis (¬H)
    is modelled as a very broad uniform distribution over a wide interval.
    """
    var = evidence.noise_std ** 2
    gauss = np.exp(-0.5 * ((evidence.measurement - expected) ** 2) / var) / np.sqrt(2 * np.pi * var)

    uniform_width = max(1e-6, 2 * expected)
    uniform = 1.0 / uniform_width

    return gauss / uniform

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    edge_posteriors: Dict[Tuple[str, str], float],
    path_weight: float = 0.2,
) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    material = 0.0
    for a, b in edges:
        w = edge_posteriors.get((a, b), edge_posteriors.get((b, a), 1.0))
        material += w * length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge = (cur, nxt) if (cur, nxt) in edge_posteriors else (nxt, cur)
                w = edge_posteriors.get(edge, 1.0)
                step = w * length(nodes[cur], nodes[nxt])
                dist[nxt] = dist[cur] + step
                stack.append(nxt)

    path_term = path_weight * sum(dist.values())
    return material + path_term

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def compute_edge_hypotheses(
    edges: List[Tuple[str, str]],
    nodes: Dict[str, Tuple[float, float]],
    evidences: List[MathEvidence],
    prior: float = 0.5,
) -> Dict[Tuple[str, str], MathHypothesis]:
    hyps: Dict[Tuple[str, str], MathHypothesis] = {}
    for a, b in edges:
        hyp_id = f"{a}_{b}"
        hyps[(a, b)] = MathHypothesis(id=hyp_id, prior=prior, posterior=prior, evidence_ids=())

    for ev in evidences:
        for edge, hyp in hyps.items():
            if edge[0] in ev.id and edge[1] in ev.id:
                expected_len = length(nodes[edge[0]], nodes[edge[1]])
                lr = gaussian_likelihood_ratio(ev, expected_len)
                hyps[edge] = update_hypothesis(hyp, ev, lr)

    return hyps

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    evidences: List[MathEvidence],
    prior: float = 0.5,
    path_weight: float = 0.2,
) -> float:
    hyps = compute_edge_hypotheses(edges, nodes, evidences, prior)
    edge_post = {edge: hyp.posterior for edge, hyp in hyps.items()}
    return tree_cost(nodes, edges, root, edge_post, path_weight)

def sample_random_tree(num_nodes: int = 5) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]], str]:
    points: Dict[str, Tuple[float, float]] = {}
    for i in range(num_nodes):
        points[f"N{i}"] = (random.uniform(0, 10), random.uniform(0, 10))

    nodes = list(points.keys())
    root = nodes[0]
    visited = {root}
    edges: List[Tuple[str, str]] = []
    while len(visited) < len(nodes):
        a = random.choice(list(visited))
        b = random.choice([n for n in nodes if n not in visited])
        edges.append((a, b))
        visited.add(b)

    return points, edges, root

# Test
if __name__ == "__main__":
    nodes, edges, root = sample_random_tree()
    evidences = [MathEvidence(f"A_B", 5.0, 0.5), MathEvidence(f"B_C", 6.0, 0.6)]
    cost = hybrid_tree_cost(nodes, edges, root, evidences)
    print(cost)