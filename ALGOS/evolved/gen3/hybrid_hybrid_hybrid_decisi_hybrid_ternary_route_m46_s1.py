# DARWIN HAMMER — match 46, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_rete_bandit_gate_m28_s1.py (gen2)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s0.py (gen2)
# born: 2026-05-29T23:25:27Z

"""
Hybrid Algorithm: hybrid_hybrid_decision_hygi_rete_bandit_gate_m28_s1 + hybrid_ternary_router_hybrid_minimum_cost__m36_s0

Mathematical Bridge
-------------------
Parent A provides a Shannon entropy computation over categorical evidence extracted
from free‑form text using regular expressions.  Parent B builds a minimum‑cost
spanning‑tree where each edge carries a *prior* probability that is later updated
with Bayesian evidence.

The fusion treats the entropy **H** of the extracted evidence as a global uncertainty
measure and maps it to a set of edge priors **πₑ** :

    πₑ = exp( -H ) / Σₑ' exp( -H )   (uniformly scaled by the same H)

Thus higher entropy (more uncertainty) yields lower edge priors, increasing the
expected material cost of the tree.  These priors are then used in the Bayesian
update equations from Parent B to refine routing decisions in a ternary router
style function.

The resulting system integrates:
* Regex‑based evidence extraction (Parent A)
* Shannon entropy **H** (Parent A)
* Edge‑weighted minimum‑cost tree **C** (Parent B)
* Bayesian marginal and update (Parent B)

Three core functions demonstrate the hybrid operation:
1. `extract_evidence_features`
2. `entropy_weighted_edge_priors`
3. `hybrid_tree_cost` (with Bayesian updates via `bayes_update`)
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np
import re
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – evidence extraction & Shannon entropy
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)

CATEGORY_REGEXES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
}


def extract_evidence_features(text: str) -> Dict[str, int]:
    """
    Count occurrences of each evidence category in *text*.
    Returns a dict mapping category name → count.
    """
    counts: Dict[str, int] = {}
    for name, regex in CATEGORY_REGEXES.items():
        matches = regex.findall(text)
        counts[name] = len(matches)
    return counts


def shannon_entropy(counts: Dict[str, int]) -> float:
    """
    Compute Shannon entropy H = - Σ p_i log2(p_i) where p_i are normalized counts.
    If total count is zero, entropy is defined as 0.0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    # filter zero probabilities to avoid log2(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# ----------------------------------------------------------------------
# Parent B – tree cost, Bayesian update, ternary‑router style routing
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = likelihood * prior + false_positive * (1‑prior)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior = prior * likelihood / marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def entropy_weighted_edge_priors(
    edges: List[Edge], entropy: float, base_prior: float = 0.5
) -> Dict[Edge, float]:
    """
    Produce a prior for each edge based on global entropy.
    The formula is:

        πₑ = base_prior * exp( -entropy ) / Z

    where Z normalises the distribution over edges.
    """
    if entropy < 0.0:
        raise ValueError("Entropy cannot be negative")
    unnorm = np.array([math.exp(-entropy) for _ in edges])
    Z = unnorm.sum()
    if Z == 0:
        # fallback to uniform base prior
        return {e: base_prior for e in edges}
    normalized = (base_prior * unnorm) / Z
    return {e: float(p) for e, p in zip(edges, normalized)}


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    path_weight: float = 0.2,
) -> float:
    """
    Minimum‑cost tree where each edge length is scaled by its prior probability.
    The cost is:

        C = Σ_e  length(e) * πₑ   +   path_weight * Σ_{v} dist(root, v)

    where dist is the shortest‑path distance in the *unweighted* graph.
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        prior = edge_priors.get((a, b), edge_priors.get((b, a), 0.5))
        material += length(nodes[a], nodes[b]) * prior

    # BFS/DFS to compute distances from root (unweighted)
    dist: Dict[str, float] = {root: 0.0}
    stack: List[str] = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1  # unit weight for hop count
                stack.append(nb)

    return material + path_weight * sum(dist.values())


def ternary_router(
    current: str,
    candidates: List[str],
    edge_priors: Dict[Edge, float],
    entropy: float,
) -> str:
    """
    Choose among up to three candidate nodes based on a weighted score:

        score(v) = -log(π_{(current, v)}) * (1 + entropy)

    The node with the smallest score is selected (i.e., highest prior, low uncertainty).
    """
    if not candidates:
        raise ValueError("Candidate list cannot be empty")
    if len(candidates) > 3:
        candidates = candidates[:3]  # enforce ternary limit

    best = candidates[0]
    best_score = math.inf
    for cand in candidates:
        prior = edge_priors.get((current, cand), edge_priors.get((cand, current), 0.5))
        # protect against zero prior
        prior = max(prior, 1e-9)
        score = -math.log(prior) * (1.0 + entropy)
        if score < best_score:
            best = cand
            best_score = score
    return best


def bayesian_edge_update(
    edge_priors: Dict[Edge, float],
    evidence_counts: Dict[str, int],
    likelihood_map: Dict[str, float],
    false_positive: float = 0.01,
) -> Dict[Edge, float]:
    """
    Update each edge prior using Bayesian inference.
    For simplicity we treat each category count as independent evidence
    and derive a *likelihood* per edge by averaging the category likelihoods
    weighted by the normalized counts.
    """
    total = sum(evidence_counts.values())
    if total == 0:
        return edge_priors.copy()  # no evidence to update

    # Normalised evidence contribution per category
    cat_weights = {cat: cnt / total for cat, cnt in evidence_counts.items()}

    updated = {}
    for edge, prior in edge_priors.items():
        # Composite likelihood for this edge
        composite_likelihood = sum(
            cat_weights.get(cat, 0.0) * likelihood_map.get(cat, 0.5)
            for cat in CATEGORY_REGEXES.keys()
        )
        marginal = bayes_marginal(prior, composite_likelihood, false_positive)
        posterior = bayes_update(prior, composite_likelihood, marginal)
        updated[edge] = posterior
    return updated


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text containing a mix of categories
    sample_text = """
    The evidence was verified and the source hash sha256 matches.
    We need to plan the next steps and create a checklist.
    However, we must wait until tomorrow before we can proceed.
    If needed, ask a friend or a lawyer for support.
    It's important to respect boundaries and keep the data safe.
    Finally, the task is completed and verified.
    """

    # 1. Extract categorical evidence and compute entropy
    features = extract_evidence_features(sample_text)
    H = shannon_entropy(features)
    print("Feature counts:", features)
    print("Shannon entropy (bits):", H)

    # 2. Define a tiny graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]

    # 3. Compute entropy‑weighted priors
    priors = entropy_weighted_edge_priors(edges, H, base_prior=0.6)
    print("Initial edge priors:", priors)

    # 4. Compute hybrid tree cost
    cost_before = hybrid_tree_cost(nodes, edges, root="A", edge_priors=priors)
    print("Hybrid tree cost before Bayesian update:", cost_before)

    # 5. Simulate a ternary routing decision from node A
    candidates = ["B", "C", "D"]
    next_node = ternary_router("A", candidates, priors, H)
    print("Ternary router selects:", next_node)

    # 6. Bayesian update of priors using the extracted evidence
    # Define a simple likelihood map per category (higher for evidence, lower for delay)
    likelihood_map = {
        "evidence": 0.9,
        "planning": 0.7,
        "delay": 0.2,
        "support": 0.6,
        "boundary": 0.4,
        "outcome": 0.8,
    }
    updated_priors = bayesian_edge_update(priors, features, likelihood_map)
    print("Updated edge priors after Bayesian evidence:", updated_priors)

    # 7. Re‑compute cost with updated priors
    cost_after = hybrid_tree_cost(nodes, edges, root="A", edge_priors=updated_priors)
    print("Hybrid tree cost after Bayesian update:", cost_after)

    # Ensure the script runs without exception
    sys.exit(0)