# DARWIN HAMMER — match 7, survivor 0
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_decision_hygiene_shannon_entropy_m12_s1.py (gen1)
# born: 2026-05-29T23:22:20Z

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

"""
This module integrates the hybrid_minimum_cost_tree_bayes_update_m6_s2 and 
hybrid_decision_hygiene_shannon_entropy_m12_s1 algorithms into a single hybrid system.
The bridge between the two structures is the concept of information entropy applied to 
the decision hygiene scoring system, and the expected cost of the minimum-cost tree 
computed using Bayesian update.
"""

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P(H|E) = p·L / P(E)
    """
    if np.any(marginal <= 0):
        raise ValueError("All marginal probabilities must be > 0")
    return prior * likelihood / marginal

def bayes_edge_posteriors(
    prior_dict: Dict[Tuple[str, str], float],
    likelihood_dict: Dict[Tuple[str, str], float],
    false_positive: float,
) -> Dict[Tuple[str, str], float]:
    """
    Compute posterior probability for each edge using Bayesian update.

    Parameters
    ----------
    prior_dict, likelihood_dict : dict mapping Edge → probability in [0,1]
    false_positive : scalar false‑positive rate (FP) in [0,1]

    Returns
    -------
    posterior_dict : dict mapping Edge → posterior probability
    """
    edges = list(prior_dict.keys())
    priors = np.array([prior_dict[e] for e in edges], dtype=float)
    likes = np.array([likelihood_dict[e] for e in edges], dtype=float)

    marginal = bayes_marginal(priors, likes, false_positive)
    post = bayes_update(priors, likes, marginal)

    return {e: float(p) for e, p in zip(edges, post)}

def counts(text: str) -> Dict[str, int]:
    EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
    PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
    DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
    SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
    BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
    OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
    IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
    SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
    RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }

def shannon_entropy(observations: List[int | float], is_distribution: bool = False) -> float:
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(xs)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def score_features(c: Dict[str, int]) -> Tuple[int, str]:
    positive = (
        c["evidence_count"] * 1600
        + c["planning_count"] * 1200
        + c["delay_count"] * 1400
        + c["support_count"] * 1000
        + c["boundary_count"] * 1200
        + c["outcome_count"] * 800
    )
    negative = c["impulsive_count"] * 1500 + c["scarcity_count"] * 700 + c["risk_count"] * 1200
    score = max(-10000, min(10000, positive - negative))
    if c["risk_count"] and score < 2500:
        label = "critical_risk_or_pain_signal"
    elif score >= 7000:
        label = "high_decision_hygiene"
    elif score >= 3000:
        label = "improving_decision_hygiene"
    elif score <= -2500:
        label = "strained_decision_context"
    else:
        label = "neutral_or_unclear"
    return score, label

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float,
    edge_priors: Dict[Tuple[str, str], float],
    edge_likelihoods: Dict[Tuple[str, str], float],
    false_positive: float,
    text: str,
) -> float:
    # 1. Geometry
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # 2. Bayesian edge posteriors
    posteriors = bayes_edge_posteriors(edge_priors, edge_likelihoods, false_positive)

    # 3. Decision hygiene features
    feature_counts = counts(text)
    decision_hygiene_score, label = score_features(feature_counts)

    # 4. Assemble the expected material and path components
    expected_cost = sum(posteriors[e] * edge_len[e] for e in edges) + path_weight * sum(dist[v] for v in nodes)

    return expected_cost

def hybrid_score(text: str) -> Tuple[float, int, str]:
    feature_counts = counts(text)
    decision_hygiene_score, label = score_features(feature_counts)
    shannon_entropy_score = shannon_entropy(list(feature_counts.values()))
    return shannon_entropy_score, decision_hygiene_score, label

if __name__ == "__main__":
    text = "This is a test text with some decision hygiene features."
    nodes = {
        "A": (0.0, 0.0),
        "B": (3.0, 4.0),
        "C": (6.0, 8.0),
    }
    edges = [
        ("A", "B"),
        ("B", "C"),
    ]
    root = "A"
    path_weight = 1.0
    edge_priors = {
        ("A", "B"): 0.8,
        ("B", "C"): 0.7,
    }
    edge_likelihoods = {
        ("A", "B"): 0.9,
        ("B", "C"): 0.8,
    }
    false_positive = 0.1
    print(hybrid_tree_cost(nodes, edges, root, path_weight, edge_priors, edge_likelihoods, false_positive, text))
    print(hybrid_score(text))