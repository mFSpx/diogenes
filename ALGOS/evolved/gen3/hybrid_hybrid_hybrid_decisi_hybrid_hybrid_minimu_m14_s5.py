# DARWIN HAMMER — match 14, survivor 5
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0.py (gen2)
# born: 2026-05-29T23:25:17Z

import math
import numpy as np
from typing import Dict, List, Tuple

# Helper mathematics
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# Feature extraction
FEATURE_REGEXES = [
    ("evidence", r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b"),
    ("planning", r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b"),
    ("delay", r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|b)\b"),
    ("quality", r"\b(?:quality|high|low|grade|rating)\b"),
    ("security", r"\b(?:security|secure|vulnerability|exploit)\b"),
    ("performance", r"\b(?:performance|fast|slow|latency)\b"),
    ("compliance", r"\b(?:compliance|regulation|standard)\b"),
    ("cost", r"\b(?:cost|price|budget|expense)\b"),
    ("generic", r"\b\w{7,}\b"),
]

def extract_features(text: str) -> np.ndarray:
    import re
    counts = []
    for _, pattern in FEATURE_REGEXES:
        matches = re.findall(pattern, text, re.I)
        counts.append(len(matches))
    return np.array(counts, dtype=int)

# Epistemic certainty flags
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
FLAG_CERTAINTY = {
    "FACT": 0.95,
    "PROBABLE": 0.75,
    "POSSIBLE": 0.50,
    "SURE_MAYBE": 0.30,
    "BULLSHIT": 0.05,
}

# Hybrid decision-hygiene score
def hybrid_hygiene_score(v: np.ndarray, w_pos: np.ndarray, w_neg: np.ndarray) -> float:
    if v.shape != (9,) or w_pos.shape != (9,) or w_neg.shape != (9,):
        raise ValueError("All vectors must be of length 9")
    s = float(w_pos @ v - w_neg @ v)
    total = v.sum()
    if total == 0:
        H = 0.0
    else:
        p = v / total
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log2(p[mask])))
    H_max = math.log2(9)
    hybrid = s * (1.0 + H / H_max)
    return hybrid

# Edge effective weight
def edge_effective_weight(
    node_i: str,
    node_j: str,
    coords: Dict[str, Tuple[float, float]],
    hybrid_scores: Dict[str, float],
    flag: str,
    epsilon: float = 1e-9,
) -> float:
    d = length(coords[node_i], coords[node_j])
    Si = hybrid_scores.get(node_i, 0.0)
    Sj = hybrid_scores.get(node_j, 0.0)
    prior = Si / (Si + Sj + epsilon)
    certainty = FLAG_CERTAINTY.get(flag.upper(), 0.0)
    likelihood = 1.0 - certainty
    fp = certainty * 0.1
    marginal = bayes_marginal(prior, likelihood, fp)
    weight = d * (1.0 - marginal) + epsilon
    return weight

# Build epistemic tree
def build_epistemic_tree(
    nodes: Dict[str, Tuple[float, float]],
    texts: Dict[str, str],
    edge_flags: Dict[Tuple[str, str], str],
    w_pos: np.ndarray,
    w_neg: np.ndarray,
) -> Tuple[List[Tuple[str, str]], float]:
    import networkx as nx
    G = nx.Graph()
    hybrid_scores = {}
    for node, text in texts.items():
        v = extract_features(text)
        hybrid_scores[node] = hybrid_hygiene_score(v, w_pos, w_neg)
    for (node_i, node_j), flag in edge_flags.items():
        weight = edge_effective_weight(node_i, node_j, nodes, hybrid_scores, flag)
        G.add_edge(node_i, node_j, weight=weight)
    mst = nx.minimum_spanning_tree(G)
    edges_in_tree = list(mst.edges)
    total_weight = sum(mst[u][v]['weight'] for u, v in edges_in_tree)
    return edges_in_tree, total_weight

# Example usage
if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 2),
    }
    texts = {
        'A': 'This is a sample text with evidence and planning.',
        'B': 'This text has delay and quality.',
        'C': 'This text has security and performance.',
    }
    edge_flags = {
        ('A', 'B'): 'FACT',
        ('B', 'C'): 'PROBABLE',
        ('A', 'C'): 'POSSIBLE',
    }
    w_pos = np.array([1.0] * 9)
    w_neg = np.array([-1.0] * 9)
    edges_in_tree, total_weight = build_epistemic_tree(nodes, texts, edge_flags, w_pos, w_neg)
    print(edges_in_tree, total_weight)