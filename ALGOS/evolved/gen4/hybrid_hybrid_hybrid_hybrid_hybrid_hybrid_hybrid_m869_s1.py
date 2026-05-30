# DARWIN HAMMER — match 869, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s2.py (gen3)
# born: 2026-05-29T23:31:17Z

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Types from Parent A
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Types from Parent B
@dataclass(frozen=True)
class BurstSignal:
    count: int
    timestamp: datetime

# ----------------------------------------------------------------------
# Module Docstring
"""
This module fuses the mathematical structures of the hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s1 and hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s2 algorithms.
The mathematical bridge between these two algorithms lies in the use of Bayesian marginalisation and update, and matrix operations. In hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s1,
the edge weights are updated using Bayesian marginalisation, while in hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s2, the weight matrix W is updated recurrently using gradient descent.
This fusion module integrates these two concepts by using the Bayesian marginalisation as a pre-processing step to initialise the weight matrix W in the hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s2 algorithm.
"""

# ----------------------------------------------------------------------
# Types
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

# ----------------------------------------------------------------------
# Functions

def compute_node_priors(event_sessions: Dict[str, List[BurstSignal]]) -> Dict[str, float]:
    """
    Builds node priors from temporal motifs.

    Args:
    event_sessions: A dictionary where keys are node IDs and values are lists of BurstSignal objects.

    Returns:
    A dictionary where keys are node IDs and values are their corresponding priors.
    """
    node_priors = {}
    for node, sessions in event_sessions.items():
        motif_counts = Counter()
        for session in sessions:
            motif_counts[session.count] += 1
        most_frequent_motif = motif_counts.most_common(1)[0][0]
        total_motif_support = sum(motif_counts.values())
        node_priors[node] = most_frequent_motif / total_motif_support
    return node_priors

def hybrid_edge_weight(node_priors: Dict[str, float], edge_likelihood: float, false_positive_rate: float) -> float:
    """
    Computes the hybrid edge weight.

    Args:
    node_priors: A dictionary where keys are node IDs and values are their corresponding priors.
    edge_likelihood: The likelihood of the edge.
    false_positive_rate: The false positive rate of the edge.

    Returns:
    The hybrid edge weight.
    """
    node_prior = node_priors[edge[0]]
    posterior = edge_likelihood * node_prior + false_positive_rate * (1 - node_prior)
    motif_similarity = jaccard_similarity(node_priors[edge[0]], node_priors[edge[1]])
    return posterior * motif_similarity

def hybrid_tree_cost(node_priors: Dict[str, float], edge_likelihoods: Dict[str, float], false_positive_rates: Dict[str, float]) -> float:
    """
    Evaluates the full cost of a rooted tree using the hybrid weights.

    Args:
    node_priors: A dictionary where keys are node IDs and values are their corresponding priors.
    edge_likelihoods: A dictionary where keys are edge IDs and values are their corresponding likelihoods.
    false_positive_rates: A dictionary where keys are edge IDs and values are their corresponding false positive rates.

    Returns:
    The total cost of the tree.
    """
    total_cost = 0
    for edge, likelihood in edge_likelihoods.items():
        false_positive_rate = false_positive_rates[edge]
        hybrid_weight = hybrid_edge_weight(node_priors, likelihood, false_positive_rate)
        total_cost += hybrid_weight
    return total_cost

def jaccard_similarity(x: float, y: float) -> float:
    """
    Computes the Jaccard similarity between two sets.

    Args:
    x: The first set.
    y: The second set.

    Returns:
    The Jaccard similarity between the two sets.
    """
    intersection = min(x, y)
    union = max(x, y)
    return intersection / union

# ----------------------------------------------------------------------
# Smoke Test

if __name__ == "__main__":
    event_sessions = {
        "A": [BurstSignal(1, datetime(2022, 1, 1, tzinfo=timezone.utc))],
        "B": [BurstSignal(2, datetime(2022, 1, 2, tzinfo=timezone.utc))],
        "C": [BurstSignal(3, datetime(2022, 1, 3, tzinfo=timezone.utc))]
    }
    node_priors = compute_node_priors(event_sessions)
    edge_likelihoods = {"AB": 0.5, "BC": 0.7}
    false_positive_rates = {"AB": 0.2, "BC": 0.3}
    print(hybrid_tree_cost(node_priors, edge_likelihoods, false_positive_rates))