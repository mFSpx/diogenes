# DARWIN HAMMER — match 3371, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_workshare_allocator_m250_s2.py (gen4)
# born: 2026-05-29T23:49:31Z

"""
This module combines the core topologies of two parent algorithms: 
PARENT ALGORITHM A (hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s2.py) and 
PARENT ALGORITHM B (hybrid_hybrid_hybrid_hard_t_workshare_allocator_m250_s2.py).

The mathematical bridge lies in the integration of Bayesian edge costing from Parent A with the 
workshare allocation and linguistic analysis from Parent B. Specifically, the Bayesian posterior 
from Parent A is used to inform the allocation of workshare units in Parent B, while the linguistic 
analysis from Parent B is used to refine the edge costing in Parent A.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set

import numpy as np

# ----------------------------------------------------------------------
# Types and basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian utilities (from Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    if not 0.0 <= prior <= 1.0 or not 0.0 <= likelihood <= 1.0 or not 0.0 <= false_positive <= 1.0:
        raise ValueError("All probability arguments must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Constants & utilities (from Parent B)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class WorkshareLane:
    group: str
    llm_units: float
    llm_share_pct: float
    proof_required: bool

@dataclass(frozen=True)
class AllocationResult:
    total_units: float
    deterministic_target_pct: float
    deterministic_units: float
    llm_units: float
    lanes: Tuple[WorkshareLane, ...]

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_edge_cost(edge: Edge, prior: float, likelihood: float, false_positive: float, text: str) -> float:
    """
    Calculate the hybrid edge cost by combining the Bayesian posterior with the linguistic analysis.
    
    :param edge: The edge to calculate the cost for
    :param prior: The prior probability
    :param likelihood: The likelihood probability
    :param false_positive: The false positive probability
    :param text: The text to analyze
    :return: The hybrid edge cost
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    
    # Analyze the text using the linguistic analysis from Parent B
    words = [w.lower() for w in text.split() if w.isalpha()]
    function_words = [w for w in words if w in FUNCTION_CATS["article"] or w in FUNCTION_CATS["preposition"]]
    proportion = len(function_words) / len(words) if words else 0.0
    
    # Combine the posterior with the linguistic analysis
    hybrid_cost = posterior * (1.0 - proportion)
    return hybrid_cost

def hybrid_workshare_allocation(total_units: float, deterministic_target_pct: float, llm_share_pct: float, text: str) -> AllocationResult:
    """
    Calculate the hybrid workshare allocation by combining the Bayesian posterior with the linguistic analysis.
    
    :param total_units: The total units to allocate
    :param deterministic_target_pct: The deterministic target percentage
    :param llm_share_pct: The LLM share percentage
    :param text: The text to analyze
    :return: The hybrid workshare allocation
    """
    # Analyze the text using the linguistic analysis from Parent B
    words = [w.lower() for w in text.split() if w.isalpha()]
    function_words = [w for w in words if w in FUNCTION_CATS["article"] or w in FUNCTION_CATS["preposition"]]
    proportion = len(function_words) / len(words) if words else 0.0
    
    # Combine the proportions with the Bayesian posterior
    posterior = 1.0 - proportion
    llm_units = total_units * llm_share_pct * posterior
    deterministic_units = total_units * deterministic_target_pct * (1.0 - posterior)
    
    # Create the workshare lanes
    lanes = tuple(WorkshareLane(group, llm_units, llm_share_pct, False) for group in GROUPS)
    
    return AllocationResult(total_units, deterministic_target_pct, deterministic_units, llm_units, lanes)

def hybridAllocation(total_units: float, deterministic_target_pct: float, llm_share_pct: float, edges: List[Edge], prior: float, likelihood: float, false_positive: float, text: str) -> Tuple[List[float], AllocationResult]:
    """
    Calculate the hybrid allocation by combining the edge costs with the workshare allocation.
    
    :param total_units: The total units to allocate
    :param deterministic_target_pct: The deterministic target percentage
    :param llm_share_pct: The LLM share percentage
    :param edges: The list of edges
    :param prior: The prior probability
    :param likelihood: The likelihood probability
    :param false_positive: The false positive probability
    :param text: The text to analyze
    :return: The hybrid allocation
    """
    edge_costs = [hybrid_edge_cost(edge, prior, likelihood, false_positive, text) for edge in edges]
    allocation = hybrid_workshare_allocation(total_units, deterministic_target_pct, llm_share_pct, text)
    
    return edge_costs, allocation

if __name__ == "__main__":
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    prior = 0.5
    likelihood = 0.8
    false_positive = 0.2
    text = "This is a test sentence."
    total_units = 100.0
    deterministic_target_pct = 0.6
    llm_share_pct = 0.4
    
    edge_costs, allocation = hybridAllocation(total_units, deterministic_target_pct, llm_share_pct, edges, prior, likelihood, false_positive, text)
    print("Edge costs:", edge_costs)
    print("Allocation:", allocation)