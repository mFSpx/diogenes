# DARWIN HAMMER — match 1378, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py (gen3)
# born: 2026-05-29T23:35:43Z

"""
Hybrid algorithm combining the principles of 
hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py and 
hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py.

The mathematical bridge between these two systems is established by 
incorporating the epistemic certainty flags into the edge weights of 
the symmetric cost matrix, and using the ternary routing step to 
select an intermediate node that minimises the cost.

The core idea is to use the epistemic certainty flags to modify 
the path weights in the tree scoring function, and use the ternary 
routing step to evaluate the hygiene score and Shannon entropy of 
each candidate.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Iterable, List, Tuple

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def _shingles(text: str, width: int = 5) -> List[str]:
    """Generate overlapping substrings (shingles) of given width."""
    return [text[i : i + width] for i in range(len(text) - width + 1)]

def minhash_signature(text: str, k: int = 64, width: int = 5) -> List[int]:
    """
    Very small minhash implementation.
    Returns the k smallest hash values of the shingles.
    """
    if not text:
        return [0] * k
    sh = _shingles(text.lower(), width)
    # deterministic hash: use built‑in hash mixed with a fixed seed
    hashes = [hash(s) & 0xFFFFFFFFFFFFFFFF for s in sh]
    hashes.sort()
    # pad if fewer than k shingles
    return (hashes[:k] + [0] * k)[:k]

def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of the character distribution (up to 10 000 chars)."""
    if not text:
        return 0.0
    text = text[:10000]
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def hybrid_cost_matrix(texts: List[str], epistemic_certainties: List[float]) -> np.ndarray:
    """
    Compute the symmetric cost matrix with epistemic certainty flags.

    Args:
    texts (List[str]): List of input texts.
    epistemic_certainties (List[float]): List of epistemic certainty flags.

    Returns:
    np.ndarray: Symmetric cost matrix.
    """
    num_texts = len(texts)
    cost_matrix = np.zeros((num_texts, num_texts))
    for i in range(num_texts):
        for j in range(i+1, num_texts):
            vector_i = np.concatenate((minhash_signature(texts[i]), [shannon_entropy(texts[i])]))
            vector_j = np.concatenate((minhash_signature(texts[j]), [shannon_entropy(texts[j])]))
            distance = np.linalg.norm(vector_i - vector_j)
            cost_matrix[i, j] = distance * (1 - epistemic_certainties[i]) * (1 - epistemic_certainties[j])
            cost_matrix[j, i] = cost_matrix[i, j]
    return cost_matrix

def ternary_routing(cost_matrix: np.ndarray, source: int, destination: int) -> int:
    """
    Perform ternary routing to select an intermediate node.

    Args:
    cost_matrix (np.ndarray): Symmetric cost matrix.
    source (int): Source node index.
    destination (int): Destination node index.

    Returns:
    int: Index of the intermediate node.
    """
    num_nodes = cost_matrix.shape[0]
    min_cost = float('inf')
    intermediate_node = -1
    for k in range(num_nodes):
        cost = cost_matrix[source, k] + cost_matrix[k, destination]
        if cost < min_cost:
            min_cost = cost
            intermediate_node = k
    return intermediate_node

def hybrid_operation(texts: List[str], epistemic_certainties: List[float]) -> Tuple[np.ndarray, int]:
    """
    Perform the hybrid operation.

    Args:
    texts (List[str]): List of input texts.
    epistemic_certainties (List[float]): List of epistemic certainty flags.

    Returns:
    Tuple[np.ndarray, int]: Symmetric cost matrix and index of the intermediate node.
    """
    cost_matrix = hybrid_cost_matrix(texts, epistemic_certainties)
    source = 0
    destination = len(texts) - 1
    intermediate_node = ternary_routing(cost_matrix, source, destination)
    return cost_matrix, intermediate_node

if __name__ == "__main__":
    texts = ["This is a sample text.", "This is another sample text."]
    epistemic_certainties = [0.9, 0.8]
    cost_matrix, intermediate_node = hybrid_operation(texts, epistemic_certainties)
    print("Cost Matrix:")
    print(cost_matrix)
    print("Intermediate Node:", intermediate_node)