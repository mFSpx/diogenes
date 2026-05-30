# DARWIN HAMMER — match 1378, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py (gen3)
# born: 2026-05-29T23:35:43Z

"""
Hybrid algorithm combining the principles of 
`hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py` and 
`hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py`.

The mathematical bridge between these two systems is established by 
incorporating the epistemic certainty flags into the edge weights of 
the symmetric cost matrix C, and using the ternary routing step to 
select an intermediate node that minimises the cost function 
modified by the Bayesian update rule.

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
from typing import List, Tuple

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
    hashes = [hash(s) & 0xFFFFFFFFFFFFFFFF for s in sh]
    hashes.sort()
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

def hybrid_cost_matrix(texts: List[str], 
                      prior: float, 
                      likelihood: float, 
                      false_positive: float) -> np.ndarray:
    """
    Compute the hybrid cost matrix C that combines the 
    symmetric cost matrix from the ternary routing step 
    with the epistemic certainty flags.
    """
    num_texts = len(texts)
    C = np.zeros((num_texts, num_texts))
    for i in range(num_texts):
        for j in range(i+1, num_texts):
            v_i = np.array(minhash_signature(texts[i]) + [shannon_entropy(texts[i])])
            v_j = np.array(minhash_signature(texts[j]) + [shannon_entropy(texts[j])])
            dist = np.linalg.norm(v_i - v_j)**2
            marginal = bayes_marginal(prior, likelihood, false_positive)
            updated_cost = dist * bayes_update(prior, likelihood, marginal)
            C[i, j] = updated_cost
            C[j, i] = updated_cost
    return C

def ternary_routing(C: np.ndarray, source: int, destination: int) -> int:
    """
    Select an intermediate node that minimises the cost function 
    C_{source,k}+C_{k,destination}.
    """
    num_nodes = C.shape[0]
    min_cost = float('inf')
    best_node = -1
    for k in range(num_nodes):
        if k != source and k != destination:
            cost = C[source, k] + C[k, destination]
            if cost < min_cost:
                min_cost = cost
                best_node = k
    return best_node

def evaluate_hybrid_system(texts: List[str], 
                          prior: float, 
                          likelihood: float, 
                          false_positive: float) -> Tuple[np.ndarray, int]:
    C = hybrid_cost_matrix(texts, prior, likelihood, false_positive)
    best_node = ternary_routing(C, 0, 1)
    return C, best_node

if __name__ == "__main__":
    texts = ["This is a sample text.", "This is another sample text."]
    prior = 0.5
    likelihood = 0.8
    false_positive = 0.2
    C, best_node = evaluate_hybrid_system(texts, prior, likelihood, false_positive)
    print("Hybrid Cost Matrix:")
    print(C)
    print("Best Node:", best_node)