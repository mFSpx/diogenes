# DARWIN HAMMER — match 939, survivor 1
# gen: 4
# parent_a: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s2.py (gen3)
# born: 2026-05-29T23:31:40Z

"""
Module merging hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py and 
hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s2.py. 
The mathematical bridge between the two structures is the application of the 
Shannon entropy H(P_nei) from hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py 
to the Bayesian updated features from hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s2.py, 
enabling the analysis of the compatibility between text-derived feature vectors and 
model-resource vectors with uncertain probabilities.

The governing equation of the hybrid algorithm is:
s = vᵀ P m * bayes_update(prior, likelihood) * (1 - H(P_nei))

where v is the text-derived feature vector, m is the model-resource vector, 
P is the projection matrix, bayes_update is the Bayesian update function, 
and H(P_nei) is the Shannon entropy of the semantic neighbourhood.
"""

import numpy as np
import random
import math
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

_FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
_PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

_ENCLAVE: dict[str, np.ndarray] = {}

def clear_enclave() -> None:
    """Remove all registered documents."""
    _ENCLAVE.clear()

def register_document(doc_id: str, vector: list[float]) -> None:
    """Store a document vector as a NumPy array for fast linear algebra."""
    _ENCLAVE[doc_id] = np.array(vector, dtype=float)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)

def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """Return the k most similar documents (excluding the query)."""
    v = _ENCLAVE[doc_id]
    sims = [(d, _cosine(v, w)) for d, w in _ENCLAVE.items() if d != doc_id]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]

def shannon_entropy(p: np.ndarray) -> float:
    """Compute the Shannon entropy of a probability distribution."""
    return -np.sum(p * np.log2(p))

def words(text: str) -> List[str]:
    return [w.lower() for w in text.split() if w.isalpha()]

def lsm_vector(text: str) -> Dict[str, float]:
    word_counts = Counter(words(text))
    return {cat: sum(word_counts.get(w, 0) for w in words) / len(words) for cat in _FUNCTION_CATS}

def bayes_update(prior: float, likelihood: float) -> float:
    """Perform a Bayesian update."""
    return prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood))

def hybrid_analysis(text: str, doc_id: str, model_vector: np.ndarray) -> float:
    """Perform the hybrid analysis."""
    # Compute the semantic neighbourhood
    sims = semantic_neighbors(doc_id)
    p_nei = np.array([sim[1] for sim in sims])
    p_nei /= p_nei.sum()
    nei_entropy = shannon_entropy(p_nei)

    # Compute the text-derived feature vector
    text_vector = np.array(list(lsm_vector(text).values()))

    # Compute the projection matrix
    P = np.eye(len(text_vector))

    # Perform the Bayesian update
    prior = 0.5
    likelihood = 0.8
    bayes_factor = bayes_update(prior, likelihood)

    # Compute the hybrid score
    s = np.dot(text_vector.T, np.dot(P, model_vector)) * bayes_factor * (1 - nei_entropy)
    return s

if __name__ == "__main__":
    clear_enclave()
    register_document("doc1", [1.0, 2.0, 3.0])
    register_document("doc2", [4.0, 5.0, 6.0])
    print(hybrid_analysis("This is a test.", "doc1", np.array([7.0, 8.0, 9.0])))