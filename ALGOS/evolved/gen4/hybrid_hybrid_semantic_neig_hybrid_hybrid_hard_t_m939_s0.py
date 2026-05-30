# DARWIN HAMMER — match 939, survivor 0
# gen: 4
# parent_a: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s2.py (gen3)
# born: 2026-05-29T23:31:40Z

"""
Module merging hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py and hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s2.py.
The mathematical bridge between the two structures is the application of the cosine similarity from hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py to the Bayesian updated features from hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s2.py.
The governing equation of the hybrid algorithm is:
s = vᵀ P_nei m * bayes_update(prior, likelihood)
where v is the text-derived feature vector, m is the model-resource vector, P_nei is the neighbourhood certainty matrix, and bayes_update is the Bayesian update function.
"""

import numpy as np
import random
import math
import sys
import pathlib
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# ----------------------------------------------------------------------
# In-memory semantic enclave (parent A)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Bayesian update function (parent B)
# ----------------------------------------------------------------------
def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Bayesian update function."""
    return prior * likelihood / np.sum(prior * likelihood)

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
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

def words(text: str) -> List[str]:
    return [w.lower() for w in text.split() if w.isalpha()]

def lsm_vector(text: str) -> Dict[str, float]:
    word_counts = Counter(words(text))
    return {cat: sum(word_counts.get(w, 0) for w in words) / len(words) for cat in FUNCTION_CATS}

def extract_full_features(text: str) -> np.ndarray:
    """Extract full features from text."""
    return np.array([lsm_vector(text)])

def hybrid_semantic_bayes(doc_id: str, k: int = 5) -> np.ndarray:
    """Hybrid semantic-bayes algorithm."""
    v = _ENCLAVE[doc_id]
    sims = semantic_neighbors(doc_id, k)
    P_nei = np.zeros((k,))
    for i, (d, s) in enumerate(sims):
        p = _ENCLAVE[d]
        P_nei[i] = _cosine(v, p)
    P_nei = P_nei / np.sum(P_nei)
    prior = extract_full_features(doc_id)
    likelihood = extract_full_features(doc_id)
    return prior * np.dot(P_nei, likelihood)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    register_document("doc1", [1.0, 2.0, 3.0])
    register_document("doc2", [4.0, 5.0, 6.0])
    register_document("doc3", [7.0, 8.0, 9.0])
    print(hybrid_semantic_bayes("doc1"))