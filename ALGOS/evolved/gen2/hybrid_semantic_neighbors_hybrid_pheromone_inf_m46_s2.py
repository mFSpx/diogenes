# DARWIN HAMMER — match 46, survivor 2
# gen: 2
# parent_a: semantic_neighbors.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s0.py (gen1)
# born: 2026-05-29T23:23:42Z

"""Hybrid Semantic-Pheromone Infotaxis Module.
Parents:
- semantic_neighbors.py (cosine similarity based neighbor retrieval)
- hybrid_pheromone_infotaxis_m3_s0.py (pheromone probability weighting + entropy‑based action selection)

Mathematical bridge:
Cosine similarities between document vectors are interpreted as raw pheromone signals.
These signals are decayed (half‑life) and normalized to obtain a probability distribution
P_nei over the k‑nearest neighbours.  The Shannon entropy H(P_nei) quantifies the
uncertainty of the semantic neighbourhood.  For each candidate action we compute an
expected entropy  

    E = p_hit * H(hit_state) + (1‑p_hit) * H(miss_state)

where p_hit is modulated by the neighbourhood certainty (1‑H(P_nei)).  The action
with minimal E is selected.  Thus the core linear algebra of cosine similarity and
the probabilistic‑entropy machinery of pheromone‑infotaxis are fused into a single
decision‑making pipeline.
"""

import math
import random
import sys
import pathlib
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------
# In‑memory semantic enclave (parent A)
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
# In‑memory pheromone store (simplified parent B)
# ----------------------------------------------------------------------
_PheromoneStore: dict[str, list[tuple[float, datetime]]] = defaultdict(list)

def _decay_factor(age_seconds: float, half_life: float) -> float:
    """Exponential decay based on half‑life."""
    return 0.5 ** (age_seconds / half_life)

def signal_pheromone(
    surface_key: str,
    signal_value: float,
    half_life_seconds: int,
    timestamp: datetime | None = None,
) -> None:
    """Record a pheromone signal with its emission time."""
    ts = timestamp or datetime.now(timezone.utc)
    _PheromoneStore[surface_key].append((signal_value, ts))

def calculate_pheromone_probabilities(
    surface_key: str,
    limit: int,
    half_life_seconds: int,
) -> list[float]:
    """Return a normalized probability vector from the most recent `limit` signals,
    applying half‑life decay."""
    now = datetime.now(timezone.utc)
    raw = []
    for value, ts in _PheromoneStore[surface_key][-limit:]:
        age = (now - ts).total_seconds()
        raw.append(value * _decay_factor(age, half_life_seconds))
    total = sum(raw)
    if total == 0.0:
        return [0.0] * len(raw)
    return [v / total for v in raw]

# ----------------------------------------------------------------------
# Entropy utilities (parent B)
# ----------------------------------------------------------------------
def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError('positive probability mass required')
    ent = 0.0
    for p in probabilities:
        if p > 0.0:
            q = max(p / total, eps)
            ent -= q * math.log(q)
    return ent

def expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float]) -> float:
    """Expected entropy of an action given hit probability."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_neighbor_pheromones(
    doc_id: str,
    k: int = 5,
    half_life_seconds: int = 86400,
) -> list[float]:
    """
    Treat cosine similarities of the k‑nearest neighbours as raw pheromone signals,
    decay them, and return a normalized probability distribution.
    """
    neighbors = semantic_neighbors(doc_id, k)
    if not neighbors:
        return []
    # Use similarity as signal_value; timestamp = now (no aging yet)
    surface_key = f"neigh_{doc_id}"
    # Reset store for this query to avoid cross‑talk
    _PheromoneStore[surface_key].clear()
    for _, sim in neighbors:
        signal_pheromone(surface_key, signal_value=sim, half_life_seconds=half_life_seconds)
    return calculate_pheromone_probabilities(surface_key, limit=k, half_life_seconds=half_life_seconds)

def hybrid_action_score(
    doc_id: str,
    actions: dict[str, tuple[float, list[float], list[float]]],
    k: int = 5,
    half_life_seconds: int = 86400,
) -> dict[str, float]:
    """
    Compute a hybrid expected‑entropy score for each action.
    The hit probability is scaled by the certainty of the semantic neighbourhood:
        p_hit' = base_p_hit * (1 - H(P_nei))
    where H(P_nei) is the entropy of the neighbour‑derived probability vector.
    """
    neigh_probs = compute_neighbor_pheromones(doc_id, k, half_life_seconds)
    if not neigh_probs:
        neighbourhood_certainty = 0.0
    else:
        neighbourhood_certainty = 1.0 - entropy(neigh_probs)
    scores: dict[str, float] = {}
    for act, (base_p_hit, hit_state, miss_state) in actions.items():
        p_hit_mod = base_p_hit * neighbourhood_certainty
        scores[act] = expected_entropy(p_hit_mod, hit_state, miss_state)
    return scores

def best_hybrid_action(
    doc_id: str,
    actions: dict[str, tuple[float, list[float], list[float]]],
    k: int = 5,
    half_life_seconds: int = 86400,
) -> str:
    """
    Select the action with the minimal hybrid expected entropy.
    """
    scores = hybrid_action_score(doc_id, actions, k, half_life_seconds)
    # Tie‑break by lexical order for reproducibility
    return min(scores, key=lambda a: (scores[a], a))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Register a tiny corpus
    register_document("docA", [1, 0, 0, 1])
    register_document("docB", [0.9, 0.1, 0, 1])
    register_document("docC", [0, 1, 0, 0])
    register_document("docD", [0, 0, 1, 0])
    register_document("docE", [0.5, 0.5, 0, 0.5])

    # Define dummy actions: (base_p_hit, hit_state_distribution, miss_state_distribution)
    actions = {
        "explore": (0.6, [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]),
        "exploit": (0.8, [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]),
        "idle":    (0.2, [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]),
    }

    # Run hybrid decision for docA
    chosen = best_hybrid_action("docA", actions, k=3, half_life_seconds=3600)
    print(f"Chosen action for 'docA': {chosen}")

    # Show intermediate scores for transparency
    scores = hybrid_action_score("docA", actions, k=3, half_life_seconds=3600)
    for act, sc in scores.items():
        print(f"  {act}: expected entropy = {sc:.6f}")

    sys.exit(0)