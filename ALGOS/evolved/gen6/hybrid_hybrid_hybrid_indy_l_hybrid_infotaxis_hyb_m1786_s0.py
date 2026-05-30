# DARWIN HAMMER — match 1786, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_indy_learning_hybrid_pheromone_inf_m541_s2.py (gen5)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s2.py (gen3)
# born: 2026-05-29T23:38:45Z

"""
Fusion of hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s1.py and hybrid_pheromone_infotaxis_m3_s0.py
Mathematical Bridge:
The hybrid affinity function from the hybrid_pheromone_infotaxis_m3_s0.py algorithm is used to modulate
the recovery priority of the candidate neighbors in the hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py algorithm.
The governing equations of both parents are integrated through the following steps:
1. Compute the hybrid affinity using the infotaxis algorithm.
2. Use the hybrid affinity to modulate the recovery priority of the candidate neighbors.
3. Compute the expected entropy of the hit and miss states using the hybrid affinity.
4. Normalize the expected entropy to a value in [0,1].
"""

import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def sha256_json(value: Any) -> str:
    """Deterministic hash of any JSON‑serialisable object."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()

def tokenize(text: str) -> List[Dict[str, Any]]:
    """Very light tokenizer returning token text and character offsets."""
    word_re = re.compile(r"\S+")
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in word_re.finditer(text)
    ]

def chunk_text_tokens(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
    source_ref: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Split a token list into overlapping windows."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be >=0 and < max_tokens"
        )
    toks = tokenize(text)
    source_ref = dict(source_ref or {})
    if not toks:
        return []

    chunks: List[Dict[str, Any]] = []
    step = max_tokens - overlap_tokens
    for i in range(0, len(toks), step):
        chunk = toks[i : i + max_tokens]
        chunks.append({"tokens": chunk, "source_ref": source_ref})
    return chunks

# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------

def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy, robust to zero probabilities."""
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = probs / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))

def expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

def hybrid_affinity(cosine_similarity: float, p_other: float) -> float:
    return cosine_similarity * (1 - p_other)

def normalized_expected_entropy(expected_entropy: float) -> float:
    return expected_entropy / (1 + expected_entropy)

def morphology_similarity(m1: Morphology, m2: Morphology) -> float:
    return sphericity_index(m1.length, m1.width, m1.height) * flatness_index(m2.length, m2.width, m2.height)

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------

def hybrid_algorithm(hit_state: list[float], miss_state: list[float], p_hit: float, neighbors: List[Morphology]) -> List[Tuple[Morphology, float]]:
    """Compute the expected entropy of the hit and miss states and use it to modulate the recovery priority of the candidate neighbors."""
    expected_entropies = [expected_entropy(p_hit, hit_state, miss_state) for _ in neighbors]
    normalized_entropies = [normalized_expected_entropy(e) for e in expected_entropies]
    hybrid_affinities = [hybrid_affinity(morphology_similarity(m1, m2), p_other) for m1, m2, p_other in zip(neighbors, neighbors[1:], np.linspace(0, 1, len(neighbors)))]
    return list(zip(neighbors, hybrid_affinities))

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    hit_state = [0.5, 0.3, 0.2]
    miss_state = [0.7, 0.2, 0.1]
    p_hit = 0.8
    neighbors = [Morphology(length=1.0, width=2.0, height=3.0, mass=4.0), Morphology(length=2.0, width=3.0, height=4.0, mass=5.0)]
    results = hybrid_algorithm(hit_state, miss_state, p_hit, neighbors)
    print(results)