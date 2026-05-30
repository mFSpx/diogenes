# DARWIN HAMMER — match 1786, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_indy_learning_hybrid_pheromone_inf_m541_s2.py (gen5)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s2.py (gen3)
# born: 2026-05-29T23:38:45Z

"""
This module presents a novel HYBRID algorithm, mathematically fusing the core topologies of 
PARENT ALGORITHM A (hybrid_hybrid_indy_learning_hybrid_pheromone_inf_m541_s2.py) and 
PARENT ALGORITHM B (hybrid_infotaxis_hybrid_semantic_neig_m739_s2.py) into a single unified system.
The mathematical bridge between these two algorithms is based on the concept of entropy and its impact on 
the recovery priority of candidate neighbors in the hybrid semantic-morphology system.
The entropy calculations from the infotaxis algorithm are used to modulate the recovery priority, 
which in turn affects the hybrid affinity between neighbors.
"""

import hashlib
import json
import math
import random
import re
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

def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(np.array(hit_state)) + (1.0 - p_hit) * entropy(np.array(miss_state))

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Dict[str, float], b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    return (m["mass"] * b) / (k * neck_lever)

def hybrid_affinity(length: float, width: float, height: float, p_other: float, E_normalized: float) -> float:
    cosine_similarity = sphericity_index(length, width, height)
    return cosine_similarity * p_other * (1 - E_normalized)

def calculate_expected_entropy(hit_state: List[float], miss_state: List[float], p_hit: float) -> float:
    return expected_entropy(p_hit, hit_state, miss_state)

def calculate_hybrid_affinity(length: float, width: float, height: float, p_other: float, E_normalized: float) -> float:
    return hybrid_affinity(length, width, height, p_other, E_normalized)

def calculate_recovery_priority(hit_state: List[float], miss_state: List[float], p_hit: float) -> float:
    E_expected = calculate_expected_entropy(hit_state, miss_state, p_hit)
    return 1 - (E_expected / entropy(np.array([0.5, 0.5])))

if __name__ == "__main__":
    # Test the functions
    hit_state = [0.6, 0.4]
    miss_state = [0.3, 0.7]
    p_hit = 0.7
    E_normalized = 0.5
    p_other = 0.8
    length = 10.0
    width = 5.0
    height = 2.0

    E_expected = calculate_expected_entropy(hit_state, miss_state, p_hit)
    recovery_priority = calculate_recovery_priority(hit_state, miss_state, p_hit)
    hybrid_affinity_value = calculate_hybrid_affinity(length, width, height, recovery_priority, E_normalized)
    print("Expected Entropy:", E_expected)
    print("Recovery Priority:", recovery_priority)
    print("Hybrid Affinity:", hybrid_affinity_value)