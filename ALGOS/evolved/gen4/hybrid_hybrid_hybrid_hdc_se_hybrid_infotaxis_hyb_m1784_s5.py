# DARWIN HAMMER — match 1784, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_hybrid_decisi_m162_s0.py (gen3)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s1.py (gen3)
# born: 2026-05-29T23:38:47Z

"""
Hybrid Serpentina‑Infotaxis Morphology‑Entropy Engine
Parents:
- hybrid_hdc_serpentin_hybrid_hybrid_decisi_m162_s0.py (hyper‑dimensional serpentina morphology)
- hybrid_infotaxis_hybrid_semantic_neig_m739_s1.py (entropy‑driven infotaxis with morphology‑based recovery priority)

Mathematical Bridge
The serpentina morphology is embedded as a high‑dimensional vector 𝑣∈ℝᴰ.  Pairwise similarities
    s_i = 𝑣·a_i   (dot product with candidate action vector a_i)
are turned into a probability distribution via a soft‑max:
    p_i = exp(β·s_i) / Σ_j exp(β·s_j)
The Shannon entropy of this distribution,
    H = − Σ_i p_i log p_i,
captures the uncertainty of the action space.
The morphology‑driven recovery priority
    p̂ = min(1, RTI / max_index)   with   RTI = m.mass**b·exp(k·FI)/neck_lever
scales the entropy, yielding the hybrid affinity
    h = H · p̂ .
Thus the hyper‑dimensional topology (vector similarities) modulates the entropy
search, while the physical‑morphology topology (righting‑time index) rescales it.
"""

import re
import math
import random
import hashlib
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

Vector = List[float]

# ----------------------------------------------------------------------
# Morphology definition (shared)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

# ----------------------------------------------------------------------
# Hyper‑dimensional utilities (from Parent A)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]

def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    """Deterministic hyper‑dimensional embedding of a morphology."""
    seed_bytes = hashlib.sha256(
        f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    vec = np.array(random_vector(dim, seed))
    scaling = np.array([m.length, m.width, m.height, m.mass] * (dim // 4 + 1))[:dim]
    return (vec * scaling).tolist()

# ----------------------------------------------------------------------
# Geometry & recovery priority (from Parent B)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """Physical right‑ing time index (RTI)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority p̂ ∈ [0,1] derived from RTI."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ----------------------------------------------------------------------
# Entropy‑driven infotaxis utilities (adapted from Parent B)
# ----------------------------------------------------------------------
def softmax(similarities: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Stable soft‑max yielding a probability distribution."""
    max_val = np.max(beta * similarities)
    exp_vals = np.exp(beta * similarities - max_val)
    return exp_vals / exp_vals.sum()

def action_entropy(action_vectors: List[Vector], query_vector: Vector, beta: float = 1.0) -> float:
    """
    Compute Shannon entropy of the similarity‑derived distribution between
    `query_vector` (the morphology embedding) and a set of candidate action vectors.
    """
    if not action_vectors:
        return 0.0
    sims = np.dot(np.array(action_vectors), np.array(query_vector))
    probs = softmax(sims, beta)
    # Guard against log(0) by adding a tiny epsilon
    eps = np.finfo(float).eps
    return -float(np.sum(probs * np.log(probs + eps)))

# ----------------------------------------------------------------------
# Hybrid operations (core of the fused algorithm)
# ----------------------------------------------------------------------
def hybrid_affinity(
    action_vectors: List[Vector],
    morphology: Morphology,
    beta: float = 1.0,
    max_index: float = 10.0,
) -> float:
    """
    Compute the hybrid affinity h = H · p̂ where
        H = entropy of the action similarity distribution,
        p̂ = recovery priority from morphology.
    """
    query_vec = morphology_vector(morphology, dim=len(action_vectors[0]))
    H = action_entropy(action_vectors, query_vec, beta=beta)
    p_hat = recovery_priority(morphology, max_index=max_index)
    return H * p_hat

def rank_actions_by_hybrid(
    action_vectors: List[Vector],
    morphology: Morphology,
    beta: float = 1.0,
    max_index: float = 10.0,
) -> List[Tuple[int, float]]:
    """
    Return a list of (action_index, hybrid_score) sorted ascending.
    Lower hybrid_score indicates a more decisive (lower entropy) and higher‑priority action.
    """
    scores = []
    for idx, _ in enumerate(action_vectors):
        # For ranking we compute the entropy of the *full* set (as in infotaxis)
        # and then weight it by the priority – the same score for every action.
        # To break ties we add a tiny term proportional to the similarity of the
        # candidate to the morphology vector.
        query_vec = morphology_vector(morphology, dim=len(action_vectors[0]))
        sims = np.dot(np.array(action_vectors), np.array(query_vec))
        probs = softmax(sims, beta=beta)
        H = -float(np.sum(probs * np.log(probs + np.finfo(float).eps)))
        p_hat = recovery_priority(morphology, max_index=max_index)
        # Tie‑breaker: higher similarity → slightly lower score
        tie_break = -0.001 * probs[idx]
        score = H * p_hat + tie_break
        scores.append((idx, score))
    return sorted(scores, key=lambda x: x[1])

def select_best_action(
    action_vectors: List[Vector],
    morphology: Morphology,
    beta: float = 1.0,
    max_index: float = 10.0,
) -> Tuple[int, float]:
    """
    Choose the action with the minimal hybrid score.
    Returns (action_index, hybrid_score).
    """
    ranked = rank_actions_by_hybrid(action_vectors, morphology, beta, max_index)
    return ranked[0] if ranked else (-1, float("inf"))

# ----------------------------------------------------------------------
# Simple evidence regex utility (from Parent A, retained for completeness)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

def contains_evidence(text: str) -> bool:
    """Return True if the text contains any evidence‑related keyword."""
    return bool(EVIDENCE_RE.search(text))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a sample morphology
    sample_morph = Morphology(length=2.5, width=1.8, height=0.9, mass=3.2)

    # Generate a small pool of candidate action vectors (dim = 1024 for speed)
    dim = 1024
    rng_seed = 42
    random.seed(rng_seed)
    action_pool = [random_vector(dim, seed=rng_seed + i) for i in range(5)]

    # Compute hybrid affinity for the whole set
    h = hybrid_affinity(action_pool, sample_morph, beta=0.5)
    print(f"Hybrid affinity (global): {h:.6f}")

    # Rank actions
    ranked = rank_actions_by_hybrid(action_pool, sample_morph, beta=0.5)
    print("Ranked actions (index, score):")
    for idx, score in ranked:
        print(f"  {idx}: {score:.6f}")

    # Select best action
    best_idx, best_score = select_best_action(action_pool, sample_morph, beta=0.5)
    print(f"Best action: index {best_idx} with hybrid score {best_score:.6f}")

    # Evidence detection demo
    demo_text = "The experiment was verified and the log file contains a SHA256 hash."
    print(f"Evidence detected: {contains_evidence(demo_text)}")