# DARWIN HAMMER — match 5315, survivor 1
# gen: 7
# parent_a: hybrid_korpus_text_hybrid_hybrid_regret_m21_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2643_s6.py (gen6)
# born: 2026-05-30T00:01:11Z

"""Hybrid Algorithm: MinHash‑Regret + Pheromone Curvature Fusion

Parents:
- **Parent A** (`hybrid_korpus_text_hybrid_hybrid_regret_m21_s4.py`): provides MinHash signatures,
  Jaccard similarity, and a regret‑weighted action score.
- **Parent B** (`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2643_s6.py`): provides exponential
  pheromone decay modulated by an Ollivier‑Ricci curvature term derived from a linear model
  and a geometric product coefficient.

Mathematical Bridge:
Both parents manipulate scalar scores that are later used for probabilistic selection.
We treat the **MinHash similarity** from Parent A as a *feature vector* `x` for the curvature
computation of Parent B. The curvature `C = ||W·x – t||²` scales the decay factor
`exp(-α·C)`. The final hybrid score for an action `a` is


score(a) = sim(ref, a) · exp(−regret(a)) · exp(−α·C(x_a)) / (1 + risk_a)


where `sim` is the Jaccard similarity of MinHash signatures, `regret(a)` is a simple
regret‑weighting based on expected value and cost, and `C(x_a)` is the curvature obtained
from a randomly generated weight matrix `W`.  The resulting scores feed both a softmax
action selector and a pheromone‑update routine that decays stored signals with the same
curvature‑scaled exponential factor.

The module implements the full hybrid pipeline:
1. MinHash signature generation.
2. Curvature computation from the signature‑derived feature vector.
3. Hybrid scoring, softmax selection, and pheromone decay update.
"""

import math
import random
import sys
import pathlib
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Iterable, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – MinHash & regret utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action description used by the regret component."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    # Use built‑in hashlib for reproducibility without external deps
    import hashlib

    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def shingles(text: str, width: int = 5) -> List[str]:
    """Create overlapping substrings (shingles) of a given width."""
    return [text[i : i + width] for i in range(len(text) - width + 1)]


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    hashes = []
    for seed in range(k):
        hash_values = [_hash(seed, t) for t in toks]
        hashes.append(min(hash_values))
    return hashes


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    intersection = sum(1 for a, b in zip(sig1, sig2) if a == b)
    union = len(sig1)  # For MinHash, union size equals signature length
    return intersection / union


def text_entropy(text: str) -> float:
    """Shannon entropy of character distribution in a string."""
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def regret_weight(action: MathAction) -> float:
    """Simple exponential regret weighting."""
    # Larger expected value or cost → higher regret → smaller weight
    return math.exp(-(action.expected_value + action.cost))


# ----------------------------------------------------------------------
# Parent B – Curvature & pheromone decay utilities
# ----------------------------------------------------------------------


def random_weight_matrix(dim: int) -> np.ndarray:
    """Generate a random square weight matrix for curvature computation."""
    rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=1.0, size=(dim, dim))


def curvature(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """
    Ollivier‑Ricci curvature proxy: C = ||W·x – t||².
    If target is None, use the zero vector.
    """
    if target is None:
        target = np.zeros_like(x)
    diff = W @ x - target
    return float(np.linalg.norm(diff) ** 2)


@dataclass
class PheromoneEntry:
    """Lightweight pheromone record with exponential decay."""
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_decay: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def decay_factor(age_seconds: float, half_life: int) -> float:
    """Standard exponential half‑life decay."""
    if half_life <= 0:
        return 0.0
    return 0.5 ** (age_seconds / half_life)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_score(
    action: MathAction,
    ref_sig: List[int],
    action_sig: List[int],
    entropy: float,
    W: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Compute the fused score for an action.

    Components:
    - Jaccard similarity between reference and action signatures.
    - Regret weighting based on expected value and cost.
    - Curvature‑scaled exponential decay using the signature as feature vector.
    - Optional entropy damping (higher entropy → slightly lower score).

    Returns a non‑negative scalar.
    """
    sim = jaccard_similarity(ref_sig, action_sig)
    if sim == 0.0:
        return 0.0

    # Transform the MinHash signature into a normalized feature vector
    sig_array = np.array(action_sig, dtype=float)
    if sig_array.max() > 0:
        sig_norm = sig_array / sig_array.max()
    else:
        sig_norm = sig_array

    C = curvature(W, sig_norm)
    curvature_term = math.exp(-alpha * C)

    regret = regret_weight(action)
    entropy_term = math.exp(-entropy * 0.01)  # mild damping

    risk_term = 1.0 / (1.0 + action.risk)

    return sim * regret * curvature_term * entropy_term * risk_term


def softmax_selection(actions: List[MathAction], scores: Dict[str, float]) -> MathAction:
    """Select an action proportionally to softmax of its hybrid scores."""
    max_score = max(scores.values()) if scores else 0.0
    # Stabilize exponentials
    exp_vals = [
        math.exp(scores.get(a.id, -1e9) - max_score) for a in actions
    ]
    total = sum(exp_vals)
    if total == 0.0:
        # Fallback: uniform random choice
        return random.choice(actions)
    probs = [v / total for v in exp_vals]
    return random.choices(actions, weights=probs, k=1)[0]


def update_pheromones(
    pheromones: List[PheromoneEntry],
    action_scores: Dict[str, float],
    alpha: float = 0.1,
) -> None:
    """
    Apply curvature‑scaled decay to each pheromone entry.
    The decay factor is multiplied by the hybrid score of the matching action
    (if any). This ties the pheromone dynamics directly to the MinHash‑regret
    evaluation.
    """
    now = datetime.now(timezone.utc)
    for p in pheromones:
        age = (now - p.created_at).total_seconds()
        base_decay = decay_factor(age, p.half_life_seconds)

        # Use the surface_key as a proxy for an action identifier
        score = action_scores.get(p.surface_key, 1.0)
        # Curvature term re‑used with a tiny random matrix (to avoid extra state)
        dim = 8  # small fixed dimension for the auxiliary matrix
        W_aux = random_weight_matrix(dim)
        # Build a dummy feature vector from the pheromone uuid hash
        uid_hash = _hash(0, p.uuid)
        x_vec = np.array([uid_hash % (2 ** i) for i in range(dim)], dtype=float)
        if x_vec.max() > 0:
            x_vec /= x_vec.max()
        C = curvature(W_aux, x_vec)
        curvature_decay = math.exp(-alpha * C)

        # Composite decay
        p.signal_value *= base_decay * curvature_decay * score


# ----------------------------------------------------------------------
# Demonstration / Smoke test
# ----------------------------------------------------------------------


def _demo():
    # Sample text corpus
    reference_text = "The quick brown fox jumps over the lazy dog."
    actions = [
        MathAction(id="a1", expected_value=2.5, cost=0.3, risk=0.1),
        MathAction(id="a2", expected_value=1.0, cost=0.1, risk=0.5),
        MathAction(id="a3", expected_value=3.0, cost=0.7, risk=0.2),
    ]

    # Generate MinHash signatures
    ref_sig = signature(shingles(reference_text))

    # Random weight matrix for curvature (dimension matches signature length)
    dim = len(ref_sig) if ref_sig else 1
    W = random_weight_matrix(dim)

    # Compute hybrid scores
    scores: Dict[str, float] = {}
    for act in actions:
        # For demo purposes, create a synthetic text per action
        act_text = f"{act.id} example payload for MinHash."
        act_sig = signature(shingles(act_text), k=dim)
        ent = text_entropy(act_text)
        sc = hybrid_score(act, ref_sig, act_sig, ent, W)
        scores[act.id] = sc

    # Select an action via softmax
    chosen = softmax_selection(actions, scores)
    print(f"Chosen action: {chosen.id}")

    # Initialise pheromones (one per action)
    pheromones = [
        PheromoneEntry(
            surface_key=act.id,
            signal_kind="reward",
            signal_value=1.0,
            half_life_seconds=60,
        )
        for act in actions
    ]

    # Update pheromones using the same scores
    update_pheromones(pheromones, scores)

    # Show updated pheromone values
    for p in pheromones:
        print(f"Pheromone {p.surface_key}: value={p.signal_value:.6f}")


if __name__ == "__main__":
    _demo()