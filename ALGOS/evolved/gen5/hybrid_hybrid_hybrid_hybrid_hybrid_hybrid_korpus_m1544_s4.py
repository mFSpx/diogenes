# DARWIN HAMMER — match 1544, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_regret_m267_s1.py (gen4)
# parent_b: hybrid_hybrid_korpus_text_h_gliner_zero_shot_ext_m118_s1.py (gen3)
# born: 2026-05-29T23:37:17Z

"""Hybrid Algorithm: Quaternion Regret Rotor with MinHash Bridge

Parents:
- hybrid_hybrid_hybrid_model__hybrid_hybrid_regret_m267_s1.py (Quaternion GA‑TTT scheduler + Regret Engine)
- hybrid_hybrid_korpus_text_hybrid_krampus_brain_m43_s0.py (MinHash text signature + Span extractor)

Mathematical Bridge:
The minhash signature of a text fragment is a deterministic integer vector `h ∈ ℤ^k`.  
We treat `h` as a scalar weighting factor `α = (sum(h) mod 1) + 1e-3` that modulates the bivector
`b = x ∧ (y−x)` used to update the quaternion rotor `R`.  In the regret engine, the
regret‑weighted probability of selecting an action is multiplied by the same `α`,
thereby coupling the textual minhash information with the geometric‑algebra update.
Thus the rotor selection and its subsequent sandwich transformation are driven jointly
by geometric, regret‑based, and textual minhash information."""

import numpy as np
import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Tuple

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

# ----------------------------------------------------------------------
# MinHash utilities (from Parent B)
# ----------------------------------------------------------------------
def _clean_text(text: str) -> str:
    return " ".join(text.lower().split()) if text else ""

def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Deterministic minhash signature for a given text."""
    text = _clean_text(text)
    if len(text) < 5:
        return [0] * k
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = [sys.maxsize] * k
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h & 0xFFFFFFFFFFFFFFFF)
    return signature

def minhash_weight(minhash: List[int]) -> float:
    """Map a minhash vector to a positive scaling factor α ∈ (0, 2)."""
    total = sum(minhash) & 0xFFFFFFFFFFFFFFFF
    return (total % 1000) / 500.0 + 1e-3  # avoid zero

# ----------------------------------------------------------------------
# Quaternion (rotor) utilities (from Parent A)
# ----------------------------------------------------------------------
def quaternion_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)

def quaternion_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def rotor_sandwich(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply rotor R to 3‑vector v via y = R * (0,v) * ~R and return the rotated 3‑vector."""
    v_quat = np.concatenate([[0.0], v])
    temp = quaternion_mul(R, v_quat)
    out = quaternion_mul(temp, quaternion_conj(R))
    return out[1:]  # drop scalar part

def bivector_from_vectors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simple bivector proxy: cross product (represents x ∧ y)."""
    return np.cross(x, y)

def rotor_update(R: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Update rotor using bivector b = x ∧ (y−x) scaled by α (minhash weight).
    The update rule is R' = normalize(R + α * b_quat) where b_quat = (0, b).
    """
    b = bivector_from_vectors(x, y - x)
    b_quat = np.concatenate([[0.0], b])
    R_new = R + alpha * b_quat
    norm = np.linalg.norm(R_new)
    return R_new / norm if norm != 0 else R_new

# ----------------------------------------------------------------------
# Regret‑based selection (from Parent A) coupled with MinHash (bridge)
# ----------------------------------------------------------------------
def regret_weights(actions: List[MathAction], regrets: List[float], alpha: float) -> np.ndarray:
    """
    Compute regret‑weighted probabilities and blend with α.
    Returns a normalized probability vector over actions.
    """
    if len(actions) != len(regrets):
        raise ValueError("actions and regrets must have same length")
    raw = np.array(regrets, dtype=np.float64) * alpha
    # Softmax for probabilities
    max_raw = np.max(raw)
    exp = np.exp(raw - max_raw)
    probs = exp / exp.sum()
    return probs

def select_action(actions: List[MathAction], probs: np.ndarray) -> MathAction:
    """Sample an action according to the provided probability distribution."""
    idx = np.random.choice(len(actions), p=probs)
    return actions[idx]

# ----------------------------------------------------------------------
# Hybrid Functions (demonstrate the fused system)
# ----------------------------------------------------------------------
def compute_rotor_for_action(action: MathAction, text: str) -> np.ndarray:
    """
    Build a rotor R for a given action using the action's expected value and the
    minhash‑derived weight α from the supplied text.
    """
    alpha = minhash_weight(minhash_for_text(text))
    # Initialise rotor as unit quaternion scaled by expected value
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) * (1.0 + 0.1 * action.expected_value)
    # Random 3‑vector derived from text hash for geometric update
    rng = np.random.default_rng(hash(text) & 0xFFFFFFFFFFFFFFFF)
    x = rng.random(3)
    y = x + rng.random(3) * 0.5
    R = rotor_update(base, x, y, alpha)
    return R

def transform_text_vectors(text: str, R: np.ndarray) -> List[np.ndarray]:
    """
    Split the text into word‑level vectors (simple character count embeddings),
    then rotate each vector with the provided rotor.
    Returns the list of rotated vectors.
    """
    words = [w for w in text.split() if w]
    rotated = []
    for w in words:
        # Very cheap embedding: length + vowel count + consonant count
        v = np.array([
            len(w),
            sum(ch in "aeiou" for ch in w.lower()),
            sum(ch.isalpha() and ch.lower() not in "aeiou" for ch in w.lower())
        ], dtype=np.float64)
        rotated.append(rotor_sandwich(R, v))
    return rotated

def hybrid_regret_rotor_pipeline(actions: List[MathAction],
                                 regrets: List[float],
                                 text: str,
                                 labels: List[str]) -> Tuple[MathAction, np.ndarray, List[np.ndarray]]:
    """
    Full hybrid pipeline:
    1. Derive α from the text's minhash.
    2. Compute regret‑weighted probabilities and sample an action.
    3. Build a rotor for the sampled action.
    4. Rotate word embeddings of the text.
    5. Return the chosen action, rotor, and rotated embeddings.
    """
    alpha = minhash_weight(minhash_for_text(text))
    probs = regret_weights(actions, regrets, alpha)
    chosen = select_action(actions, probs)
    R = compute_rotor_for_action(chosen, text)
    rotated_embeddings = transform_text_vectors(text, R)
    return chosen, R, rotated_embeddings

# ----------------------------------------------------------------------
# Simple signature utilities (fixed bug from Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A", expected_value=1.2, cost=0.1, risk=0.05),
        MathAction(id="B", expected_value=0.8, cost=0.2, risk=0.1),
        MathAction(id="C", expected_value=1.5, cost=0.05, risk=0.02),
    ]
    # Mock regrets (higher = more regret)
    regrets = [0.3, 0.7, 0.2]

    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_labels = ["quick", "fox", "dog"]

    chosen_action, rotor, rotated = hybrid_regret_rotor_pipeline(
        actions, regrets, sample_text, sample_labels
    )

    print(f"Chosen action: {chosen_action.id}")
    print(f"Rotor (quaternion): {rotor}")
    print(f"Number of rotated word vectors: {len(rotated)}")
    # Verify signature bridge
    sig1 = signature([chosen_action.id, sample_text])
    sig2 = signature(sample_labels)
    print(f"Signature similarity: {similarity(sig1, sig2):.3f}")