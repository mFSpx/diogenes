# DARWIN HAMMER — match 1544, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_regret_m267_s1.py (gen4)
# parent_b: hybrid_hybrid_korpus_text_h_gliner_zero_shot_ext_m118_s1.py (gen3)
# born: 2026-05-29T23:37:17Z

"""Hybrid Algorithm: Quaternion‑GA Rotor driven by MinHash Text Signature

Parents:
- hybrid_hybrid_hybrid_model__hybrid_hybrid_regret_m267_s1.py (Quaternion GA rotor & regret)
- hybrid_hybrid_korpus_text_h_gliner_zero_shot_ext_m118_s1.py (MinHash text signature & span extraction)

Mathematical Bridge:
The MinHash signature of a text is reduced to a 4‑component vector and normalised to a
unit quaternion R.  This quaternion acts as the GA rotor that transforms an
action‑quaternion x = [0, expected, cost, risk] via the sandwich product
    y = R * x * ~R
The transformed component y₁ encodes a regret‑weighted expected value.  The
similarity between the text signature and a reference signature (derived from
labels) modulates the final hybrid score, thus tightly coupling the two parent
topologies in a single unified system.
"""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared structures from Parent A
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(
            np.uint8(hashlib.blake2b(data, digest_size=8).digest()), dtype=np.uint8
        ),
        "big",
    )


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
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# MinHash utilities from Parent B
# ----------------------------------------------------------------------


def _clean_text(text: str) -> str:
    return " ".join(text.lower().split()) if text else ""


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Deterministic MinHash signature for a given text."""
    clean = _clean_text(text)
    if len(clean) < 5:
        return [0] * k
    shingles = [clean[i : i + 5] for i in range(len(clean) - 4)]
    # deterministic pseudo‑random base signature
    signature = [2 ** 31 - 1] * k
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h & 0x7FFFFFFF)
    return signature


# ----------------------------------------------------------------------
# Quaternion utilities (core of Parent A)
# ----------------------------------------------------------------------


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions a and b (shape (4,))."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_normalise(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def rotor_from_text(text: str, k: int = 4) -> np.ndarray:
    """
    Create a unit quaternion rotor from a MinHash signature.
    The first *k* hash values are taken, shifted to positive range and normalised.
    """
    mh = minhash_for_text(text, k=64)
    # take first k entries, ensure they are non‑negative floats
    raw = np.array([abs(int(v)) for v in mh[:k]], dtype=float)
    if raw.max() == 0:
        raw = np.ones_like(raw)
    rotor = quat_normalise(raw)
    return rotor


def action_to_quat(action: MathAction) -> np.ndarray:
    """
    Encode a MathAction as a pure quaternion (scalar part zero).
    Order: [0, expected, cost, risk]
    """
    return np.array([0.0, action.expected_value, action.cost, action.risk], dtype=float)


def transform_action(action: MathAction, rotor: np.ndarray) -> np.ndarray:
    """
    Apply the GA rotor to the action quaternion via the sandwich product:
        y = R * x * ~R
    Returns the transformed quaternion y.
    """
    x = action_to_quat(action)
    y = quat_mul(quat_mul(rotor, x), quat_conj(rotor))
    return y


def hybrid_regret_score(
    actions: List[MathAction],
    text: str,
    label_set: List[str],
) -> float:
    """
    Compute a hybrid score that blends:
    1. Regret‑like contribution from rotor‑transformed actions.
    2. Similarity between the text MinHash signature and a signature built from the labels.
    """
    # 1️⃣ Rotor from the main text
    rotor = rotor_from_text(text, k=4)

    # 2️⃣ Transform each action and extract the scalar‑free component (index 1)
    transformed_vals = []
    for act in actions:
        y = transform_action(act, rotor)
        # y[1] corresponds to the expected component after rotation
        transformed_vals.append(y[1])

    # Regret term: mean absolute deviation from original expected values
    original = np.array([a.expected_value for a in actions])
    transformed = np.array(transformed_vals)
    regret_term = np.mean(np.abs(transformed - original))

    # 3️⃣ Signature from label set (treated as tokens)
    label_sig = signature(label_set, k=64)

    # 4️⃣ Text signature
    text_sig = minhash_for_text(text, k=64)

    # Similarity term (0..1)
    sim_term = similarity(label_sig, text_sig)

    # Final hybrid score – higher is better, combines regret reduction and semantic similarity
    return (1.0 - regret_term) * sim_term


# ----------------------------------------------------------------------
# Demonstration utilities (not part of the core algorithm)
# ----------------------------------------------------------------------


def extract_spans(text: str, labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    Simple span extractor mirroring Parent B's behaviour.
    Returns list of (start, end, label) tuples.
    """
    spans = []
    for label in labels:
        start = text.find(label)
        if start != -1:
            spans.append((start, start + len(label), label))
    return spans


def demo():
    # Sample actions
    actions = [
        MathAction(id="a1", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="a2", expected_value=5.5, cost=1.5, risk=0.5),
        MathAction(id="a3", expected_value=7.2, cost=3.0, risk=2.0),
    ]

    # Sample text and labels
    text = "The quick brown fox jumps over the lazy dog."
    labels = ["quick", "fox", "dog"]

    # Compute hybrid score
    score = hybrid_regret_score(actions, text, labels)
    print(f"Hybrid regret‑similarity score: {score:.4f}")

    # Show transformed actions
    rotor = rotor_from_text(text, k=4)
    for act in actions:
        y = transform_action(act, rotor)
        print(f"Action {act.id} -> transformed quaternion {y}")

    # Extract spans
    spans = extract_spans(text, labels)
    print("Extracted spans:", spans)


if __name__ == "__main__":
    demo()