# DARWIN HAMMER — match 1341, survivor 4
# gen: 5
# parent_a: hybrid_korpus_text_hybrid_hybrid_regret_m21_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_hybrid_m805_s1.py (gen4)
# born: 2026-05-29T23:35:24Z

"""HybridMinHashHDCEngine
Combines:
- Parent A: KORPUS low‑level text MinHash helpers.
- Parent B: Hyperdimensional Computing (HDC) with variational free‑energy.

Mathematical bridge:
A MinHash signature (list of integers) is transformed into a bipolar hypervector
by bundling symbol‑vectors derived from each integer hash value.  The resulting
hypervector lives in the same space as the HDC morphology vectors.  Regret‑
weighted action scores (from Parent A) are modulated by both the Jaccard similarity
between MinHash signatures **and** the cosine similarity between the derived
hypervectors, while a variational free‑energy term (from Parent B) penalises
morphologies that are unlikely under the observed data.  The unified hybrid
score for action *i* is

    S_i = σ(R_i) · (1 + J(sig_i, sig_ref)) · cos(hv_i, hv_ref) · exp(‑F(morph_i))

where σ is a sigmoid, J is Jaccard similarity of MinHash signatures,
cos is cosine similarity of hypervectors, and F is variational free energy.
"""

import hashlib
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – MinHash utilities
# ----------------------------------------------------------------------
INT16_MAX = 2**15 - 1


def shingles(text: str, width: int = 5) -> List[str]:
    """Return overlapping substrings (shingles) of given width."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    return [text[i : i + width] for i in range(len(text) - width + 1)]


def minhash(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Classic min‑hash over a set of tokens."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [0] * k
    hashes: List[int] = []
    for seed in range(k):
        hash_values = []
        for t in toks:
            data = seed.to_bytes(4, "big") + t.encode("utf-8", errors="ignore")
            hv = int.from_bytes(
                hashlib.blake2b(data, digest_size=8).digest(), "big"
            )
            hash_values.append(hv)
        hash_values.sort()
        hashes.append(hash_values[0])
    return hashes


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Convenience wrapper – shingle then minhash."""
    return minhash(shingles(text), k=k)


def jaccard_similarity(sig_i: List[int], sig_ref: List[int]) -> float:
    """Jaccard‑like similarity for two MinHash signatures."""
    intersection = sum(1 for a, b in zip(sig_i, sig_ref) if a == b)
    union = sum(1 for a, b in zip(sig_i, sig_ref) if a != b) + intersection
    return intersection / union if union != 0 else 0.0


# ----------------------------------------------------------------------
# Parent B – Hyperdimensional Computing utilities
# ----------------------------------------------------------------------
Vector = np.ndarray  # bipolar hypervector of dtype int8


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    """Generate a random bipolar vector (elements ∈ {‑1, 1})."""
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=np.int8,
        count=dim,
    )
    return data


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic hypervector for a symbolic token."""
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Binding (XOR‑like) operation – element‑wise multiplication."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: List[Vector]) -> Vector:
    """Superposition (majority vote) of bipolar vectors."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vectors, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity for bipolar vectors (range [‑1, 1])."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = int(np.dot(a, b))
    norm = math.sqrt(int(np.dot(a, a)) * int(np.dot(b, b)))
    return dot / norm if norm != 0 else 0.0


# ----------------------------------------------------------------------
# Domain specific data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action with economic attributes and a textual description."""
    id: str
    description: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass
class Morphology:
    """Geometric entity used for free‑energy evaluation."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Hybrid core functions (the new unified system)
# ----------------------------------------------------------------------
def signature_to_hypervector(sig: List[int], dim: int = 10000) -> Vector:
    """
    Map a MinHash signature to a bipolar hypervector.
    Each integer hash is turned into a deterministic symbol vector;
    the final hypervector is the bundle of all symbol vectors.
    """
    symbols = [f"hash_{h}" for h in sig]
    vecs = [symbol_vector(sym, dim) for sym in symbols]
    return bundle(vecs)


def regret_weight(R: float) -> float:
    """Sigmoid‑shaped regret weighting function."""
    return 1.0 / (1.0 + math.exp(-R))


def sphericity_index(morph: Morphology) -> float:
    """Morphology‑specific geometric index (range (0, 1])."""
    if min(morph.length, morph.width, morph.height) <= 0:
        raise ValueError("dimensions must be positive")
    geo_mean = (morph.length * morph.width * morph.height) ** (1.0 / 3.0)
    return geo_mean / max(morph.length, morph.width, morph.height)


def variational_free_energy(observation: np.ndarray, model_mean: np.ndarray) -> float:
    """
    Simple Gaussian free‑energy approximation:
        F = 0.5 * ||observation – model_mean||²
    """
    diff = observation.astype(np.float64) - model_mean.astype(np.float64)
    return 0.5 * np.dot(diff, diff)


def hybrid_action_score(
    action: MathAction,
    ref_text: str,
    morphology: Morphology,
    observation: np.ndarray,
    dim: int = 10000,
    k: int = 64,
) -> float:
    """
    Compute the unified hybrid score for an action.

    Steps:
    1. Build MinHash signatures for the reference text and the action description.
    2. Convert both signatures to hypervectors.
    3. Compute regret term R_i = EV – cost – risk and pass through sigmoid.
    4. Evaluate Jaccard similarity of signatures.
    5. Evaluate cosine similarity of hypervectors.
    6. Compute free‑energy penalty from morphology and observation.
    7. Combine everything as:
           S_i = σ(R_i) · (1 + J) · cos(hv_i, hv_ref) · exp(‑F)
    """
    # 1. Signatures
    sig_ref = minhash_for_text(ref_text, k=k)
    sig_act = minhash_for_text(action.description, k=k)

    # 2. Hypervectors
    hv_ref = signature_to_hypervector(sig_ref, dim=dim)
    hv_act = signature_to_hypervector(sig_act, dim=dim)

    # 3. Regret weighting
    R_i = action.expected_value - action.cost - action.risk
    w_regret = regret_weight(R_i)

    # 4. Jaccard similarity
    j_sim = jaccard_similarity(sig_act, sig_ref)

    # 5. Cosine similarity of hypervectors
    c_sim = cosine_similarity(hv_act, hv_ref)

    # 6. Free‑energy term (model mean derived from morphology)
    #    Use a simple mapping: each geometric dimension contributes proportionally.
    model_mean = np.array(
        [
            morphology.length,
            morphology.width,
            morphology.height,
            morphology.mass,
        ],
        dtype=np.float64,
    )
    # Pad or truncate observation to length 4 for compatibility
    if observation.size != 4:
        raise ValueError("observation must be a vector of length 4")
    F = variational_free_energy(observation, model_mean)

    # 7. Combine
    score = w_regret * (1.0 + j_sim) * c_sim * math.exp(-F)
    return score


def batch_hybrid_scores(
    actions: List[MathAction],
    ref_text: str,
    morphology: Morphology,
    observation: np.ndarray,
    dim: int = 10000,
    k: int = 64,
) -> Dict[str, float]:
    """
    Compute hybrid scores for a batch of actions and return a mapping
    from action id to its score.
    """
    scores: Dict[str, float] = {}
    for act in actions:
        scores[act.id] = hybrid_action_score(
            act, ref_text, morphology, observation, dim=dim, k=k
        )
    return scores


def top_n_actions(
    scores: Dict[str, float], n: int = 3
) -> List[Tuple[str, float]]:
    """Return the n actions with highest hybrid scores."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reference context
    reference = "The quick brown fox jumps over the lazy dog."

    # Sample actions
    actions = [
        MathAction(
            id="A1",
            description="A swift brown fox leaps",
            expected_value=12.0,
            cost=3.0,
            risk=1.0,
        ),
        MathAction(
            id="A2",
            description="A lazy dog sleeps",
            expected_value=8.0,
            cost=2.0,
            risk=0.5,
        ),
        MathAction(
            id="A3",
            description="An agile cat climbs",
            expected_value=10.0,
            cost=4.0,
            risk=2.0,
        ),
    ]

    # Morphology and observation (simple 4‑dimensional sensor reading)
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=2.3)
    obs = np.array([1.0, 0.9, 0.6, 2.5], dtype=np.float64)

    # Compute scores
    scores = batch_hybrid_scores(actions, reference, morph, obs)

    # Display results
    print("Hybrid scores per action:")
    for aid, sc in scores.items():
        print(f"  {aid}: {sc:.6f}")

    print("\nTop actions:")
    for aid, sc in top_n_actions(scores, n=2):
        print(f"  {aid}: {sc:.6f}")

    sys.exit(0)