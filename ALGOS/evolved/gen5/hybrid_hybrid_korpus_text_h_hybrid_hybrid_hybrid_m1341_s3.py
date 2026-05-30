# DARWIN HAMMER — match 1341, survivor 3
# gen: 5
# parent_a: hybrid_korpus_text_hybrid_hybrid_regret_m21_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_hybrid_m805_s1.py (gen4)
# born: 2026-05-29T23:35:24Z

"""Hybrid MinHash‑HDC Regret Engine
Parents:
- hybrid_korpus_text_hybrid_hybrid_regret_m21_s3.py (MinHash text helpers + regret weighting)
- hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_hybrid_m805_s1.py (Hyperdimensional computing + morphology & variational free energy)

Mathematical bridge:
A MinHash signature is a compact set‑sketch. Each integer of the signature can be used as a seed to
generate a bipolar hypervector (HDC).  By binding (element‑wise multiplication) the hypervectors of
all signature components we obtain a single high‑dimensional representation `hv_sig`.  This hypervector
acts as a *reference* in the HDC space, analogous to the reference MinHash signature in the original
regret engine.  The hybrid score for an action combines three ingredients:

1. Regret term `R_i = expected_value – cost – risk`.
2. Jaccard‑like similarity between the action’s MinHash signature and the reference signature.
3. Variational free‑energy evaluated on the hypervector of the action, the reference hypervector and a
   morphology‑derived observation vector.

The final hybrid score is
    S_i = σ(R_i) · (1 + sim_minhash) · exp( – F_free(i) )
where σ is the sigmoid regret‑weighting, `sim_minhash` is the Jaccard similarity of the two signatures,
and `F_free(i)` is the variational free energy of the hypervector representation of the action.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

INT16_MAX = 2**15 - 1
DEFAULT_DIM = 10000  # dimensionality for hypervectors


# ----------------------------------------------------------------------
# Parent A – MinHash utilities
# ----------------------------------------------------------------------
def shingles(text: str, width: int = 5) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    return [text[i:i + width] for i in range(len(text) - width + 1)]


def minhash(tokens: Iterable[str], k: int = 64) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [0] * k
    hashes = []
    for seed in range(k):
        hash_values = []
        for t in toks:
            data = seed.to_bytes(4, "big") + t.encode("utf-8", errors="ignore")
            hv = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")
            hash_values.append(hv)
        hash_values.sort()
        hashes.append(hash_values[0])
    return hashes


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    return minhash(shingles(text), k=k)


def jaccard_similarity(sig_i: List[int], sig_ref: List[int]) -> float:
    intersection = sum(1 for a, b in zip(sig_i, sig_ref) if a == b)
    union = sum(1 for a, b in zip(sig_i, sig_ref) if a != b) + intersection
    return intersection / union if union != 0 else 0.0


# ----------------------------------------------------------------------
# Parent B – Hyperdimensional computing & morphology utilities
# ----------------------------------------------------------------------
Vector = np.ndarray  # bipolar hypervector of dtype int8


def random_vector(dim: int = DEFAULT_DIM, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)), dtype=np.int8, count=dim
    )
    return data


def symbol_vector(symbol: str, dim: int = DEFAULT_DIM) -> Vector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vectors, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(morph: Morphology) -> float:
    if min(morph.length, morph.width, morph.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (morph.length * morph.width * morph.height) ** (1.0 / 3.0) / max(
        morph.length, morph.width, morph.height
    )


def observation_vector(morph: Morphology, dim: int = DEFAULT_DIM) -> Vector:
    """Create a deterministic hypervector from morphology parameters."""
    # Encode each numeric attribute as a symbol and bind them.
    length_vec = symbol_vector(f"len:{morph.length:.2f}", dim)
    width_vec = symbol_vector(f"wid:{morph.width:.2f}", dim)
    height_vec = symbol_vector(f"hgt:{morph.height:.2f}", dim)
    mass_vec = symbol_vector(f"mass:{morph.mass:.2f}", dim)
    return bind(bind(bind(length_vec, width_vec), height_vec), mass_vec)


def variational_free_energy(obs: Vector, hv: Vector) -> float:
    """
    Simple Gaussian free‑energy surrogate:
        F = 0.5 * ||obs - hv||^2  -  log_prior
    Prior is uniform over bipolar vectors, so log_prior is constant and omitted.
    """
    diff = obs.astype(np.int16) - hv.astype(np.int16)
    return 0.5 * np.sum(diff * diff)


# ----------------------------------------------------------------------
# Hybrid layer – bridging MinHash and HDC
# ----------------------------------------------------------------------
def signature_to_hypervector(sig: List[int], dim: int = DEFAULT_DIM) -> Vector:
    """
    Convert each integer of a MinHash signature into a bipolar hypervector using the
    integer as a seed, then bundle (majority‑vote) all component vectors.
    """
    component_vectors = [random_vector(dim, seed=i) for i in sig]
    return bundle(component_vectors)


def regret_weight(R: float) -> float:
    """Sigmoid mapping of regret term."""
    return 1.0 / (1.0 + math.exp(-R))


def hybrid_action_score(
    action_sig: List[int],
    ref_sig: List[int],
    action: "MathAction",
    morphology: Morphology,
    dim: int = DEFAULT_DIM,
) -> float:
    """
    Compute the hybrid score S_i = σ(R_i)·(1+sim_minhash)·exp(-F_free).
    """
    # 1. Regret term
    R_i = action.expected_value - action.cost - action.risk
    w_regret = regret_weight(R_i)

    # 2. MinHash similarity term
    sim_minhash = jaccard_similarity(action_sig, ref_sig)

    # 3. Hypervector conversion & free‑energy term
    hv_action = signature_to_hypervector(action_sig, dim)
    hv_ref = signature_to_hypervector(ref_sig, dim)
    obs = observation_vector(morphology, dim)
    # Bind the reference hypervector with the observation to obtain a context vector
    context = bind(hv_ref, obs)
    F = variational_free_energy(obs, bind(hv_action, context))

    # Combine
    return w_regret * (1.0 + sim_minhash) * math.exp(-F)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Demonstration functions (required ≥3)
# ----------------------------------------------------------------------
def compute_all_hybrid_scores(
    actions: List[MathAction],
    reference_text: str,
    morphology: Morphology,
    k: int = 64,
    dim: int = DEFAULT_DIM,
) -> Dict[str, float]:
    """Return a dict mapping action.id to its hybrid score."""
    ref_sig = minhash_for_text(reference_text, k)
    scores = {}
    for act in actions:
        act_sig = minhash_for_text(act.id, k)  # using id as proxy text
        scores[act.id] = hybrid_action_score(act_sig, ref_sig, act, morphology, dim)
    return scores


def top_n_actions(
    scores: Dict[str, float], n: int = 3
) -> List[Tuple[str, float]]:
    """Return the top‑n actions sorted by descending hybrid score."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


def simulate_random_actions(
    count: int = 10, seed: int = 42
) -> List[MathAction]:
    """Create a list of random MathAction objects for testing."""
    rng = random.Random(seed)
    actions = []
    for i in range(count):
        ev = rng.uniform(0, 100)
        cost = rng.uniform(0, 20)
        risk = rng.uniform(0, 10)
        actions.append(MathAction(id=f"action_{i}", expected_value=ev, cost=cost, risk=risk))
    return actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    REF_TEXT = "The quick brown fox jumps over the lazy dog."
    morph = Morphology(length=12.4, width=7.8, height=3.5, mass=2.1)

    actions = simulate_random_actions(count=8)
    scores = compute_all_hybrid_scores(actions, REF_TEXT, morph)

    print("Hybrid scores:")
    for aid, sc in scores.items():
        print(f"  {aid}: {sc:.6f}")

    top = top_n_actions(scores, n=3)
    print("\nTop 3 actions:")
    for aid, sc in top:
        print(f"  {aid}: {sc:.6f}")