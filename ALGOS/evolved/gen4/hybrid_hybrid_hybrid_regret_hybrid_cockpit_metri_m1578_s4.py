# DARWIN HAMMER — match 1578, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_hdc_serpentin_m347_s0.py (gen3)
# parent_b: hybrid_cockpit_metrics_hybrid_pheromone_inf_m64_s1.py (gen2)
# born: 2026-05-29T23:37:40Z

"""Hybrid Regret‑Weighted Liquid Time‑Constant MinHash + Cockpit‑Pheromone Metrics.

Parent A contributes:
  - MinHash signatures (`signature`), random hyperdimensional vectors, and the bind operation.
  - Actions are represented as high‑dimensional vectors (`HybridAction`).

Parent B contributes:
  - Normalized cockpit ratios (`anti_slop_ratio`, `cockpit_honesty`, `audit_debt`).
  - Pheromone decay (`calculate_pheromone_signal`) and Shannon entropy (`calculate_entropy`).

Mathematical bridge:
  The cockpit ratios are probabilities ∈[0,1] that weight pheromone signals.
  Those weighted signals are used as a scalar multiplier for the hyperdimensional
  vector of an action before the MinHash binding step.  Consequently the
  similarity of two actions (MinHash overlap) is modulated by cockpit‑derived
  trust, and a global “trust‑entropy” score combines pairwise MinHash similarity
  with the entropy of the cockpit probability distribution.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – hyperdimensional & MinHash utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a set of string tokens."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for i in range(k)) for t in toks]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[float]:
    """Deterministic pseudo‑random vector in [0,1)."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def bind(a: List[float], b: List[float]) -> List[float]:
    """Element‑wise (Hadamard) product – the HD “bind” operation."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return [x * y for x, y in zip(a, b)]


# ----------------------------------------------------------------------
# Core domain objects (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class HybridAction:
    action: MathAction
    vector: List[float]


def hybrid_morphology_vector(
    action: MathAction, dim: int = 1024, seed: str | int | None = None
) -> List[float]:
    """
    Produce a hyperdimensional vector for *action*.

    The raw random vector is scaled by a weight derived from the
    action's attributes (expected value, cost, risk) and by a pheromone
    signal computed from the same attributes (see ``weighted_pheromone``).
    """
    base = random_vector(dim, seed or action.id)

    # Normalised attribute factors (logistic / reciprocal transforms)
    ev_norm = 1.0 / (1.0 + math.exp(-action.expected_value))
    cost_norm = 1.0 / (1.0 + action.cost)
    risk_norm = 1.0 / (1.0 + action.risk)

    # Attribute‑derived scalar weight
    attr_weight = ev_norm * (1.0 - cost_norm) * (1.0 - risk_norm)

    # Pheromone‑derived scalar weight
    pheromone = weighted_pheromone(action)

    total_weight = attr_weight * pheromone
    return [x * total_weight for x in base]


# ----------------------------------------------------------------------
# Parent B – cockpit metrics & pheromone / infotaxis utilities
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Fraction of claims that have supporting evidence."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known to be OK."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int) -> float:
    """Raw count of missing audit steps (non‑negative)."""
    return float(max(0, exports_missing_audit_step))


def calculate_pheromone_signal(
    base_signal: float, half_life_seconds: float, elapsed_seconds: float
) -> float:
    """Exponential decay of a pheromone signal."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be > 0")
    decay = math.pow(0.5, elapsed_seconds / half_life_seconds)
    return base_signal * decay


def calculate_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    total = probabilities.sum()
    if total <= 0:
        return 0.0
    probs = probabilities / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Fusion utilities – the mathematical bridge
# ----------------------------------------------------------------------
def weighted_pheromone(action: MathAction) -> float:
    """
    Compute a scalar pheromone signal for *action*.

    The base signal is the expected value; the half‑life grows with risk,
    and the elapsed time is approximated by the cost.
    """
    base = max(0.0, action.expected_value)
    half_life = 10.0 + 5.0 * action.risk          # more risk → slower decay
    elapsed = action.cost                         # larger cost → more elapsed time
    return calculate_pheromone_signal(base, half_life, elapsed)


def make_hybrid_action(action: MathAction, dim: int = 1024) -> HybridAction:
    """Create a ``HybridAction`` that embeds cockpit‑scaled pheromone into the HD vector."""
    vec = hybrid_morphology_vector(action, dim=dim)
    return HybridAction(action=action, vector=vec)


def hybrid_minhash_similarity(a: HybridAction, b: HybridAction, k: int = 128) -> float:
    """
    Pairwise similarity between two ``HybridAction`` objects.

    1. Convert each hyperdimensional vector to a token list (stringified floats).
    2. Compute MinHash signatures.
    3. Return the Jaccard‑like overlap of the signatures.
    """
    tokens_a = [f"{v:.8f}" for v in a.vector]
    tokens_b = [f"{v:.8f}" for v in b.vector]

    sig_a = signature(tokens_a, k=k)
    sig_b = signature(tokens_b, k=k)

    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / k


def trust_entropy_score(
    hybrid_actions: List[HybridAction],
    cockpit_counts: Dict[str, int],
    k: int = 128,
) -> float:
    """
    Global trust score that merges:

    * The entropy of cockpit‑derived probability distribution.
    * The average pairwise MinHash similarity of the hyperdimensional actions.

    The score lies in [0,1]; higher values indicate trustworthy, low‑entropy,
    and mutually similar actions.
    """
    # --- Cockpit probability vector -------------------------------------------------
    m1 = anti_slop_ratio(
        cockpit_counts.get("claims_with_evidence", 0),
        cockpit_counts.get("total_claims_emitted", 0),
    )
    m2 = cockpit_honesty(
        cockpit_counts.get("displayed_ok", 0),
        cockpit_counts.get("unknown_displayed_as_ok", 0),
    )
    m3 = 1.0 / (1.0 + audit_debt(cockpit_counts.get("exports_missing_audit_step", 0)))
    prob_vec = np.array([m1, m2, m3], dtype=float)

    if prob_vec.sum() == 0:
        prob_vec = np.ones_like(prob_vec) / prob_vec.size
    else:
        prob_vec = prob_vec / prob_vec.sum()

    entropy = calculate_entropy(prob_vec)

    # --- Average pairwise MinHash similarity ---------------------------------------
    n = len(hybrid_actions)
    if n < 2:
        avg_sim = 0.0
    else:
        sims = [
            hybrid_minhash_similarity(hybrid_actions[i], hybrid_actions[j], k=k)
            for i in range(n)
            for j in range(i + 1, n)
        ]
        avg_sim = sum(sims) / len(sims)

    # Normalise entropy to [0,1] (max entropy = log(len(prob_vec)))
    max_entropy = math.log(len(prob_vec))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Trust = similarity * (1 - normalised entropy)
    return avg_sim * (1.0 - norm_entropy)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few sample actions
    actions = [
        MathAction(id="A1", expected_value=2.3, cost=1.0, risk=0.2),
        MathAction(id="A2", expected_value=1.8, cost=0.5, risk=0.5),
        MathAction(id="A3", expected_value=3.0, cost=2.0, risk=0.1),
    ]

    # Convert to hybrid actions (HD vectors with pheromone weighting)
    hybrid_actions = [make_hybrid_action(a, dim=2048) for a in actions]

    # Example cockpit counts
    cockpit_counts = {
        "claims_with_evidence": 80,
        "total_claims_emitted": 100,
        "displayed_ok": 45,
        "unknown_displayed_as_ok": 5,
        "exports_missing_audit_step": 2,
    }

    # Compute trust‑entropy score
    score = trust_entropy_score(hybrid_actions, cockpit_counts, k=64)
    print(f"Trust‑entropy score: {score:.4f}")

    # Demonstrate pairwise similarity
    for i in range(len(hybrid_actions)):
        for j in range(i + 1, len(hybrid_actions)):
            sim = hybrid_minhash_similarity(hybrid_actions[i], hybrid_actions[j], k=64)
            print(f"Similarity between {hybrid_actions[i].action.id} and {hybrid_actions[j].action.id}: {sim:.3f}")