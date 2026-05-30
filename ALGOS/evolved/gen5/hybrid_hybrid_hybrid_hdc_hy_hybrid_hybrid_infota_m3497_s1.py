# DARWIN HAMMER — match 3497, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_worksh_m162_s1.py (gen4)
# parent_b: hybrid_hybrid_infotaxis_min_rbf_surrogate_m103_s1.py (gen2)
# born: 2026-05-29T23:50:25Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A provides high‑dimensional deterministic random vectors and a rich
feature extractor based on a text‑derived RNG.  
Parent B supplies min‑hash signatures, Jaccard‑like similarity, entropy
calculations and a Gaussian kernel.

**Mathematical bridge** – The bridge is the probability distribution over
binary token occurrences derived from A’s deterministic vector.  By treating
each index with a +1 entry as a token, we can feed this token set into B’s
min‑hash machinery, obtaining a compact signature that preserves the
high‑dimensional structure.  The similarity of two signatures becomes a
proxy for the inner‑product similarity of the original vectors.  This
similarity is then interpreted as a Bernoulli success probability *pₕᵢₜ*,
which feeds into B’s entropy formulas together with the feature vectors
extracted by A, yielding a hybrid expected‑entropy reward that can drive a
policy update.

The module implements three representative hybrid operations:
1. `hybrid_vector_signature` – builds a min‑hash signature from A’s vector.
2. `hybrid_similarity` – similarity of two texts via the hybrid signatures.
3. `hybrid_expected_entropy` – expected entropy reward using similarity as
   *pₕᵢₜ* and A’s feature vectors as state distributions.

All core equations from both parents are retained and mathematically
intertwined.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from datetime import date
from typing import Iterable, List, Set, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – deterministic random vectors and feature extraction
# ----------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")


def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    """Generate a deterministic ±1 vector."""
    rng = random.Random(seed)
    return np.array([1 if rng.getrandbits(1) else -1 for _ in range(dim)], dtype=int)


def symbol_vector(symbol: str, dim: int = 10000) -> np.ndarray:
    """Hash a symbol into a seed and produce its deterministic vector."""
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def _rng_from_text(text: str) -> random.Random:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_full_features(text: str) -> Dict[str, float]:
    """Create a flat feature dictionary from a text‑derived RNG."""
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index",
        "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() for k in keys}


# ----------------------------------------------------------------------
# Parent B – min‑hash signatures, similarity, entropy, kernels
# ----------------------------------------------------------------------

MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Min‑hash signature of a token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def shingles(text: str, width: int = 5) -> Set[str]:
    """Generate fixed‑width word shingles."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Entropy expectation conditioned on a binary outcome."""
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------------------------------------------------
# Hybrid Layer – marrying A’s vectors with B’s signature/entropy tools
# ----------------------------------------------------------------------


def _vector_to_tokens(vec: np.ndarray) -> List[str]:
    """
    Convert a ±1 vector into a token list.
    Positive indices become tokens "pos_i", negative become "neg_i".
    """
    tokens: List[str] = []
    for i, val in enumerate(vec):
        if val > 0:
            tokens.append(f"pos_{i}")
        else:
            tokens.append(f"neg_{i}")
    return tokens


def hybrid_vector_signature(text: str, dim: int = 1000, k: int = 128) -> List[int]:
    """
    Produce a min‑hash signature from the deterministic vector of *text*.
    The vector is first turned into a token set (positive/negative index tokens)
    and then fed to the standard B‑signature routine.
    """
    vec = symbol_vector(text, dim=dim)
    tokens = _vector_to_tokens(vec)
    return signature(tokens, k=k)


def hybrid_similarity(text_a: str, text_b: str, dim: int = 1000, k: int = 128) -> float:
    """
    Similarity between two texts via hybrid signatures.
    This approximates the cosine similarity of the original ±1 vectors.
    """
    sig_a = hybrid_vector_signature(text_a, dim=dim, k=k)
    sig_b = hybrid_vector_signature(text_b, dim=dim, k=k)
    return similarity(sig_a, sig_b)


def hybrid_expected_entropy(
    text_a: str,
    text_b: str,
    dim: int = 1000,
    k: int = 128,
) -> float:
    """
    Compute the expected entropy reward for a pair of texts.
    *p_hit* is taken as the hybrid similarity (in [0,1]).
    The *hit_state* distribution uses features extracted from *text_a*,
    while *miss_state* uses features from *text_b*.
    """
    p_hit = hybrid_similarity(text_a, text_b, dim=dim, k=k)

    feats_a = extract_full_features(text_a)
    feats_b = extract_full_features(text_b)

    # Convert feature dicts to probability‑like lists (normalize to sum=1)
    vals_a = list(feats_a.values())
    vals_b = list(feats_b.values())

    total_a = sum(vals_a)
    total_b = sum(vals_b)

    # Guard against degenerate zero‑sum (should not happen with random RNG)
    if total_a == 0:
        probs_a = [1.0 / len(vals_a)] * len(vals_a)
    else:
        probs_a = [v / total_a for v in vals_a]

    if total_b == 0:
        probs_b = [1.0 / len(vals_b)] * len(vals_b)
    else:
        probs_b = [v / total_b for v in vals_b]

    return expected_entropy(p_hit, probs_a, probs_b)


def hybrid_policy_update(
    action: str,
    policy: Dict[str, List[float]],
    text_a: str,
    text_b: str,
    dim: int = 1000,
    k: int = 128,
) -> None:
    """
    Update a simple count‑based policy using the hybrid expected entropy as reward.
    The policy dict maps actions to [cumulative_reward, count].
    """
    reward = hybrid_expected_entropy(text_a, text_b, dim=dim, k=k)
    total, cnt = policy.get(action, [0.0, 0.0])
    policy[action] = [total + reward, cnt + 1.0]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    txt1 = "The quick brown fox jumps over the lazy dog."
    txt2 = "A swift auburn fox leapt above a sleepy canine."

    sig1 = hybrid_vector_signature(txt1, dim=2000, k=256)
    sig2 = hybrid_vector_signature(txt2, dim=2000, k=256)

    sim = hybrid_similarity(txt1, txt2, dim=2000, k=256)
    exp_ent = hybrid_expected_entropy(txt1, txt2, dim=2000, k=256)

    policy: Dict[str, List[float]] = {}
    hybrid_policy_update("compare", policy, txt1, txt2, dim=2000, k=256)

    print(f"Signature length: {len(sig1)}")
    print(f"Hybrid similarity: {sim:.4f}")
    print(f"Hybrid expected entropy: {exp_ent:.6f}")
    print(f"Policy after update: {policy}")