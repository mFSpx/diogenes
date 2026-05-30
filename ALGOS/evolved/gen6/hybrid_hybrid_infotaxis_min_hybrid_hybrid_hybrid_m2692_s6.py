# DARWIN HAMMER — match 2692, survivor 6
# gen: 6
# parent_a: hybrid_infotaxis_minhash_m63_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_hybrid_krampu_m535_s0.py (gen5)
# born: 2026-05-29T23:43:32Z

"""Hybrid Infotaxis‑MinHash & Pheromone Decay (Hybrid_A + Hybrid_B)

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** supplies a MinHash based *signature* for a token set and the
  *signature entropy* – the Shannon entropy of the multiset of hash minima.
* **Parent B** defines a *pheromone entry* whose `signal_value` decays
  exponentially with a half‑life and whose `signal_value` is interpreted as a
  strength or probability.

**Mathematical bridge** – the signature entropy is used as the pheromone’s
signal strength.  The decayed signal value modulates the probability
`p_hit` in the expected‑entropy calculation of Parent A, thereby coupling the
information‑theoretic uncertainty of the token set with the temporal decay
of a pheromone.  Additionally a bilinear form maps the high‑dimensional
category‑frequency vector (from Parent B) and the low‑dimensional MinHash
signature (from Parent A) into a scalar score, providing a unified
low‑dimensional representation.

The resulting hybrid system can:
1. Build a pheromone entry from a token set using MinHash signature entropy.
2. Update the pheromone with new tokens while respecting decay.
3. Compute a hybrid expected entropy that incorporates the decayed signal.
4. Produce a bilinear similarity score between categorical features and the
   MinHash signature.

All code is pure Python 3, using only the allowed standard‑library modules and
NumPy.
"""

import hashlib
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Dict

import numpy as np

MAX64 = (1 << 64) - 1
DEFAULT_K = 128
DEFAULT_HALF_LIFE = 60  # seconds, arbitrary default


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by the MinHash signature."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = DEFAULT_K) -> List[int]:
    """MinHash signature of a token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability mass."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def signature_entropy(sig: List[int]) -> float:
    """Entropy of a MinHash signature (counts of equal hash minima)."""
    if not sig:
        raise ValueError("signature must not be empty")
    counts = Counter(sig)
    probs = list(counts.values())
    return entropy(probs)


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted average of two entropy values."""
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


# ----------------------------------------------------------------------
# Pheromone data structure (Parent B)
# ----------------------------------------------------------------------
@dataclass
class PheromoneEntry:
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: pathlib.Path
    last_decay: pathlib.Path

    def age_seconds(self) -> float:
        """Simulated age – in a real system this would be a timestamp diff."""
        return random.uniform(0, 100)

    def decay_factor(self) -> float:
        """Exponential decay factor based on half‑life."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply decay in‑place to the signal value."""
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def create_pheromone_from_tokens(
    tokens: Iterable[str],
    uuid: str,
    surface_key: str = "default",
    signal_kind: str = "entropy",
    k: int = DEFAULT_K,
    half_life_seconds: int = DEFAULT_HALF_LIFE,
) -> PheromoneEntry:
    """
    Build a PheromoneEntry whose initial signal value is the entropy of the
    MinHash signature of *tokens*.
    """
    sig = signature(tokens, k=k)
    sig_ent = signature_entropy(sig)
    now = pathlib.Path.cwd()
    return PheromoneEntry(
        uuid=uuid,
        surface_key=surface_key,
        signal_kind=signal_kind,
        signal_value=sig_ent,
        half_life_seconds=half_life_seconds,
        created_at=now,
        last_decay=now,
    )


def hybrid_expected_entropy(
    pheromone: PheromoneEntry,
    token_set: Iterable[str],
    k: int = DEFAULT_K,
) -> float:
    """
    Compute the expected entropy of adding a new token to *token_set*,
    where the probability of a “hit” is the decayed pheromone signal
    (scaled to [0,1]).

    The hit state entropy is the entropy of the signature after adding the
    token; the miss state entropy is the current signature entropy.
    """
    # Apply decay to obtain the current effective signal.
    pheromone.apply_decay()
    # Normalise signal to a probability (entropy lies in [0, log(k)]).
    max_possible = math.log(k)
    p_hit = min(max(pheromone.signal_value / max_possible, 0.0), 1.0)

    # Current signature entropy (miss state)
    current_sig = signature(token_set, k=k)
    miss_entropy = signature_entropy(current_sig)

    # Simulate adding each possible new token (here we just pick one random token)
    # For demonstration we compute the entropy after adding a single new token.
    # In practice this could be enumerated over a candidate set.
    new_token = f"new_{random.randint(0, 1_000_000)}"
    augmented_set = set(token_set) | {new_token}
    hit_sig = signature(augmented_set, k=k)
    hit_entropy = signature_entropy(hit_sig)

    # Expected entropy using the decayed signal as p_hit
    return expected_entropy(p_hit, [hit_entropy], [miss_entropy])


def bilinear_score(
    tokens: Iterable[str],
    sig: List[int],
    M: np.ndarray,
) -> float:
    """
    Compute a bilinear form vᵀ M w where:
    * v – vector of token‑category frequencies (size C)
    * w – MinHash signature vector (size K)
    * M – a C×K matrix (pre‑initialised)

    This projects the high‑dimensional categorical feature space onto the
    low‑dimensional MinHash space.
    """
    # Build category frequency vector (C = number of FUNCTION_CATS)
    cat_names = list(FUNCTION_CATS.keys())
    cat_counts = np.zeros(len(cat_names), dtype=float)
    token_set = set(tokens)
    for idx, cat in enumerate(cat_names):
        cat_counts[idx] = len(token_set & FUNCTION_CATS[cat])

    w = np.array(sig, dtype=float)
    if M.shape != (len(cat_names), len(sig)):
        raise ValueError("Shape mismatch between M and feature vectors")
    return float(cat_counts @ M @ w)


# ----------------------------------------------------------------------
# Example categorical dictionary from Parent B (used by bilinear_score)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, Set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although whoever that which what how why when where who whom since as until long".split()
    ),
    "adverb": set("how very rather more".split()),
}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample token set
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    # Create pheromone entry from tokens
    pher = create_pheromone_from_tokens(tokens, uuid="test-001")
    print(f"Pheromone initial signal (entropy): {pher.signal_value:.4f}")

    # Compute hybrid expected entropy
    hyb_ent = hybrid_expected_entropy(pher, tokens)
    print(f"Hybrid expected entropy after decay: {hyb_ent:.4f}")

    # Prepare bilinear matrix M (categories × signature length)
    np.random.seed(0)
    M = np.random.normal(scale=0.01, size=(len(FUNCTION_CATS), DEFAULT_K))

    # Compute bilinear score
    sig = signature(tokens, k=DEFAULT_K)
    score = bilinear_score(tokens, sig, M)
    print(f"Bilinear category‑signature score: {score:.6f}")

    # Update pheromone with new tokens and recompute
    new_tokens = ["jumps", "high", "over", "the", "moon"]
    pher.signal_value = signature_entropy(signature(set(tokens) | set(new_tokens), k=DEFAULT_K))
    pher.apply_decay()
    print(f"Pheromone signal after update & decay: {pher.signal_value:.4f}")

    sys.exit(0)