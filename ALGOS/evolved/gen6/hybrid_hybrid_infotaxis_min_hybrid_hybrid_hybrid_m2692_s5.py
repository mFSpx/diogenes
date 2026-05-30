# DARWIN HAMMER — match 2692, survivor 5
# gen: 6
# parent_a: hybrid_infotaxis_minhash_m63_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_hybrid_krampu_m535_s0.py (gen5)
# born: 2026-05-29T23:43:32Z

"""Hybrid Infotaxis–MinHash & Pheromone Decay (Parents: hybrid_infotaxis_minhash_m63_s4.py, hybrid_hybrid_hybrid_label__hybrid_hybrid_krampu_m535_s0.py)

Mathematical bridge
-------------------
* **From parent A** we obtain a *MinHash signature* `σ ∈ ℕᵏ` for a token set and its
  *signature entropy* `H(σ)`.  The entropy quantifies the information content of the
  textual observation.
* **From parent B** we have a *pheromone entry* whose `signal_value` decays exponentially
  with a half‑life `τ`.  The decay factor is `0.5^{age/τ}`.
* The hybrid connects the two by **using the signature entropy as the initial
  pheromone signal** and **modulating the decay rate with a bilinear similarity**
  between signatures of two entries.  The similarity is expressed as a bilinear form


S(σ_i, σ_j) = σ_iᵀ W σ_j / (‖σ_i‖‖σ_j‖)


  where `W` is a (optional) weight matrix, defaulting to the identity.  The decay
  factor becomes `0.5^{age / (τ·(1+S))}`, i.e. more similar entries retain their signal
  longer.

The module therefore fuses the *information‑theoretic* view of text (entropy) with
the *stigmergic* view of pheromones (exponential decay), yielding a unified hybrid
system.

"""

import hashlib
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Dict, Optional

import numpy as np

MAX64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# Parent‑A utilities (MinHash & entropy)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return the MinHash signature of length *k* for the given token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


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


def signature_entropy(sig: List[int]) -> float:
    """Entropy of the multiset of signature components."""
    if not sig:
        raise ValueError("signature must not be empty")
    counts = Counter(sig)
    probs = list(counts.values())
    return entropy(probs)


# ----------------------------------------------------------------------
# Parent‑B utilities (Pheromone entry & decay)
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
    signature: List[int]

    def age_seconds(self) -> float:
        """Simulated age (seconds) – in a real system this would be a timestamp diff."""
        # For reproducibility in tests we use a deterministic pseudo‑random value.
        rng = random.Random(hash(self.uuid))
        return rng.uniform(0, 100)

    def decay_factor(self, similarity: float = 0.0) -> float:
        """
        Exponential decay factor adjusted by similarity.
        similarity ∈ [0, 1]; larger similarity slows decay.
        """
        effective_half_life = self.half_life_seconds * (1.0 + similarity)
        if effective_half_life <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / effective_half_life)

    def apply_decay(self, similarity: float = 0.0) -> None:
        """Apply decay to the stored signal value."""
        factor = self.decay_factor(similarity)
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def bilinear_similarity(sig_a: List[int], sig_b: List[int],
                        weight_matrix: Optional[np.ndarray] = None) -> float:
    """
    Bilinear similarity S(σ_a, σ_b) = σ_aᵀ W σ_b / (‖σ_a‖‖σ_b‖).
    The signatures are interpreted as column vectors.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    vec_a = np.asarray(sig_a, dtype=np.float64)
    vec_b = np.asarray(sig_b, dtype=np.float64)

    if weight_matrix is None:
        weight_matrix = np.eye(len(sig_a), dtype=np.float64)

    if weight_matrix.shape != (len(sig_a), len(sig_b)):
        raise ValueError("weight matrix shape mismatch")

    numerator = float(vec_a @ weight_matrix @ vec_b)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-12)
    return max(0.0, min(1.0, numerator / denom))


def create_hybrid_entry(uuid: str,
                        surface_key: str,
                        tokens: Iterable[str],
                        half_life_seconds: int = 60,
                        k: int = 128) -> PheromoneEntry:
    """
    Build a `PheromoneEntry` whose initial `signal_value` is the entropy of the
    MinHash signature derived from *tokens*.
    """
    sig = signature(tokens, k=k)
    ent = signature_entropy(sig)  # entropy in nats
    # Normalise entropy to a convenient scale (0‑1) using a softmax‑like mapping.
    norm_signal = 1.0 - math.exp(-ent)  # monotone mapping to (0,1)
    entry = PheromoneEntry(
        uuid=uuid,
        surface_key=surface_key,
        signal_kind="text_entropy",
        signal_value=norm_signal,
        half_life_seconds=half_life_seconds,
        created_at=pathlib.Path.cwd(),
        last_decay=pathlib.Path.cwd(),
        signature=sig,
    )
    return entry


def decay_hybrid_entries(entries: List[PheromoneEntry],
                         weight_matrix: Optional[np.ndarray] = None) -> None:
    """
    Apply decay to each entry, using pairwise bilinear similarity with the most
    similar neighbour to modulate the decay factor.
    """
    n = len(entries)
    if n == 0:
        return

    # Pre‑compute similarity matrix (upper triangular)
    sims = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sims[i, j] = bilinear_similarity(
                entries[i].signature,
                entries[j].signature,
                weight_matrix=weight_matrix,
            )
            sims[j, i] = sims[i, j]

    # For each entry, pick the maximum similarity to any other entry.
    max_sims = np.max(sims + np.eye(n), axis=1)  # add identity to keep self‑similarity = 1
    for entry, sim in zip(entries, max_sims):
        entry.apply_decay(similarity=sim)


def hybrid_expected_entropy(entries: List[PheromoneEntry]) -> float:
    """
    Compute the expected entropy of the hybrid system as the Shannon entropy of
    the (decayed) signal values, weighted by their current magnitude.
    """
    if not entries:
        raise ValueError("no entries to evaluate")
    values = np.array([e.signal_value for e in entries], dtype=np.float64)
    # Normalise to a probability distribution.
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    return -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _demo() -> None:
    # Create three synthetic text token sets.
    tokens_a = "the quick brown fox jumps over the lazy dog".split()
    tokens_b = "lorem ipsum dolor sit amet consectetur adipiscing elit".split()
    tokens_c = "data science machine learning artificial intelligence".split()

    # Build hybrid pheromone entries.
    entry_a = create_hybrid_entry("A", "surface1", tokens_a, half_life_seconds=80)
    entry_b = create_hybrid_entry("B", "surface2", tokens_b, half_life_seconds=80)
    entry_c = create_hybrid_entry("C", "surface3", tokens_c, half_life_seconds=80)

    entries = [entry_a, entry_b, entry_c]

    print("Initial signal values:", [e.signal_value for e in entries])
    print("Initial hybrid entropy:", hybrid_expected_entropy(entries))

    # Apply one decay step.
    decay_hybrid_entries(entries)

    print("Post‑decay signal values:", [e.signal_value for e in entries])
    print("Post‑decay hybrid entropy:", hybrid_expected_entropy(entries))


if __name__ == "__main__":
    _demo()