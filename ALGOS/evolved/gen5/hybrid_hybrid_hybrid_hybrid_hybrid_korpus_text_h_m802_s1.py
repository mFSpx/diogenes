# DARWIN HAMMER — match 802, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s1.py (gen3)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s1.py (gen4)
# born: 2026-05-29T23:32:21Z

"""Hybrid Allocation & Text Similarity Engine

This module fuses **Parent Algorithm A** (`hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s1.py`)
and **Parent Algorithm B** (`hybrid_korpus_text_hybrid_hybrid_regret_m21_s1.py`).

Mathematical bridge
-------------------
*Parent A* produces a sinusoidal **weekday‑weight vector** **w**∈ℝⁿ (row‑stochastic) that
distributes a resource budget among *n* groups.
*Parent B* computes a **regret‑weighted MinHash similarity** **s**∈[0,1] between two
texts.

The hybrid treats **s** as a *modulation scalar* for each group’s weight.
For a group *g* we evaluate `s(g) = Jaccard(MinHash(text_g), MinHash(reference))`.
The final allocation weight is  


ŵ_i = w_i · s_i
ŵ = ŵ / Σ ŵ          (renormalised)


Thus the sinusoidal topology of A is mathematically blended with the
entropy‑like similarity topology of B.  The VRAM‑aware GPU filter from A is
applied before the allocation, guaranteeing that only capable GPUs receive
the budget.

The public API consists of three core functions that demonstrate this
integration.
"""

import datetime as dt
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – weekday weight vector & VRAM‑aware GPU selection
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised sinusoidal weight vector for *groups* on a given weekday.

    Parameters
    ----------
    groups: sequence of group identifiers
    dow: weekday index where 0 = Sunday … 6 = Saturday

    Returns
    -------
    np.ndarray of shape (len(groups),) with sum == 1.0
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def vram_aware_gpu_selection(
    gpus: List[Dict[str, Any]],
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> List[Dict[str, Any]]:
    """
    Keep only GPUs whose free VRAM satisfies ``budget_mb + reserve_mb``.
    """
    needed = budget_mb + reserve_mb
    return [gpu for gpu in gpus if gpu.get("memory.free", 0) >= needed]


# ----------------------------------------------------------------------
# Parent B – MinHash based regret‑weighted similarity
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    return sorted([_hash(i, t) for i, t in enumerate(toks)])[:k]


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """MinHash signature of a text (lower‑cased word tokens)."""
    words = re.findall(r"\b\w+\b", text.lower())
    return signature(words, k=k)


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Classic Jaccard index between two integer signatures."""
    set1, set2 = set(sig1), set(sig2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def regret_weighted_similarity(text1: str, text2: str, k: int = 64) -> float:
    """
    Regret‑weighted similarity = Jaccard similarity of MinHash signatures.
    """
    return jaccard_similarity(minhash_for_text(text1, k), minhash_for_text(text2, k))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def modulated_weights(
    groups: Sequence[str],
    date: dt.date,
    reference_text: str,
    k: int = 64,
) -> np.ndarray:
    """
    Combine the sinusoidal weekday weights with regret‑weighted similarity.

    For each group ``g`` we compute
        s_g = regret_weighted_similarity(g, reference_text)
    and then form the element‑wise product with the weekday vector.
    The result is renormalised to sum to 1.
    """
    dow = (date.weekday() + 1) % 7  # 0 = Sunday … 6 = Saturday
    base_weights = weekday_weight_vector(groups, dow)

    # similarity per group (treat group name as a short text)
    sims = np.array(
        [regret_weighted_similarity(g, reference_text, k=k) for g in groups],
        dtype=np.float64,
    )
    # element‑wise modulation
    raw = base_weights * sims
    if raw.sum() == 0:
        # fallback to uniform distribution if everything zero
        return np.full_like(raw, 1.0 / len(raw))
    return raw / raw.sum()


def hybrid_allocate(
    total_units: float,
    date: dt.date,
    reference_text: str,
    groups: Sequence[str] = GROUPS,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    gpus: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Allocate ``total_units`` across *groups* using the modulated weight vector.
    Returns a mapping:
        {
            "allocation": {group: units, ...},
            "selected_gpus": [...],
            "weights": np.ndarray,
        }
    """
    if gpus is None:
        gpus = []

    selected = vram_aware_gpu_selection(gpus, budget_mb, reserve_mb)
    weights = modulated_weights(groups, date, reference_text)

    allocation = {g: float(total_units) * float(w) for g, w in zip(groups, weights)}
    return {
        "allocation": allocation,
        "selected_gpus": selected,
        "weights": weights,
    }


def hybrid_vector_literal(
    text: str,
    reference_text: str,
    k: int = 64,
) -> str:
    """
    Produce a numeric literal for *text* whose embedding magnitude is
    scaled by the regret‑weighted similarity to ``reference_text``.
    The scaling factor is the same similarity used in the allocation step.
    """
    similarity = regret_weighted_similarity(text, reference_text, k=k)
    embedding = [ord(c) for c in text]
    # Apply similarity as a linear gain to each component
    modulated = [float(v) * similarity for v in embedding]
    # Normalise to the 0‑1 range using 65535 (max 16‑bit char code)
    normalized = [v / 65535.0 for v in modulated]
    return "[" + ",".join(f"{x:.8f}" for x in normalized) + "]"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)

    # Dummy GPU list
    dummy_gpus = [
        {"id": 0, "memory.free": 16384},
        {"id": 1, "memory.free": 4096},
        {"id": 2, "memory.free": 2048},
    ]

    today = dt.date.today()
    ref = "The quick brown fox jumps over the lazy dog."

    result = hybrid_allocate(
        total_units=1000.0,
        date=today,
        reference_text=ref,
        gpus=dummy_gpus,
    )

    print("=== Allocation Result ===")
    for grp, units in result["allocation"].items():
        print(f"{grp:12s}: {units:8.3f} units")
    print("\nSelected GPUs:", [gpu["id"] for gpu in result["selected_gpus"]])
    print("\nWeight vector:", result["weights"])

    # Vector literal demonstration
    sample_text = "Hello, world!"
    lit = hybrid_vector_literal(sample_text, ref)
    print("\nHybrid vector literal for sample text:")
    print(lit)

    sys.exit(0)