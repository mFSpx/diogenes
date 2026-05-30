# DARWIN HAMMER — match 576, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_percyphon_hyb_honeybee_store_m100_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s1.py (gen4)
# born: 2026-05-29T23:29:52Z

"""Hybrid Algorithm: Morphic-Stylometric Resource Optimizer

Parents:
- hybrid_hybrid_percyphon_hyb_honeybee_store_m100_s0 (procedural slots, morphology indices, store dynamics)
- hybrid_hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s1 (stylometric categorisation and social‑interaction optimisation)

Mathematical Bridge:
The bridge is built on the observation that the *sphericity* (S) and *flatness* (F) indices of a
Morphology can be interpreted as shape‑derived confidence weights.  These weights are used to
scale the contribution of stylometric category frequencies (C_i) when computing a global
stylometric score Σ w_i·C_i.  The resulting score modulates the classic store update
Δ = α·Σ(inflow) – β·Σ(outflow) by an additive term γ·Score·S·F, producing a hybrid resource
allocation rule that couples geometric morphology with language‑based optimisation.

The module provides three core functions that demonstrate this fusion:
1. `compute_morphology_indices` – returns S and F.
2. `stylometric_score` – computes a weighted category score using the indices.
3. `hybrid_update` – updates the store using both the original store dynamics and the
   morphology‑stylometry term.
"""

import hashlib
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – procedural slot & morphology utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> Tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = [
        "ledger",
        "runner",
        "witness",
        "archivist",
        "carrier",
        "scribe",
    ][int(h[10:12], 16) % 6]
    return name, alias, persona


def generate_procedural_slot(seed: str, idx: int) -> ProceduralSlot:
    name, alias, persona = _slot_name(seed, idx)
    uuid = _uuid_from_sha256(f"{seed}:{idx}")
    ternary_offset = (idx * 3) % 7
    return ProceduralSlot(
        slot_index=idx,
        name=name,
        alias=alias,
        persona=persona,
        uuid=uuid,
        ternary_offset=ternary_offset,
    )


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: (V)^(1/3) / longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness: (average of two largest dimensions) / (2 * smallest dimension)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    dims = sorted([length, width, height], reverse=True)
    return (dims[0] + dims[1]) / (2.0 * dims[2])


def compute_morphology_indices(morph: Morphology) -> Tuple[float, float]:
    """Return (sphericity, flatness) for a given Morphology."""
    s = sphericity_index(morph.length, morph.width, morph.height)
    f = flatness_index(morph.length, morph.width, morph.height)
    return s, f


def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """Classic Darvin‑Hammer store dynamics."""
    delta = alpha * sum(inflow) - beta * sum(outflow)
    return max(0.0, store + dt * delta), delta


# ----------------------------------------------------------------------
# Parent B – stylometric categorisation utilities
# ----------------------------------------------------------------------


FUNCTION_CATS: Dict[str, set[str]] = {
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
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _clean_token(tok: str) -> str:
    return tok.strip(PUNCT).lower()


def words(text: str) -> List[str]:
    """Very lightweight tokeniser – splits on whitespace and strips punctuation."""
    return [_clean_token(tok) for tok in text.split() if _clean_token(tok)]


def category_frequencies(tokens: List[str]) -> Dict[str, int]:
    """Count how many tokens belong to each FUNCTION_CATS category."""
    freq: Dict[str, int] = {cat: 0 for cat in FUNCTION_CATS}
    for tok in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if tok in vocab:
                freq[cat] += 1
                break  # each token counted at most once
    return freq


def stylometric_score(text: str, weight_map: Dict[str, float]) -> float:
    """
    Compute a weighted stylometric score.

    score = Σ_i weight_map[i] * count_i
    where i iterates over FUNCTION_CATS keys.
    """
    toks = words(text)
    freqs = category_frequencies(toks)
    score = sum(weight_map.get(cat, 0.0) * cnt for cat, cnt in freqs.items())
    return score


# ----------------------------------------------------------------------
# Hybrid Core – bridging morphology & stylometry into store dynamics
# ----------------------------------------------------------------------


def hybrid_weight_map(morph: Morphology) -> Dict[str, float]:
    """
    Derive a weight map for stylometric categories from morphology.

    The sphericity (S) boosts categories related to structure (article, preposition),
    while flatness (F) boosts fluid categories (adverb_common, pronoun).  The weights
    are normalised to sum to 1.
    """
    S, F = compute_morphology_indices(morph)
    raw = {
        "article": S,
        "preposition": S * 0.8,
        "adverb_common": F,
        "pronoun": F * 0.9,
        "auxiliary": (S + F) * 0.5,
        "conjunction": (S + F) * 0.3,
        "negation": 0.1,
        "quantifier": 0.1,
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def hybrid_stylometric_term(text: str, morph: Morphology) -> float:
    """
    Compute the stylometric term that will be injected into the store update.

    term = γ * Score * S * F
    where γ is a fixed scaling constant (set to 0.05).
    """
    γ = 0.05
    weight_map = hybrid_weight_map(morph)
    score = stylometric_score(text, weight_map)
    S, F = compute_morphology_indices(morph)
    return γ * score * S * F


def hybrid_update(
    store: float,
    inflow: List[float],
    outflow: List[float],
    morph: Morphology,
    text: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """
    Perform a store update that merges the classic Darvin‑Hammer dynamics with a
    morphology‑stylometry driven term.

    Returns (new_store, total_delta) where total_delta = classic_delta + stylometric_term.
    """
    classic_new, classic_delta = update_store(store, inflow, outflow, alpha, beta, dt)
    styl_term = hybrid_stylometric_term(text, morph)
    new_store = max(0.0, classic_new + dt * styl_term)
    total_delta = classic_delta + styl_term
    return new_store, total_delta


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------


def demo_hybrid_cycle():
    """Run a short demo cycle showcasing the hybrid algorithm."""
    # Create a random morphology
    rng = random.Random(42)
    morph = Morphology(
        length=rng.uniform(1.0, 10.0),
        width=rng.uniform(1.0, 10.0),
        height=rng.uniform(1.0, 10.0),
        mass=rng.uniform(0.5, 5.0),
    )

    # Sample text (could be any user‑generated content)
    sample_text = (
        "I quickly ran through the forest, but the birds were not singing. "
        "All the runners have already left, yet some carriers still carry the ledger."
    )

    # Initial store and flows
    store = 100.0
    inflow = [rng.uniform(0, 5) for _ in range(3)]
    outflow = [rng.uniform(0, 4) for _ in range(2)]

    # Perform hybrid update
    new_store, delta = hybrid_update(
        store,
        inflow,
        outflow,
        morph,
        sample_text,
        alpha=1.2,
        beta=0.9,
        dt=1.0,
    )

    # Produce a procedural slot for illustration
    slot = generate_procedural_slot("demo-seed", 7)

    # Output results (purely for demo – not part of core API)
    print("Morphology:", morph)
    print("Sphericity, Flatness:", compute_morphology_indices(morph))
    print("Weight map:", hybrid_weight_map(morph))
    print("Classic inflow:", inflow, "outflow:", outflow)
    print("Stylometric term added:", delta - (alpha * sum(inflow) - beta * sum(outflow)))
    print("Store before:", store, "after:", new_store)
    print("Procedural slot:", slot.as_dict())


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_cycle()