# DARWIN HAMMER — match 576, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_percyphon_hyb_honeybee_store_m100_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s1.py (gen4)
# born: 2026-05-29T23:29:52Z

"""Hybrid Algorithm integrating Morphology‑based Resource Dynamics (Parent A) with Stylometric Interaction (Parent B).

Parent A contributes geometric indices (sphericity, flatness) and a linear store update rule.
Parent B contributes a stylometric categorisation of text into functional word classes.

Mathematical bridge:
    * The geometric indices are used to adapt the α (inflow gain) and β (outflow loss) coefficients
      of the store dynamics: α = 1 + sphericity, β = 1 + flatness.
    * The stylometric profile yields a scalar interaction factor ϕ ∈ [0.5, 1.5] that scales the
      raw inflow/outflow vectors before they enter the store update.
    * The final hybrid update is therefore
          Δ = α·Σ(ϕ·inflow) – β·Σ((2‑ϕ)·outflow)
          store_{new} = max(0, store + dt·Δ)
    This fuses the morphology‑driven coefficients with the text‑driven social interaction
    scaling, yielding a unified system that reacts to both physical shape and linguistic
    context.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Geometry & Store primitives
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
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][
        int(h[10:12], 16) % 6
    ]
    return name, alias, persona


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity – ratio of geometric mean to the longest edge."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Geometric flatness – average of the two shorter edges divided by the longest."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    edges = sorted([length, width, height])
    return (edges[0] + edges[1]) / (2.0 * edges[2])


def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """Linear store dynamics."""
    delta = alpha * sum(inflow) - beta * sum(outflow)
    return max(0.0, store + dt * delta), delta


# ----------------------------------------------------------------------
# Parent B – Stylometric categorisation
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
    """Very lightweight tokenizer – split on whitespace and strip punctuation."""
    return [_clean_token(tok) for tok in text.split() if _clean_token(tok)]


def stylometric_profile(text: str) -> Dict[str, float]:
    """
    Return normalized frequencies of each functional word category.
    Frequencies sum to 1 across all recognised tokens.
    """
    toks = words(text)
    total = len(toks) or 1
    cat_counts: Dict[str, int] = {cat: 0 for cat in FUNCTION_CATS}
    for tok in toks:
        for cat, vocab in FUNCTION_CATS.items():
            if tok in vocab:
                cat_counts[cat] += 1
                break
    return {cat: cnt / total for cat, cnt in cat_counts.items()}


# ----------------------------------------------------------------------
# Hybrid layer – mathematical fusion
# ----------------------------------------------------------------------
def compute_morphology_indices(morph: Morphology) -> Tuple[float, float]:
    """Return (sphericity, flatness) for a given morphology."""
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flt = flatness_index(morph.length, morph.width, morph.height)
    return sph, flt


def interaction_factor(profile: Dict[str, float]) -> float:
    """
    Derive a scalar φ ∈ [0.5, 1.5] from stylometric frequencies.
    The heuristic favours pronoun usage (social interaction) and penalises
    preposition abundance (spatial anchoring). The formula is:

        φ = 1 + 0.5·(pronoun_freq – preposition_freq)

    The result is clamped to the interval [0.5, 1.5].
    """
    pron = profile.get("pronoun", 0.0)
    prep = profile.get("preposition", 0.0)
    phi = 1.0 + 0.5 * (pron - prep)
    return max(0.5, min(1.5, phi))


def hybrid_update_store(
    store: float,
    morph: Morphology,
    text: str,
    inflow: List[float],
    outflow: List[float],
    dt: float = 1.0,
) -> Tuple[float, float]:
    """
    Hybrid store update that blends geometry‑driven coefficients with
    stylometric interaction scaling.

    Steps
    -----
    1. Compute sphericity (σ) and flatness (φ_g).
    2. Set α = 1 + σ, β = 1 + φ_g.
    3. Compute stylometric profile → interaction scalar ϕ ∈ [0.5, 1.5].
    4. Scale inflow by ϕ and outflow by (2‑ϕ) (conserves total weighting).
    5. Apply the classic linear store update.
    """
    sph, flt = compute_morphology_indices(morph)
    alpha = 1.0 + sph
    beta = 1.0 + flt

    profile = stylometric_profile(text)
    phi = interaction_factor(profile)

    inflow_scaled = [phi * v for v in inflow]
    outflow_scaled = [(2.0 - phi) * v for v in outflow]

    new_store, delta = update_store(store, inflow_scaled, outflow_scaled, alpha, beta, dt)
    return new_store, delta


def generate_procedural_slots(
    seed: str, count: int, morph: Morphology
) -> List[ProceduralSlot]:
    """
    Produce a list of ProceduralSlot objects whose ternary_offset is
    influenced by the morphology indices (σ, φ_g). The offset is defined as

        offset = int( (σ + φ_g) * 10 ) % 3

    This creates a deterministic yet morphology‑aware distribution of offsets.
    """
    sigma, phi_g = compute_morphology_indices(morph)
    base_offset = int((sigma + phi_g) * 10) % 3

    slots: List[ProceduralSlot] = []
    for i in range(count):
        name, alias, persona = _slot_name(seed, i)
        uid = _uuid_from_sha256(f"{seed}:{i}")
        # Introduce a small per‑slot variation using the index
        offset = (base_offset + i) % 3
        slots.append(
            ProceduralSlot(
                slot_index=i,
                name=name,
                alias=alias,
                persona=persona,
                uuid=uid,
                ternary_offset=offset,
            )
        )
    return slots


def resource_allocation_score(morph: Morphology, text: str) -> float:
    """
    A composite metric that could guide higher‑level decisions.
    It is the dot product between the normalized morphology vector
    (length, width, height) and the stylometric frequency vector
    (ordered by FUNCTION_CATS keys). Both vectors are L2‑normalised.
    """
    vec_morph = np.array([morph.length, morph.width, morph.height], dtype=float)
    if np.linalg.norm(vec_morph) == 0:
        norm_morph = vec_morph
    else:
        norm_morph = vec_morph / np.linalg.norm(vec_morph)

    profile = stylometric_profile(text)
    vec_style = np.array([profile.get(cat, 0.0) for cat in FUNCTION_CATS], dtype=float)
    if np.linalg.norm(vec_style) == 0:
        norm_style = vec_style
    else:
        norm_style = vec_style / np.linalg.norm(vec_style)

    # Pad the shorter vector with zeros to make lengths match
    if norm_morph.size < norm_style.size:
        norm_morph = np.pad(norm_morph, (0, norm_style.size - norm_morph.size))
    elif norm_style.size < norm_morph.size:
        norm_style = np.pad(norm_style, (0, norm_morph.size - norm_style.size))

    return float(np.dot(norm_morph, norm_style))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample morphology
    morph = Morphology(length=4.2, width=2.8, height=1.5, mass=12.0)

    # Sample text
    sample_text = (
        "I think therefore I am. The quick brown fox jumps over the lazy dog, "
        "but not all foxes are quick."
    )

    # Store dynamics
    store = 100.0
    inflow = [5.0, 3.2, 1.1]
    outflow = [2.5, 4.0]

    # Hybrid update
    new_store, delta = hybrid_update_store(
        store, morph, sample_text, inflow, outflow, dt=1.0
    )
    print(f"Hybrid update → new store: {new_store:.2f}, delta: {delta:.2f}")

    # Generate procedural slots
    slots = generate_procedural_slots("seed-xyz", 5, morph)
    for s in slots:
        print(s)

    # Allocation score
    score = resource_allocation_score(morph, sample_text)
    print(f"Resource allocation score: {score:.4f}")