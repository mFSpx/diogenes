# DARWIN HAMMER — match 250, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s0.py (gen3)
# parent_b: workshare_allocator.py (gen0)
# born: 2026-05-29T23:27:57Z

import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Constants & utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _pct(value: float) -> float:
    """Round a float to six decimal places for stable JSON‑compatible output."""
    return round(float(value), 6)


def _safe_split_words(text: str) -> List[str]:
    """Return a list of alphabetic lower‑cased words, ignoring punctuation."""
    return [w.lower() for w in text.split() if w.isalpha()]


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class WorkshareLane:
    group: str
    llm_units: float
    llm_share_pct: float
    proof_required: bool


@dataclass(frozen=True)
class AllocationResult:
    total_units: float
    deterministic_target_pct: float
    deterministic_units: float
    llm_units: float
    lanes: Tuple[WorkshareLane, ...]


# ----------------------------------------------------------------------
# Core algorithms
# ----------------------------------------------------------------------
def allocate_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> AllocationResult:
    """
    Split `total_units` into a deterministic portion and a LLM‑share portion.
    The LLM share is distributed equally among the supplied `groups`.
    """
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if total_units < 0:
        raise ValueError("total_units must be non‑negative")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups) if groups else 0.0

    lanes = tuple(
        WorkshareLane(
            group=g,
            llm_units=_pct(per_group),
            llm_share_pct=_pct(100.0 / len(groups)) if groups else 0.0,
            proof_required=True,
        )
        for g in groups
    )

    return AllocationResult(
        total_units=_pct(total_units),
        deterministic_target_pct=_pct(deterministic_target_pct),
        deterministic_units=_pct(deterministic_units),
        llm_units=_pct(llm_units),
        lanes=lanes,
    )


def stylometry_vector(text: str) -> np.ndarray:
    """
    Produce a normalized frequency vector over the functional‑category
    dictionary `FUNCTION_CATS`.  If the text contains no alphabetic words,
    a zero vector is returned (avoiding division‑by‑zero).
    """
    words = _safe_split_words(text)
    if not words:
        return np.zeros(len(FUNCTION_CATS), dtype=float)

    vec = np.zeros(len(FUNCTION_CATS), dtype=float)
    cat_index = {cat: idx for idx, cat in enumerate(FUNCTION_CATS.keys())}
    for w in words:
        for cat, vocab in FUNCTION_CATS.items():
            if w in vocab:
                vec[cat_index[cat]] += 1.0
    return vec / len(words)


def _default_category_to_group_weights() -> np.ndarray:
    """
    Create a simple uniform weighting matrix (categories × groups).
    More sophisticated schemes could be injected by the caller.
    """
    n_cats = len(FUNCTION_CATS)
    n_groups = len(GROUPS)
    return np.full((n_cats, n_groups), 1.0 / n_groups, dtype=float)


def compute_modulation_factors(
    stylometry: np.ndarray,
    weight_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Convert a stylometry vector into a per‑group modulation factor.
    The weight matrix maps categories → groups.  If omitted, a uniform
    matrix is used, guaranteeing that the factors sum to 1.
    """
    if weight_matrix is None:
        weight_matrix = _default_category_to_group_weights()
    if weight_matrix.shape != (len(FUNCTION_CATS), len(GROUPS)):
        raise ValueError(
            f"weight_matrix must be of shape ({len(FUNCTION_CATS)},{len(GROUPS)})"
        )
    # Linear combination of category frequencies for each group
    raw = stylometry @ weight_matrix  # shape: (len(GROUPS),)
    # Guard against all‑zero stylometry (returns uniform distribution)
    if raw.sum() == 0:
        return np.full(len(GROUPS), 1.0 / len(GROUPS), dtype=float)
    return raw / raw.sum()


def hybrid_operation(
    text: str,
    total_units: float,
    *,
    deterministic_target_pct: float = 90.0,
    weight_matrix: np.ndarray = None,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid computation:
      1. Compute stylometry.
      2. Allocate deterministic vs. LLM workshare.
      3. Modulate each lane according to stylometry → group mapping.
    Returns a dictionary that is easy to serialise.
    """
    stylometry = stylometry_vector(text)
    allocation = allocate_workshare(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
    )
    modulation = compute_modulation_factors(stylometry, weight_matrix)

    modulated_lanes = []
    for lane, factor in zip(allocation.lanes, modulation):
        mod_lane = WorkshareLane(
            group=lane.group,
            llm_units=_pct(lane.llm_units * factor),
            llm_share_pct=lane.llm_share_pct,
            proof_required=lane.proof_required,
        )
        modulated_lanes.append(mod_lane)

    return {
        "text": text,
        "stylometry": stylometry.tolist(),
        "allocation": asdict(allocation),
        "modulated_lanes": [asdict(l) for l in modulated_lanes],
    }


def summarize_hybrid_operation(hybrid_output: Dict[str, Any]) -> str:
    """
    Human‑readable one‑liner summary for debugging / logging.
    """
    lines = [
        f"Text: {hybrid_output['text']}",
        f"Stylometry (categories): {hybrid_output['stylometry']}",
        f"Allocation: total={hybrid_output['allocation']['total_units']}, "
        f"deterministic={hybrid_output['allocation']['deterministic_units']}, "
        f"llm={hybrid_output['allocation']['llm_units']}",
        "Modulated Lanes:",
    ]
    for lane in hybrid_output["modulated_lanes"]:
        lines.append(
            f"  - {lane['group']}: units={lane['llm_units']}, share%={lane['llm_share_pct']}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Demo / simple CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "This is a sample text for demonstration purposes."
    result = hybrid_operation(sample_text, total_units=100.0)
    print(summarize_hybrid_operation(result))