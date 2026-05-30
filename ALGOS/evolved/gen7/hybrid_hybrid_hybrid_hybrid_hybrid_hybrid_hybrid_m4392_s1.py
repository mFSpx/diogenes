# DARWIN HAMMER — match 4392, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1270_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1216_s1.py (gen6)
# born: 2026-05-29T23:55:20Z

"""Hybrid Algorithm integrating Parent A (resource extraction & regret weighting) and Parent B (workshare lane allocation).

Mathematical Bridge:
- Parent A produces a 2‑dimensional ResourceVector (load, privacy) derived from a normalized cue vector.
- Parent B defines a set of LLM groups (WORKSHARE) and a router matrix that maps cues to group allocations.
- The bridge is a linear transformation: the cue‑derived vector is expanded to a 3‑dimensional “cue space” (evidence, planning, delay) and multiplied by a router matrix **R** (3 × G, G = number of groups) to obtain raw allocation scores.
- These scores are then modulated by a regret‑weighted factor λ ∈ [0,1] (from Parent A) producing the final allocation probabilities.
- The router matrix itself can be updated via a regret‑weighted strategy, blending the previous router with a gradient‑like correction derived from observed outcomes.

The resulting system jointly extracts textual cues, converts them into resource demands, allocates LLM units across groups, and iteratively adapts the allocation policy using regret feedback.
"""

import re
import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A structures
# ----------------------------------------------------------------------
@dataclass
class ResourceVector:
    load: float
    privacy: float


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


def extract_master_vector(text: str) -> Dict[str, float]:
    """Generate a pseudo‑random master vector for demonstration."""
    if not text.strip():
        return {}
    return {
        "visceral_ratio": random.random(),
        "tech_ratio": random.random(),
        "legal_osint_ratio": random.random(),
        "ledger_density": random.random(),
        "recursion_score": random.random(),
    }


def extract_text_features(text: str, master_vector: Dict[str, float]) -> ResourceVector:
    """Map textual cues to a 2‑dimensional resource demand."""
    evidence_re = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    )
    planning_re = re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    )
    delay_re = re.compile(r"\b(?:pause|sleep|wait|tomorrow|tonight|next|week|month|year)\b", re.I)

    matches = {
        "evidence": evidence_re.findall(text),
        "planning": planning_re.findall(text),
        "delay": delay_re.findall(text),
    }

    cue_vector = np.array([len(matches["evidence"]), len(matches["planning"]), len(matches["delay"])], dtype=float)
    norm = np.linalg.norm(cue_vector)
    scaled_cue_vector = cue_vector / norm if norm > 0 else cue_vector

    load = float(
        np.dot(
            scaled_cue_vector,
            np.array(
                [
                    master_vector.get("visceral_ratio", 0.0),
                    master_vector.get("tech_ratio", 0.0),
                    master_vector.get("legal_osint_ratio", 0.0),
                ]
            ),
        )
    )
    privacy = float(
        np.dot(
            scaled_cue_vector,
            np.array(
                [
                    master_vector.get("ledger_density", 0.0),
                    master_vector.get("recursion_score", 0.0),
                    master_vector.get("recursion_score", 0.0),
                ]
            ),
        )
    )
    return ResourceVector(load, privacy)


# ----------------------------------------------------------------------
# Parent B structures
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"
GROUPS = ("codex", "groq", "cohere", "local_models")


def words(text: str) -> List[str]:
    """Tokenise a string into lower‑case alphabetic words."""
    return [word.lower() for word in text.split() if word.isalpha()]


@dataclass
class WorkshareLane:
    group: str
    llm_units: float = 0.0


# ----------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------
def build_router_matrix(master_vector: Dict[str, float]) -> np.ndarray:
    """
    Construct a 3 × G router matrix R that maps cue space (evidence, planning, delay)
    to the GROUPS workshare lanes.

    The entries are derived from the master vector to keep a deterministic yet
    stochastic relationship between the two parent algorithms.
    """
    G = len(GROUPS)
    rng = np.random.default_rng(seed=int(sum(master_vector.values()) * 1e6) % (2**32 - 1))
    # Base random matrix
    R = rng.random((3, G))
    # Scale rows by specific master‑vector components to embed Parent A information
    scaling = np.array(
        [
            master_vector.get("visceral_ratio", 1.0),
            master_vector.get("tech_ratio", 1.0),
            master_vector.get("legal_osint_ratio", 1.0),
        ]
    ).reshape(3, 1)
    R = R * scaling
    # Normalise columns so each column sums to 1 (probability distribution per group)
    col_sums = R.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    R = R / col_sums
    return R


def regret_weighted_strategy(
    router: np.ndarray, regret: float, master_vector: Dict[str, float]
) -> np.ndarray:
    """
    Update the router matrix using a simple regret‑weighted rule.

    NewRouter = (1‑λ) * router + λ * Δ,
    where λ = regret (clipped to [0,1]) and Δ is a gradient‑like correction
    derived from the master vector (higher recursion_score pushes allocation toward
    groups with larger current share).
    """
    λ = max(0.0, min(1.0, regret))
    G = router.shape[1]

    # Gradient‑like term: push weight toward groups with larger recursion_score
    recursion = master_vector.get("recursion_score", 0.5)
    bias = np.full((3, G), recursion)
    Δ = bias / bias.sum(axis=0, keepdims=True)

    new_router = (1.0 - λ) * router + λ * Δ
    # Re‑normalise columns to retain stochasticity
    col_sums = new_router.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    new_router = new_router / col_sums
    return new_router


def allocate_resources(
    resource: ResourceVector, router: np.ndarray, regret: float
) -> Dict[str, WorkshareLane]:
    """
    Convert a ResourceVector into allocation probabilities across GROUPS.

    1. Form a cue vector C = [load, privacy, (load+privacy)/2] to obtain a 3‑dim input.
    2. Compute raw scores S = C · R   (shape: (G,)).
    3. Apply regret weighting: S' = (1‑λ)·S + λ·mean(S)   (λ = regret).
    4. Normalise to obtain a probability distribution and instantiate WorkshareLane objects.
    """
    λ = max(0.0, min(1.0, regret))

    # Step 1 – build a 3‑dim cue vector; the third component is a simple blend.
    cue = np.array([resource.load, resource.privacy, (resource.load + resource.privacy) / 2.0])
    # Step 2 – raw scores per group
    raw_scores = np.dot(cue, router)  # shape (G,)

    # Step 3 – regret smoothing towards the mean allocation
    mean_score = raw_scores.mean()
    adjusted = (1.0 - λ) * raw_scores + λ * mean_score

    # Step 4 – normalise to probabilities
    total = adjusted.sum()
    if total == 0:
        probs = np.full_like(adjusted, 1.0 / len(adjusted))
    else:
        probs = adjusted / total

    allocation = {
        grp: WorkshareLane(group=grp, llm_units=float(probs[i]))
        for i, grp in enumerate(GROUPS)
    }
    return allocation


def hybrid_process(text: str, regret: float = 0.1) -> Dict[str, WorkshareLane]:
    """
    End‑to‑end hybrid pipeline:
    - Extract master vector and resource demands from the input text.
    - Build (or retrieve) a router matrix.
    - Optionally update the router using regret weighting.
    - Allocate LLM units across groups.
    """
    master = extract_master_vector(text)
    resource = extract_text_features(text, master)
    router = build_router_matrix(master)
    router = regret_weighted_strategy(router, regret, master)
    allocation = allocate_resources(resource, router, regret)
    return allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The report includes evidence and a detailed plan. "
        "We will verify the source tomorrow and schedule a follow‑up."
    )
    alloc = hybrid_process(sample_text, regret=0.25)
    print("Hybrid allocation results:")
    for grp, lane in alloc.items():
        print(f"  {grp:12s} -> llm_units: {lane.llm_units:.4f}")
    # Simple sanity check: probabilities sum to ~1
    total_units = sum(lane.llm_units for lane in alloc.values())
    print(f"Total allocated units (should be 1.0): {total_units:.6f}")