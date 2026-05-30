# DARWIN HAMMER — match 2663, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s7.py (gen3)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s3.py (gen2)
# born: 2026-05-29T23:43:22Z

"""Hybrid Algorithm combining Morphology-based Recovery Metrics (Parent A) with
Regex‑driven Text Feature Scoring (Parent B).

Mathematical Bridge
-------------------
Parent A provides a scalar *recovery priority* `R ∈ [0,1]` derived from the
morphology of a physical entity:


R = clip( righting_time_index(m) / max_index )


Parent B produces a feature count vector **c** ∈ ℕ⁹ from a text string using
nine regular expressions (evidence, planning, …, risk) and a weight vector
**w** ∈ ℝ⁹ (the `_POSITIVE_WEIGHTS`). The raw text score is the dot product
`S = w·c`.

The hybrid algorithm fuses the two by scaling the text score with the
recovery priority, yielding a unified decision metric


H = R * S


Thus the morphological state modulates the influence of textual evidence,
creating a single scalar that can drive downstream actions (e.g. model
loading, alerting, etc.). The implementation below provides three public
functions that expose this combined computation."""


import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology and ModelPool
# ----------------------------------------------------------------------


class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier


class ModelPool:
    """Simple RAM‑constrained pool for loading/unloading model descriptors."""

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        if self.is_loaded(model.name):
            return
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Scalar in [0,1] describing how quickly the entity can self‑right."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent B – Regex feature extraction and weighted scoring
# ----------------------------------------------------------------------


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array(
    [1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64
)


def _regex_for_name(name: str) -> re.Pattern:
    """Map feature name to compiled regex."""
    return {
        "evidence": EVIDENCE_RE,
        "planning": PLANNING_RE,
        "delay": DELAY_RE,
        "support": SUPPORT_RE,
        "boundary": BOUNDARY_RE,
        "outcome": OUTCOME_RE,
        "impulsive": IMPULSIVE_RE,
        "scarcity": SCARCITY_RE,
        "risk": RISK_RE,
    }[name]


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------


def compute_morphology_metrics(m: Morphology) -> Dict[str, float]:
    """
    Return all intermediate morphology metrics as a dictionary.
    """
    s = sphericity_index(m.length, m.width, m.height)
    f = flatness_index(m.length, m.width, m.height)
    rti = righting_time_index(m)
    rp = recovery_priority(m)
    return {
        "sphericity": s,
        "flatness": f,
        "righting_time_index": rti,
        "recovery_priority": rp,
    }


def text_feature_vector(text: str) -> np.ndarray:
    """
    Count occurrences of each regex feature in `text` and return a length‑9 vector.
    """
    counts = []
    for name in _FEATURE_ORDER:
        pattern = _regex_for_name(name)
        matches = pattern.findall(text)
        counts.append(len(matches))
    return np.array(counts, dtype=np.int64)


def hybrid_evaluation(m: Morphology, text: str) -> Dict[str, Any]:
    """
    Combine morphology‑derived recovery priority with regex‑based textual scoring.

    Returns a dictionary containing:
        - morphology metrics
        - raw feature counts
        - raw text score (dot product)
        - hybrid score H = R * S
    """
    morph_metrics = compute_morphology_metrics(m)
    R = morph_metrics["recovery_priority"]

    counts = text_feature_vector(text)
    S = int(_POSITIVE_WEIGHTS.dot(counts))

    H = R * S

    return {
        "morphology": morph_metrics,
        "feature_counts": dict(zip(_FEATURE_ORDER, counts.tolist())),
        "text_score": S,
        "hybrid_score": H,
    }


def conditional_model_load(pool: ModelPool, hybrid_score: float, threshold: float = 500.0) -> None:
    """
    Demonstrates a decision based on the hybrid score:
    - If the score exceeds `threshold`, attempt to load a high‑priority model.
    - Otherwise, load a low‑priority fallback model.
    """
    high = ModelTier(name="high_priority_model", ram_mb=2500, tier="high")
    low = ModelTier(name="fallback_model", ram_mb=800, tier="low")

    if hybrid_score >= threshold:
        model = high
    else:
        model = low

    try:
        pool.load(model)
        print(f"Loaded model '{model.name}' (tier={model.tier}, ram={model.ram_mb} MB).")
    except RuntimeError as e:
        print(f"Failed to load model '{model.name}': {e}", file=sys.stderr)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a representative morphology
    demo_morph = Morphology(length=2.5, width=1.8, height=0.9, mass=75.0)

    # Sample text containing a mixture of features
    demo_text = (
        "I have evidence that the plan failed. "
        "We need to wait until tomorrow before we can verify the outcome. "
        "Please call a friend for support. "
        "If things get risky, we must set a boundary."
    )

    # Run the hybrid evaluation
    result = hybrid_evaluation(demo_morph, demo_text)

    print("Hybrid Evaluation Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Use the hybrid score to decide which model to load
    pool = ModelPool(ram_ceiling_mb=4000)
    conditional_model_load(pool, result["hybrid_score"], threshold=500.0)

    # Show currently loaded models
    print("Currently loaded models:", list(pool.loaded.keys()))