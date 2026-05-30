# DARWIN HAMMER — match 2663, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s7.py (gen3)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s3.py (gen2)
# born: 2026-05-29T23:43:22Z

"""Hybrid Module combining Morphology-based biomechanics (Parent A) with
regex‑driven textual risk assessment (Parent B).

Mathematical bridge:
Both parents expose vector‑like structures.
- Parent A provides a scalar “recovery priority” ρ ∈ [0,1] derived from
  morphology (righting_time_index normalized).
- Parent B provides a weighted feature vector w ∈ ℝ⁹ and a count vector
  c ∈ ℕ⁹ obtained from regex matches.

The hybrid score is defined as the bilinear form

    S = ρ · (w · c)

where “·” is the standard dot product. ρ scales the textual risk score
according to the physical recovery capability of the subject,
producing a unified decision metric.

The module implements three core functions that realise this fusion,
plus a tiny ModelPool utility from Parent A for demonstration."""
import math
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – biomechanics utilities
# ----------------------------------------------------------------------


class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier


class ModelPool:
    """Simple RAM‑aware container for ModelTier objects."""

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
    """Geometric sphericity: cubic root of volume divided by length."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width)/(2·height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical right‑ing time proxy."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority ∈ [0,1] from righting_time_index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent B – regex‑driven feature extraction and weighting
# ----------------------------------------------------------------------


# Compile regex patterns (identical to Parent B)
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|"
    r"filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|"
    r"tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|"
    r"rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|"
    r"crisis|collapse)\b",
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

# Positive weights (aligned with feature order)
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)


def _count_matches(pattern: re.Pattern, text: str) -> int:
    """Return number of non‑overlapping matches of pattern in text."""
    return len(pattern.findall(text))


def extract_feature_counts(text: str) -> np.ndarray:
    """Return a length‑9 integer vector with counts for each feature."""
    patterns = [
        EVIDENCE_RE,
        PLANNING_RE,
        DELAY_RE,
        SUPPORT_RE,
        BOUNDARY_RE,
        OUTCOME_RE,
        IMPULSIVE_RE,
        SCARCITY_RE,
        RISK_RE,
    ]
    counts = [_count_matches(p, text) for p in patterns]
    return np.array(counts, dtype=np.int64)


# ----------------------------------------------------------------------
# Hybrid core – mathematical fusion of the two parents
# ----------------------------------------------------------------------


def compute_hybrid_score(morph: Morphology, text: str) -> float:
    """
    Compute the fused score S = ρ · (w·c).

    Parameters
    ----------
    morph: Morphology
        Physical description used to obtain recovery priority ρ.
    text: str
        Input text to be analysed by the regex feature extractor.

    Returns
    -------
    float
        Hybrid risk/priority score.
    """
    rho = recovery_priority(morph)                     # scalar ∈ [0,1]
    counts = extract_feature_counts(text)             # vector c
    weighted_sum = float(np.dot(_POSITIVE_WEIGHTS, counts))  # scalar w·c
    return rho * weighted_sum


def decide_action(morph: Morphology, text: str,
                  threshold: float = 500.0) -> Tuple[bool, float]:
    """
    Decide whether intervention is required.

    Returns a tuple (intervene, score) where ``intervene`` is True
    if the hybrid score exceeds ``threshold``.
    """
    score = compute_hybrid_score(morph, text)
    intervene = score >= threshold
    return intervene, score


def simulate_model_pool(models: List[Tuple[str, int, str]]) -> ModelPool:
    """
    Load a list of models into a ModelPool, demonstrating compatibility
    of the biomechanics side with the hybrid workflow.

    Each model is a tuple (name, ram_mb, tier).
    """
    pool = ModelPool()
    for name, ram, tier in models:
        model = ModelTier(name=name, ram_mb=ram, tier=tier)
        try:
            pool.load(model)
        except RuntimeError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
    return pool


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example morphology (a small robotic unit)
    example_morph = Morphology(length=0.4, width=0.3, height=0.2, mass=2.5)

    # Example textual input containing various keywords
    example_text = """
    I have evidence that the plan failed. We need to pause and wait for tomorrow.
    Please call a friend for support. No contact with the risky area.
    The issue is fixed and verified.
    """

    score = compute_hybrid_score(example_morph, example_text)
    intervene, final_score = decide_action(example_morph, example_text, threshold=800.0)

    print(f"Hybrid score: {score:.2f}")
    print(f"Decision – intervene? {intervene} (final score {final_score:.2f})")

    # Demonstrate ModelPool usage
    demo_models = [
        ("vision_core", 1500, "A"),
        ("speech_recognizer", 2000, "B"),
        ("navigation", 1800, "C"),
        ("heavy_model", 1200, "D")  # may exceed ceiling depending on previous loads
    ]
    pool = simulate_model_pool(demo_models)
    print(f"Loaded models ({len(pool.loaded)}): {list(pool.loaded.keys())}")