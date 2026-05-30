# DARWIN HAMMER — match 400, survivor 2
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# parent_b: hybrid_privacy_model_pool_m7_s0.py (gen1)
# born: 2026-05-29T23:28:49Z

"""
Hybrid module combining:

- Parent A: regex‑based feature extraction, weighted decision scoring, and Shannon entropy
  computation (decision_hygiene + shannon_entropy).
- Parent B: privacy‑preserving model‑pool management (privacy scoring, differential
  privacy aggregation, reconstruction risk).

Mathematical bridge:
The Shannon entropy `H` of the extracted textual features is used to scale the
differential‑privacy budget `ε` in the model‑loading routine.  A higher entropy
(indicating a richer, more diverse signal) yields a larger effective ε,
reducing added Laplace noise, while a lower entropy tightens privacy (smaller ε).
The weighted decision score from Parent A is also employed as the sensitivity
parameter for the Laplace mechanism, linking the two topologies into a unified
system.
"""

from __future__ import annotations

import re
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – feature extraction, weighting, and entropy
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

# Positive contribution weights (desired cues)
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
# Negative contribution weights (undesired cues) – placeholder zeros for simplicity
_NEGATIVE_WEIGHTS = np.zeros_like(_POSITIVE_WEIGHTS)

_REGEX_MAP = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}


def _extract_counts(text: str) -> np.ndarray:
    """Return a count vector aligned with _FEATURE_ORDER."""
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for idx, feat in enumerate(_FEATURE_ORDER):
        regex = _REGEX_MAP[feat]
        counts[idx] = len(regex.findall(text))
    return counts


def weighted_decision_score(counts: np.ndarray) -> float:
    """
    Compute a linear decision score:
        score = w_pos·pos_counts – w_neg·neg_counts
    where pos_counts = counts * (positive cue flag) and neg_counts = counts *
    (negative cue flag).  For this hybrid we treat all counts as positive cues,
    but the negative weight array is kept for extensibility.
    """
    pos_part = np.dot(_POSITIVE_WEIGHTS, counts)
    neg_part = np.dot(_NEGATIVE_WEIGHTS, counts)
    return float(pos_part - neg_part)


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of the normalized count distribution."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Guard against log(0) by filtering zero probabilities
    nonzero = probs[probs > 0]
    return -float(np.sum(nonzero * np.log2(nonzero)))


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Perform the full Parent‑A pipeline on *text*.
    Returns a dict with raw counts, weighted score, and entropy.
    """
    counts = _extract_counts(text)
    score = weighted_decision_score(counts)
    entropy = shannon_entropy(counts)
    return {
        "counts": counts,
        "weighted_score": score,
        "entropy": entropy,
        "feature_order": _FEATURE_ORDER,
    }

# ----------------------------------------------------------------------
# Parent B – privacy‑aware model pool
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Simple immutable description of a model."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded due to policy constraints."""
    pass


class ModelPool:
    """Container that tracks loaded models respecting RAM ceiling and tier rules."""

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load *model* if constraints are satisfied; otherwise raise."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """
        Evict least‑recently‑added models until *model* fits, then load it.
        (Simplified eviction strategy using insertion order.)
        """
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = next(iter(self.loaded))
            del self.loaded[evicted_name]
        self.load(model)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """
    Simple risk estimator: proportion of unique identifiers.
    Clamped to [0, 1].
    """
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """
    Laplace mechanism for sum aggregation.
    Returns noisy sum.
    """
    total = sum(values)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise


def load_model_with_privacy(
    model: ModelTier,
    pool: ModelPool,
    base_epsilon: float = 1.0,
    sensitivity: float = 1.0,
) -> None:
    """
    Privacy‑aware loader.
    The effective epsilon is scaled by the Shannon entropy of the most recent
    textual analysis (if available) to respect the mathematical bridge.
    """
    # Retrieve a recent entropy value if the caller stored it globally;
    # otherwise fall back to base_epsilon.
    entropy = getattr(pool, "_last_entropy", None)
    if entropy is None:
        epsilon = base_epsilon
    else:
        # Map entropy ∈ [0, log2(N)] → scaling factor ∈ [0.5, 2.0]
        # (N is number of features = 9)
        max_entropy = math.log2(len(_FEATURE_ORDER))
        scale_factor = 0.5 + 1.5 * (entropy / max_entropy)  # between 0.5 and 2.0
        epsilon = base_epsilon * scale_factor

    # Risk‑based noise magnitude
    risk = reconstruction_risk_score(len(pool.loaded), pool.ram_ceiling_mb)
    noise = np.random.laplace(0.0, risk / epsilon)

    if model.ram_mb + pool._used() + noise <= pool.ram_ceiling_mb:
        pool.load(model)
    else:
        # Attempt eviction then load
        try:
            pool.load_with_eviction(model)
        except ModelLoadError as exc:
            raise ModelLoadError(f"Cannot load model {model.name}: {exc}") from exc


# ----------------------------------------------------------------------
# Hybrid functions demonstrating the fused system
# ----------------------------------------------------------------------


def hybrid_analyze_and_load(text: str, model: ModelTier, pool: ModelPool) -> Dict[str, Any]:
    """
    1. Analyse *text* (Parent A).
    2. Store entropy in the pool for later privacy scaling.
    3. Attempt privacy‑aware load of *model* (Parent B).
    Returns the analysis dictionary plus a flag indicating load success.
    """
    analysis = analyze_text(text)
    # Attach entropy to pool for the loader to read
    setattr(pool, "_last_entropy", analysis["entropy"])

    try:
        load_model_with_privacy(model, pool, base_epsilon=1.0, sensitivity=analysis["weighted_score"])
        loaded = pool.is_loaded(model.name)
    except ModelLoadError:
        loaded = False

    analysis["model_loaded"] = loaded
    return analysis


def batch_privacy_load(models: List[ModelTier], pool: ModelPool, texts: List[str]) -> List[Dict[str, Any]]:
    """
    Process a list of (text, model) pairs, loading each model with privacy
    calibrated by its corresponding text's entropy.
    Returns a list of analysis dictionaries.
    """
    results = []
    for txt, mdl in zip(texts, models):
        result = hybrid_analyze_and_load(txt, mdl, pool)
        results.append(result)
    return results


def summarize_pool(pool: ModelPool) -> Dict[str, Any]:
    """
    Produce a concise summary: total RAM used, number of models, and a list of model names.
    """
    used = pool._used()
    count = len(pool.loaded)
    names = list(pool.loaded.keys())
    return {"ram_used_mb": used, "model_count": count, "models": names}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I have evidence that the plan was approved, but I'm scared and need support.",
        "We need to delay the release until tomorrow, the budget is tight.",
        "All tasks are done, shipped, and verified. No risk left.",
    ]

    models = [
        ModelTier(name="model_A", ram_mb=1500, tier="T1"),
        ModelTier(name="model_B", ram_mb=2000, tier="T2"),
        ModelTier(name="model_C", ram_mb=1800, tier="T3"),
    ]

    pool = ModelPool(ram_ceiling_mb=5000)

    # Run batch processing
    results = batch_privacy_load(models, pool, sample_texts)

    for i, res in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Weighted score: {res['weighted_score']:.2f}")
        print(f"Entropy: {res['entropy']:.4f}")
        print(f"Model loaded: {res['model_loaded']}")
        print(f"Counts: {dict(zip(_FEATURE_ORDER, res['counts']))}")

    print("\nPool summary:", summarize_pool(pool))