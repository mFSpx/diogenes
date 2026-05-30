# DARWIN HAMMER — match 2327, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hard_t_m845_s1.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py (gen2)
# born: 2026-05-29T23:43:15Z

"""HybridFusionAlgorithm
Combines:
- Parent A: NLMS adaptive weight update and hard‑truth compatibility s = vᵀ·P·m
- Parent B: Regex‑driven textual feature extraction with positive/negative weight scoring

Mathematical bridge:
The NLMS weight vector **w** (size 10) is split into two parts:
    v = w[:2]               – provides the high‑dimensional feature vector for the hard‑truth model
    x = w[:9] (padded)      – serves as the NLMS input vector for learning from the
                              feature‑based score produced by Parent B.
The score from Parent B, 𝜎 = f·p – f·n (where f is the raw feature count vector,
p and n are the positive/negative weight arrays), is multiplied by a scalar
α = vᵀ·P·m (compatibility) to obtain the target signal for the NLMS update:
    target = α * σ
Thus the adaptive NLMS learns to align its weight vector **w** with the
text‑driven compatibility measure, fusing both topologies into a single
learning loop.
"""

import re
import math
import random
import sys
from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Regex feature extraction (Parent B)
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

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)


def _raw_counts(text: str) -> np.ndarray:
    """Return a count vector (len 9) for the defined regex categories."""
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    regexes = [
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
    for i, reg in enumerate(regexes):
        counts[i] = len(reg.findall(text))
    return counts


def feature_score(counts: np.ndarray) -> float:
    """Compute the Parent‑B scalar score σ = f·p – f·n."""
    pos = np.dot(counts, _POSITIVE_WEIGHTS)
    neg = np.dot(counts, _NEGATIVE_WEIGHTS)
    return float(pos - neg)


# ----------------------------------------------------------------------
# NLMS + Hard‑Truth model (Parent A)
# ----------------------------------------------------------------------
@dataclass
class ModelResource:
    vector: np.ndarray          # shape (2,)
    reliability: float
    curvature: float


class HybridFusion:
    """Unified algorithm merging NLMS adaptation with regex‑driven scoring."""

    def __init__(self, dim: int = 10, mu: float = 0.5, eps: float = 1e-9):
        self.dim = dim
        self.weights = np.random.rand(dim)          # NLMS weight vector w
        self.mu = mu
        self.eps = eps
        self.audit_manifest = Counter()

    # ---------- NLMS core ----------
    def predict(self, x: np.ndarray) -> float:
        """NLMS linear prediction y = w·x."""
        return float(np.dot(self.weights, x))

    def nlms_update(self, x: np.ndarray, target: float) -> None:
        """Standard NLMS weight update."""
        y = self.predict(x)
        error = target - y
        power = np.dot(x, x) + self.eps
        self.weights += self.mu * error * x / power

    # ---------- Hard‑truth compatibility ----------
    @staticmethod
    def compatibility(v: np.ndarray, m: ModelResource) -> float:
        """
        Compute s = vᵀ·P·m where P selects the first two columns of I.
        v must have length ≥2, m.vector length ==2.
        """
        P = np.eye(len(v))[:, :2]          # shape (len(v),2)
        return float(np.dot(v.T, np.dot(P, m.vector)))

    # ---------- Hybrid operation ----------
    def hybrid_step(self, text: str, model_res: ModelResource) -> Tuple[float, float]:
        """
        Perform a full hybrid iteration:
          1. Extract regex feature counts → f
          2. Compute Parent‑B score σ
          3. Build NLMS input x from f (padded to self.dim)
          4. Derive v = self.weights[:2] for compatibility
          5. Compute α = compatibility(v, model_res)
          6. Target = α * σ
          7. NLMS weight update with (x, target)
        Returns (target, prediction_before_update)
        """
        # 1‑2
        f = _raw_counts(text)
        sigma = feature_score(f)

        # 3 – pad/truncate to dim
        x = np.zeros(self.dim)
        length = min(len(f), self.dim)
        x[:length] = f[:length]

        # 4‑5
        v = self.weights[:2]
        alpha = self.compatibility(v, model_res)

        # 6
        target = alpha * sigma

        # 7
        pred_before = self.predict(x)
        self.nlms_update(x, target)

        # bookkeeping
        self.audit_manifest.update({k: int(v) for k, v in zip(_FEATURE_ORDER, f)})

        return target, pred_before

    # ---------- Convenience wrappers ----------
    def evaluate_text(self, text: str, model_res: ModelResource) -> float:
        """Return the current compatibility‑scaled score without updating weights."""
        f = _raw_counts(text)
        sigma = feature_score(f)
        v = self.weights[:2]
        alpha = self.compatibility(v, model_res)
        return alpha * sigma

    def get_weights(self) -> np.ndarray:
        """Expose current NLMS weight vector."""
        return self.weights.copy()


# ----------------------------------------------------------------------
# Demonstration functions (require at least three)
# ----------------------------------------------------------------------
def demo_extraction() -> None:
    sample = "The evidence was verified, but we need more planning and support."
    counts = _raw_counts(sample)
    print("Feature counts:", dict(zip(_FEATURE_ORDER, counts)))
    print("Parent‑B score σ:", feature_score(counts))


def demo_compatibility() -> None:
    rng = np.random.default_rng(42)
    v = rng.random(2)
    m = ModelResource(vector=rng.random(2), reliability=0.9, curvature=0.7)
    s = HybridFusion.compatibility(v, m)
    print("Compatibility s =", s)


def demo_hybrid_loop() -> None:
    hf = HybridFusion()
    model_res = ModelResource(vector=np.array([0.6, 0.4]), reliability=0.95, curvature=0.8)
    texts = [
        "Evidence confirmed, plan ready, no delay.",
        "I feel panic and scarcity, need support now.",
        "All tasks are done, shipped, and verified.",
    ]
    for i, txt in enumerate(texts, 1):
        target, pred = hf.hybrid_step(txt, model_res)
        print(f"Iter {i}: target={target:.3f}, pred_before={pred:.3f}, w[:4]={hf.get_weights()[:4]}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Feature Extraction ===")
    demo_extraction()
    print("\n=== Demo: Compatibility Computation ===")
    demo_compatibility()
    print("\n=== Demo: Hybrid Learning Loop ===")
    demo_hybrid_loop()
    print("\nFinal NLMS weights:", HybridFusion().get_weights())