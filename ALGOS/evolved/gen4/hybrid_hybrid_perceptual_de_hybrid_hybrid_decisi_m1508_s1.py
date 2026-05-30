# DARWIN HAMMER — match 1508, survivor 1
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s5.py (gen3)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s3.py (gen2)
# born: 2026-05-29T23:38:28Z

"""Hybrid Perceptual‑RBF Decision Engine.

Parents:
- hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s5.py (RBF surrogate model)
- hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s3.py (regex‑based feature extraction with
  positive/negative weight vectors)

Mathematical bridge:
The regex engine converts raw text into a numeric feature vector **f** ∈ ℝ⁹
(counts of nine semantic categories).  The RBF surrogate treats each **f**
as a centre in a radial‑basis space and learns a linear combination of Gaussian
kernels that maps **f** → a decision score **y**.  By solving the linear system
K w = y (K_ij = exp(−‖f_i−f_j‖² ε²)), we fuse the discrete lexical scoring of
parent B with the continuous interpolation power of parent A, yielding a unified
predictor that respects both handcrafted weights and data‑driven smoothness.

The module implements:
1. feature extraction (regex counts → vector)
2. RBF surrogate training on those vectors
3. hybrid prediction that combines the surrogate output with the original
   handcrafted weighted sum.
"""

import math
import re
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Radial Basis Function utilities
# ----------------------------------------------------------------------
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve linear system A·x = b using Gauss‑Jordan elimination."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # pivot selection
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    """Radial basis function surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Predict a scalar value for input vector x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Parent B – Regex feature extraction and handcrafted scoring
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

def _raw_counts(text: str) -> List[int]:
    """Return raw occurrence counts for each feature in order."""
    return [
        len(_REGEX_MAP[name].findall(text or ""))
        for name in _FEATURE_ORDER
    ]

def handcrafted_score(text: str) -> float:
    """Linear combination of counts using positive/negative weight vectors."""
    counts = np.array(_raw_counts(text), dtype=np.int64)
    return float(counts @ _POSITIVE_WEIGHTS - counts @ _NEGATIVE_WEIGHTS)

# ----------------------------------------------------------------------
# Hybrid layer – combine feature vectors with an RBF surrogate
# ----------------------------------------------------------------------
def extract_feature_vector(text: str) -> Tuple[float, ...]:
    """Convert raw text into a 9‑dimensional feature tuple."""
    return tuple(_raw_counts(text))

def train_rbf_surrogate(
    training_data: List[Tuple[str, float]],
    epsilon: float = 1.0,
) -> RBFSurrogate:
    """
    Train an RBF surrogate on (text, target) pairs.

    The feature vectors become RBF centres.  The linear system
    K·w = y is solved for the weights w, where
        K_ij = exp(−ε²‖f_i−f_j‖²).
    """
    if not training_data:
        raise ValueError("training_data must contain at least one sample")
    centers = [extract_feature_vector(txt) for txt, _ in training_data]
    targets = [y for _, y in training_data]

    # Build kernel matrix K
    n = len(centers)
    K = [[gaussian(euclidean(centers[i], centers[j]), epsilon) for j in range(n)] for i in range(n)]

    # Solve for weights
    weights = solve_linear(K, targets)
    return RBFSurrogate(centers=centers, weights=weights, epsilon=epsilon)

def hybrid_predict(text: str, surrogate: RBFSurrogate) -> float:
    """
    Produce a decision score that blends:
    1. The handcrafted score (direct linear weighting of counts).
    2. The surrogate's smooth interpolation over the same feature space.

    The final score is a weighted sum: 0.6·handcrafted + 0.4·surrogate.
    """
    vec = extract_feature_vector(text)
    h_score = handcrafted_score(text)
    s_score = surrogate.predict(vec)
    return 0.6 * h_score + 0.4 * s_score

def batch_hybrid_scores(
    texts: Iterable[str],
    surrogate: RBFSurrogate,
) -> List[float]:
    """Compute hybrid scores for an iterable of texts."""
    return [hybrid_predict(t, surrogate) for t in texts]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic training set: texts with an intuitive target.
    samples = [
        ("I have evidence that the plan succeeded and the outcome is good.", 3000.0),
        ("I am scared, I might kill myself. No support, just panic.", -4000.0),
        ("We need to schedule the next steps, the deadline is tomorrow.", 1200.0),
        ("I am broke and can't afford rent, feeling desperate.", -2500.0),
        ("The document is verified, the bug is fixed, everything is green.", 2500.0),
    ]

    # Train surrogate
    surrogate = train_rbf_surrogate(samples, epsilon=0.5)

    # Test on new sentences
    test_texts = [
        "I have proof and the issue is resolved, all good.",
        "I'm feeling hopeless, no money, thinking about ending it.",
        "Let's plan the roadmap and set a timeline, no delays.",
    ]

    for txt in test_texts:
        score = hybrid_predict(txt, surrogate)
        print(f"Text: {txt!r}\nHybrid Score: {score:.2f}\n")