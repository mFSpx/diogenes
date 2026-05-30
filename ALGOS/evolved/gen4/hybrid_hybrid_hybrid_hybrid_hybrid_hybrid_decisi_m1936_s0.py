# DARWIN HAMMER — match 1936, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_bandit_router_m111_s2.py (gen3)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s5.py (gen2)
# born: 2026-05-29T23:39:52Z

"""Hybrid Decision-Bandit Algorithm
Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_bandit_router_m111_s2.py (entropy, Count‑Min sketch, bandit)
- hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s5.py (regex feature set, positive weights)

Mathematical Bridge:
Both parents operate on discrete feature counts extracted from text.  
Parent A computes Shannon entropy  H = ‑∑pᵢ log pᵢ and approximates a log‑likelihood
∑log(1+ĉᵢ) via a Count‑Min sketch (ĉᵢ ≈ true count).  
Parent B provides a deterministic weighted score S = w·c using a fixed positive‑weight
vector.  

The hybrid fuses these by defining a composite utility  

    U = S · (1 + H)  

which couples deterministic importance (S) with uncertainty (H).  
The bandit module then uses U as the observed reward; action selection follows the
Upper‑Confidence‑Bound (UCB) rule where the exploration term is proportional to √(H).

The code below implements:
1. Feature extraction via the shared regex set.
2. Entropy, weighted score, and Count‑Min sketch log‑likelihood.
3. A multi‑armed bandit that consumes the composite utility.
"""

import re
import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------------
# Shared regex feature set (identical to parents)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
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

# Positive weights from parent B (negative weights are zero for simplicity)
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)

# ----------------------------------------------------------------------
# 1. Feature extraction
# ----------------------------------------------------------------------
def extract_feature_counts(text: str) -> Counter:
    """Return a Counter mapping each feature name to its occurrence count in *text*."""
    counts = Counter()
    mapping = {
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
    for name, regex in mapping.items():
        matches = regex.findall(text)
        if matches:
            counts[name] = len(matches)
    return counts

# ----------------------------------------------------------------------
# 2. Entropy and weighted score
# ----------------------------------------------------------------------
def shannon_entropy(counts: Counter) -> float:
    """Compute Shannon entropy of the normalized feature distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p, 2)
    return entropy

def weighted_score(counts: Counter, weights: np.ndarray = _POSITIVE_WEIGHTS) -> float:
    """Deterministic linear score S = w·c using the fixed positive weight vector."""
    vector = np.array([counts.get(name, 0) for name in _FEATURE_ORDER], dtype=np.float64)
    return float(weights @ vector)

# ----------------------------------------------------------------------
# 3. Count‑Min Sketch for log‑likelihood approximation
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min Sketch with additive updates."""
    def __init__(self, depth: int = 5, width: int = 2000, seed: int = 0):
        self.depth = depth
        self.width = width
        rng = np.random.default_rng(seed)
        # hash functions: a·x + b (mod prime) then mod width
        self.prime = 2 ** 31 - 1
        self.a = rng.integers(1, self.prime, size=depth, dtype=np.int64)
        self.b = rng.integers(0, self.prime, size=depth, dtype=np.int64)
        self.table = np.zeros((depth, width), dtype=np.int64)

    def _hash(self, item: str, i: int) -> int:
        h = (self.a[i] * hash(item) + self.b[i]) % self.prime
        return h % self.width

    def update(self, item: str, increment: int = 1):
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += increment

    def estimate(self, item: str) -> int:
        """Return the minimum count across hash rows (over‑estimate of true count)."""
        estimates = [
            self.table[i, self._hash(item, i)] for i in range(self.depth)
        ]
        return min(estimates)

def sketch_log_likelihood(sketch: CountMinSketch, counts: Counter) -> float:
    """
    Approximate ∑ log(1 + ĉᵢ) where ĉᵢ is the sketch estimate for feature i.
    This mirrors the empirical log‑likelihood used in parent A.
    """
    total = 0.0
    for name, true_c in counts.items():
        est = sketch.estimate(name)
        total += math.log(1 + est)
    return total

# ----------------------------------------------------------------------
# 4. Multi‑armed bandit with UCB that incorporates entropy as exploration
# ----------------------------------------------------------------------
@dataclass
class UCBBandit:
    n_arms: int
    total_counts: int = 0
    pulls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __post_init__(self):
        self.pulls = np.zeros(self.n_arms, dtype=np.int64)
        self.values = np.zeros(self.n_arms, dtype=np.float64)

    def select_arm(self, entropy: float) -> int:
        """
        Upper‑Confidence‑Bound selection.
        Exploration term is scaled by sqrt(entropy) to tie uncertainty to the
        entropy computed from the current text.
        """
        if 0 in self.pulls:
            # ensure each arm is tried at least once
            return int(np.argmin(self.pulls))
        avg_reward = self.values / self.pulls
        confidence = np.sqrt(2 * math.log(self.total_counts) / self.pulls)
        # Inject entropy as additional scaling
        exploration = confidence * math.sqrt(entropy + 1e-9)
        ucb = avg_reward + exploration
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        """Incorporate observed *reward* for *arm*."""
        self.pulls[arm] += 1
        self.values[arm] += reward
        self.total_counts += 1

# ----------------------------------------------------------------------
# 5. Hybrid operation
# ----------------------------------------------------------------------
def hybrid_decision(text: str, bandit: UCBBandit, sketch: CountMinSketch) -> Tuple[int, dict]:
    """
    Process *text* through the fused pipeline and return the selected arm.
    The returned dictionary contains intermediate diagnostics.
    """
    # Feature extraction
    counts = extract_feature_counts(text)

    # Entropy and deterministic score
    H = shannon_entropy(counts)
    S = weighted_score(counts)

    # Composite utility (bridge)
    U = S * (1.0 + H)

    # Update sketch and compute log‑likelihood approximation
    for name, c in counts.items():
        sketch.update(name, c)
    loglik = sketch_log_likelihood(sketch, counts)

    # Final reward combines composite utility and log‑likelihood
    reward = U + loglik

    # Bandit arm selection
    arm = bandit.select_arm(H)
    bandit.update(arm, reward)

    diagnostics = {
        "counts": dict(counts),
        "entropy": H,
        "weighted_score": S,
        "composite_utility": U,
        "log_likelihood_est": loglik,
        "reward": reward,
    }
    return arm, diagnostics

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I have evidence that the plan failed because of a delay. Need support and a new roadmap.",
        "Urgent! I can't afford rent, I'm broke and need immediate help. No money left.",
        "The project is done, shipped, and verified. All outcomes are green.",
        "I feel panic, rage, and want to destroy everything right now.",
        "Please check the audit log and source documents for verification."
    ]

    bandit = UCBBandit(n_arms=3)
    sketch = CountMinSketch(depth=4, width=1024, seed=42)

    for idx, txt in enumerate(sample_texts, 1):
        arm, info = hybrid_decision(txt, bandit, sketch)
        print(f"--- Sample {idx} ---")
        print(f"Selected arm: {arm}")
        for key, val in info.items():
            print(f"{key}: {val}")
        print()
    sys.exit(0)