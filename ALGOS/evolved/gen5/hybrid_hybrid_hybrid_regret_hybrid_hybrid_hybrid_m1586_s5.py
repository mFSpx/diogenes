# DARWIN HAMMER — match 1586, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1091_s3.py (gen4)
# born: 2026-05-29T23:37:37Z

"""Hybrid Regret‑Weighted Ternary‑Decision & Edge‑Weighted Hygiene Analyzer.

Parents:
- hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s3.py (Regret‑Weighted
  ternary decision space)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1091_s3.py (Edge‑length
  expected values weighting of hygiene feature counts)

Mathematical bridge:
The Regret‑Weighted engine produces a signed ternary decision vector **d**
∈ {‑1,0,1}ⁿ whose magnitude is modulated by a regret‑based sigmoid weight **w**
∈ [0,1]ⁿ, giving a continuous decision vector **r = w ∘ d**.
The hygiene analyzer extracts a discrete feature‑count vector **c** ∈ ℕᵐ from
text and weights each component by the expected edge length **e** ∈ ℝ₊ᵐ,
producing **h = e ∘ c**.

The hybrid system fuses the two by projecting **r** onto the hygiene space:
the final hybrid score is the (scaled) inner product  

    S = (h · r) / (‖h‖·‖r‖ + ε)

which is a cosine‑like similarity that respects both regret‑weighted decisions
and edge‑weighted hygiene evidence.

The implementation below provides three public functions demonstrating the
hybrid operation together with the original utility routines (signature,
similarity, sigmoid)."""

import math
import random
import sys
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Regret‑Weighted ternary decision utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Atomic action with expected value, cost and risk."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Min‑hash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two min‑hash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Parent B – Hygiene feature extraction & edge‑length expectations
# ----------------------------------------------------------------------


# Regexes for the seven hygiene categories used by the original parent B
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

FEATURE_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
    ("outcome", OUTCOME_RE),
    ("impulsive", IMPULSIVE_RE),
]


def extract_feature_counts(text: str) -> np.ndarray:
    """Return a count vector (len = 7) for the hygiene categories."""
    counts = []
    for _, regex in FEATURE_REGEXES:
        counts.append(len(regex.findall(text)))
    return np.array(counts, dtype=float)


def expected_edge_weights(num_features: int = 7, seed: int = 0) -> np.ndarray:
    """
    Simulate expected edge lengths for each feature.
    In a real system these would come from a posterior belief; here we use
    a deterministic pseudo‑random generator for reproducibility.
    """
    rng = random.Random(seed)
    # Base length 1.0 plus a small random perturbation
    return np.array([1.0 + rng.gauss(0, 0.1) for _ in range(num_features)], dtype=float)


# ----------------------------------------------------------------------
# Hybrid core – mathematical fusion of the two parents
# ----------------------------------------------------------------------


def ternary_decision(action: MathAction, margin: float = 0.0) -> int:
    """
    Map an action to a ternary decision:
        -1  if (EV - cost - risk) < -margin
         0  if |EV - cost - risk| ≤ margin
        +1  otherwise
    """
    score = action.expected_value - action.cost - action.risk
    if score < -margin:
        return -1
    if score > margin:
        return 1
    return 0


def regret_weighted_vector(actions: List[MathAction]) -> np.ndarray:
    """
    Build the Regret‑Weighted decision vector **r**.
    Steps:
        1. Compute regrets = max(EV) - EV for each action.
        2. Convert regrets to sigmoid weights w ∈ (0,1).
        3. Obtain ternary decisions d ∈ {‑1,0,1}.
        4. Return r = w ∘ d as a float numpy array.
    """
    if not actions:
        raise ValueError("actions list cannot be empty")
    evs = np.array([a.expected_value for a in actions], dtype=float)
    max_ev = evs.max()
    regrets = max_ev - evs
    w = sigmoid(regrets)  # higher regret → lower weight
    d = np.array([ternary_decision(a) for a in actions], dtype=float)
    return w * d


def hybrid_score(text: str, actions: List[MathAction]) -> float:
    """
    Compute the hybrid similarity score S between a text's hygiene evidence
    and a set of actions.

    1. Extract raw feature counts c from the text.
    2. Weight them by expected edge lengths e → h = e ∘ c.
    3. Build the regret‑weighted decision vector r from actions.
    4. Return the cosine‑like similarity between h and r.

    The function is fully deterministic given the same random seed used in
    `expected_edge_weights`.
    """
    # 1‑2. Hygiene side
    c = extract_feature_counts(text)                     # shape (7,)
    e = expected_edge_weights(num_features=c.size)      # shape (7,)
    h = e * c                                            # weighted hygiene vector

    # 3. Decision side
    r = regret_weighted_vector(actions)                 # shape (n_actions,)

    # Align dimensions: if the number of actions differs from 7 we project
    # r onto the 7‑dimensional space by averaging (a simple bridge).
    if r.size != h.size:
        # Simple projection: repeat or truncate to match length
        if r.size < h.size:
            repeats = int(math.ceil(h.size / r.size))
            r_proj = np.tile(r, repeats)[: h.size]
        else:
            r_proj = r[: h.size]
    else:
        r_proj = r

    # 4. Cosine‑like similarity (adds epsilon to avoid division by zero)
    eps = 1e-9
    numerator = np.dot(h, r_proj)
    denominator = (np.linalg.norm(h) * np.linalg.norm(r_proj)) + eps
    return numerator / denominator


# ----------------------------------------------------------------------
# Additional utility demonstrating the bridge
# ----------------------------------------------------------------------


def hybrid_signature_similarity(text_a: str, text_b: str, actions_a: List[MathAction],
                                actions_b: List[MathAction]) -> float:
    """
    Combine min‑hash signature similarity of two texts with the hybrid
    decision similarity. The final metric is the product of the two
    similarities, emphasizing agreement on both lexical and decision‑level
    dimensions.
    """
    sig_a = signature(text_a.split())
    sig_b = signature(text_b.split())
    lexical_sim = similarity(sig_a, sig_b)

    decision_sim = hybrid_score(text_a, actions_a) * hybrid_score(text_b, actions_b)
    # Normalize decision_sim to [0,1] via sigmoid to keep the product bounded
    decision_sim_norm = sigmoid(np.array([decision_sim]))[0]

    return lexical_sim * decision_sim_norm


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We have verified the source and attached the hash. The plan is to "
        "release the update tomorrow after a short pause. If any issues arise, "
        "call the support team. Remember the boundary conditions and keep the "
        "data safe."
    )

    actions = [
        MathAction(id="A1", expected_value=0.8, cost=0.1, risk=0.05),
        MathAction(id="A2", expected_value=0.4, cost=0.2, risk=0.1),
        MathAction(id="A3", expected_value=0.6, cost=0.05, risk=0.2),
    ]

    # Compute hybrid score for the sample
    score = hybrid_score(sample_text, actions)
    print(f"Hybrid score: {score:.4f}")

    # Demonstrate combined signature similarity
    text2 = "The evidence was recorded and the checklist is ready. Wait for approval."
    actions2 = [
        MathAction(id="B1", expected_value=0.7, cost=0.15, risk=0.05),
        MathAction(id="B2", expected_value=0.5, cost=0.1, risk=0.1),
        MathAction(id="B3", expected_value=0.3, cost=0.2, risk=0.15),
    ]
    combined_sim = hybrid_signature_similarity(sample_text, text2, actions, actions2)
    print(f"Combined lexical‑decision similarity: {combined_sim:.4f}")