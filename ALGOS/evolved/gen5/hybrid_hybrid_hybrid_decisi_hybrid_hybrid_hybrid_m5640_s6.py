# DARWIN HAMMER — match 5640, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_fracti_m265_s1.py (gen4)
# born: 2026-05-30T00:03:45Z

"""Hybrid Decision-Hygiene & Hyperdimensional Morphology Fusion

Parents:
- hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s2.py (Decision hygiene, spatial‑signature filtering, privacy linear model)
- hybrid_hybrid_hdc_se_hybrid_hybrid_fracti_m265_s1.py (Hyperdimensional computing, morphology vectors, binding operator, min‑hash text encoding)

Mathematical Bridge:
The decision‑hygiene feature extraction yields a scalar *decision score* S for each textual entity.
S is used as a multiplicative scaling factor for the morphology‑derived hypervector Vₘ (produced by the
parent‑B random‑seeded generator).  The scaled vector S·Vₘ is then *bound* with a min‑hash
text hypervector Vₜ via element‑wise multiplication (the HCHDC binding operator).  The resulting
bound vector encodes both the semantic hygiene information and the physical morphology,
while a simple linear privacy budget filter selects a feasible subset of entities.

The module implements this fused pipeline and provides three core functions demonstrating the
integration.
"""

import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Decision‑hygiene regex patterns (parent A)
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
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford)\b",
    re.I,
)

_REGEX_LIST = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
    ("outcome", OUTCOME_RE),
    ("impulsive", IMPULSIVE_RE),
    ("scarcity", SCARCITY_RE),
]

# ----------------------------------------------------------------------
# Morphology dataclass (parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

# ----------------------------------------------------------------------
# Helper: deterministic random vector generator
# ----------------------------------------------------------------------
def _random_vector(dim: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.fromiter((rng.random() for _ in range(dim)), dtype=np.float64)

# ----------------------------------------------------------------------
# 1. Decision‑hygiene feature extraction
# ----------------------------------------------------------------------
def extract_decision_features(text: str) -> Dict[str, int]:
    """Count occurrences of each decision‑hygiene regex in *text*."""
    counts = {}
    for name, pattern in _REGEX_LIST:
        matches = pattern.findall(text)
        counts[name] = len(matches)
    return counts

# ----------------------------------------------------------------------
# 2. Decision score (Shannon‑entropy‑like weighting)
# ----------------------------------------------------------------------
def decision_hygiene_score(features: Dict[str, int]) -> float:
    """Convert feature counts into a scalar score S ∈ (0, ∞)."""
    total = sum(features.values())
    if total == 0:
        return 0.0
    # probability distribution over categories
    probs = np.array(list(features.values())) / total
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    # Scale by total count to give higher weight to richer texts
    return entropy * total

# ----------------------------------------------------------------------
# 3. Morphology hypervector (parent B)
# ----------------------------------------------------------------------
def morphology_hypervector(m: Morphology, dim: int = 8192) -> np.ndarray:
    """
    Produce a deterministic hypervector for a morphology.
    Seed is derived from the four physical parameters.
    """
    seed_bytes = f"{m.length:.6f}{m.width:.6f}{m.height:.6f}{m.mass:.6f}".encode()
    seed = int.from_bytes(
        np.frombuffer(seed_bytes, dtype=np.uint8)[:8].tobytes(), "big", signed=False
    )
    base_vec = _random_vector(dim, seed)

    # scaling factors (broadcast to dim)
    factors = np.array([m.length, m.width, m.height, m.mass], dtype=np.float64)
    repeat = dim // len(factors) + 1
    scaling = np.tile(factors, repeat)[:dim]
    return base_vec * scaling

# ----------------------------------------------------------------------
# 4. Min‑hash text hypervector (parent B)
# ----------------------------------------------------------------------
def _minhash_signature(tokens: List[str], dim: int, num_perm: int, seed: int) -> np.ndarray:
    """Simple min‑hash: for each permutation compute minimal hash value."""
    rng = random.Random(seed)
    # generate a list of random salts
    salts = [rng.randint(0, 2**31 - 1) for _ in range(num_perm)]
    signature = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        token_bytes = token.encode()
        for i, salt in enumerate(salts):
            h = np.uint64(hash((salt, token_bytes)))
            if h < signature[i]:
                signature[i] = h
    # Normalize to [0,1) and expand to dim via repetition
    norm = signature.astype(np.float64) / np.iinfo(np.uint64).max
    repeat = dim // num_perm + 1
    vec = np.tile(norm, repeat)[:dim]
    return vec

def text_hypervector(text: str, dim: int = 8192, num_perm: int = 128) -> np.ndarray:
    """Create a hypervector from *text* using a min‑hash sketch."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return np.zeros(dim, dtype=np.float64)
    seed = hash(text) & 0xFFFFFFFF
    return _minhash_signature(tokens, dim, num_perm, seed)

# ----------------------------------------------------------------------
# 5. Binding operator (element‑wise multiplication)
# ----------------------------------------------------------------------
def bind_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HCHDC binding: element‑wise multiplication followed by L2 normalisation."""
    bound = a * b
    norm = np.linalg.norm(bound) + 1e-12
    return bound / norm

# ----------------------------------------------------------------------
# 6. Privacy‑aware linear selection (parent A)
# ----------------------------------------------------------------------
def privacy_budget_filter(
    scores: List[float],
    costs: List[float],
    budget: float,
) -> List[int]:
    """
    Greedy selection of indices maximizing total score while respecting a linear
    privacy budget Σ cost ≤ budget.
    Returns the list of chosen indices in descending score order.
    """
    if len(scores) != len(costs):
        raise ValueError("scores and costs must have the same length")
    # compute score‑per‑cost ratio, treat zero cost as infinite ratio
    ratios = [
        (s / c if c > 0 else float("inf"), idx) for idx, (s, c) in enumerate(zip(scores, costs))
    ]
    ratios.sort(reverse=True)
    chosen = []
    total_cost = 0.0
    for _, idx in ratios:
        if total_cost + costs[idx] <= budget:
            chosen.append(idx)
            total_cost += costs[idx]
    return chosen

# ----------------------------------------------------------------------
# 7. Hybrid pipeline exposing three public functions
# ----------------------------------------------------------------------
def compute_hybrid_vector(text: str, morph: Morphology) -> np.ndarray:
    """
    Core hybrid operation:
    1. Extract decision‑hygiene features → score S.
    2. Generate morphology hypervector Vₘ and scale by S.
    3. Generate text hypervector Vₜ.
    4. Bind (S·Vₘ) ⊗ Vₜ.
    Returns the bound, L2‑normalised vector.
    """
    features = extract_decision_features(text)
    S = decision_hygiene_score(features)
    if S == 0.0:
        # fallback to pure text representation if no hygiene signal
        return text_hypervector(text)
    V_m = morphology_hypervector(morph)
    V_m_scaled = V_m * S
    V_t = text_hypervector(text)
    return bind_vectors(V_m_scaled, V_t)


def batch_hybrid_vectors(
    texts: List[str],
    morphologies: List[Morphology],
    privacy_costs: List[float],
    budget: float,
) -> List[np.ndarray]:
    """
    Process a batch, apply privacy budget filtering, and return bound vectors
    for the selected items.
    """
    if not (len(texts) == len(morphologies) == len(privacy_costs)):
        raise ValueError("All input lists must have the same length")
    # Compute raw scores for filtering
    raw_scores = [
        decision_hygiene_score(extract_decision_features(t)) for t in texts
    ]
    selected_idx = privacy_budget_filter(raw_scores, privacy_costs, budget)
    result = [
        compute_hybrid_vector(texts[i], morphologies[i]) for i in selected_idx
    ]
    return result


def similarity_between_hybrid_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity (dot product since vectors are L2‑normalised).
    """
    if v1.shape != v2.shape:
        raise ValueError("Vector shapes must match")
    return float(np.dot(v1, v2))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The plan was verified and the evidence was logged before deployment.",
        "I need support now, can't wait any longer!",
        "Massive scarcity of resources caused delay in the schedule.",
    ]
    sample_morph = [
        Morphology(2.5, 1.2, 0.8, 3.0),
        Morphology(1.0, 1.0, 1.0, 0.5),
        Morphology(5.0, 2.0, 2.0, 10.0),
    ]
    privacy_costs = [0.3, 0.5, 0.4]
    budget = 1.0

    vectors = batch_hybrid_vectors(sample_texts, sample_morph, privacy_costs, budget)
    print(f"Generated {len(vectors)} hybrid vectors within budget {budget}")

    if len(vectors) >= 2:
        sim = similarity_between_hybrid_vectors(vectors[0], vectors[1])
        print(f"Cosine similarity between first two selected vectors: {sim:.4f}")

    # Demonstrate single‑item computation
    single_vec = compute_hybrid_vector(sample_texts[0], sample_morph[0])
    print(f"Single hybrid vector norm (should be 1.0): {np.linalg.norm(single_vec):.6f}")

    sys.exit(0)