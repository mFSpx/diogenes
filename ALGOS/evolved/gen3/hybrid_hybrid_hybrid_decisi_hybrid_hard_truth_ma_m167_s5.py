# DARWIN HAMMER — match 167, survivor 5
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py (gen2)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s1.py (gen2)
# born: 2026-05-29T23:27:18Z

"""Hybrid Decision Hygiene, Shannon Entropy, Stylometry & Minimum‑Cost Tree.

Parents:
- hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py (regex‑based hygiene + Shannon entropy)
- hybrid_hard_truth_math_hybrid_minimum_cost__m12_s1.py (stylometry + Bayesian minimum‑cost tree)

Mathematical bridge:
The hygiene regex count vector **c** (length 8) is turned into a scalar scaling factor
    s = (‖c‖₁ + 1) / (len(c) + 1) .
The stylometry frequency vector **f** (length N) is transformed into an *expected* vector
    **f̂** = **f** ⊙ **p** ,
where **p** is the posterior edge‑belief probability vector (length N) obtained from the
minimum‑cost tree Bayes update.  
The Shannon entropy **H** of the token distribution of a text is normalised by its
maximum possible entropy **Hₘₐₓ** for the given vocabulary size, yielding a factor
    h = H / Hₘₐₓ ∈ [0, 1].

The final hybrid score for a pair of texts (t₁, t₂) and a tree **T** is

    score = s₁ · s₂ · h₁ · h₂ · (f̂₁·f̂₂) · C(T) ,

where **sᵢ** is the hygiene scaling of text *i*, **hᵢ** its normalised entropy,
**f̂ᵢ** the expected stylometry vector, “·” denotes the dot product and **C(T)**
is the expected minimum‑cost of the tree, i.e. the sum over edges of
    E[length] = Σₑ pₑ·ℓₑ
weighted by node‑beliefs derived from incident edge posteriors.

The implementation below follows this formulation and provides three public
functions that illustrate the hybrid operation."""


import math
import random
import sys
from pathlib import Path
from collections import Counter
import re
import numpy as np
from typing import Dict, List, Tuple, Iterable

# ----------------------------------------------------------------------
# Parent A – hygiene regexes and entropy utilities
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

HYGIENE_REGEXES = [
    EVIDENCE_RE,
    PLANNING_RE,
    DELAY_RE,
    SUPPORT_RE,
    BOUNDARY_RE,
    OUTCOME_RE,
    IMPULSIVE_RE,
    SCARCITY_RE,
]

def extract_hygiene_counts(text: str) -> List[int]:
    """Return a list of hit counts for each hygiene regex."""
    return [len(regex.findall(text)) for regex in HYGIENE_REGEXES]

def hygiene_scaling_factor(counts: List[int]) -> float:
    """Scalar s = (sum(counts)+1)/(len(counts)+1) used to weight stylometry."""
    return (sum(counts) + 1) / (len(counts) + 1)

def shannon_entropy(tokens: List[str]) -> Tuple[float, float]:
    """Return (entropy, max_entropy) for a token list."""
    if not tokens:
        return 0.0, 0.0
    freq = Counter(tokens)
    total = len(tokens)
    probs = np.array([c / total for c in freq.values()], dtype=float)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 0.0
    return entropy, max_entropy

def normalized_entropy(text: str) -> float:
    """Normalised Shannon entropy h = H / Hmax ∈ [0,1]."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    H, Hmax = shannon_entropy(tokens)
    return H / Hmax if Hmax > 0 else 0.0

# ----------------------------------------------------------------------
# Parent B – stylometry categories and Bayesian tree utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most".split()),
    # Additional categories can be added here.
}

CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)

def stylometry_vector(text: str) -> np.ndarray:
    """Frequency vector of function‑word categories (length = NUM_CATS)."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    total = len(tokens) if tokens else 1
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        count = sum(1 for tok in tokens if tok in cat_words)
        vec[idx] = count / total
    return vec

# Simple Bayesian edge belief model ------------------------------------
Edge = Tuple[str, str]
Point = Tuple[float, float]

def random_edge_beliefs(edges: List[Edge]) -> np.ndarray:
    """Draw a Dirichlet‑distributed probability vector over edges."""
    alpha = np.ones(len(edges))
    return np.random.dirichlet(alpha)

def expected_edge_lengths(edges: List[Edge],
                         positions: Dict[str, Point],
                         beliefs: np.ndarray) -> np.ndarray:
    """Return E[length] for each edge given posterior beliefs."""
    lengths = np.array([
        math.hypot(positions[e[0]][0] - positions[e[1]][0],
                   positions[e[0]][1] - positions[e[1]][1])
        for e in edges
    ], dtype=float)
    return beliefs * lengths  # element‑wise product gives expected contribution

def node_belief_weights(edges: List[Edge],
                        beliefs: np.ndarray,
                        nodes: List[str]) -> Dict[str, float]:
    """Aggregate incident edge beliefs to each node."""
    weights = {n: 0.0 for n in nodes}
    for (e, b) in zip(edges, beliefs):
        a, b_node = e
        weights[a] += b
        weights[b_node] += b
    # Normalise so that sum(weights)=1
    total = sum(weights.values())
    if total > 0:
        for n in weights:
            weights[n] /= total
    return weights

def expected_tree_cost(edges: List[Edge],
                       positions: Dict[str, Point],
                       beliefs: np.ndarray) -> float:
    """Σₑ pₑ·ℓₑ, the expected minimum‑cost of the tree."""
    exp_lengths = expected_edge_lengths(edges, positions, beliefs)
    return float(np.sum(exp_lengths))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_expected_stylometry(text: str,
                               edge_beliefs: np.ndarray) -> np.ndarray:
    """
    Transform the raw stylometry vector f into an expected vector f̂ = f ⊙ p,
    where p are the edge‑belief probabilities (broadcast to length NUM_CATS).
    If the belief vector is shorter, it is tiled; if longer, it is truncated.
    """
    f = stylometry_vector(text)
    p = edge_beliefs
    if len(p) < NUM_CATS:
        # Tile to match size
        repeats = math.ceil(NUM_CATS / len(p))
        p = np.tile(p, repeats)[:NUM_CATS]
    elif len(p) > NUM_CATS:
        p = p[:NUM_CATS]
    return f * p

def hybrid_hygiene_entropy_factor(text: str) -> float:
    """
    Combine hygiene scaling s and normalised entropy h into a single factor:
        g = s * h
    """
    counts = extract_hygiene_counts(text)
    s = hygiene_scaling_factor(counts)
    h = normalized_entropy(text)
    return s * h

def hybrid_score(text_a: str,
                 text_b: str,
                 edges: List[Edge],
                 positions: Dict[str, Point]) -> float:
    """
    Full hybrid score for two texts and a tree.
    Steps:
    1. Sample posterior edge beliefs (Dirichlet).
    2. Compute expected stylometry vectors f̂₁, f̂₂.
    3. Compute dot product d = f̂₁·f̂₂.
    4. Compute hygiene‑entropy factors g₁, g₂.
    5. Compute expected tree cost C.
    6. Return score = g₁·g₂·d·C.
    """
    # 1. Edge beliefs
    beliefs = random_edge_beliefs(edges)

    # 2. Expected stylometry vectors
    fhat_a = hybrid_expected_stylometry(text_a, beliefs)
    fhat_b = hybrid_expected_stylometry(text_b, beliefs)

    # 3. Dot product (similarity)
    d = float(np.dot(fhat_a, fhat_b))

    # 4. Hygiene‑entropy factors
    g_a = hybrid_hygiene_entropy_factor(text_a)
    g_b = hybrid_hygiene_entropy_factor(text_b)

    # 5. Expected tree cost
    C = expected_tree_cost(edges, positions, beliefs)

    # 6. Combine
    return g_a * g_b * d * C

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text1 = (
        "I verified the source and recorded the log. "
        "The plan is to schedule the next test tomorrow. "
        "Please ask a friend for support if needed."
    )
    sample_text2 = (
        "The document was confirmed and the audit is complete. "
        "We will prioritize the roadmap and allocate budget. "
        "If you feel unsafe, contact a therapist."
    )

    # Simple tree: three nodes forming a line
    positions = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (2.0, 0.0),
    }
    edges = [("A", "B"), ("B", "C")]

    score = hybrid_score(sample_text1, sample_text2, edges, positions)
    print(f"Hybrid score: {score:.6f}")