# DARWIN HAMMER — match 1474, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_liquid_m39_s2.py (gen3)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s2.py (gen2)
# born: 2026-05-29T23:36:49Z

"""
Hybrid Decision–Diffusion–Fractional HDC Algorithm

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_liquid_m39_s2.py (regex‑based decision system +
  Liquid‑Time‑Constant (LTC) cell with similarity term `s ∈ [0,1]`).
- hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s2.py (Fractional Hyperdimensional
  Computing (HDC) + Hoeffding‑Gini decision making).

Mathematical Bridge
-------------------
The similarity term `s` produced by the LTC cell is used in two places:

1. **Weight Modulation** – The decision‑matrix weights of the regex system are scaled by
   `α = s` before being applied to the feature counts.  This creates a feedback loop where
   the current input similarity influences the importance of each decision feature.

2. **Fractional Exponent & Hoeffding Scale** – The same `α` is employed as the fractional
   exponent in the HDC binding (`fractional_power`) and also as the multiplicative factor
   `r = α` inside the Hoeffding bound `sqrt( (r²·ln(1/δ)) / (2n) )`.  Consequently the
   uncertainty estimate adapts to the diffusion similarity.

The three core functions below demonstrate this fused topology:
`compute_similarity`, `ltc_step`, and `hybrid_decision`.  Additional helper functions
support hypervector generation, binding, and Hoeffding‑based split decisions.
"""

import re
import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Regex feature definitions (Parent A)
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
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|work)\b",
    re.I,
)

REGEX_PATTERNS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
}

# ----------------------------------------------------------------------
# Hyperdimensional Computing utilities (Parent B)
# ----------------------------------------------------------------------
def random_hv(d: int = 4096, kind: str = "bipolar", seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")

def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (binding)."""
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))

def fractional_power(X: np.ndarray, alpha: float) -> np.ndarray:
    """Element‑wise fractional power preserving sign."""
    return np.abs(X) ** alpha * np.sign(X)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound with `r` as the range (here supplied by similarity)."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    cum = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cum / (n * sum(xs))

# ----------------------------------------------------------------------
# LTC‑style similarity (Parent A) ------------------------------------
# ----------------------------------------------------------------------
def _minhash_signature(tokens: List[str], num_perm: int = 64) -> np.ndarray:
    """Very lightweight MinHash‑like signature: for each permutation a hash
    of the concatenated token string is taken and the minimum over tokens
    is stored.  Returns an integer array of length `num_perm`."""
    rng = np.random.default_rng(42)
    coeffs = rng.integers(1, 2**31 - 1, size=num_perm, dtype=np.int64)
    perms = rng.integers(0, 2**31 - 1, size=num_perm, dtype=np.int64)

    sig = np.full(num_perm, 2**63 - 1, dtype=np.int64)
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        combined = (coeffs * h + perms) % (2**31 - 1)
        sig = np.minimum(sig, combined)
    return sig

def compute_similarity(input_text: str, reference_text: str) -> float:
    """
    Compute a Jaccard‑like similarity `s ∈ [0,1]` between two texts using
    MinHash signatures.  This value will be used as the bridge between the
    LTC cell and the fractional HDC/Hoefding components.
    """
    import hashlib  # local import to keep top‑level namespace tidy

    tokens_a = re.findall(r"\w+", input_text.lower())
    tokens_b = re.findall(r"\w+", reference_text.lower())
    if not tokens_a or not tokens_b:
        return 0.0

    sig_a = _minhash_signature(tokens_a)
    sig_b = _minhash_signature(tokens_b)
    matches = np.sum(sig_a == sig_b)
    return matches / len(sig_a)

def ltc_step(state: np.ndarray, input_vec: np.ndarray, similarity: float, T: int = 10) -> np.ndarray:
    """
    Perform one diffusion step of a Liquid‑Time‑Constant cell.
    The diffusion horizon `t` is derived from similarity:
        t = round((1 - s) * T)
    The new state is a simple linear interpolation towards the input.
    """
    if state.shape != input_vec.shape:
        raise ValueError("state and input_vec must share shape")
    t = int(round((1.0 - similarity) * T))
    alpha = t / max(T, 1)  # fraction of diffusion applied
    return (1 - alpha) * state + alpha * input_vec

# ----------------------------------------------------------------------
# Hybrid Decision Engine
# ----------------------------------------------------------------------
def extract_regex_features(text: str) -> Dict[str, int]:
    """
    Count matches for each regex category defined in `REGEX_PATTERNS`.
    Returns a dict mapping category → count.
    """
    counts = {}
    for name, pattern in REGEX_PATTERNS.items():
        counts[name] = len(pattern.findall(text))
    return counts

def build_weight_matrix(similarity: float) -> np.ndarray:
    """
    Base weight matrix (categories × hypervector dimensions) is random.
    It is then scaled by the similarity term `α = similarity` to modulate
    importance of each feature according to the LTC feedback.
    """
    dim = 4096
    rng = np.random.default_rng(1234)
    base = rng.normal(loc=0.0, scale=1.0, size=(len(REGEX_PATTERNS), dim))
    return similarity * base

def hybrid_decision(text: str, reference: str, delta: float = 0.05) -> Tuple[bool, float]:
    """
    End‑to‑end hybrid operation:

    1. Compute similarity `s` between `text` and a reference corpus.
    2. Run one LTC diffusion step on a latent state (initially zero).
    3. Convert regex feature counts into a hypervector via binding with a random
       hypervector per category, using `fractional_power` with exponent `α = s`.
    4. Apply the similarity‑scaled weight matrix to the hypervector and obtain a
       scalar decision score.
    5. Compute a Hoeffding bound using `r = s` and decide whether to “split”
       (i.e., take action) based on whether the score exceeds the bound.

    Returns (should_split, decision_score).
    """
    # 1. similarity
    s = compute_similarity(text, reference)

    # 2. LTC diffusion (state starts at zero)
    dim = 4096
    state = np.zeros(dim)
    # Input to LTC: concatenate regex counts as a float vector
    feats = extract_regex_features(text)
    input_vec = np.array(list(feats.values()), dtype=float)
    # Pad/crop to match dimensionality
    if input_vec.size < dim:
        input_vec = np.pad(input_vec, (0, dim - input_vec.size))
    else:
        input_vec = input_vec[:dim]
    state = ltc_step(state, input_vec, s)

    # 3. Hypervector construction
    hv = np.zeros(dim, dtype=complex)
    rng = np.random.default_rng(2026)
    for i, (cat, count) in enumerate(feats.items()):
        cat_hv = random_hv(d=dim, kind="complex", seed=rng.integers(0, 1 << 30))
        bound_hv = bind(cat_hv, np.full(dim, count, dtype=complex))
        hv += bound_hv
    hv = fractional_power(hv, s)  # α = similarity

    # 4. Weighted decision score
    W = build_weight_matrix(s)  # shape (categories, dim)
    # Collapse W across categories (mean) to obtain a single vector
    w_vec = W.mean(axis=0)
    decision_score = np.real(np.vdot(w_vec, hv))  # real part as scalar

    # 5. Hoeffding bound based split decision
    n = max(1, sum(feats.values()))  # sample size proxy
    bound = hoeffding_bound(r=s, delta=delta, n=n)
    should_split = decision_score > bound
    return should_split, decision_score

# ----------------------------------------------------------------------
# Additional demonstration utilities
# ----------------------------------------------------------------------
def fractional_hv_demo(text: str, reference: str) -> np.ndarray:
    """
    Produce a fractional hypervector for `text` using similarity `s`
    as the exponent.  Useful for visual inspection or downstream HDC
    operations.
    """
    s = compute_similarity(text, reference)
    feats = extract_regex_features(text)
    dim = 4096
    hv = np.zeros(dim, dtype=complex)
    rng = np.random.default_rng(777)
    for cat, count in feats.items():
        cat_hv = random_hv(d=dim, kind="complex", seed=rng.integers(0, 1 << 30))
        hv += bind(cat_hv, np.full(dim, count, dtype=complex))
    return fractional_power(hv, s)

def gini_split_assessment(scores: List[float]) -> Tuple[bool, float]:
    """
    Given a list of decision scores, compute the Gini coefficient.
    If the coefficient exceeds 0.4 we deem the distribution sufficiently
    heterogeneous to warrant a split.
    """
    g = gini_coefficient(scores)
    return g > 0.4, g

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We need to verify the source and provide a screenshot of the log. "
        "The plan is to schedule the next steps tomorrow, but we might have to "
        "delay due to resource constraints."
    )
    reference_corpus = (
        "Please confirm the evidence, attach the hash, and record the outcome. "
        "A clear roadmap and timeline are essential."
    )

    split, score = hybrid_decision(sample_text, reference_corpus)
    print(f"Similarity‑driven split decision: {split} (score = {score:.4f})")

    hv = fractional_hv_demo(sample_text, reference_corpus)
    print(f"Fractional hypervector norm: {np.linalg.norm(hv):.4f}")

    # Gini assessment on a dummy score list
    dummy_scores = [score, score * 0.8, score * 1.2, score * 0.5]
    need_split, gini_val = gini_split_assessment(dummy_scores)
    print(f"Gini coefficient = {gini_val:.4f} → split needed: {need_split}")